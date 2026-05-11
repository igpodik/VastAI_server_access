#!/usr/bin/env python3
"""
AvitoTech ML CUP 2026 — retrieval + ranking baseline (Polars + CatBoost).

Memory discipline:
  - scan_parquet / LazyFrame, collect(streaming=True) on heavy steps
  - gc.collect() after large collects
  - POLARS_MAX_THREADS capped for 32-core hosts

Layout: data/train_data/*.parquet, data/item_features.parquet,
        data/eval_users.csv, data/eval_user_events.pq, data/contact_eids.csv
"""

import argparse
import datetime as dt
import gc
import logging
import os
import random
import resource
import sys
from pathlib import Path
from typing import Iterable

import polars as pl

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[misc, assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUTOFF_UTC = dt.datetime(2026, 4, 15, 0, 0, 0, tzinfo=dt.timezone.utc)
CUTOFF_MS = int(CUTOFF_UTC.timestamp() * 1000)

EVAL_VERTICAL_IDS = [0, 2, 3, 4, 5, 7]

BUCKET_NAMES = ("v0", "v2", "v3", "v4", "v57", "holdout")
N_BUCKETS = len(BUCKET_NAMES)

COVIS_TOP_K = 50
FALLBACK_TOP_K = 160
SUBMISSION_K = 160
DOMINANT_SHARE = 0.90
# Wide pool when padding holdout users (no dominant vertical).
FB_PAD_WIDE = SUBMISSION_K * 3

RANDOM_SEED = 42


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        pl.set_random_seed(seed)
    except AttributeError:
        pass


def _log_mem(stage: str) -> None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logging.info("[mem] %s | ru_maxrss=%s", stage, usage)
    except Exception:
        logging.info("[mem] %s", stage)


def map_vertical_to_bucket_expr(col: str = "vertical_id") -> pl.Expr:
    c = pl.col(col)
    return (
        pl.when(c == 0)
        .then(pl.lit("v0"))
        .when(c == 2)
        .then(pl.lit("v2"))
        .when(c == 3)
        .then(pl.lit("v3"))
        .when(c == 4)
        .then(pl.lit("v4"))
        .when(c.is_in([5, 7]))
        .then(pl.lit("v57"))
        .otherwise(pl.lit(None))
    )


# ---------------------------------------------------------------------------
# Step 1 — contact eids + per-vertical top-160 fallback (contacts only)
# ---------------------------------------------------------------------------


def load_target_eids(path: Path) -> list[int]:
    df = pl.read_csv(path)
    col = "mapped_eid" if "mapped_eid" in df.columns else df.columns[0]
    out = df.select(pl.col(col).cast(pl.UInt32)).to_series().to_list()
    logging.info("Loaded %s contact eids from %s", f"{len(out):,}", path)
    return out


def build_vertical_top160_fallback(
    train_glob: str,
    item_features_path: Path,
    target_eids: Iterable[int],
) -> pl.DataFrame:
    """Top-160 item_id per vertical_id by contact (eid ∈ target_eids) count."""
    eid_set = list({int(x) for x in target_eids})
    items = pl.scan_parquet(str(item_features_path)).select(["item_id", "vertical_id"])

    lf = (
        pl.scan_parquet(train_glob)
        .filter(pl.col("timestamp") < CUTOFF_MS)
        .filter(pl.col("eid").is_in(eid_set))
        .join(items, on="item_id", how="inner")
        .group_by(["vertical_id", "item_id"])
        .agg(pl.len().alias("contact_cnt"))
    )

    ranked = (
        lf.with_columns(
            pl.col("contact_cnt")
            .rank(method="ordinal", descending=True)
            .over("vertical_id")
            .alias("rk")
        )
        .filter(pl.col("rk") <= FALLBACK_TOP_K)
        .select(["vertical_id", "item_id", "contact_cnt", "rk"])
    )

    df = ranked.collect(streaming=True)
    gc.collect()
    _log_mem("after build_vertical_top160_fallback")
    logging.info("Fallback rows=%s (<=160 per vertical_id)", f"{df.height:,}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — co-visitation + semantic pools
# ---------------------------------------------------------------------------


def build_co_visitation(train_glob: str) -> pl.DataFrame:
    """Next-item co-vis: shift(-1) over user_id; top-50 next_item per item_id."""
    base = (
        pl.scan_parquet(train_glob)
        .filter(pl.col("timestamp") < CUTOFF_MS)
        .select(["user_id", "item_id", "timestamp"])
    )

    nu = (
        base.group_by("item_id")
        .agg(pl.col("user_id").n_unique().alias("nu"))
        .filter(pl.col("nu") >= 2)
        .select("item_id")
    )

    seq = (
        base.join(nu, on="item_id", how="inner")
        .sort(["user_id", "timestamp"])
        .with_columns(pl.col("item_id").shift(-1).over("user_id").alias("next_item"))
        .filter(pl.col("next_item").is_not_null())
        .filter(pl.col("item_id") != pl.col("next_item"))
    )

    edges = (
        seq.group_by(["item_id", "next_item"])
        .len()
        .rename({"len": "cnt"})
        .sort(["item_id", "cnt"], descending=[False, True])
        .group_by("item_id", maintain_order=True)
        .head(COVIS_TOP_K)
        .rename({"next_item": "cand_item_id"})
    )

    df = edges.collect(streaming=True)
    gc.collect()
    _log_mem("after build_co_visitation")
    logging.info("Co-visitation edges: %s", f"{df.height:,}")
    return df


def build_semantic_pools(
    item_features_path: Path,
    train_glob: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Top-50 items per sid_0_y / sid_1_y cluster by train impression count."""
    item_pop = (
        pl.scan_parquet(train_glob)
        .filter(pl.col("timestamp") < CUTOFF_MS)
        .group_by("item_id")
        .len()
        .rename({"len": "pop"})
    )

    items = pl.scan_parquet(str(item_features_path)).select(["item_id", "sid_0_y", "sid_1_y"])

    joined = items.join(item_pop, on="item_id", how="left").with_columns(
        pl.col("pop").fill_null(0).cast(pl.Int64)
    )

    def _top50_by(key: str) -> pl.DataFrame:
        df = (
            joined.sort([key, "pop"], descending=[False, True])
            .group_by(key, maintain_order=True)
            .head(COVIS_TOP_K)
            .select([pl.lit(key).alias("pool_key"), pl.col(key), pl.col("item_id"), pl.col("pop")])
            .collect(streaming=True)
        )
        gc.collect()
        return df

    pool0 = _top50_by("sid_0_y")
    pool1 = _top50_by("sid_1_y")
    _log_mem("after build_semantic_pools")
    logging.info("Semantic pools sid_0=%s sid_1=%s rows", pool0.height, pool1.height)
    return pool0, pool1


def generate_candidates(
    co_vis: pl.DataFrame,
    pool_sid0: pl.DataFrame,
    pool_sid1: pl.DataFrame,
    item_features_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    _ = pl.scan_parquet(str(item_features_path))
    return co_vis, pool_sid0, pool_sid1


# ---------------------------------------------------------------------------
# Step 3 — user bucket (dominant raw vertical among eval verticals)
# ---------------------------------------------------------------------------


def build_user_bucket_map(
    eval_users_path: Path,
    eval_events_path: Path,
    item_features_path: Path,
) -> pl.DataFrame:
    """
    Per eval user: dominant vertical_id among {0,2,3,4,5,7} by event share.
    If max_share >= 0.9 → bucket = map(vertical); else holdout.
    """
    users = pl.scan_csv(str(eval_users_path)).select(pl.col("user_id").cast(pl.UInt32).unique())
    items = pl.scan_parquet(str(item_features_path)).select(["item_id", "vertical_id"])

    hist = (
        pl.scan_parquet(str(eval_events_path))
        .join(users, on="user_id", how="inner")
        .join(items, on="item_id", how="inner")
        .filter(pl.col("vertical_id").is_in(EVAL_VERTICAL_IDS))
    )

    counts = hist.group_by(["user_id", "vertical_id"]).len().rename({"len": "n"})
    totals = counts.group_by("user_id").agg(pl.col("n").sum().alias("tot"))
    shares = counts.join(totals, on="user_id").with_columns((pl.col("n") / pl.col("tot")).alias("share"))

    dominant = shares.sort(["user_id", "share"], descending=[False, True]).group_by(
        "user_id", maintain_order=True
    ).first()

    dom = dominant.join(users, on="user_id", how="right").with_columns(
        pl.when(pl.col("share").fill_null(0.0) >= DOMINANT_SHARE)
        .then(pl.col("vertical_id"))
        .otherwise(pl.lit(None))
        .cast(pl.Int32)
        .alias("dominant_vertical_id"),
    )

    out = dom.with_columns(
        pl.when(pl.col("dominant_vertical_id").is_not_null())
        .then(map_vertical_to_bucket_expr("dominant_vertical_id"))
        .otherwise(pl.lit("holdout"))
        .alias("bucket_name")
    ).select(["user_id", "bucket_name", "dominant_vertical_id"])

    df = out.collect(streaming=True)
    gc.collect()
    logging.info("User buckets:\n%s", df.group_by("bucket_name").len().sort("len", descending=True))
    return df


# ---------------------------------------------------------------------------
# Step 4 — candidates + anti-join (no seen items)
# ---------------------------------------------------------------------------


def _last_items_per_user(eval_events_path: Path, user_ids: pl.Series) -> pl.DataFrame:
    ev = pl.scan_parquet(str(eval_events_path)).filter(pl.col("user_id").is_in(user_ids))
    last_ts = ev.group_by("user_id").agg(pl.max("timestamp").alias("mx"))
    last_items = (
        ev.join(last_ts, on="user_id")
        .filter(pl.col("timestamp") == pl.col("mx"))
        .group_by("user_id")
        .agg(pl.col("item_id").last().alias("last_item_id"))
    )
    return last_items.collect(streaming=True)


def candidates_for_bucket(
    bucket_users: pl.DataFrame,
    eval_events_path: Path,
    co_vis: pl.DataFrame,
    pool_sid0: pl.DataFrame,
    pool_sid1: pl.DataFrame,
    item_features_path: Path,
) -> pl.DataFrame:
    empty = pl.DataFrame(
        schema={"user_id": pl.UInt32, "item_id": pl.UInt32},
    )
    if bucket_users.is_empty():
        return empty

    uids = bucket_users["user_id"]
    last_items = _last_items_per_user(eval_events_path, uids)
    if last_items.is_empty():
        return empty

    feats = pl.read_parquet(
        item_features_path,
        columns=["item_id", "sid_0_y", "sid_1_y"],
    )
    last_enriched = last_items.join(feats, left_on="last_item_id", right_on="item_id", how="left")

    c1 = last_enriched.join(co_vis, left_on="last_item_id", right_on="item_id", how="inner").select(
        [pl.col("user_id"), pl.col("cand_item_id").alias("item_id")]
    )

    p0 = pool_sid0.rename({"item_id": "cand_item_id"})
    c2 = (
        last_enriched.select(["user_id", "sid_0_y"])
        .join(p0, on="sid_0_y", how="inner")
        .select([pl.col("user_id"), pl.col("cand_item_id").alias("item_id")])
    )

    p1 = pool_sid1.rename({"item_id": "cand_item_id"})
    c3 = (
        last_enriched.select(["user_id", "sid_1_y"])
        .join(p1, on="sid_1_y", how="inner")
        .select([pl.col("user_id"), pl.col("cand_item_id").alias("item_id")])
    )

    cand = pl.concat([c1, c2, c3], how="vertical").with_columns(
        [pl.col("user_id").cast(pl.UInt32), pl.col("item_id").cast(pl.UInt32)]
    ).unique()

    hist = (
        pl.scan_parquet(str(eval_events_path))
        .filter(pl.col("user_id").is_in(uids))
        .select(["user_id", "item_id"])
        .unique()
        .with_columns([pl.col("user_id").cast(pl.UInt32), pl.col("item_id").cast(pl.UInt32)])
    )

    out = cand.join(hist.collect(streaming=True), on=["user_id", "item_id"], how="anti").unique()
    gc.collect()
    return out


def fallback_rows_for_users(
    users_df: pl.DataFrame,
    fb_vert: pl.DataFrame,
) -> pl.DataFrame:
    """Per user: up to SUBMISSION_K items from fb_vert for dominant_vertical_id (or global top)."""
    if users_df.is_empty():
        return pl.DataFrame(schema={"user_id": pl.UInt32, "item_id": pl.UInt32})

    u = users_df.select(["user_id", "dominant_vertical_id"]).unique()
    parts: list[pl.DataFrame] = []

    for row in u.iter_rows(named=True):
        uid = int(row["user_id"])
        dv = row["dominant_vertical_id"]
        if dv is not None:
            items = (
                fb_vert.filter(pl.col("vertical_id") == int(dv))
                .sort("contact_cnt", descending=True)
                .head(SUBMISSION_K)["item_id"]
                .to_list()
            )
        else:
            items = fb_vert.sort("contact_cnt", descending=True).head(FB_PAD_WIDE)["item_id"].to_list()
        items = items[:SUBMISSION_K]
        if not items:
            continue
        parts.append(
            pl.DataFrame(
                {
                    "user_id": pl.Series([uid] * len(items), dtype=pl.UInt32),
                    "item_id": pl.Series(items, dtype=pl.UInt32),
                }
            )
        )

    if not parts:
        return pl.DataFrame(schema={"user_id": pl.UInt32, "item_id": pl.UInt32})
    return pl.concat(parts, how="vertical").unique(["user_id", "item_id"])


# ---------------------------------------------------------------------------
# Step 5 — features + CatBoost stubs
# ---------------------------------------------------------------------------


def build_global_user_item_stats(train_glob: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    lf = pl.scan_parquet(train_glob).filter(pl.col("timestamp") < CUTOFF_MS)
    ucnt = lf.group_by("user_id").len().rename({"len": "user_event_cnt"}).collect(streaming=True)
    icnt = lf.group_by("item_id").len().rename({"len": "item_event_cnt"}).collect(streaming=True)
    gc.collect()
    _log_mem("after build_global_user_item_stats")
    return ucnt, icnt


def attach_features(candidates: pl.DataFrame, ucnt: pl.DataFrame, icnt: pl.DataFrame) -> pl.DataFrame:
    return (
        candidates.join(ucnt, on="user_id", how="left")
        .join(icnt, on="item_id", how="left")
        .with_columns(
            [
                pl.col("user_event_cnt").fill_null(0).cast(pl.Float32),
                pl.col("item_event_cnt").fill_null(0).cast(pl.Float32),
            ]
        )
    )


def init_catboost_stubs(*, use_gpu: bool = True) -> list[CatBoostClassifier]:
    if CatBoostClassifier is None:
        raise RuntimeError("catboost is not installed")
    task = "GPU" if use_gpu else "CPU"
    try:
        models: list[CatBoostClassifier] = []
        for i in range(N_BUCKETS):
            m = CatBoostClassifier(
                iterations=3,
                depth=4,
                loss_function="Logloss",
                verbose=False,
                task_type=task,
                allow_writing_files=False,
                random_seed=RANDOM_SEED + i,
            )
            tiny_X = pl.DataFrame(
                {"user_event_cnt": [0.0, 100.0, 5.0], "item_event_cnt": [0.0, 2.0, 50.0]}
            ).to_pandas()
            m.fit(tiny_X, [0, 1, 0])
            models.append(m)
        logging.info("CatBoost stubs ready (%s)", task)
        return models
    except Exception as e:  # pragma: no cover
        logging.warning("CatBoost GPU failed (%s); CPU fallback.", e)
        return init_catboost_stubs(use_gpu=False)


def score_candidates(df: pl.DataFrame, model: CatBoostClassifier) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(pl.lit(0.5).alias("score"))
    X = df.select(["user_event_cnt", "item_event_cnt"]).to_pandas()
    proba = model.predict_proba(X)[:, 1]
    return df.with_columns(pl.Series("score", proba))


# ---------------------------------------------------------------------------
# Step 6 — pad to 160 / vertical fallback / full user coverage
# ---------------------------------------------------------------------------


def pad_scored_to_submission(
    scored: pl.DataFrame,
    user_meta: pl.DataFrame,
    fb_vert: pl.DataFrame,
) -> pl.DataFrame:
    """
    Per user: top SUBMISSION_K by score; pad with Step-1 top for dominant_vertical_id
    (or global top if holdout / null).

    Iterates every user in ``user_meta`` so users with no retrieval rows still get
    up to SUBMISSION_K fallback items.
    """
    meta = user_meta.select(["user_id", "dominant_vertical_id"]).unique()
    if meta.is_empty():
        return pl.DataFrame(schema={"user_id": pl.UInt32, "item_id": pl.UInt32})

    if scored.is_empty():
        ranked = pl.DataFrame(
            schema={
                "user_id": pl.UInt32,
                "item_id": pl.UInt32,
                "score": pl.Float64,
                "dominant_vertical_id": pl.Int32,
            }
        )
    else:
        ranked = (
            scored.join(meta, on="user_id", how="left")
            .sort(["user_id", "score"], descending=[False, True])
            .select(["user_id", "item_id", "score", "dominant_vertical_id"])
        )

    rows: list[pl.DataFrame] = []
    for row in meta.iter_rows(named=True):
        uid = int(row["user_id"])
        dv = row["dominant_vertical_id"]
        sub = ranked.filter(pl.col("user_id") == uid).head(SUBMISSION_K)
        need = SUBMISSION_K - sub.height
        taken: set[int] = set(int(x) for x in sub["item_id"].to_list()) if sub.height else set()
        if need > 0:
            if dv is not None:
                pad_list = (
                    fb_vert.filter(pl.col("vertical_id") == int(dv))
                    .sort("contact_cnt", descending=True)["item_id"]
                    .to_list()
                )
            else:
                pad_list = fb_vert.sort("contact_cnt", descending=True).head(FB_PAD_WIDE)["item_id"].to_list()
            extra = [int(x) for x in pad_list if int(x) not in taken][:need]
            pad = pl.DataFrame(
                {"user_id": [uid] * len(extra), "item_id": extra},
                schema={"user_id": pl.UInt32, "item_id": pl.UInt32},
            )
            if sub.height:
                sub = pl.concat([sub.select(["user_id", "item_id"]), pad], how="vertical")
            else:
                sub = pad
        else:
            sub = sub.select(["user_id", "item_id"])
        if sub.is_empty():
            sub = fallback_rows_for_users(
                pl.DataFrame(
                    {
                        "user_id": pl.Series([uid], dtype=pl.UInt32),
                        "dominant_vertical_id": pl.Series([dv], dtype=pl.Int32),
                    }
                ),
                fb_vert,
            )
        rows.append(sub)

    return pl.concat(rows, how="vertical").unique(["user_id", "item_id"])


def merge_bucket_submissions(
    parts: list[pl.DataFrame],
    eval_users_path: Path,
    user_bucket: pl.DataFrame,
    fb_vert: pl.DataFrame,
) -> pl.DataFrame:
    if not parts:
        base = pl.DataFrame(schema={"user_id": pl.UInt32, "item_id": pl.UInt32})
    else:
        base = pl.concat(parts, how="vertical").unique(["user_id", "item_id"])

    eval_users = pl.read_csv(eval_users_path).select(pl.col("user_id").cast(pl.UInt32).unique())
    present = base["user_id"].unique() if not base.is_empty() else pl.Series([], dtype=pl.UInt32)
    missing = eval_users.filter(~pl.col("user_id").is_in(present))
    if missing.is_empty():
        return base

    um = user_bucket.join(missing, on="user_id", how="inner")
    fill = fallback_rows_for_users(um, fb_vert)
    out = pl.concat([base, fill], how="vertical").unique(["user_id", "item_id"])
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(data_dir: Path, out_csv: Path, *, use_gpu: bool = True) -> None:
    train_glob = str(data_dir / "train_data" / "*.parquet")
    item_features = data_dir / "item_features.parquet"
    eval_users = data_dir / "eval_users.csv"
    eval_events = data_dir / "eval_user_events.pq"
    contact_eids = data_dir / "contact_eids.csv"

    for p in (item_features, eval_users, eval_events, contact_eids):
        if not p.exists():
            raise FileNotFoundError(p)

    target_eids = load_target_eids(contact_eids)

    logging.info("Step 1: vertical popularity fallback…")
    fb_vert = build_vertical_top160_fallback(train_glob, item_features, target_eids)

    logging.info("Step 2: co-visitation + semantic pools…")
    co_vis = build_co_visitation(train_glob)
    pool0, pool1 = build_semantic_pools(item_features, train_glob)
    generate_candidates(co_vis, pool0, pool1, item_features)

    logging.info("Step 3: user buckets…")
    user_bucket = build_user_bucket_map(eval_users, eval_events, item_features)

    logging.info("Step 4–5: global stats + CatBoost…")
    ucnt, icnt = build_global_user_item_stats(train_glob)
    models = init_catboost_stubs(use_gpu=use_gpu)
    bucket_to_model = {name: models[i] for i, name in enumerate(BUCKET_NAMES)}

    all_parts: list[pl.DataFrame] = []
    for bname in BUCKET_NAMES:
        bu = user_bucket.filter(pl.col("bucket_name") == bname)
        logging.info("Bucket %s users=%s", bname, bu.height)
        if bu.is_empty():
            continue

        cand = candidates_for_bucket(bu, eval_events, co_vis, pool0, pool1, item_features)
        if cand.is_empty():
            part = fallback_rows_for_users(bu, fb_vert)
        else:
            feat = attach_features(cand, ucnt, icnt)
            scored = score_candidates(feat, bucket_to_model[bname])
            part = pad_scored_to_submission(scored, bu, fb_vert)

        all_parts.append(part)
        gc.collect()
        _log_mem(f"after bucket {bname}")

    out = merge_bucket_submissions(all_parts, eval_users, user_bucket, fb_vert)
    out.write_csv(out_csv)
    logging.info("Wrote %s rows → %s", f"{out.height:,}", out_csv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        pl.Config.set_fmt_str_lengths(200)
    except Exception:
        pass
    os.environ.setdefault("POLARS_MAX_THREADS", "28")
    set_random_seed(RANDOM_SEED)
    run_pipeline(args.data_dir.resolve(), args.out.resolve(), use_gpu=not args.cpu_only)


if __name__ == "__main__":
    main(sys.argv[1:])
