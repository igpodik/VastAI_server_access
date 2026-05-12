#!/usr/bin/env python3
"""
AvitoTech ML CUP 2026 — retrieval + ranking baseline (Polars + CatBoost).

Memory discipline:
  - train parquet processed in sorted file chunks (peak RAM ~ one shard + aggregates)
  - co-vis raw window: [START_MS, CUTOFF_MS) (14 days); folded edges capped per item_id between flushes
  - POLARS_MAX_THREADS = TRAIN_HOST_RAM_GB // 6 (default RAM 128 GiB → 21 threads)
  - gc.collect() after major steps
  - CatBoost: fit on subsample (CB_MAX_FIT_ROWS), predict in chunks (CB_PRED_CHUNK); proxy RMSE target

Layout: data/train_data/*.parquet, data/item_features.parquet,
        data/eval_users.csv, data/eval_user_events.pq (или .parquet), data/contact_eids.csv
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
from typing import Iterable, Optional

import numpy as np
import polars as pl

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore[misc, assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUTOFF_UTC = dt.datetime(2026, 4, 15, 0, 0, 0, tzinfo=dt.timezone.utc)
CUTOFF_MS = int(CUTOFF_UTC.timestamp() * 1000)
# Окно для co-visitation (и смежных агрегатов): последние 14 суток до cutoff — меньше строк, ниже пик RAM.
START_MS = CUTOFF_MS - (14 * 24 * 60 * 60 * 1000)

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

# CatBoost: subsample fit + chunked predict (ограничение пика RAM на больших бакетах).
CB_MAX_FIT_ROWS = 350_000
CB_PRED_CHUNK = 200_000
CB_ITERATIONS = 120
CB_DEPTH = 5
CB_LEARNING_RATE = 0.08
CB_PROXY_ICNT_WEIGHT = 0.55
CB_PROXY_UCNT_WEIGHT = 0.45
CB_FEATURE_COLS: tuple[str, ...] = (
    "user_event_cnt",
    "item_event_cnt",
    "log_user",
    "log_item",
    "ui_ratio",
    "vert_h",
)

# Сколько файлов train накапливать в списке рёбер co-vis перед слиянием (пик RAM).
COVIS_EDGE_FOLD_EVERY = 6
# Жёсткий потолок рёбер на item_id в накопителе co-vis между flush (до финального top-K).
COVIS_FOLDED_TOP_PER_ITEM = 150


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


def _train_parquet_paths(data_dir: Path) -> list[Path]:
    d = data_dir / "train_data"
    paths = sorted(d.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Нет parquet в {d}")
    return paths


def _aggregate_train_counts_by_file(paths: list[Path]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Глобальные item_pop, ucnt, icnt за один проход по файлам (без scan по glob)."""
    acc_ip: Optional[pl.DataFrame] = None
    acc_uc: Optional[pl.DataFrame] = None
    for path in paths:
        chunk = (
            pl.scan_parquet(str(path))
            .filter(pl.col("timestamp") < CUTOFF_MS)
            .select(
                [
                    pl.col("user_id").cast(pl.UInt32),
                    pl.col("item_id").cast(pl.UInt32),
                ]
            )
        )
        ip = (
            chunk.group_by("item_id")
            .len()
            .rename({"len": "pop"})
            .collect(streaming=True)
        )
        uc = (
            chunk.group_by("user_id")
            .len()
            .rename({"len": "user_event_cnt"})
            .collect(streaming=True)
        )
        acc_ip = ip if acc_ip is None else (
            pl.concat([acc_ip, ip], how="vertical").group_by("item_id").agg(pl.col("pop").sum())
        )
        acc_uc = uc if acc_uc is None else (
            pl.concat([acc_uc, uc], how="vertical")
            .group_by("user_id")
            .agg(pl.col("user_event_cnt").sum())
        )
        del chunk, ip, uc
        gc.collect()
    assert acc_ip is not None and acc_uc is not None
    icnt = acc_ip.rename({"pop": "item_event_cnt"})
    _log_mem("after _aggregate_train_counts_by_file")
    return acc_ip, acc_uc, icnt


def _nu_items_multi_user(paths: list[Path]) -> pl.DataFrame:
    """
    item_id с числом уникальных user_id ≥ 2 по train.

    По одному parquet за раз (без pl.concat(scans) на 100 LazyFrame — иначе раздувание графа).
    После каждого файла: локальный ``nu`` = n_unique(user_id) внутри файла; аккумулятор суммирует ``nu``
    по ``item_id`` (корректно, если пользователи не дублируются между партициями; иначе — консервативная
    оценка сверху для порога ``nu >= 2`` при типичном шардировании по user).
    """
    acc: Optional[pl.DataFrame] = None
    for path in paths:
        chunk = (
            pl.scan_parquet(str(path))
            .filter(pl.col("timestamp") < CUTOFF_MS)
            .select(
                [
                    pl.col("item_id").cast(pl.UInt32),
                    pl.col("user_id").cast(pl.UInt32),
                ]
            )
        )
        part = (
            chunk.group_by("item_id")
            .agg(pl.col("user_id").n_unique().alias("nu"))
            .collect(streaming=True)
        )
        del chunk
        gc.collect()
        if part.is_empty():
            del part
            gc.collect()
            continue
        acc = part if acc is None else (
            pl.concat([acc, part], how="vertical")
            .group_by("item_id")
            .agg(pl.col("nu").sum())
        )
        del part
        gc.collect()
    if acc is None or acc.is_empty():
        gc.collect()
        return pl.DataFrame(schema={"item_id": pl.UInt32})
    out = acc.filter(pl.col("nu") >= 2).select("item_id")
    del acc
    gc.collect()
    return out


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


def build_vertical_top160_fallback_paths(
    paths: list[Path],
    item_features_path: Path,
    target_eids: Iterable[int],
) -> pl.DataFrame:
    """Top-160 item_id per vertical_id by contact (eid ∈ target_eids), по файлам train."""
    eid_set = list({int(x) for x in target_eids})
    items = pl.scan_parquet(str(item_features_path)).select(
        [
            pl.col("item_id").cast(pl.UInt32),
            pl.col("vertical_id"),
        ]
    )
    acc: Optional[pl.DataFrame] = None
    for path in paths:
        part = (
            pl.scan_parquet(str(path))
            .filter(pl.col("timestamp") < CUTOFF_MS)
            .filter(pl.col("eid").is_in(eid_set))
            .select(
                [
                    pl.col("item_id").cast(pl.UInt32),
                    pl.col("eid"),
                ]
            )
            .join(items, on="item_id", how="inner")
            .group_by(["vertical_id", "item_id"])
            .agg(pl.len().alias("contact_cnt"))
            .collect(streaming=True)
        )
        if part.is_empty():
            del part
            gc.collect()
            continue
        acc = part if acc is None else (
            pl.concat([acc, part], how="vertical")
            .group_by(["vertical_id", "item_id"])
            .agg(pl.col("contact_cnt").sum())
        )
        del part
        gc.collect()
    if acc is None:
        acc = pl.DataFrame(
            schema={
                "vertical_id": pl.Int32,
                "item_id": pl.UInt32,
                "contact_cnt": pl.UInt32,
            }
        )
    df = (
        acc.with_columns(
            pl.col("contact_cnt")
            .rank(method="ordinal", descending=True)
            .over("vertical_id")
            .alias("rk")
        )
        .filter(pl.col("rk") <= FALLBACK_TOP_K)
        .select(["vertical_id", "item_id", "contact_cnt", "rk"])
    )
    del acc
    gc.collect()
    _log_mem("after build_vertical_top160_fallback")
    logging.info("Fallback rows=%s (<=160 per vertical_id)", f"{df.height:,}")
    return df


def _merge_co_edge_parts(parts: list[pl.DataFrame]) -> pl.DataFrame:
    if not parts:
        return pl.DataFrame(
            schema={
                "item_id": pl.UInt32,
                "next_item": pl.UInt32,
                "cnt": pl.UInt64,
            }
        )
    return (
        pl.concat(parts, how="vertical")
        .group_by(["item_id", "next_item"])
        .agg(pl.col("cnt").sum())
    )


def build_co_visitation_paths(paths: list[Path], nu_items: pl.DataFrame) -> pl.DataFrame:
    """Co-vis по файлам; строки в каждом файле уже отсортированы по (user_id, timestamp),
    каждый пользователь живёт ровно в одной партиции — carry не нужен.
    nu_items — глобальный фильтр item_id (встречался у ≥ 2 уникальных пользователей).
    """
    edge_fold_buf: list[pl.DataFrame] = []
    folded: Optional[pl.DataFrame] = None

    def flush_edges() -> None:
        nonlocal folded, edge_fold_buf
        if not edge_fold_buf:
            return
        chunk = _merge_co_edge_parts(edge_fold_buf)
        edge_fold_buf.clear()
        merged = chunk if folded is None else (
            pl.concat([folded, chunk], how="vertical")
            .group_by(["item_id", "next_item"])
            .agg(pl.col("cnt").sum())
        )
        del chunk
        if folded is not None:
            del folded
        gc.collect()
        folded = (
            merged.sort(["item_id", "cnt"], descending=[False, True])
            .group_by("item_id", maintain_order=True)
            .head(COVIS_FOLDED_TOP_PER_ITEM)
        )
        del merged
        gc.collect()

    for path in paths:
        raw = (
            pl.scan_parquet(str(path))
            .filter((pl.col("timestamp") >= START_MS) & (pl.col("timestamp") < CUTOFF_MS))
            .select(
                [
                    pl.col("user_id").cast(pl.UInt32),
                    pl.col("item_id").cast(pl.UInt32),
                ]
            )
            .collect(streaming=True)
        )
        if raw.is_empty():
            del raw
            gc.collect()
            continue

        seq = raw.join(nu_items, on="item_id", how="inner")
        del raw
        gc.collect()
        if seq.is_empty():
            del seq
            gc.collect()
            continue

        within = seq.with_columns(
            pl.col("item_id").shift(-1).over("user_id").alias("next_item"),
        )
        chunk_edges = (
            within.filter(
                pl.col("next_item").is_not_null() & (pl.col("item_id") != pl.col("next_item"))
            )
            .group_by(["item_id", "next_item"])
            .len()
            .rename({"len": "cnt"})
            .with_columns(pl.col("cnt").cast(pl.UInt64))
        )
        del seq, within

        if not chunk_edges.is_empty():
            edge_fold_buf.append(chunk_edges)
            if len(edge_fold_buf) >= COVIS_EDGE_FOLD_EVERY:
                flush_edges()
        del chunk_edges
        gc.collect()

    flush_edges()
    if folded is None:
        all_e = pl.DataFrame(
            schema={"item_id": pl.UInt32, "next_item": pl.UInt32, "cnt": pl.UInt64},
        )
    else:
        all_e = folded

    if all_e.is_empty():
        df = pl.DataFrame(schema={"item_id": pl.UInt32, "cand_item_id": pl.UInt32})
    else:
        df = (
            all_e.sort(["item_id", "cnt"], descending=[False, True])
            .group_by("item_id", maintain_order=True)
            .head(COVIS_TOP_K)
            .rename({"next_item": "cand_item_id"})
        )
    del all_e, folded
    gc.collect()
    _log_mem("after build_co_visitation")
    logging.info("Co-visitation edges: %s", f"{df.height:,}")
    return df


def build_semantic_pools_from_item_pop(
    item_features_path: Path,
    item_pop: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Top-50 items per sid_0_y / sid_1_y по уже посчитанному item_pop (без повторного чтения train)."""
    items = pl.scan_parquet(str(item_features_path)).select(
        [
            pl.col("item_id").cast(pl.UInt32),
            pl.col("sid_0_y"),
            pl.col("sid_1_y"),
        ]
    )
    joined = items.join(item_pop.lazy(), on="item_id", how="left").with_columns(
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
    _ = pl.scan_parquet(str(item_features_path)).select(pl.col("item_id").cast(pl.UInt32))
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
    items = pl.scan_parquet(str(item_features_path)).select(
        [
            pl.col("item_id").cast(pl.UInt32),
            pl.col("vertical_id"),
        ]
    )

    hist = (
        pl.scan_parquet(str(eval_events_path))
        .select(
            [
                pl.col("user_id").cast(pl.UInt32),
                pl.col("item_id").cast(pl.UInt32),
            ]
        )
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
    ev = (
        pl.scan_parquet(str(eval_events_path))
        .filter(pl.col("user_id").is_in(user_ids))
        .select(
            [
                pl.col("user_id").cast(pl.UInt32),
                pl.col("item_id").cast(pl.UInt32),
                pl.col("timestamp"),
            ]
        )
    )
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
        .select(
            [
                pl.col("user_id").cast(pl.UInt32),
                pl.col("item_id").cast(pl.UInt32),
            ]
        )
        .unique()
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
# Step 5 — features + CatBoost (subsample RMSE proxy, chunked predict)
# ---------------------------------------------------------------------------


def attach_features(
    candidates: pl.DataFrame,
    ucnt: pl.DataFrame,
    icnt: pl.DataFrame,
    item_features_path: Path,
) -> pl.DataFrame:
    """Join ucnt/icnt + item vertical; числовые фичи Float32 (без лишних колонок в CB_FEATURE_COLS)."""
    items = pl.read_parquet(
        str(item_features_path),
        columns=["item_id", "vertical_id"],
    )
    return (
        candidates.join(ucnt, on="user_id", how="left")
        .join(icnt, on="item_id", how="left")
        .join(items, on="item_id", how="left")
        .with_columns(
            [
                pl.col("user_event_cnt").fill_null(0).cast(pl.Float32),
                pl.col("item_event_cnt").fill_null(0).cast(pl.Float32),
            ]
        )
        .with_columns(
            [
                (pl.col("user_event_cnt") + 1.0).log().cast(pl.Float32).alias("log_user"),
                (pl.col("item_event_cnt") + 1.0).log().cast(pl.Float32).alias("log_item"),
                ((pl.col("user_event_cnt") + 1.0) / (pl.col("item_event_cnt") + 1.0))
                .cast(pl.Float32)
                .alias("ui_ratio"),
                pl.col("vertical_id")
                .fill_null(-1)
                .hash(seed=RANDOM_SEED)
                .mod(243)
                .cast(pl.Float32)
                .alias("vert_h"),
            ]
        )
        .drop("vertical_id")
    )


def fit_catboost_bucket_model(
    feat: pl.DataFrame,
    bname: str,
    *,
    use_gpu: bool,
) -> CatBoostRegressor:
    """Обучение на subsample; proxy-y без разметки соревнования (лог-смесь счётчиков)."""
    if CatBoostRegressor is None:
        raise RuntimeError("catboost is not installed")
    seed = RANDOM_SEED + (hash(bname) & 0xFFFF)
    n_fit = min(CB_MAX_FIT_ROWS, feat.height)
    subs = feat.sample(n=n_fit, shuffle=True, seed=seed)
    uc = subs["user_event_cnt"].to_numpy().astype(np.float64, copy=False)
    ic = subs["item_event_cnt"].to_numpy().astype(np.float64, copy=False)
    y = CB_PROXY_ICNT_WEIGHT * np.log1p(ic) + CB_PROXY_UCNT_WEIGHT * np.log1p(uc)
    X = subs.select(CB_FEATURE_COLS).to_numpy().astype(np.float64, copy=False)
    del subs
    gc.collect()

    def _fit(task: str) -> CatBoostRegressor:
        m = CatBoostRegressor(
            iterations=CB_ITERATIONS,
            depth=CB_DEPTH,
            learning_rate=CB_LEARNING_RATE,
            loss_function="RMSE",
            verbose=False,
            task_type=task,
            allow_writing_files=False,
            random_seed=seed,
        )
        m.fit(X, y)
        return m

    try:
        model = _fit("GPU" if use_gpu else "CPU")
    except Exception as e:  # pragma: no cover
        if use_gpu:
            logging.warning("CatBoostRegressor GPU failed (%s); CPU fallback.", e)
            model = _fit("CPU")
        else:
            logging.exception("CatBoostRegressor failed on CPU")
            raise
    del X, y
    gc.collect()
    logging.info("CatBoost bucket=%s fit rows=%s", bname, n_fit)
    return model


def score_candidates(df: pl.DataFrame, model: CatBoostRegressor) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(pl.lit(0.5).alias("score"))
    parts: list[np.ndarray] = []
    for start in range(0, df.height, CB_PRED_CHUNK):
        h = min(CB_PRED_CHUNK, df.height - start)
        sl = df.slice(start, h)
        x = sl.select(CB_FEATURE_COLS).cast(pl.Float32).to_numpy().astype(np.float64, copy=False)
        parts.append(np.asarray(model.predict(x), dtype=np.float64))
    proba = np.concatenate(parts)
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
    item_features = data_dir / "item_features.parquet"
    eval_users = data_dir / "eval_users.csv"
    eval_events = data_dir / "eval_user_events.pq"
    if not eval_events.exists():
        _alt_ev = data_dir / "eval_user_events.parquet"
        if _alt_ev.exists():
            eval_events = _alt_ev
    contact_eids = data_dir / "contact_eids.csv"

    for p in (item_features, eval_users, eval_events, contact_eids):
        if not p.exists():
            raise FileNotFoundError(p)

    paths = _train_parquet_paths(data_dir)
    target_eids = load_target_eids(contact_eids)

    logging.info("Train parquet files=%s (sorted)", len(paths))
    logging.info("Aggregating item_pop + ucnt + icnt (one pass per file)…")
    item_pop, ucnt, icnt = _aggregate_train_counts_by_file(paths)
    gc.collect()
    _log_mem("after counts aggregate")

    logging.info("Computing nu_items for co-vis…")
    nu_items = _nu_items_multi_user(paths)
    gc.collect()
    _log_mem("after nu_items")

    logging.info("Step 1: vertical popularity fallback…")
    fb_vert = build_vertical_top160_fallback_paths(paths, item_features, target_eids)
    gc.collect()
    _log_mem("after fb_vert")

    logging.info("Step 2: co-visitation + semantic pools…")
    co_vis = build_co_visitation_paths(paths, nu_items)
    del nu_items
    gc.collect()
    _log_mem("after co_vis")

    pool0, pool1 = build_semantic_pools_from_item_pop(item_features, item_pop)
    del item_pop
    gc.collect()
    _log_mem("after semantic pools")

    generate_candidates(co_vis, pool0, pool1, item_features)
    gc.collect()
    _log_mem("after generate_candidates")

    logging.info("Step 3: user buckets…")
    user_bucket = build_user_bucket_map(eval_users, eval_events, item_features)
    gc.collect()
    _log_mem("after user_bucket")

    logging.info("Step 4–5: CatBoost…")
    if CatBoostRegressor is None:
        raise RuntimeError("catboost is not installed")

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
            feat = attach_features(cand, ucnt, icnt, item_features)
            model = fit_catboost_bucket_model(feat, bname, use_gpu=use_gpu)
            scored = score_candidates(feat, model)
            del model
            gc.collect()
            part = pad_scored_to_submission(scored, bu, fb_vert)
            del feat, scored
            gc.collect()

        all_parts.append(part)
        gc.collect()
        _log_mem(f"after bucket {bname}")

    del co_vis, pool0, pool1, ucnt, icnt
    gc.collect()
    _log_mem("after all buckets")

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
    ram_gb = int(os.environ.get("TRAIN_HOST_RAM_GB", "128"))
    n_threads = max(1, ram_gb // 6)
    os.environ["POLARS_MAX_THREADS"] = str(n_threads)
    logging.info(
        "TRAIN_HOST_RAM_GB=%s POLARS_MAX_THREADS=%s",
        ram_gb,
        n_threads,
    )
    try:
        pl.Config.set_fmt_str_lengths(200)
    except Exception:
        pass
    set_random_seed(RANDOM_SEED)
    run_pipeline(
        args.data_dir.expanduser().resolve(),
        args.out.expanduser().resolve(),
        use_gpu=not args.cpu_only,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
