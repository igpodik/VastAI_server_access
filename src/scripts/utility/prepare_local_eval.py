"""
prepare_local_eval.py — competitor-facing template for building a local
validation set out of df_train.

The official `eval.csv` (private/public ground truth) is not shipped to
competitors. This script reproduces the official eval pipeline against a
synthetic threshold inside train.

Pipeline (constants below match the official v4 eval — DO NOT change them
unless you know what you're doing):

  synthetic_train := events with timestamp <  synth_threshold
  synthetic_eval  := events with timestamp >= synth_threshold + 12h gap

  Filters on synthetic_eval:
    1. eid in contact_eids.csv,
    2. item_id appears in synthetic_train with >= 2 unique users,
    3. (user_id, item_id) ∉ synthetic_train (novelty filter).

  Bucket sampling (5 vertical × 10k + 5 holdout × 10k):
    For each of the 5 named verticals, sample 10k users whose unique-train
    items in that vertical's mapped_vertical_id(s) cover >= 90% of their
    total. Then 5 stratified holdout pools of 10k from the remaining
    eligible users.

  Per-user target = first 160 chronologically-unique novel contact items.

Usage (only the dataset paths and where to write are configurable):
    python -m bin.datafest_2026_v2.prepare_local_eval \\
        --train data/train.parquet \\
        --item-features data/item_features.parquet \\
        --contact-eids data/contact_eids.csv \\
        --out data/local_eval.csv

Output:
  * local_eval.csv        — flat (user_id, item_id), score with calc_metric.py
  * local_eval_users.csv  — user_id → bucket map (per-bucket recall analysis)
"""

import argparse
from datetime import datetime, timedelta

import polars as pl
from loguru import logger

# ── Constants frozen by the official v4 eval spec ─────────────────────────
DEFAULT_SYNTH_THRESHOLD = "2026-04-08T00:00:00"  # 1 week before real threshold
GAP_HOURS = 12
K = 160
MIN_USERS_PER_ITEM = 2
QUOTA_PER_VERTICAL = 10_000
HOLDOUT_QUOTA = 10_000
N_HOLDOUT_BUCKETS = 5
VERTICAL_THRESHOLD = 0.9
# 5 named buckets: 4 single verticals + 1 concat of two related verticals.
# vertical_id values come from item_features.parquet's mapped_vertical_id column.
BUCKET_SPECS: tuple[tuple[str, frozenset[int]], ...] = (
    ("v0", frozenset({0})),
    ("v2", frozenset({2})),
    ("v3", frozenset({3})),
    ("v4", frozenset({4})),
    ("v57", frozenset({5, 7})),
)


def _build_candidates(
    train_path: str,
    contact_eids: list[int],
    threshold_ms: int,
    eval_start_ms: int,
) -> pl.DataFrame:
    train = pl.scan_parquet(train_path)
    synth_train = train.filter(pl.col("timestamp") < threshold_ms)

    train_items_2u = (
        synth_train.group_by("item_id")
        .agg(pl.col("user_id").n_unique().alias("n_users"))
        .filter(pl.col("n_users") >= MIN_USERS_PER_ITEM)
        .select("item_id")
    )
    seen = synth_train.select(["user_id", "item_id"]).unique()

    candidates = (
        train.filter(pl.col("timestamp") >= eval_start_ms)
        .filter(pl.col("eid").is_in(contact_eids))
        .join(train_items_2u, on="item_id", how="inner")
        .join(seen, on=["user_id", "item_id"], how="anti")
        .collect()
    )
    logger.info(
        f"Candidate events: {candidates.height:,} rows, "
        f"{candidates['user_id'].n_unique():,} eligible users"
    )
    return candidates


def _build_user_sample(
    train_path: str,
    item_features_path: str,
    threshold_ms: int,
    eligible_users: pl.DataFrame,
) -> pl.DataFrame:
    synth_train = pl.scan_parquet(train_path).filter(
        pl.col("timestamp") < threshold_ms
    )
    items_v = pl.scan_parquet(item_features_path).select(["item_id", "vertical_id"])

    user_vertical = (
        synth_train.select(["user_id", "item_id"])
        .unique()
        .join(items_v, on="item_id", how="inner")
        .group_by(["user_id", "vertical_id"])
        .agg(pl.len().alias("n_in_v"))
        .join(eligible_users.lazy(), on="user_id", how="inner")
        .collect()
    )
    user_total = user_vertical.group_by("user_id").agg(
        pl.col("n_in_v").sum().alias("n_total")
    )

    parts: list[pl.DataFrame] = []
    for bucket_name, vertical_ids in BUCKET_SPECS:
        bucket_n = (
            user_vertical.filter(pl.col("vertical_id").is_in(list(vertical_ids)))
            .group_by("user_id")
            .agg(pl.col("n_in_v").sum().alias("n_in_bucket"))
        )
        focused = (
            bucket_n.join(user_total, on="user_id")
            .with_columns((pl.col("n_in_bucket") / pl.col("n_total")).alias("frac"))
            .filter(pl.col("frac") >= VERTICAL_THRESHOLD)
            .select("user_id")
            .unique()
            .sort("user_id")
        )
        n = min(QUOTA_PER_VERTICAL, focused.height)
        seed = 42 + sum(int(v) for v in vertical_ids)
        sample = focused.sample(n=n, seed=seed)
        logger.info(
            f"  bucket {bucket_name} (verticals={sorted(vertical_ids)}): "
            f"{n} / {focused.height} eligible focused users (seed={seed})"
        )
        parts.append(sample.with_columns(pl.lit(bucket_name).alias("bucket")))

    bucketed = pl.concat([p.select("user_id") for p in parts]).unique()
    holdout_pool = (
        eligible_users.join(bucketed, on="user_id", how="anti").sort("user_id")
    )
    logger.info(
        f"Holdout pool: {holdout_pool.height:,} eligible users not in any vertical bucket"
    )
    for i in range(N_HOLDOUT_BUCKETS):
        if holdout_pool.height == 0:
            break
        n = min(HOLDOUT_QUOTA, holdout_pool.height)
        seed = 100 + i
        sample = holdout_pool.sample(n=n, seed=seed)
        logger.info(f"  bucket h{i}: {n} / {holdout_pool.height} (seed={seed})")
        parts.append(sample.with_columns(pl.lit(f"h{i}").alias("bucket")))
        holdout_pool = holdout_pool.join(sample, on="user_id", how="anti").sort(
            "user_id"
        )

    combined = pl.concat(parts)
    n_unique = combined["user_id"].n_unique()
    logger.info(f"Total sampled: {combined.height:,} rows, {n_unique:,} unique users")
    if n_unique != combined.height:
        logger.warning(f"{combined.height - n_unique} duplicate user→bucket assignments")
    return combined


def _build_eval_rows(candidates: pl.DataFrame, sampled: pl.DataFrame) -> pl.DataFrame:
    return (
        candidates.join(sampled.select("user_id"), on="user_id", how="inner")
        .sort(["user_id", "timestamp"])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").unique(maintain_order=True).head(K))
        .explode("item_id")
    )


def prepare_local_eval(
    train_path: str,
    item_features_path: str,
    contact_eids_path: str,
    out_path: str,
    synth_threshold: str,
):
    threshold_date = datetime.fromisoformat(synth_threshold)
    eval_start_date = threshold_date + timedelta(hours=GAP_HOURS)
    threshold_ms = int(threshold_date.timestamp() * 1000)
    eval_start_ms = int(eval_start_date.timestamp() * 1000)
    logger.info(
        f"synth_train: timestamp < {threshold_ms} ({threshold_date}) | "
        f"synth_eval: timestamp >= {eval_start_ms} ({eval_start_date}, gap {GAP_HOURS}h)"
    )

    contact_eids = pl.read_csv(contact_eids_path).get_column("mapped_eid").to_list()
    logger.info(f"Contact eids: {contact_eids}")

    candidates = _build_candidates(
        train_path, contact_eids, threshold_ms, eval_start_ms
    )
    eligible_users = candidates.select("user_id").unique().sort("user_id")

    sampled = _build_user_sample(
        train_path, item_features_path, threshold_ms, eligible_users
    )

    users_path = out_path.replace(".csv", "_users.csv")
    sampled.write_csv(users_path)
    logger.info(f"User → bucket map saved to {users_path}")

    eval_df = _build_eval_rows(candidates, sampled)
    eval_df.write_csv(out_path)
    logger.info(
        f"{out_path}: {eval_df.height:,} rows, "
        f"{eval_df['user_id'].n_unique():,} users with >=1 target"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=str, required=True,
        help="Path to train.parquet — full pre-threshold clickstream.",
    )
    parser.add_argument(
        "--item-features", type=str, required=True,
        help="Path to item_features.parquet (vertical_id lookup for bucketing).",
    )
    parser.add_argument(
        "--contact-eids", type=str, required=True,
        help="Path to contact_eids.csv.",
    )
    parser.add_argument(
        "--out", type=str, default="local_eval.csv",
        help="Output CSV path (columns: user_id, item_id).",
    )
    parser.add_argument(
        "--synth-threshold", type=str, default=DEFAULT_SYNTH_THRESHOLD,
        help=(
            "Synthetic threshold date (ISO). Events strictly before this go to "
            "synth_train, events at/after this + 12h go to synth_eval. Default "
            f"is {DEFAULT_SYNTH_THRESHOLD} (one week before the official eval)."
        ),
    )
    args = parser.parse_args()

    prepare_local_eval(
        train_path=args.train,
        item_features_path=args.item_features,
        contact_eids_path=args.contact_eids,
        out_path=args.out,
        synth_threshold=args.synth_threshold,
    )
