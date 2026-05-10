import argparse

import polars as pl
from loguru import logger

# Verticals included in the eval bucketing (5 vertical buckets):
# v0={0}, v2={2}, v3={3}, v4={4}, v57={5, 7}.
# tiny niches not represented in eval and are skipped.
USED_VERTICAL_IDS = (0, 2, 3, 4, 5, 7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-user-events", type=str, required=True,
        help="Path to eval_user_events.pq.",
    )
    parser.add_argument(
        "--item-features", type=str, required=True,
        help="Path to item_features.parquet (used for vertical filter).",
    )
    parser.add_argument(
        "--eval-users", type=str, required=True,
        help="Path to eval_users.csv (single-column user_id list).",
    )
    parser.add_argument(
        "--out", type=str, default="submission_popular.csv",
        help="Output CSV path with (user_id, item_id) pairs.",
    )
    parser.add_argument(
        "--k", type=int, default=160,
        help="Items per user (matches the Recall@160 cap).",
    )
    args = parser.parse_args()

    logger.info(f"Reading {args.eval_user_events} and {args.item_features}...")
    items = (
        pl.scan_parquet(args.item_features)
        .select(["item_id", "vertical_id"])
        .filter(pl.col("vertical_id").is_in(list(USED_VERTICAL_IDS)))
    )

    top_items = (
        pl.scan_parquet(args.eval_user_events)
        .join(items, on="item_id", how="inner")
        .group_by("item_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(args.k)
        .select("item_id")
        .collect()
    )
    logger.info(f"Selected top-{args.k} popular items (total catalog scope: {len(USED_VERTICAL_IDS)} verticals).")

    users = (
        pl.read_csv(args.eval_users)
        .select("user_id")
        .unique()
        .with_columns(pl.col("user_id").cast(pl.UInt32))
    )
    logger.info(f"Eval users: {users.height:,}")

    submission = users.join(top_items, how="cross")
    submission.write_csv(args.out)
    logger.info(
        f"Wrote {submission.height:,} (user, item) pairs to {args.out} "
        f"({users.height:,} users × {top_items.height:,} items)"
    )


if __name__ == "__main__":
    main()
