#!/usr/bin/env python3
"""
Бейзлайн рекомендаций без ML: глобальная популярность + матрица co-visitation (следующий item).

Читает огромный кликстрим через Polars LazyFrame (scan_parquet + concat).
Для каждого user_id из теста — до 160 уникальных item_id (длинный CSV как в popular.py).

При нехватке RAM на шаге co-visitation см. map-reduce в data_pipeline.py (шаг 3):
можно по шардам считать пары (item_id, next_item_id), суммировать, затем top-50.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl

LOG = logging.getLogger("matrix_baseline")

# Файл в CWD: последняя зафиксированная стадия (полезно при kill/OOM, когда stderr не успел).
_PROGRESS_SNAPSHOT = Path("matrix_baseline_last_stage.txt")


def _progress_snapshot(message: str) -> None:
    """Пишет метку времени и стадию — перезаписывает файл целиком."""
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}\n"
    try:
        _PROGRESS_SNAPSHOT.write_text(line, encoding="utf-8")
    except OSError:
        pass


def _log_phase_start(phase: str) -> float:
    LOG.info(">>> Старт: %s", phase)
    _progress_snapshot(f"СТАРТ: {phase}")
    _flush_log_handlers()
    return time.perf_counter()


def _log_phase_done(phase: str, t0: float) -> None:
    dt = time.perf_counter() - t0
    LOG.info("<<< Готово: %s (%.2f с)", phase, dt)
    _progress_snapshot(f"ГОТОВО: {phase} ({dt:.2f}s)")
    _flush_log_handlers()


def _flush_log_handlers() -> None:
    for h in logging.root.handlers:
        flush = getattr(h, "flush", None)
        if flush is not None:
            flush()

# Сколько строк на пользователя берём до отсечения seed — запас против просадки ниже 160.
_HEAD_BUFFER_BEFORE_SEED_FILTER = 160


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    """Сборка LazyFrame с потоковым движком (совместимость версий Polars)."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def scan_clicks(clicks_glob: str) -> pl.LazyFrame:
    """
    Объединяет все parquet, попавшие под glob, в один LazyFrame.

    Ожидаются колонки user_id, item_id, timestamp.
    Пути сортируются для воспроизводимости.
    """
    paths = sorted(glob.glob(clicks_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"Ни один файл не найден по шаблону: {clicks_glob!r}")
    LOG.info("Найдено %d parquet-файлов по glob", len(paths))

    lf = pl.concat([pl.scan_parquet(p) for p in paths], how="vertical")
    lf = lf.select(
        pl.col("user_id").cast(pl.UInt32),
        pl.col("item_id").cast(pl.UInt32),
        pl.col("timestamp"),
    )
    return ensure_timestamp_milliseconds(lf)


def ensure_timestamp_milliseconds(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Приводит timestamp к Int64 миллисекундам от эпохи.

    Поддержка: Int64/UInt64 (уже ms), Float64, Datetime (любая точность → ms).
    """
    try:
        sch = lf.collect_schema()
    except Exception:  # старые версии Polars
        return lf

    if "timestamp" not in sch:
        raise ValueError("В схеме нет колонки timestamp")

    ts = sch["timestamp"]
    if ts == pl.Int64 or ts == pl.UInt64:
        return lf.with_columns(pl.col("timestamp").cast(pl.Int64))
    if ts == pl.Float64 or ts == pl.Float32:
        return lf.with_columns(pl.col("timestamp").cast(pl.Int64))
    if isinstance(ts, pl.Datetime):
        return lf.with_columns(pl.col("timestamp").dt.timestamp("ms").cast(pl.Int64))
    # UInt32 и др. — к Int64
    return lf.with_columns(pl.col("timestamp").cast(pl.Int64))


def apply_time_window_if_needed(
    lf: pl.LazyFrame,
    global_days: int | None,
    since_ms: int | None,
) -> pl.LazyFrame:
    """
    Ограничивает клики по времени для глобального топа и co-vis (одинаковое окно).

    - since_ms: нижняя граница timestamp (включительно).
    - global_days: если задано, сначала считается max(timestamp), затем отсечка
      timestamp >= max_ts - global_days * 86400 * 1000 (два лёгких прохода по данным).

    Если оба None — используется весь период.
    """
    if since_ms is not None:
        LOG.info("Фильтр timestamp >= %d (since_ms)", since_ms)
        return lf.filter(pl.col("timestamp") >= since_ms)

    if global_days is None:
        return lf

    max_row = collect_streaming(lf.select(pl.col("timestamp").max().alias("max_ts")))
    if max_row.is_empty() or max_row["max_ts"][0] is None:
        LOG.warning("max(timestamp) пустой — окно по дням не применяется")
        return lf

    max_ts = int(max_row["max_ts"][0])
    cutoff = max_ts - int(global_days) * 86400 * 1000
    LOG.info(
        "Окно последних %d суток: timestamp in [%d, %d]",
        global_days,
        cutoff,
        max_ts,
    )
    return lf.filter(pl.col("timestamp") >= cutoff)


def compute_global_top(
    lf: pl.LazyFrame,
    k: int = 160,
) -> pl.DataFrame:
    """
    Глобальный топ item_id по числу кликов (fallback).

    Краевой случай: при нуле строк после фильтра вернётся пустой DataFrame —
    тогда инференс заполнит только пустыми рекомендациями (логируется в main).
    """
    return collect_streaming(
        lf.group_by("item_id")
        .agg(pl.len().alias("n_clicks"))
        .sort("n_clicks", descending=True)
        .head(k)
        .select("item_id")
    )


def compute_covisitation(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Матрица переходов: для каждого item_id — до 50 самых частых next_item_id.

    Пары (текущий клик → следующий по времени в рамках user_id).
    Эквивалент «.head(50).over(item_id)» здесь реализован через
    sort по (item_id, count desc) и group_by(..., maintain_order=True).head(50).

    Если на полном датасете этот шаг не помещается в RAM, разбейте клики по
    шардам: на каждом шарде посчитайте пары (item_id, next_item_id, count),
    сохраните во временные parquet, затем pl.concat(scan...) + group_by сумма
    count и снова top-50 на item_id (аналог map-reduce в data_pipeline.py).
    """
    # Один collect_streaming на всю цепочку: при «тишине» после этого лога чаще всего
    # падает/убивает OOM именно глобальная сортировка по (user_id, timestamp).
    LOG.info(
        "co-vis: один потоковый collect — sort(user_id,timestamp) → shift(-1) "
        "→ group_by(item_id,next) → top-50 на item (долго и тяжело по RAM)"
    )
    _progress_snapshot("co-visitation: внутри collect_streaming (sort+shift+…)")
    _flush_log_handlers()

    transitions = (
        lf.sort(["user_id", "timestamp"])
        .with_columns(
            pl.col("item_id").shift(-1).over("user_id").alias("next_item_id"),
        )
        .filter(pl.col("next_item_id").is_not_null())
    )

    pair_counts = transitions.group_by(["item_id", "next_item_id"]).agg(
        pl.len().alias("count"),
    )

    out = collect_streaming(
        pair_counts.sort(["item_id", "count"], descending=[False, True])
        .group_by("item_id", maintain_order=True)
        .head(50)
    )
    LOG.info("co-vis: collect завершён, строк в матрице: %d", out.height)
    _flush_log_handlers()
    return out


def read_test_users(path: str) -> pl.DataFrame:
    """CSV с колонкой user_id (как в popular.py)."""
    return (
        pl.read_csv(path)
        .select("user_id")
        .unique()
        .with_columns(pl.col("user_id").cast(pl.UInt32))
    )


def last_seed_per_user(lf: pl.LazyFrame, test_users: pl.DataFrame) -> pl.DataFrame:
    """
    Последний item_id для каждого тестового пользователя, у которого есть клики.

    Эквивалентно: сортировка по timestamp по убыванию и first() в group_by.
    Пользователи без истории в этом DataFrame отсутствуют — их обработает merge.
    """
    test_lf = test_users.lazy()
    return collect_streaming(
        lf.join(test_lf, on="user_id", how="inner")
        .sort(["user_id", "timestamp"])
        .group_by("user_id")
        .agg(
            pl.col("item_id").sort_by("timestamp", descending=True).first().alias("seed_item_id"),
        )
    )


def seeds_for_all_test_users(test_users: pl.DataFrame, seeds_historical: pl.DataFrame) -> pl.DataFrame:
    """
    Все тестовые user_id с left join на seed_item_id.

    seed_item_id = null, если пользователя не было в кликах — дальше только глобальный fallback.
    """
    return test_users.join(seeds_historical, on="user_id", how="left")


def build_recommendations(
    seeds_full: pl.DataFrame,
    covis: pl.DataFrame,
    global_top: pl.DataFrame,
    exclude_seed: bool,
) -> pl.DataFrame:
    """
    До 160 уникальных item_id на пользователя: co-vis (prio = count), затем глобальный топ.

    Сначала объединяем кандидатов, сортируем по убыванию prio, unique оставляет
    более приоритетный источник (co-vis выше глобального при совпадении item_id).
    """
    _progress_snapshot("build_recommendations: старт (eager join/cross)")
    LOG.info("infer: сборка global_part (cross join users × global_top)...")
    _flush_log_handlers()

    if global_top.is_empty():
        LOG.warning("Глобальный топ пуст — рекомендации будут пустыми")
        return pl.DataFrame(
            {
                "user_id": pl.Series([], dtype=pl.UInt32),
                "item_id": pl.Series([], dtype=pl.UInt32),
            }
        )

    # Глобальный fallback: приоритеты << любого реального count из co-vis,
    # внутри глобала порядок как в топе (более популярный item — выше prio).
    global_scored = global_top.with_row_index("g_order").with_columns(
        (-(pl.col("g_order") + 1).cast(pl.Float64) * 1e-9).alias("prio"),
    )

    # Таблица B: все тестовые пользователи × глобальный топ (малый cross join).
    users_keys = seeds_full.select("user_id").unique()
    global_part = users_keys.join(global_scored.select(["item_id", "prio"]), how="cross")

    # Таблица A: co-vis для пользователей с известным seed и совпадением в матрице.
    covis_lf = covis.lazy()
    covis_part = (
        seeds_full.drop_nulls("seed_item_id")
        .lazy()
        .join(
            covis_lf,
            left_on="seed_item_id",
            right_on="item_id",
            how="inner",
        )
        .select(
            pl.col("user_id"),
            pl.col("next_item_id").alias("item_id"),
            pl.col("count").cast(pl.Float64).alias("prio"),
        )
    )

    LOG.info("infer: collect_streaming concat(covis_part, global_part) + sort...")
    _progress_snapshot("infer: collect combined + sort(user_id, prio)")
    _flush_log_handlers()
    combined = collect_streaming(
        pl.concat(
            [covis_part, global_part.lazy()],
            how="vertical",
        ).sort(["user_id", "prio"], descending=[False, True])
    )
    LOG.info("infer: combined собран, строк: %d", combined.height)
    _flush_log_handlers()

    # Первое вхождение item_id на пользователя — с максимальным prio (co-vis побеждает глобальный).
    deduped = combined.unique(subset=["user_id", "item_id"], keep="first")

    if exclude_seed:
        deduped = deduped.join(
            seeds_full.select(["user_id", "seed_item_id"]),
            on="user_id",
            how="left",
        ).filter(
            pl.col("seed_item_id").is_null() | (pl.col("item_id") != pl.col("seed_item_id")),
        )
        deduped = deduped.drop("seed_item_id")

    # После удаления seed снова сортируем по prio и берём запас строк, затем финальные 160.
    ranked = deduped.sort(["user_id", "prio"], descending=[False, True])
    buffered = ranked.group_by("user_id", maintain_order=True).head(_HEAD_BUFFER_BEFORE_SEED_FILTER)
    final_df = buffered.group_by("user_id", maintain_order=True).head(160)

    out = final_df.select(["user_id", "item_id"])
    return out


def warn_short_recommendations(rec: pl.DataFrame, test_users: pl.DataFrame) -> None:
    """Предупреждение, если у части пользователей < 160 уникальных рекомендаций."""
    counts = rec.group_by("user_id").len().rename({"len": "n"})
    expected = test_users.select("user_id").unique()
    joined = expected.join(counts, on="user_id", how="left").with_columns(
        pl.col("n").fill_null(0).cast(pl.UInt32),
    )
    short = joined.filter(pl.col("n") < 160)
    if short.is_empty():
        return
    LOG.warning(
        "У %d пользователей меньше 160 рекомендаций (макс. n=%d)",
        short.height,
        int(short["n"].max()),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clicks-glob",
        type=str,
        required=True,
        help="Glob к parquet шардам кликстрима (например data/**/part_*.parquet).",
    )
    parser.add_argument(
        "--test-users",
        type=str,
        required=True,
        help="CSV с колонкой user_id.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission_matrix_baseline.csv",
        help="Выход: длинный CSV (user_id, item_id).",
    )
    parser.add_argument(
        "--global-days",
        type=int,
        default=None,
        help="Учитывать только клики за последние N суток (и для топа, и для co-vis).",
    )
    parser.add_argument(
        "--since-ms",
        type=int,
        default=None,
        help="Нижняя граница timestamp в мс (альтернатива --global-days).",
    )
    parser.add_argument(
        "--global-out",
        type=str,
        default=None,
        help="Опционально сохранить глобальный топ (parquet).",
    )
    parser.add_argument(
        "--covis-out",
        type=str,
        default=None,
        help="Опционально сохранить матрицу co-visitation (parquet).",
    )
    parser.add_argument(
        "--exclude-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Не рекомендовать последний кликнутый item (по умолчанию: да).",
    )
    args = parser.parse_args()

    if args.global_days is not None and args.since_ms is not None:
        LOG.error("Задайте только один из параметров: --global-days или --since-ms")
        sys.exit(2)

    LOG.info(
        "Снимок стадии пишется в %s (последняя строка = где остановились)",
        _PROGRESS_SNAPSHOT.resolve(),
    )

    try:
        t0 = _log_phase_start("чтение test_users CSV")
        test_users = read_test_users(args.test_users)
        _log_phase_done("чтение test_users CSV", t0)
        LOG.info("Тестовых пользователей (уникальных): %d", test_users.height)
        _flush_log_handlers()

        t0 = _log_phase_start("scan_clicks (glob + concat LazyFrame)")
        lf_base = scan_clicks(args.clicks_glob)
        _log_phase_done("scan_clicks (glob + concat LazyFrame)", t0)

        t0 = _log_phase_start("apply_time_window_if_needed (возможен max(timestamp))")
        lf_win = apply_time_window_if_needed(lf_base, args.global_days, args.since_ms)
        _log_phase_done("apply_time_window_if_needed", t0)

        t0 = _log_phase_start("compute_global_top (group_by item + sort + head 160)")
        LOG.info("Считаю глобальный топ-160...")
        _flush_log_handlers()
        global_top = compute_global_top(lf_win, k=160)
        _log_phase_done("compute_global_top", t0)
        if args.global_out:
            Path(args.global_out).parent.mkdir(parents=True, exist_ok=True)
            global_top.write_parquet(args.global_out)
            LOG.info("Записан глобальный топ: %s", args.global_out)

        if global_top.is_empty():
            LOG.warning(
                "Глобальный топ пуст (нет кликов в выбранном окне). "
                "Всем пользователям будет пустая выдача."
            )

        t0 = _log_phase_start("compute_covisitation (см. логи внутри функции)")
        LOG.info("Считаю co-visitation (топ-50 next на item_id)...")
        _flush_log_handlers()
        covis = compute_covisitation(lf_win)
        _log_phase_done("compute_covisitation", t0)
        if args.covis_out:
            Path(args.covis_out).parent.mkdir(parents=True, exist_ok=True)
            covis.write_parquet(args.covis_out)
            LOG.info("Записана матрица co-vis: %s", args.covis_out)
        LOG.info("Строк в матрице co-vis (после top-50 на item): %d", covis.height)

        t0 = _log_phase_start("last_seed_per_user (join train×test + group_by)")
        LOG.info("Последний клик (seed) для тестовых пользователей с историей...")
        _flush_log_handlers()
        seeds_hist = last_seed_per_user(lf_win, test_users)
        _log_phase_done("last_seed_per_user", t0)
        seeds_full = seeds_for_all_test_users(test_users, seeds_hist)
        n_no_history = seeds_full.filter(pl.col("seed_item_id").is_null()).height
        if n_no_history:
            LOG.info("Пользователей без истории кликов (только глобальный топ): %d", n_no_history)

        t0 = _log_phase_start("build_recommendations (join + concat + collect)")
        LOG.info("Сборка рекомендаций (до 160 уникальных item_id)...")
        _flush_log_handlers()
        rec = build_recommendations(
            seeds_full,
            covis,
            global_top,
            exclude_seed=args.exclude_seed,
        )
        _log_phase_done("build_recommendations", t0)

        t0 = _log_phase_start("дозаполнение missing_users + warn_short")
        missing_users = test_users.join(rec.select("user_id").unique(), on="user_id", how="anti")
        if not missing_users.is_empty() and not global_top.is_empty():
            fill = missing_users.join(
                global_top.with_columns(pl.lit(0.0).alias("prio")), how="cross"
            )
            rec = pl.concat([rec, fill.select(["user_id", "item_id"])], how="vertical")
        warn_short_recommendations(rec, test_users)
        _log_phase_done("дозаполнение missing_users + warn_short", t0)

        t0 = _log_phase_start("write_csv submission")
        rec = rec.sort(["user_id", "item_id"])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        rec.write_csv(args.out)
        _log_phase_done("write_csv submission", t0)
        LOG.info("Записано %d пар (user_id, item_id) в %s", rec.height, args.out)
        _progress_snapshot("УСПЕХ: запись сабмита завершена")
        _flush_log_handlers()

    except Exception:
        LOG.exception(
            "Прогон прервался с исключением. Последняя стадия в файле: %s",
            _PROGRESS_SNAPSHOT.resolve(),
        )
        _progress_snapshot("ОШИБКА: см. stderr / лог выше (traceback)")
        _flush_log_handlers()
        raise


if __name__ == "__main__":
    main()
