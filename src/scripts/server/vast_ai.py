"""Единый скрипт управления Vast.ai инстансом.

Использование:
    python vast_ai.py search               -- найти лучший оффер
    python vast_ai.py start                -- создать инстанс
    python vast_ai.py artifacts-download -- скачать артефакты локально (data/ без зеркала Яндекса)
    python vast_ai.py pause-instance       -- только stop_instance (API)
    python vast_ai.py destroy-instance     -- только destroy инстанса
    python vast_ai.py stop                 -- pause-instance + destroy-instance

При ошибке pipeline.sh: сначала **artifacts-download** (пока инстанс running и SSH жив),
затем **pause-instance**, затем **destroy-instance**. `trap` в pipeline вешается только после
успешного `eval` SSH — иначе teardown не вызывается. Таймаут ожидания `running` в **start**:
контракт удаляется через API, **SSH_URL** в config очищается.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from vastai import VastAI

from config import (
    BLACKLISTED_OFFER_IDS,
    CONFIG_JSON_PATH,
    DISK_SIZE_GB,
    FAST_DISK_BW_OK_WITHOUT_NVME_LABEL,
    IMAGE,
    MAX_HOURLY_USD,
    MAX_INET_COST_PER_GB,
    MIN_CPU_CORES_EFFECTIVE,
    MIN_CPU_RAM_GB,
    MIN_CUDA_MAX_GOOD,
    MIN_DIRECT_PORT_COUNT,
    MIN_DISK_BW,
    MIN_INET_DOWN_MBPS,
    MIN_INET_UP_MBPS,
    MIN_RELIABILITY,
    NUM_GPUS,
    load_best_server_id,
    save_best_server_id,
    save_instance_id,
    save_ssh_url,
)
from pipeline_ssh_env import load_ssh_key_path

# Локальная выгрузка с инстанса: src/results/<datetime>/<EXPERIMENT_NAME>/
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# Исключения при rsync ~/avito_cup/data/ (см. download_data_on_server.sh — не тянуть зеркало с Яндекса).
_YANDEX_DATA_RSYNC_EXCLUDES: tuple[str, ...] = (
    "item_features.parquet",
    "contact_eids.csv",
    "eval_users.csv",
    "eval_user_events.zip",
    "eval_user_events.parquet",
    "eval_user_events.pq",
    "prepare_local_eval.py",
    "popular.py",
    "submission_popular.csv",
    "train_data/",
)

_RSYNC_RETRIES = 3
_RSYNC_RETRY_DELAY_SEC = 8.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECRET_ENV = Path(__file__).resolve().parent / "secret.env"


def _load_key(path: Path) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("KEY="):
            value = line.removeprefix("KEY=").lstrip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
    raise ValueError(f"KEY not found in {path}")


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _cpu_ram_gb(o: dict[str, Any]) -> float:
    ram = _f(o.get("cpu_ram"))
    return ram / 1000.0 if ram > 500.0 else ram


def _gpu_ram_gb(o: dict[str, Any]) -> float:
    ram = _f(o.get("gpu_ram"))
    return ram / 1000.0 if ram > 500.0 else ram


def _hourly_total_usd(o: dict[str, Any]) -> float:
    if o.get("discounted_dph_total") is not None:
        return _f(o.get("discounted_dph_total"))
    search = o.get("search") or {}
    if search.get("totalHour") is not None:
        return _f(search.get("totalHour"))
    return _f(o.get("dph_total"))


def _dlperf_per_dollar(o: dict[str, Any]) -> float:
    return _f(o.get("dlperf_per_dphtotal"))


def _top_rank_tuple(o: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        _f(o.get("cpu_cores_effective")),
        _cpu_ram_gb(o),
        _f(o.get("disk_bw")),
        _f(o.get("inet_down")) + _f(o.get("inet_up")),
        _f(o.get("dlperf")),
        _dlperf_per_dollar(o),
    )


def _gpu_name(o: dict[str, Any]) -> str:
    return str(o.get("gpu_name") or "")


def _is_rtx_4090(o: dict[str, Any]) -> bool:
    return "4090" in _gpu_name(o)


def _is_rtx_5090(o: dict[str, Any]) -> bool:
    return "5090" in _gpu_name(o)


def _is_preferred_gpu(o: dict[str, Any]) -> bool:
    return _is_rtx_4090(o) or _is_rtx_5090(o)


# Xeon E3/E5/E7 v1 (Sandy Bridge, ~2012) и v2 (Ivy Bridge, ~2013) не поддерживают AVX2.
# Haswell v3/v4, EPYC, Ryzen, Xeon Scalable — все с AVX2.
# Паттерн: «E[357]-<цифры> v1» или «E[357]-<цифры> v2» (word boundary).
_NO_AVX2_RE = re.compile(r"\bE[357]-\d+\s+v[12]\b", re.IGNORECASE)


def _cpu_avx2_ok(o: dict[str, Any]) -> bool:
    """Возвращает False для заведомо pre-AVX2 CPU (Sandy/Ivy Bridge v1/v2)."""
    cpu_name = str(o.get("cpu_name") or "")
    return not bool(_NO_AVX2_RE.search(cpu_name))


def _inet_cost_per_gb(o: dict[str, Any]) -> float:
    """Максимальная стоимость интернет-трафика $/GB (upload или download).

    Vast.ai хранит её в inet_up_billed / inet_down_billed (float, $/GB).
    Если поле отсутствует или равно None — считаем трафик бесплатным (0.0).
    """
    up = _f(o.get("inet_up_billed"), default=0.0)
    down = _f(o.get("inet_down_billed"), default=0.0)
    return max(up, down)


def _disk_ok(o: dict[str, Any]) -> bool:
    disk_name = str(o.get("disk_name") or "").lower()
    if "nvme" in disk_name:
        return True
    return _f(o.get("disk_bw")) >= FAST_DISK_BW_OK_WITHOUT_NVME_LABEL


def _base_query(gpu_token: Optional[str] = None) -> str:
    """Запрос к search_offers. Без gpu_token — любая видеокарта (остальные пороги те же).

    Ограничение по CUDA (cuda_max_good) не задаётся в строке запроса — оно проверяется
    после получения офферов, см. _first_failed_criterion.
    """
    tail = (
        "num_gpus=1 "
        "verified=true "
        "direct_port_count>=1 "
        "rentable=true "
        "cpu_cores_effective>=32 "
        "cpu_ram>=128 "
        "inet_down>=800 "
        "inet_up>=800 "
        f"dph_total<={MAX_HOURLY_USD} "
        f"reliability>{MIN_RELIABILITY} "
        "rented=False"
    )
    if gpu_token:
        return f"gpu_name={gpu_token} {tail}"
    return tail


def _merge_unique(a: list[dict], b: list[dict]) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for o in a + b:
        oid = o.get("id")
        if oid in seen:
            continue
        seen.add(oid)
        out.append(o)
    return out


def _first_failed_criterion(
    o: dict[str, Any],
    *,
    require_preferred_gpu: bool = True,
) -> Optional[tuple[str, str]]:
    """Первое несоответствие: (короткий_ключ_группы, подробное сообщение). OK → None.

    Проверка cuda_max_good (совместимость с MIN_CUDA_MAX_GOOD / Docker runtime) выполняется
    после остальных критериев — отдельным шагом по уже полученным офферам, без cuda_vers в API-запросе.
    """
    oid = o.get("id")
    if oid is not None and int(oid) in BLACKLISTED_OFFER_IDS:
        return (
            "blacklisted",
            f"offer_id={oid} в BLACKLISTED_OFFER_IDS (config.json) — исключён вручную",
        )
    if _f(o.get("num_gpus")) != float(NUM_GPUS):
        return (
            "num_gpus",
            f"нужно {NUM_GPUS}, факт {_f(o.get('num_gpus')):.0f} (offer_id={oid})",
        )
    if require_preferred_gpu and not _is_preferred_gpu(o):
        return (
            "gpu_family",
            f"нужны RTX 4090 или 5090, факт {_gpu_name(o)!r} (offer_id={oid})",
        )
    if o.get("rented") is True or o.get("rentable") is not True:
        return (
            "rentable_rented",
            f"rentable={o.get('rentable')!r}, rented={o.get('rented')!r} (offer_id={oid})",
        )
    if o.get("verified") is not True and str(o.get("verification", "")).lower() != "verified":
        return (
            "verified",
            f"verified={o.get('verified')!r}, verification={o.get('verification')!r} "
            f"(offer_id={oid})",
        )
    if _f(o.get("direct_port_count")) < MIN_DIRECT_PORT_COUNT:
        return (
            "direct_port_count",
            f"мин {MIN_DIRECT_PORT_COUNT}, факт {_f(o.get('direct_port_count')):.0f} "
            f"(offer_id={oid})",
        )
    if _f(o.get("disk_space")) < DISK_SIZE_GB:
        return (
            "disk_space_gb",
            f"мин {DISK_SIZE_GB}, факт {_f(o.get('disk_space')):.1f} (offer_id={oid})",
        )
    if _f(o.get("disk_bw")) < MIN_DISK_BW:
        return (
            "disk_bw_min",
            f"мин {MIN_DISK_BW} МБ/с, факт {_f(o.get('disk_bw')):.0f} (offer_id={oid})",
        )
    if not _disk_ok(o):
        dn = str(o.get("disk_name") or "")
        dbw = _f(o.get("disk_bw"))
        return (
            "disk_nvme_or_fast_bw",
            f"NVMe в имени или disk_bw>={FAST_DISK_BW_OK_WITHOUT_NVME_LABEL}; "
            f"disk_name={dn!r}, disk_bw={dbw:.0f} (offer_id={oid})",
        )
    if _f(o.get("cpu_cores_effective")) < MIN_CPU_CORES_EFFECTIVE:
        return (
            "cpu_cores_effective",
            f"мин {MIN_CPU_CORES_EFFECTIVE}, факт {_f(o.get('cpu_cores_effective')):.3g} "
            f"(offer_id={oid})",
        )
    if _cpu_ram_gb(o) < MIN_CPU_RAM_GB:
        return (
            "cpu_ram_gb",
            f"мин {MIN_CPU_RAM_GB}, факт {_cpu_ram_gb(o):.1f} (offer_id={oid})",
        )
    inet_d = _f(o.get("inet_down"))
    inet_u = _f(o.get("inet_up"))
    if inet_d < MIN_INET_DOWN_MBPS or inet_u < MIN_INET_UP_MBPS:
        return (
            "inet_mbps",
            f"мин down {MIN_INET_DOWN_MBPS}, up {MIN_INET_UP_MBPS}; "
            f"факт down {inet_d:.0f}, up {inet_u:.0f} (offer_id={oid})",
        )
    _inet_cost = _inet_cost_per_gb(o)
    if _inet_cost > MAX_INET_COST_PER_GB:
        return (
            "inet_cost_per_gb",
            f"стоимость трафика {_inet_cost:.4f} $/GB > макс {MAX_INET_COST_PER_GB} $/GB "
            f"(inet_up_billed={o.get('inet_up_billed')}, inet_down_billed={o.get('inet_down_billed')}, "
            f"offer_id={oid})",
        )
    if _hourly_total_usd(o) > MAX_HOURLY_USD:
        return (
            "hourly_usd",
            f"макс {MAX_HOURLY_USD}, факт {_hourly_total_usd(o):.4f} (offer_id={oid})",
        )
    if _f(o.get("reliability")) <= MIN_RELIABILITY:
        return (
            "reliability",
            f"нужно >{MIN_RELIABILITY}, факт {_f(o.get('reliability')):.5f} (offer_id={oid})",
        )
    cuda_mg = o.get("cuda_max_good")
    if cuda_mg is None:
        return (
            "cuda_max_good",
            f"поле cuda_max_good отсутствует (offer_id={oid})",
        )
    if _f(cuda_mg) < MIN_CUDA_MAX_GOOD:
        return (
            "cuda_max_good",
            f"хост cuda_max_good {_f(cuda_mg):.2f} < MIN_CUDA_MAX_GOOD {MIN_CUDA_MAX_GOOD} "
            f"(runtime Docker), offer_id={oid}",
        )
    if not _cpu_avx2_ok(o):
        return (
            "cpu_no_avx2",
            f"CPU {o.get('cpu_name')!r} — Sandy/Ivy Bridge (v1/v2), нет AVX2; "
            f"Docker-образ (polars/numpy/catboost) упадёт с SIGILL (offer_id={oid})",
        )
    return None


def _matches(o: dict[str, Any], *, require_preferred_gpu: bool = True) -> bool:
    return (
        _first_failed_criterion(o, require_preferred_gpu=require_preferred_gpu)
        is None
    )


def _report_manual_filter_failures(
    offers_raw: list[dict[str, Any]],
    *,
    require_preferred_gpu: bool = True,
) -> None:
    """Печатает сводку по первым отсечениям ручного фильтра в stderr."""
    if not offers_raw:
        scope = (
            "RTX_4090 / RTX_5090"
            if require_preferred_gpu
            else "расширенный поиск (любая GPU)"
        )
        print(f"Поиск: API не вернул офферов ({scope}).", file=sys.stderr)
        return

    bucket_counts: Counter[str] = Counter()
    example_detail: dict[str, str] = {}
    none_failures = 0
    for o in offers_raw:
        failed = _first_failed_criterion(
            o, require_preferred_gpu=require_preferred_gpu
        )
        if failed is None:
            none_failures += 1
            continue
        bucket, detail = failed
        bucket_counts[bucket] += 1
        example_detail.setdefault(bucket, detail)

    if none_failures:
        print(
            f"Внимание: {none_failures} кандидат(ов) без отсечения по этой функции "
            "(ожидалось 0 при пустом filtered).",
            file=sys.stderr,
        )

    if not bucket_counts:
        print(
            "Поиск: нет данных для диагностики отсечений.",
            file=sys.stderr,
        )
        return

    print(
        "\nРучной фильтр: ни один оффер не прошёл. "
        "Частоты первого несоответствия (по типу критерия):",
        file=sys.stderr,
    )
    for bucket, cnt in bucket_counts.most_common():
        ex = example_detail.get(bucket, "")
        print(f"  {cnt:>3}×  [{bucket}]  пример: {ex}", file=sys.stderr)


def _print_top(o: dict[str, Any]) -> None:
    rk = _top_rank_tuple(o)
    print("\n=== TOP 1 ===")
    print(f"  rank (cpu_eff, ram_gb, disk_bw, inet, dlperf, dlperf/$) = ({rk[0]:.3g}, {rk[1]:.1f}, {rk[2]:.0f}, {rk[3]:.0f}, {rk[4]:.2f}, {rk[5]:.2f})")
    print(f"  offer_id={o.get('id')}  machine_id={o.get('machine_id')}  host_id={o.get('host_id')}")
    print(
        f"  gpu: {o.get('gpu_name')}  vram_gb={_gpu_ram_gb(o):.1f}  "
        f"cuda_max_good={_f(o.get('cuda_max_good')):.2f}"
    )
    print(f"  cpu: {o.get('cpu_name')}  cores_eff={_f(o.get('cpu_cores_effective')):.3g}  ram_gb={_cpu_ram_gb(o):.1f}")
    print(f"  disk: {o.get('disk_name')}  space_gb={_f(o.get('disk_space')):.1f}  bw={_f(o.get('disk_bw')):.0f}")
    print(f"  net: down={_f(o.get('inet_down')):.0f}  up={_f(o.get('inet_up')):.0f}  inet_cost={_inet_cost_per_gb(o):.4f}$/GB")
    print(f"  price={_hourly_total_usd(o):.4f}$/h  rel={_f(o.get('reliability')):.5f}  geo={o.get('geolocation')}")


def _make_vast(key: str) -> VastAI:
    return VastAI(api_key=key)


def vast_stop_instance(vast: VastAI, instance_id: int) -> dict[str, Any]:
    """Остановить инстанс через API (state=stopped), без уничтожения диска."""
    return vast.stop_instance(instance_id)


def _read_instance_id_optional() -> Optional[int]:
    """INSTANCE_ID из config.json; None если нет или файл битый."""
    try:
        data = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
        raw = data.get("INSTANCE_ID")
        if raw is None:
            return None
        return int(raw)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_search(_args: argparse.Namespace) -> None:
    key = _load_key(_SECRET_ENV)
    vast = _make_vast(key)
    kw = dict(type="on-demand", no_default=False, limit=100,
               disable_bundling=False, storage=DISK_SIZE_GB, order="score")

    print(
        "После выборки офферов: доп. проверка cuda_max_good "
        f"(≥ {MIN_CUDA_MAX_GOOD}, как у CUDA runtime в Docker-образе из config).",
    )

    raw_4090 = vast.search_offers(query=_base_query("RTX_4090"), **kw)
    raw_5090 = vast.search_offers(query=_base_query("RTX_5090"), **kw)
    offers_pref = _merge_unique(raw_4090, raw_5090)

    filtered = [o for o in offers_pref if _matches(o)]
    filtered.sort(key=_top_rank_tuple, reverse=True)

    used_fallback = False
    raw_any: list[dict[str, Any]] = []
    offers_raw: list[dict[str, Any]] = offers_pref

    if not filtered:
        print(
            "Нет подходящих RTX 4090/5090 — расширяем поиск (любая GPU, без gpu_name в запросе).",
            file=sys.stderr,
        )
        raw_any = vast.search_offers(query=_base_query(), **kw)
        offers_raw = raw_any
        filtered = [
            o for o in raw_any if _matches(o, require_preferred_gpu=False)
        ]
        filtered.sort(key=_top_rank_tuple, reverse=True)
        used_fallback = True

    if used_fallback:
        print(
            f"candidates={len(offers_raw)} (любая GPU) filtered={len(filtered)}",
        )
    else:
        print(
            f"candidates={len(offers_pref)} (4090={len(raw_4090)}, 5090={len(raw_5090)}) "
            f"filtered={len(filtered)}",
        )

    best_4090 = max((o for o in filtered if _is_rtx_4090(o)), key=_top_rank_tuple, default=None)
    best_5090 = max((o for o in filtered if _is_rtx_5090(o)), key=_top_rank_tuple, default=None)
    champs = []
    if best_4090:
        champs.append(("RTX 4090", best_4090))
    if best_5090:
        champs.append(("RTX 5090", best_5090))
    if champs:
        label, win = max(champs, key=lambda t: _top_rank_tuple(t[1]))
        wr = _top_rank_tuple(win)
        print(
            f"Best family winner: {label}  id={win.get('id')}  "
            f"cpu_eff={wr[0]:.3g}  disk_bw={wr[2]:.0f}  $/h={_hourly_total_usd(win):.4f}"
        )

    if not filtered:
        print("No offers match manual filter criteria.", file=sys.stderr)
        _report_manual_filter_failures(
            offers_raw,
            require_preferred_gpu=not used_fallback,
        )
        sys.exit(1)

    if used_fallback:
        print(
            "Использован расширенный поиск: BEST_SERVER_ID — лучший среди любых GPU.",
            file=sys.stderr,
        )

    _print_top(filtered[0])
    save_best_server_id(int(filtered[0]["id"]))

    if len(filtered) > 1:
        print("\n--- остальные ---")
        for o in filtered[1:]:
            r = _top_rank_tuple(o)
            print(
                o.get("id"), o.get("gpu_name"),
                f"cpu_eff={r[0]:.3g}", f"ram={r[1]:.1f}GB",
                o.get("disk_name"),
                f"dph={_hourly_total_usd(o):.4f}",
                f"rel={o.get('reliability')}",
            )


def cmd_start(_args: argparse.Namespace) -> None:
    key = _load_key(_SECRET_ENV)
    offer_id = load_best_server_id()
    vast = _make_vast(key)
    print(f"OFFER_ID={offer_id}")

    result = vast.create_instance(
        id=offer_id,
        image=IMAGE,
        disk=DISK_SIZE_GB,
        onstart_cmd="echo started && nvidia-smi",
        python_utf8=True,
        lang_utf8=True,
        cancel_unavail=True,
    )
    instance_id = result["new_contract"]
    print(f"INSTANCE_ID={instance_id}")
    save_instance_id(int(instance_id))
    print(f"Waiting for instance to start...")
    timeout = time.time() + 60 * 5  # 5 minutes
    while True:
        info = vast.show_instance(id=instance_id)
        status = info.get("actual_status")
        print(f"Status: {status} (timeout={timeout - time.time():.0f}s)")
        if status == "running":
            break
        if time.time() > timeout:
            print(f"Instance {instance_id} did not start within 5 minutes. Exiting.")
            try:
                vast.destroy_instance(id=int(instance_id))
                print("destroy_instance: контракт зависшего инстанса удалён.")
            except Exception as de:  # pragma: no cover
                print(f"destroy_instance после timeout: {de}", file=sys.stderr)
            try:
                save_ssh_url("")
            except Exception:  # pragma: no cover
                pass
            sys.exit(1)
        time.sleep(10)

    print(f"Instance {instance_id} started. Getting SSH URL...")
    ssh_url = vast.ssh_url(id=instance_id)
    save_ssh_url(ssh_url)
    print(f"SSH_URL={ssh_url}")


def cmd_artifacts_download(_args: argparse.Namespace) -> None:
    """Скачать ~/avito_cup/data (без зеркала Яндекса) и ~/avito_cup/run/<experiment> (rsync по SSH)."""
    cfg = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
    url_s = cfg.get("SSH_URL")
    if not url_s:
        print("SSH_URL отсутствует в config.json.", file=sys.stderr)
        sys.exit(1)
    u = urlparse(str(url_s))
    if u.scheme != "ssh":
        print(f"Нужен ssh:// URL в SSH_URL, получено {url_s!r}", file=sys.stderr)
        sys.exit(1)
    host = u.hostname or ""
    port = u.port or 22
    user = u.username or "root"
    if not host:
        print("В SSH_URL нет hostname.", file=sys.stderr)
        sys.exit(1)

    exp = (os.environ.get("EXPERIMENT_NAME") or cfg.get("EXPERIMENT_NAME") or "").strip()
    leaf = exp.replace("/", "_").replace("\\", "_") if exp else "run"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = _RESULTS_DIR / ts / leaf
    dest.mkdir(parents=True, exist_ok=True)

    if shutil.which("rsync") is None:
        print(
            "Не найден rsync в PATH. Установите rsync (Git Bash / WSL / Linux).",
            file=sys.stderr,
        )
        sys.exit(1)

    ssh_parts = ["ssh"]
    identity = load_ssh_key_path(_SECRET_ENV)
    if identity:
        ssh_parts.extend(["-i", identity])
    ssh_parts.extend(
        [
            "-p",
            str(port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=45",
            "-o",
            "ServerAliveInterval=25",
            "-o",
            "ServerAliveCountMax=6",
        ]
    )
    rsh = shlex.join(ssh_parts)
    remote = f"{user}@{host}"
    pulls: list[tuple[str, Path, bool]] = [
        (f"{remote}:~/avito_cup/data/", dest / "avito_cup_data", True),
    ]
    if exp:
        pulls.append(
            (f"{remote}:~/avito_cup/run/{exp}/", dest / "run_bundle", False),
        )

    for src_url, dpath, exclude_yandex_mirror in pulls:
        dpath.mkdir(parents=True, exist_ok=True)
        rsync_opts: list[str] = ["-avz"]
        if exclude_yandex_mirror:
            for ex in _YANDEX_DATA_RSYNC_EXCLUDES:
                rsync_opts.extend(["--exclude", ex])
        cmd = ["rsync", *rsync_opts, "-e", rsh, src_url, str(dpath) + "/"]
        last_err: Optional[BaseException] = None
        for attempt in range(1, _RSYNC_RETRIES + 1):
            print(" ".join(cmd), f"(rsync {attempt}/{_RSYNC_RETRIES})")
            try:
                subprocess.run(cmd, check=True)
                last_err = None
                break
            except subprocess.CalledProcessError as e:
                last_err = e
                if attempt < _RSYNC_RETRIES:
                    print(
                        f"rsync не удался, повтор через {_RSYNC_RETRY_DELAY_SEC:.0f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(_RSYNC_RETRY_DELAY_SEC)
        if last_err is not None:
            raise last_err

    print(f"Артефакты сохранены в {dest}")


def cmd_pause_instance(_args: argparse.Namespace) -> None:
    """Только API stop_instance (state=stopped), без destroy."""
    iid = _read_instance_id_optional()
    if iid is None:
        print("pause-instance: INSTANCE_ID нет в config — пропуск.", file=sys.stderr)
        return
    key = _load_key(_SECRET_ENV)
    vast = _make_vast(key)
    print(f"Stopping instance {iid} (API, state=stopped)...")
    try:
        out = vast_stop_instance(vast, iid)
        print(f"stop_instance: {out}")
    except Exception as e:  # pragma: no cover
        print(f"stop_instance warning: {e}", file=sys.stderr)


def cmd_destroy_instance(_args: argparse.Namespace) -> None:
    """Только destroy_instance (удаление контракта)."""
    iid = _read_instance_id_optional()
    if iid is None:
        print("destroy-instance: INSTANCE_ID нет в config — пропуск.", file=sys.stderr)
        return
    key = _load_key(_SECRET_ENV)
    vast = _make_vast(key)
    print(f"Destroying instance {iid}...")
    try:
        result = vast.destroy_instance(id=iid)
        print(f"Destroyed: {result}")
    except Exception as e:  # pragma: no cover
        print(f"destroy_instance: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_stop(_args: argparse.Namespace) -> None:
    cmd_pause_instance(_args)
    cmd_destroy_instance(_args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Управление Vast.ai инстансом: поиск, старт, артефакты, удаление",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("search", help="Найти лучший оффер и сохранить BEST_SERVER_ID")
    sub.add_parser("start", help="Создать инстанс по BEST_SERVER_ID и дождаться running")
    sub.add_parser(
        "artifacts-download",
        help="Скачать артефакты в src/results/... (data/ без Яндекса, run/ целиком)",
    )
    sub.add_parser(
        "pause-instance",
        help="Только stop_instance (API), без destroy",
    )
    sub.add_parser(
        "destroy-instance",
        help="Только destroy инстанса (после pause или если SSH уже недоступен)",
    )
    sub.add_parser(
        "stop",
        help="pause-instance + destroy-instance",
    )

    return parser


_COMMANDS = {
    "search": cmd_search,
    "start": cmd_start,
    "artifacts-download": cmd_artifacts_download,
    "pause-instance": cmd_pause_instance,
    "destroy-instance": cmd_destroy_instance,
    "stop": cmd_stop,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
