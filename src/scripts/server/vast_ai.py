"""Единый скрипт управления Vast.ai инстансом.

Использование:
    python vast_ai.py search              -- найти лучший оффер
    python vast_ai.py start               -- создать инстанс
    python vast_ai.py data-download       -- выгрузить данные с Яндекс-облака на инстанс
    python vast_ai.py train               -- запустить обучение
    python vast_ai.py artifacts-download  -- скачать артефакты локально
    python vast_ai.py stop                -- удалить инстанс
"""

from __future__ import annotations

import argparse
import json
import os
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
    CONFIG_JSON_PATH,
    DISK_SIZE_GB,
    FAST_DISK_BW_OK_WITHOUT_NVME_LABEL,
    IMAGE,
    MAX_HOURLY_USD,
    MIN_CPU_CORES_EFFECTIVE,
    MIN_CPU_RAM_GB,
    MIN_DIRECT_PORT_COUNT,
    MIN_DISK_BW,
    MIN_INET_DOWN_MBPS,
    MIN_INET_UP_MBPS,
    MIN_RELIABILITY,
    NUM_GPUS,
    load_best_server_id,
    load_instance_id,
    save_best_server_id,
    save_instance_id,
    save_ssh_url,
)

# ---------------------------------------------------------------------------
# Yandex Storage dataset URLs
# ---------------------------------------------------------------------------
_BASE_URL = "https://storage.yandexcloud.net/datafest2026/datafest_2026_v2_v4"
_SINGLE_FILES = [
    "item_features.parquet",
    "contact_eids.csv",
    "eval_users.csv",
    "eval_user_events.zip",
    "prepare_local_eval.py",
    "popular.py",
    "submission_popular.csv",
]
_TRAIN_SHARDS = ["000-019", "020-039", "040-059", "060-079", "080-099"]

# Remote workspace paths (on the instance)
_REMOTE_DATA_DIR = "/workspace/data"
_REMOTE_RESULTS_DIR = "/workspace/results"
_REMOTE_TRAIN_SCRIPT = "/workspace/train.py"

# Локальный каталог артефактов (выгрузка с инстанса): repo/src/experiments/<timestamp>_<run>/
_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"

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


def _allowed_gpu(o: dict[str, Any]) -> bool:
    return _is_rtx_4090(o) or _is_rtx_5090(o)


def _disk_ok(o: dict[str, Any]) -> bool:
    disk_name = str(o.get("disk_name") or "").lower()
    if "nvme" in disk_name:
        return True
    return _f(o.get("disk_bw")) >= FAST_DISK_BW_OK_WITHOUT_NVME_LABEL


def _base_query(gpu_token: str) -> str:
    return (
        f"gpu_name={gpu_token} "
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


def _first_failed_criterion(o: dict[str, Any]) -> Optional[tuple[str, str]]:
    """Первое несоответствие: (короткий_ключ_группы, подробное сообщение). OK → None."""
    oid = o.get("id")
    if _f(o.get("num_gpus")) != float(NUM_GPUS):
        return (
            "num_gpus",
            f"нужно {NUM_GPUS}, факт {_f(o.get('num_gpus')):.0f} (offer_id={oid})",
        )
    if not _allowed_gpu(o):
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
    return None


def _matches(o: dict[str, Any]) -> bool:
    return _first_failed_criterion(o) is None


def _report_manual_filter_failures(offers_raw: list[dict[str, Any]]) -> None:
    """Печатает сводку по первым отсечениям ручного фильтра в stderr."""
    if not offers_raw:
        print(
            "Поиск: API не вернул офферов по запросам RTX_4090 / RTX_5090.",
            file=sys.stderr,
        )
        return

    bucket_counts: Counter[str] = Counter()
    example_detail: dict[str, str] = {}
    none_failures = 0
    for o in offers_raw:
        failed = _first_failed_criterion(o)
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
    print(f"  gpu: {o.get('gpu_name')}  vram_gb={_gpu_ram_gb(o):.1f}")
    print(f"  cpu: {o.get('cpu_name')}  cores_eff={_f(o.get('cpu_cores_effective')):.3g}  ram_gb={_cpu_ram_gb(o):.1f}")
    print(f"  disk: {o.get('disk_name')}  space_gb={_f(o.get('disk_space')):.1f}  bw={_f(o.get('disk_bw')):.0f}")
    print(f"  net: down={_f(o.get('inet_down')):.0f}  up={_f(o.get('inet_up')):.0f}")
    print(f"  price={_hourly_total_usd(o):.4f}$/h  rel={_f(o.get('reliability')):.5f}  geo={o.get('geolocation')}")


def _make_vast(key: str) -> VastAI:
    return VastAI(api_key=key)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_search(_args: argparse.Namespace) -> None:
    key = _load_key(_SECRET_ENV)
    vast = _make_vast(key)
    kw = dict(type="on-demand", no_default=False, limit=100,
               disable_bundling=False, storage=DISK_SIZE_GB, order="score")

    raw_4090 = vast.search_offers(query=_base_query("RTX_4090"), **kw)
    raw_5090 = vast.search_offers(query=_base_query("RTX_5090"), **kw)
    offers_raw = _merge_unique(raw_4090, raw_5090)

    filtered = [o for o in offers_raw if _matches(o)]
    filtered.sort(key=_top_rank_tuple, reverse=True)

    print(
        f"candidates={len(offers_raw)} (4090={len(raw_4090)} + 5090={len(raw_5090)}) "
        f"filtered={len(filtered)}"
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
        _report_manual_filter_failures(offers_raw)
        sys.exit(1)

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
        onstart_cmd="echo started && nvidia-smi && pip install --upgrade pip && pip install -U polars lightgbm catboost scikit-learn tqdm",
        python_utf8=True,
        lang_utf8=True,
        cancel_unavail=True,
    )
    instance_id = result["new_contract"]
    print(f"INSTANCE_ID={instance_id}")
    save_instance_id(int(instance_id))
    print(f"Waiting for instance to start...")
    timeout = time.time() + 60 * 3  # 3 minutes
    while True:
        info = vast.show_instance(id=instance_id)
        status = info.get("actual_status")
        print(f"Status: {status} (timeout={timeout - time.time():.0f}s)")
        if status == "running":
            break
        if time.time() > timeout:
            print(f"Instance {instance_id} did not start within 5 minutes. Exiting.")
            sys.exit(1)
        time.sleep(10)

    print(f"Instance {instance_id} started. Getting SSH URL...")
    ssh_url = vast.ssh_url(id=instance_id)
    save_ssh_url(ssh_url)
    print(f"SSH_URL={ssh_url}")


def cmd_data_download(_args: argparse.Namespace) -> None:
    """Скачать датасет с Яндекс-облака на инстанс по SSH."""
    instance_id = load_instance_id()
    # TODO: реализовать через SSH-вызов на инстансе:
    #   ssh <instance> "mkdir -p /workspace/data && cd /workspace/data && ..."
    #   Список URL:
    #     {_BASE_URL}/{f} для f in _SINGLE_FILES
    #     {_BASE_URL}/train_{shard}.zip для shard in _TRAIN_SHARDS
    print(f"TODO: data-download to instance {instance_id}")
    print(f"      BASE_URL={_BASE_URL}")
    print(f"      single files: {_SINGLE_FILES}")
    print(f"      train shards: {[f'train_{s}.zip' for s in _TRAIN_SHARDS]}")
    print(f"      remote target: {_REMOTE_DATA_DIR}")


def cmd_train(_args: argparse.Namespace) -> None:
    """Запустить entrypoint обучения на инстансе по SSH."""
    instance_id = load_instance_id()
    # TODO: реализовать SSH-вызов:
    #   ssh <instance> "cd /workspace && python train.py"
    print(f"TODO: train on instance {instance_id}")
    print(f"      remote script: {_REMOTE_TRAIN_SCRIPT}")


def cmd_artifacts_download(_args: argparse.Namespace) -> None:
    """Скачать ~/avito_cup/data и ~/avito_cup/run/<experiment> с инстанса (rsync по SSH)."""
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
    safe = exp.replace("/", "_").replace("\\", "_") or "run"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = _EXPERIMENTS_DIR / f"{ts}_{safe}"
    dest.mkdir(parents=True, exist_ok=True)

    if shutil.which("rsync") is None:
        print(
            "Не найден rsync в PATH. Установите rsync (Git Bash / WSL / Linux).",
            file=sys.stderr,
        )
        sys.exit(1)

    rsh = (
        f"ssh -p {port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    )
    remote = f"{user}@{host}"
    pulls: list[tuple[str, Path]] = [
        (f"{remote}:~/avito_cup/data/", dest / "avito_cup_data"),
    ]
    if exp:
        pulls.append(
            (f"{remote}:~/avito_cup/run/{exp}/", dest / "run_bundle"),
        )

    for src_url, dpath in pulls:
        dpath.mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", "-avz", "-e", rsh, src_url, str(dpath) + "/"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"Артефакты сохранены в {dest}")


def cmd_stop(_args: argparse.Namespace) -> None:
    key = _load_key(_SECRET_ENV)
    instance_id = load_instance_id()
    vast = _make_vast(key)
    print(f"Destroying instance {instance_id}...")
    result = vast.destroy_instance(id=instance_id)
    print(f"Destroyed: {result}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Управление Vast.ai инстансом (поиск / старт / данные / обучение / артефакты / удаление)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("search", help="Найти лучший оффер и сохранить BEST_SERVER_ID")
    sub.add_parser("start", help="Создать инстанс по BEST_SERVER_ID и дождаться running")
    sub.add_parser("data-download", help="Скачать датасет с Яндекс-облака на инстанс")
    sub.add_parser("train", help="Запустить entrypoint обучения на инстансе")
    sub.add_parser("artifacts-download", help="Скачать артефакты с инстанса в experiments/<datetime>/")
    sub.add_parser("stop", help="Удалить (destroy) инстанс")

    return parser


_COMMANDS = {
    "search": cmd_search,
    "start": cmd_start,
    "data-download": cmd_data_download,
    "train": cmd_train,
    "artifacts-download": cmd_artifacts_download,
    "stop": cmd_stop,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
