"""Настройки: значения читаются из config.json при импорте модуля."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Optional

CONFIG_JSON_PATH: Final[Path] = Path(__file__).resolve().parent / "config.json"

_KEYS_FLOAT = (
    "MAX_HOURLY_USD",
    "MIN_CPU_CORES_EFFECTIVE",
    "MIN_DISK_BW",
    "FAST_DISK_BW_OK_WITHOUT_NVME_LABEL",
    "MIN_RELIABILITY",
    "DISK_SIZE_GB",
    "NUM_GPUS",
    "MIN_CPU_RAM_GB",
    "MIN_INET_DOWN_MBPS",
    "MIN_INET_UP_MBPS",
    "MIN_DIRECT_PORT_COUNT",
    "MIN_CUDA_MAX_GOOD",
)
_KEYS_STR = ("IMAGE",)


def _load_raw() -> dict[str, Any]:
    if not CONFIG_JSON_PATH.is_file():
        raise FileNotFoundError(f"Отсутствует {CONFIG_JSON_PATH}")
    data: dict[str, Any] = json.loads(
        CONFIG_JSON_PATH.read_text(encoding="utf-8"),
    )
    missing = (frozenset(_KEYS_FLOAT) | frozenset(_KEYS_STR)) - frozenset(data)
    if missing:
        raise KeyError(f"В config.json не хватает ключей: {sorted(missing)}")
    for req in ("BEST_SERVER_ID", "INSTANCE_ID", "SSH_URL"):
        if req not in data:
            raise KeyError(f"В config.json должен быть ключ {req}")
    return data


_data = _load_raw()

MAX_HOURLY_USD: float = float(_data["MAX_HOURLY_USD"])
MIN_CPU_CORES_EFFECTIVE: float = float(_data["MIN_CPU_CORES_EFFECTIVE"])
MIN_DISK_BW: float = float(_data["MIN_DISK_BW"])
FAST_DISK_BW_OK_WITHOUT_NVME_LABEL: float = float(
    _data["FAST_DISK_BW_OK_WITHOUT_NVME_LABEL"],
)
MIN_RELIABILITY: float = float(_data["MIN_RELIABILITY"])
DISK_SIZE_GB: float = float(_data["DISK_SIZE_GB"])
NUM_GPUS: int = int(_data["NUM_GPUS"])
MIN_CPU_RAM_GB: float = float(_data["MIN_CPU_RAM_GB"])
MIN_INET_DOWN_MBPS: float = float(_data["MIN_INET_DOWN_MBPS"])
MIN_INET_UP_MBPS: float = float(_data["MIN_INET_UP_MBPS"])
MIN_DIRECT_PORT_COUNT: int = int(_data["MIN_DIRECT_PORT_COUNT"])
MIN_CUDA_MAX_GOOD: float = float(_data["MIN_CUDA_MAX_GOOD"])
IMAGE: str = str(_data["IMAGE"])

_best_raw: Any = _data.get("BEST_SERVER_ID")
BEST_SERVER_ID: Optional[int] = (
    None if _best_raw is None else int(_best_raw)
)

_inst_raw: Any = _data.get("INSTANCE_ID")
INSTANCE_ID: Optional[int] = (
    None if _inst_raw is None else int(_inst_raw)
)

_ssh_raw: Any = _data.get("SSH_URL")
SSH_URL: Optional[str] = None if _ssh_raw is None else str(_ssh_raw)

_experiment_raw: Any = _data.get("EXPERIMENT_NAME")
EXPERIMENT_NAME: str = "" if _experiment_raw is None else str(_experiment_raw)


def save_best_server_id(offer_id: int) -> None:
    """Пишет offer id лучшего сервера в config.json (поле BEST_SERVER_ID)."""
    path = CONFIG_JSON_PATH
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    data["BEST_SERVER_ID"] = int(offer_id)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    globals()["BEST_SERVER_ID"] = int(offer_id)


def save_instance_id(contract_id: int) -> None:
    """Пишет id созданного инстанса в config.json (поле INSTANCE_ID)."""
    path = CONFIG_JSON_PATH
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    data["INSTANCE_ID"] = int(contract_id)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    globals()["INSTANCE_ID"] = int(contract_id)


def save_ssh_url(url: str) -> None:
    """Пишет SSH URL инстанса в config.json (поле SSH_URL)."""
    path = CONFIG_JSON_PATH
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    data["SSH_URL"] = str(url)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    globals()["SSH_URL"] = str(url)


def load_best_server_id() -> int:
    """Читает BEST_SERVER_ID из config.json (после vast_ai search)."""
    path = CONFIG_JSON_PATH
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("BEST_SERVER_ID")
    if raw is None:
        raise ValueError(f"BEST_SERVER_ID не задан в {path}")
    return int(raw)


def load_instance_id() -> int:
    """Читает INSTANCE_ID из config.json (после vast_ai start)."""
    path = CONFIG_JSON_PATH
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("INSTANCE_ID")
    if raw is None:
        raise ValueError(f"INSTANCE_ID не задан в {path}")
    return int(raw)
