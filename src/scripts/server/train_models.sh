#!/usr/bin/env bash
set -euo pipefail

_ACTUAL_RAM_GB=""
if [[ -f /proc/meminfo ]]; then
    _ACTUAL_RAM_GB="$(awk '/MemTotal/ { printf "%.0f", $2/1024/1024 }' /proc/meminfo 2>/dev/null || echo "")"
fi
if [[ -z "${_ACTUAL_RAM_GB}" ]] || ! [[ "${_ACTUAL_RAM_GB}" =~ ^[0-9]+$ ]] || (( _ACTUAL_RAM_GB <= 0 )); then
    _ACTUAL_RAM_GB="${TRAIN_HOST_RAM_GB:-128}"
fi
export TRAIN_HOST_RAM_GB="${_ACTUAL_RAM_GB}"

# --- Polars / память (до запуска python): только доля RAM и лимит в байтах ---
#   POLARS_RAM_PERCENT (1–100) — доля MemTotal → бюджет в ГиБ, затем в байтах.
#   MEMORY_LIMIT и POLARS_MEMORY_LIMIT — одинаковое целое (байты), для кода/инструментов.
#   Polars OSS может не читать эти переменные сам; main.py и др. могут использовать явно.

_POLARS_RAM_PERCENT="${POLARS_RAM_PERCENT:-85}"
if ! [[ "${_POLARS_RAM_PERCENT}" =~ ^[0-9]+$ ]] || (( _POLARS_RAM_PERCENT < 1 )); then
    _POLARS_RAM_PERCENT=85
fi
if (( _POLARS_RAM_PERCENT > 100 )); then
    _POLARS_RAM_PERCENT=95
fi

_RAM_BUDGET_GB=$(( _ACTUAL_RAM_GB * _POLARS_RAM_PERCENT / 100 ))
(( _RAM_BUDGET_GB < 1 )) && _RAM_BUDGET_GB=1
_MEMORY_LIMIT_BYTES=$(( _RAM_BUDGET_GB * 1024 * 1024 * 1024 ))

export POLARS_RAM_PERCENT="${_POLARS_RAM_PERCENT}"
export MEMORY_LIMIT="${_MEMORY_LIMIT_BYTES}"
export POLARS_MEMORY_LIMIT="${_MEMORY_LIMIT_BYTES}"
echo "RAM=${TRAIN_HOST_RAM_GB}GB  POLARS_RAM_PERCENT=${POLARS_RAM_PERCENT}  ram_budget=${_RAM_BUDGET_GB}GiB  MEMORY_LIMIT=${MEMORY_LIMIT} (bytes)"

RUN_ROOT="$(pwd)"
if [[ -f experiment/main.py ]]; then
  # TRAINING_PHASE передаётся через env в pipeline.sh, но login-shell (bash -lc) может
  # сбрасывать переменные окружения при conda init. Читаем явно и экспортируем заново.
  export _PHASE="${TRAINING_PHASE:-all}"

  # Проверяем реальный импорт LightFM (не просто import lightfm — тот грузит только __init__.py).
  python3 -c "from lightfm import LightFM; from lightfm.data import Dataset" 2>/dev/null \
    && echo "lightfm: present (LightFM class importable)" \
    || echo "lightfm: LightFM class not importable — block will be skipped (non-fatal)"

  echo "Запуск new_catboost --phase=${_PHASE} (python3 -u main.py из experiment/)..."
  (
    cd "${RUN_ROOT}/experiment" || exit 1
    python3 -u main.py \
      --data-dir "${HOME}/avito_cup/data" \
      --out "${RUN_ROOT}/submission.csv" \
      --phase "${_PHASE}" \
      -v
  )
elif [[ -f experiment/baseline.py ]]; then
  echo "Запуск baseline (experiment/baseline.py)..."
  python3 -u experiment/baseline.py \
    --data-dir "${HOME}/avito_cup/data" \
    --out submission.csv \
    -v
else
  echo "Не найден experiment/main.py и experiment/baseline.py (cwd=$(pwd))." >&2
  exit 1
fi

echo "Готово! Сабмит сохранен."
