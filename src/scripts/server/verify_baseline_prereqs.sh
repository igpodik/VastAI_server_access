#!/usr/bin/env bash
# Проверка данных и окружения перед обучением (без запуска train).
# Поддерживает:
#   - baseline: experiment/baseline.py
#   - new_catboost: experiment/main.py (+ блоки block*.py)
#
# Использование:
#   DATA_DIR=~/avito_cup/data RUN_DIR=~/avito_cup/run/baseline bash verify_baseline_prereqs.sh
#   bash verify_baseline_prereqs.sh [DATA_DIR] [RUN_DIR]
set -euo pipefail

DATA_DIR="${1:-${DATA_DIR:-$HOME/avito_cup/data}}"
RUN_DIR="${2:-${RUN_DIR:-}}"
DATA_DIR="${DATA_DIR/#\~/$HOME}"
if [[ -n "${RUN_DIR}" ]]; then
  RUN_DIR="${RUN_DIR/#\~/$HOME}"
fi

echo "DATA_DIR=${DATA_DIR}"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Каталог данных не найден: ${DATA_DIR}" >&2
  exit 1
fi

need_file() {
  local p="${DATA_DIR%/}/$1"
  if [[ ! -f "${p}" ]]; then
    echo "Отсутствует файл: ${p}" >&2
    exit 1
  fi
}

need_file "item_features.parquet"
need_file "eval_users.csv"
if [[ ! -f "${DATA_DIR%/}/eval_user_events.pq" && ! -f "${DATA_DIR%/}/eval_user_events.parquet" ]]; then
  echo "Нужен eval_user_events.pq или eval_user_events.parquet в ${DATA_DIR}" >&2
  exit 1
fi
need_file "contact_eids.csv"

shopt -s nullglob
parts=( "${DATA_DIR}/train_data"/*.parquet )
shopt -u nullglob
if ((${#parts[@]} == 0)); then
  echo "Нет parquet в ${DATA_DIR}/train_data/" >&2
  exit 1
fi
echo "train_data: ${#parts[@]} parquet-файл(ов)"

if [[ -n "${RUN_DIR}" && -f "${RUN_DIR}/experiment/main.py" ]]; then
  python3 -c "import polars; import catboost; import numpy; import pyarrow; import scipy" 2>/dev/null || {
    echo "Импорт polars/catboost/numpy/pyarrow/scipy не удался (нужен обучающий образ и pip-зависимости)." >&2
    exit 1
  }
  echo "Импорт polars, catboost, numpy, pyarrow, scipy: OK (new_catboost)"
else
  python3 -c "import polars; import catboost; import numpy; import pyarrow" 2>/dev/null || {
    echo "Импорт polars/catboost/numpy/pyarrow не удался (нужен обучающий образ и pip-зависимости)." >&2
    exit 1
  }
  echo "Импорт polars, catboost, numpy, pyarrow: OK"
fi

if [[ -n "${RUN_DIR}" ]]; then
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "RUN_DIR не каталог: ${RUN_DIR}" >&2
    exit 1
  fi
  if [[ -f "${RUN_DIR}/experiment/main.py" ]]; then
    shopt -s nullglob
    py_files=( "${RUN_DIR}/experiment/"*.py )
    shopt -u nullglob
    if ((${#py_files[@]} == 0)); then
      echo "Нет Python-файлов в ${RUN_DIR}/experiment/" >&2
      exit 1
    fi
    for f in "${py_files[@]}"; do
      python3 -m py_compile "${f}"
      echo "py_compile $(basename "${f}"): OK"
    done
    echo "Проверки пройдены (new_catboost / main.py)."
  elif [[ -f "${RUN_DIR}/experiment/baseline.py" ]]; then
    (cd "${RUN_DIR}" && python3 -m py_compile experiment/baseline.py)
    echo "py_compile experiment/baseline.py: OK"
    echo "Проверки пройдены (baseline)."
  else
    echo "Нет ни ${RUN_DIR}/experiment/main.py, ни ${RUN_DIR}/experiment/baseline.py" >&2
    exit 1
  fi
else
  echo "Проверки пройдены (RUN_DIR не задан — только данные и импорты)."
fi
