#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VAST="$SCRIPT_DIR/vast_ai.py"
SSH_ENV="$SCRIPT_DIR/pipeline_ssh_env.py"

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOCAL_EXPERIMENTS="$REPO_ROOT/src/experiments"

# ---------------------------------------------------------------------------
# EXPERIMENT_NAME: из окружения или из config.json (без SSH).
# ---------------------------------------------------------------------------
if [[ -z "${EXPERIMENT_NAME:-}" ]]; then
    EXPERIMENT_NAME="$(
        PIPELINE_SCRIPT_DIR="$SCRIPT_DIR" "$PYTHON" -c "
import json
import os
from pathlib import Path

p = Path(os.environ['PIPELINE_SCRIPT_DIR']) / 'config.json'
print(json.loads(p.read_text(encoding='utf-8')).get('EXPERIMENT_NAME') or '')
"
    )"
fi
if [[ -z "${EXPERIMENT_NAME}" ]]; then
    echo "Задайте EXPERIMENT_NAME в окружении или в config.json." >&2
    exit 1
fi

LOCAL_EXP="$LOCAL_EXPERIMENTS/$EXPERIMENT_NAME"
if [[ ! -d "$LOCAL_EXP" ]]; then
    echo "Нет каталога эксперимента: $LOCAL_EXP" >&2
    exit 1
fi

if ! "$PYTHON" -c "import vastai" 2>/dev/null; then
    echo "Нет пакета vastai для $PYTHON. Установите: pip install -r ${SCRIPT_DIR}/requirements.txt" >&2
    exit 1
fi

REMOTE_RUN="~/avito_cup/run/${EXPERIMENT_NAME}"

echo ""
echo "=== [1/7] Поиск сервера ==="
"$PYTHON" "$VAST" search

echo ""
echo "=== [2/7] Запуск инстанса ==="
"$PYTHON" "$VAST" start

eval "$("$PYTHON" "$SSH_ENV")"

# Только после успешного SSH: teardown не трогает чужие INSTANCE_ID из прошлых прогонов.
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo "!!! Ошибка (exit_code=$exit_code)."
        if [[ -z "${PIPELINE_SSH_HOST:-}" ]]; then
            echo "PIPELINE_SSH_HOST пуст — пропуск artifacts/pause/destroy (инстанс не дошёл до eval SSH)."
            return 0
        fi
        echo "artifacts-download → pause-instance → destroy-instance..."
        EXPERIMENT_NAME="${EXPERIMENT_NAME:-}" "$PYTHON" "$VAST" artifacts-download || echo "Не удалось artifacts-download (проверьте SSH_URL и доступ по SSH)."
        "$PYTHON" "$VAST" pause-instance || echo "Не удалось pause-instance."
        "$PYTHON" "$VAST" destroy-instance || echo "Не удалось destroy-instance — удалите инстанс вручную на Vast."
    fi
}
trap cleanup EXIT

SSH_BATCH=(ssh)
if [[ -n "${PIPELINE_SSH_IDENTITY_FILE:-}" ]]; then
    SSH_BATCH+=(-i "$PIPELINE_SSH_IDENTITY_FILE")
fi
SSH_BATCH+=(
    -p "${PIPELINE_SSH_PORT}"
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=45
    -o ServerAliveInterval=25
    -o ServerAliveCountMax=6
    "${PIPELINE_SSH_USER}@${PIPELINE_SSH_HOST}"
)
# Псевдо-TTY: потоковый stdout/stderr в локальный терминал (download/train).
SSH_STREAM=(ssh -tt)
if [[ -n "${PIPELINE_SSH_IDENTITY_FILE:-}" ]]; then
    SSH_STREAM+=(-i "$PIPELINE_SSH_IDENTITY_FILE")
fi
SSH_STREAM+=(
    -p "${PIPELINE_SSH_PORT}"
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=45
    -o ServerAliveInterval=25
    -o ServerAliveCountMax=6
    "${PIPELINE_SSH_USER}@${PIPELINE_SSH_HOST}"
)

RSYNC_SSH_CMD="ssh"
if [[ -n "${PIPELINE_SSH_IDENTITY_FILE:-}" ]]; then
    RSYNC_SSH_CMD+=" -i $(printf '%q' "$PIPELINE_SSH_IDENTITY_FILE")"
fi
RSYNC_SSH_CMD+=" -p $(printf '%q' "$PIPELINE_SSH_PORT") -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o ServerAliveInterval=25 -o ServerAliveCountMax=6"

echo ""
echo "=== Синхронизация experiments/${EXPERIMENT_NAME} и скриптов на сервер ==="
"${SSH_BATCH[@]}" "mkdir -p ${REMOTE_RUN}/experiment"
rsync -avz -e "$RSYNC_SSH_CMD" \
    "${LOCAL_EXP}/" \
    "${PIPELINE_SSH_USER}@${PIPELINE_SSH_HOST}:${REMOTE_RUN}/experiment/"
for _f in download_data_on_server.sh train_models.sh verify_baseline_prereqs.sh; do
    rsync -avz -e "$RSYNC_SSH_CMD" \
        "${SCRIPT_DIR}/${_f}" \
        "${PIPELINE_SSH_USER}@${PIPELINE_SSH_HOST}:${REMOTE_RUN}/${_f}"
done
"${SSH_BATCH[@]}" "chmod +x ${REMOTE_RUN}/download_data_on_server.sh ${REMOTE_RUN}/train_models.sh ${REMOTE_RUN}/verify_baseline_prereqs.sh"

echo ""
echo "=== [2.5/7] Проверка совместимости CPU (AVX2) ==="
"${SSH_BATCH[@]}" "grep -q avx2 /proc/cpuinfo || { echo 'ОШИБКА: CPU не поддерживает AVX2 — несовместим с Docker-образом (polars/numpy/catboost требуют AVX2). Выберите другой сервер.' >&2; exit 1; }; echo CPU_AVX2_OK"

echo ""
echo "=== [3/7] Скачивание данных на инстансе (SSH, лог в реальном времени) ==="
"${SSH_STREAM[@]}" env PYTHONUNBUFFERED=1 bash -lc \
    "set -euo pipefail; cd ${REMOTE_RUN} && exec bash ./download_data_on_server.sh"

echo ""
echo "=== [4/7] Проверка данных и окружения (без обучения) ==="
"${SSH_STREAM[@]}" env PYTHONUNBUFFERED=1 EXPERIMENT_NAME="${EXPERIMENT_NAME}" bash -lc \
    "set -euo pipefail; cd ${REMOTE_RUN} && DATA_DIR=\$HOME/avito_cup/data RUN_DIR=\$HOME/avito_cup/run/${EXPERIMENT_NAME} bash ./verify_baseline_prereqs.sh"

echo ""
echo "=== [5a/7] Фаза features: retrieval + feature engineering → сохранение матриц CatBoost ==="
"${SSH_STREAM[@]}" env PYTHONUNBUFFERED=1 TRAINING_PHASE=features EXPERIMENT_NAME="${EXPERIMENT_NAME}" bash -lc \
    "set -euo pipefail; cd ${REMOTE_RUN} && exec bash ./train_models.sh"

echo ""
echo "=== [5b/7] Скачивание матриц CatBoost локально (перед обучением) ==="
_CB_MATRICES_DIR="${SCRIPT_DIR}/../../results/catboost_matrices"
mkdir -p "${_CB_MATRICES_DIR}"
for _snap in catboost_train_matrix.parquet catboost_feat_matrix.parquet; do
    rsync -az \
        -e "$RSYNC_SSH_CMD" \
        "${PIPELINE_SSH_USER}@${PIPELINE_SSH_HOST}:${REMOTE_RUN}/${_snap}" \
        "${_CB_MATRICES_DIR}/" \
        && echo "  OK: ${_snap}" \
        || echo "  WARN: не удалось скачать ${_snap} (продолжаем)"
done
echo "Матрицы CatBoost → $(cd "${_CB_MATRICES_DIR}" && pwd)"

echo ""
echo "=== [5c/7] Фаза rank: обучение CatBoost + предсказание → submission ==="
"${SSH_STREAM[@]}" env PYTHONUNBUFFERED=1 TRAINING_PHASE=rank EXPERIMENT_NAME="${EXPERIMENT_NAME}" bash -lc \
    "set -euo pipefail; cd ${REMOTE_RUN} && exec bash ./train_models.sh"

echo ""
echo "=== [6/7] Выгрузка артефактов локально ==="
EXPERIMENT_NAME="${EXPERIMENT_NAME}" "$PYTHON" "$VAST" artifacts-download

echo ""
echo "=== [7/7] Удаление инстанса ==="
"$PYTHON" "$VAST" stop

trap - EXIT

echo ""
echo "=== Пайплайн завершён ==="
