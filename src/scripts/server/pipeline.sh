#!/usr/bin/env bash
# Локальный оркестратор Vast.ai: search → start → rsync на сервер → SSH download/train →
# выгрузка артефактов → destroy инстанса.
#
# Переменные:
#   EXPERIMENT_NAME — имя каталога под src/experiments/<имя> (иначе из config.json).
#   PYTHON — интерпретатор (по умолчанию python3).
#
# Запуск (Git Bash / WSL / Linux):
#   export EXPERIMENT_NAME=baseline
#   bash pipeline.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VAST="$SCRIPT_DIR/vast_ai.py"
SSH_ENV="$SCRIPT_DIR/pipeline_ssh_env.py"

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOCAL_EXPERIMENTS="$REPO_ROOT/src/experiments"

# ---------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo "!!! Ошибка (exit_code=$exit_code). Пытаемся удалить инстанс..."
        "$PYTHON" "$VAST" stop || echo "Не удалось выполнить stop — удалите инстанс вручную."
    fi
}
trap cleanup EXIT

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

REMOTE_RUN="~/avito_cup/run/${EXPERIMENT_NAME}"

echo ""
echo "=== [1/6] Поиск сервера ==="
"$PYTHON" "$VAST" search

echo ""
echo "=== [2/6] Запуск инстанса ==="
"$PYTHON" "$VAST" start

eval "$("$PYTHON" "$SSH_ENV")"

SSH_BATCH=(
    ssh
    -p "${PIPELINE_SSH_PORT}"
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    "${PIPELINE_SSH_USER}@${PIPELINE_SSH_HOST}"
)
# Псевдо-TTY: потоковый stdout/stderr в локальный терминал (download/train).
SSH_STREAM=(
    ssh
    -tt
    -p "${PIPELINE_SSH_PORT}"
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    "${PIPELINE_SSH_USER}@${PIPELINE_SSH_HOST}"
)

RSYNC_SSH_CMD="ssh -p ${PIPELINE_SSH_PORT} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

echo ""
echo "=== Синхронизация experiments/${EXPERIMENT_NAME} и скриптов на сервер ==="
"${SSH_BATCH[@]}" "mkdir -p ${REMOTE_RUN}/experiment"
rsync -avz --mkpath -e "$RSYNC_SSH_CMD" \
    "${LOCAL_EXP}/" \
    "${PIPELINE_SSH_USER}@${PIPELINE_SSH_HOST}:${REMOTE_RUN}/experiment/"
for _f in download_data_on_server.sh train_models.sh; do
    rsync -avz -e "$RSYNC_SSH_CMD" \
        "${SCRIPT_DIR}/${_f}" \
        "${PIPELINE_SSH_USER}@${PIPELINE_SSH_HOST}:${REMOTE_RUN}/${_f}"
done
"${SSH_BATCH[@]}" "chmod +x ${REMOTE_RUN}/download_data_on_server.sh ${REMOTE_RUN}/train_models.sh"

echo ""
echo "=== [3/6] Скачивание данных на инстансе (SSH, лог в реальном времени) ==="
"${SSH_STREAM[@]}" env PYTHONUNBUFFERED=1 bash -lc \
    "set -euo pipefail; cd ${REMOTE_RUN} && exec bash ./download_data_on_server.sh"

echo ""
echo "=== [4/6] Обучение на инстансе (SSH, лог в реальном времени) ==="
"${SSH_STREAM[@]}" env PYTHONUNBUFFERED=1 bash -lc \
    "set -euo pipefail; cd ${REMOTE_RUN} && exec bash ./train_models.sh"

echo ""
echo "=== [5/6] Выгрузка артефактов локально ==="
EXPERIMENT_NAME="${EXPERIMENT_NAME}" "$PYTHON" "$VAST" artifacts-download

echo ""
echo "=== [6/6] Удаление инстанса ==="
"$PYTHON" "$VAST" stop

trap - EXIT

echo ""
echo "=== Пайплайн завершён ==="
