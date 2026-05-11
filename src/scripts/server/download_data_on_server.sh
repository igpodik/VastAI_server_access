#!/usr/bin/env bash
# Скачивание датасета DataFest с Яндекс-облака на Vast-инстанс.
# Параллельно: curl. Распаковка zip — последовательно (общий каталог train_data/).
#
# Требуется: curl и unzip в PATH (в GPU-образе они ставятся в docker/Dockerfile).
# Число параллельных загрузок: PARALLEL (по умолчанию 8).
# Запуск: bash download_data_on_server.sh

set -euo pipefail

PARALLEL="${PARALLEL:-8}"

mkdir -p ~/avito_cup/data
cd ~/avito_cup/data

if ! command -v curl >/dev/null 2>&1; then
  echo "Нужен curl в PATH (используйте обучающий Docker-образ или установите curl)." >&2
  exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "Нужен unzip в PATH (используйте обучающий Docker-образ или установите unzip)." >&2
  exit 1
fi

BASE="https://storage.yandexcloud.net/datafest2026/datafest_2026_v2_v4"

urls=(
  "${BASE}/item_features.parquet"
  "${BASE}/contact_eids.csv"
  "${BASE}/eval_users.csv"
  "${BASE}/eval_user_events.zip"
  "${BASE}/prepare_local_eval.py"
  "${BASE}/popular.py"
  "${BASE}/submission_popular.csv"
)

for i in 000-019 020-039 040-059 060-079 080-099; do
  urls+=("${BASE}/train_${i}.zip")
done

printf '%s\n' "${urls[@]}" | xargs -n 1 -P "${PARALLEL}" curl -fLSs -O

for z in eval_user_events.zip train_000-019.zip train_020-039.zip train_040-059.zip train_060-079.zip train_080-099.zip; do
  unzip -o -q "${z}"
done

# baseline ожидает eval_user_events.pq; в архиве часто eval_user_events.parquet
if [[ -f eval_user_events.parquet && ! -f eval_user_events.pq ]]; then
  mv -f eval_user_events.parquet eval_user_events.pq
fi

rm -f eval_user_events.zip train_000-019.zip train_020-039.zip train_040-059.zip train_060-079.zip train_080-099.zip

echo "Готово: ~/avito_cup/data"
