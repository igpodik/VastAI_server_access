#!/usr/bin/env bash
# Скачивание датасета DataFest с Яндекс-облака на Vast-инстанс (параллельные curl и unzip).
#
# Число параллельных загрузок: PARALLEL (по умолчанию 8).
# Запуск: bash download_data_on_server.sh

set -euo pipefail

PARALLEL="${PARALLEL:-8}"

mkdir -p ~/avito_cup/data
cd ~/avito_cup/data

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

pids=()
for z in eval_user_events.zip train_000-019.zip train_020-039.zip train_040-059.zip train_060-079.zip train_080-099.zip; do
  unzip -o -q "${z}" &
  pids+=($!)
done
_st=0
for pid in "${pids[@]}"; do
  wait "${pid}" || _st=1
done
if ((_st)); then
  exit "${_st}"
fi

rm -f eval_user_events.zip train_000-019.zip train_020-039.zip train_040-059.zip train_060-079.zip train_080-099.zip

echo "Готово: ~/avito_cup/data"
