#!/usr/bin/env bash
# Скачивание датасета DataFest с Яндекс-облака на Vast-инстанс.
# При необходимости ставит curl/unzip (apt-get или apk), затем параллельный curl и unzip.
#
# Число параллельных загрузок: PARALLEL (по умолчанию 8).
# Запуск: bash download_data_on_server.sh

set -euo pipefail

PARALLEL="${PARALLEL:-8}"

mkdir -p ~/avito_cup/data
cd ~/avito_cup/data

# Минимальные образы (PyTorch runtime и т.п.) часто без unzip/curl — ставим из репозитория.
_ensure_curl_unzip() {
  if command -v curl >/dev/null 2>&1 && command -v unzip >/dev/null 2>&1; then
    return 0
  fi
  export DEBIAN_FRONTEND=noninteractive
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl unzip
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache curl unzip ca-certificates
  fi
  command -v curl >/dev/null 2>&1 || {
    echo "curl не найден и не установлен (нужен apt-get или apk)." >&2
    exit 1
  }
}
_ensure_curl_unzip

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
  (
    if command -v unzip >/dev/null 2>&1; then
      unzip -o -q "${z}"
    else
      ZIP="${z}" python3 -c 'import os, zipfile; zipfile.ZipFile(os.environ["ZIP"]).extractall(".")'
    fi
  ) &
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
