# Vast.ai: локальный запуск пайплайна

## Нужно

- Python 3 + venv, пакет `vastai`
- `ssh`, `rsync` (Git Bash / WSL / Linux)
- SSH-ключ добавлен в аккаунт Vast.ai, локальный вход по ключу без пароля
- Файл `src/scripts/server/secret.env` (см. [`secret_example.env`](src/scripts/server/secret_example.env)):
  - `KEY="<api_key>"` — ключ Vast API
  - `SSH_KEY="/path/to/private_key"` — приватный ключ для `pipeline.sh` (rsync/SSH); в WSL удобнее путь внутри Linux (`~/.ssh/...`), не с Windows-раздела с ослабленными правами

## Docker-образ для GPU (холодный старт без pip на инстансе)

Базовый образ: [`docker/Dockerfile`](docker/Dockerfile) (`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`). В образе через **`apt-get`** (без установки на инстансе при скачивании) ставятся **`ca-certificates`**, **`curl`**, **`unzip`** — они нужны [`download_data_on_server.sh`](src/scripts/server/download_data_on_server.sh). Затем **`pip install -r`** [`docker/requirements-train.txt`](docker/requirements-train.txt): прямые пакеты (`polars-u64-idx`, **`pyarrow`** (Parquet/интеграция с pandas), `lightgbm`, `catboost`, …) и **закреплённые транзитивные** версии.

**`polars-u64-idx`** — та же библиотека Polars с 64-битными индексами; для больших таблиц (co-visitation и т.п.) обычный wheel `polars` может упасть с сообщением вроде `Polars' maximum length reached`.

При несовпадении версии CUDA driver на хосте Vast и runtime в образе подберите другой тег на [Docker Hub PyTorch](https://hub.docker.com/r/pytorch/pytorch/tags).

Сборка и публикация (из корня репозитория):

```bash
docker build -t yourdockerhub/avito-cup-train:v1 -f docker/Dockerfile .
docker push yourdockerhub/avito-cup-train:v1
```

Для GHCR: логин в `ghcr.io`, тег вида `ghcr.io/<org>/<repo>/avito-cup-train:v1`.

В [`src/scripts/server/config.json`](src/scripts/server/config.json) укажите полный тег образа в поле `IMAGE` (например `docker.io/yourdockerhub/avito-cup-train:v1`). После смены Dockerfile или `requirements-train.txt` пересоберите и запушьте образ.

Смоук-тест образа **без датасета и без Vast** (проверка `curl`/`unzip` и импортов, как на инстансе):

```bash
docker run --rm docker.io/igpodik/avito-cup-train:v1 bash -lc \
  'curl --version | head -n1; unzip -v | head -n1; python3 -c "import polars, catboost, pyarrow"'
```

Замените тег на свой из `IMAGE`, если отличается.

## Конфиг

- `config.json` рядом со скриптами: пороги поиска офферов, `IMAGE`, `DISK_SIZE_GB`, **`MIN_CUDA_MAX_GOOD`** — после ответа API у каждого оффера проверяется **`cuda_max_good`** (макс. CUDA по драйверу хоста): значение должно быть **≥** этому порогу (как правило, совпадает с minor CUDA runtime образа, например **`12.1`** для `pytorch/...-cuda12.1-...`). В строку запроса Vast `cuda_vers` **не** подставляется; опционально `EXPERIMENT_NAME`
- После `search` / `start` скриптами подставляются `BEST_SERVER_ID`, `INSTANCE_ID`, `SSH_URL` (не правьте вручную без необходимости)

## Код эксперимента

- Каталог: `src/experiments/<EXPERIMENT_NAME>/` (имя задаётся переменной окружения или полем `EXPERIMENT_NAME` в `config.json`)
- На сервер копируется в `~/avito_cup/run/<EXPERIMENT_NAME>/experiment/`

### Baseline: память и Polars

- **`TRAIN_HOST_RAM_GB`** — оценка RAM хоста в гигабайтах (по умолчанию **128**). В `baseline.py` выставляется **`POLARS_MAX_THREADS = max(1, TRAIN_HOST_RAM_GB // 6)`** (для 128 GiB → **21**), чтобы снизить пик RSS.
- Train читается **по отсортированным файлам** `train_data/*.parquet`: счётчики (`item_pop`, `ucnt`, `icnt`), fallback по `eid`, co-visitation (со стыковкой последнего `item_id` пользователя между файлами) и семантические пулы строятся без лишних полных `scan` по glob и без больших промежуточных дампов на диск.
- В [`train_models.sh`](src/scripts/server/train_models.sh) по умолчанию задаётся `TRAIN_HOST_RAM_GB=128`; при другом объёме памяти переопределите переменную перед запуском.

## Один полный прогон (локально)

```bash
cd src/scripts/server
export EXPERIMENT_NAME=baseline   # или задать в config.json
export PYTHON=python3             # или путь к venv
bash pipeline.sh
```

Шаги: `search` → `start` → rsync → на сервере [`download_data_on_server.sh`](src/scripts/server/download_data_on_server.sh) (параллельные `curl`, **последовательная** распаковка zip в общий `train_data/`, без `apt-get` на лету, при необходимости переименование `eval_user_events.parquet` → `eval_user_events.pq`) → [`verify_baseline_prereqs.sh`](src/scripts/server/verify_baseline_prereqs.sh) (файлы данных, импорты, `py_compile` baseline) → [`train_models.sh`](src/scripts/server/train_models.sh) → `artifacts-download` → `stop`. Перед `search` проверяется `import vastai`. **`artifacts-download`** тянет `~/avito_cup/run/<experiment>/` целиком и только **новые** файлы под `~/avito_cup/data/` (известные файлы с Яндекса и каталог `train_data/` исключены через `rsync --exclude`). **`stop`** = `pause-instance` + `destroy-instance`. При **ошибке** после успешного `eval` SSH `trap` выполняет **`artifacts-download` → `pause-instance` → `destroy-instance`**; если инстанс не дошёл до SSH (`PIPELINE_SSH_HOST` пуст), teardown пропускается. Если **`start`** не дождался `running`, `vast_ai start` сам вызывает `destroy` для контракта и очищает `SSH_URL` в config.

Артефакты локально: `src/results/<YYYYMMDD_HHMMSS>/<EXPERIMENT_NAME>/`.

## Только Vast API (без pipeline)

```bash
cd src/scripts/server
python vast_ai.py search
python vast_ai.py start
python vast_ai.py artifacts-download   # нужен SSH_URL; EXPERIMENT_NAME в env или json
python vast_ai.py pause-instance        # только API stop
python vast_ai.py destroy-instance      # только destroy
python vast_ai.py stop                  # pause + destroy
```

Команда **`search`** сначала ищет офферы **RTX 4090** и **RTX 5090** с теми же порогами, что в запросе; если ни один кандидат не проходит ручной фильтр, выполняется второй запрос **без привязки к имени GPU** (остальные условия те же), чтобы выбрать лучший доступный вариант.

## Полезные пути на сервере

- Данные соревнования: `~/avito_cup/data` (скрипт `download_data_on_server.sh`)
- Прогон эксперимента: `~/avito_cup/run/<EXPERIMENT_NAME>/`

## Ошибки и подсказки

- Нет офферов после `search`: смотрите stderr (сводка по первому несоответствию фильтру)
- **`Connection closed`** сразу после `start` при первом SSH: часто гонка (инстанс уже `running`, SSH ещё не готов); повторите `ssh`/`pipeline` через минуту или проверьте `ssh -vvv` с тем же ключом и портом из `SSH_URL`
- SSH/rsync: проверьте `SSH_KEY`, `SSH_URL`, firewall, `BatchMode=yes` (без интерактива)
- Нет каталога эксперимента: создайте `src/experiments/<EXPERIMENT_NAME>/`
- Обучение падает на Polars «maximum length»: используйте образ с **`polars-u64-idx`** из текущего `requirements-train.txt`
- Ошибки **`unzip` / `checkdir` / `train_data`** при параллельной распаковке: в скрипте скачивания архивы распаковываются **по очереди**; обновите скрипт на сервере и перезапустите шаг данных
