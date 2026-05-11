# Vast.ai: локальный запуск пайплайна

## Нужно

- Python 3 + venv, пакет `vastai`
- `ssh`, `rsync` (Git Bash / WSL / Linux)
- SSH-ключ добавлен в аккаунт Vast.ai, локальный вход по ключу без пароля
- Файл `src/scripts/server/secret.env` (см. [`secret_example.env`](src/scripts/server/secret_example.env)):
  - `KEY="<api_key>"` — ключ Vast API
  - `SSH_KEY="/path/to/private_key"` — приватный ключ для `pipeline.sh` (rsync/SSH); в WSL удобнее путь внутри Linux (`~/.ssh/...`), не с Windows-раздела с ослабленными правами

## Docker-образ для GPU (холодный старт без pip на инстансе)

Базовый образ: [`docker/Dockerfile`](docker/Dockerfile) (`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`). Зависимости обучения перечислены в [`docker/requirements-train.txt`](docker/requirements-train.txt): прямые пакеты (`polars-u64-idx`, `lightgbm`, `catboost`, `scikit-learn`, `tqdm` и др.) и **закреплённые транзитивные** версии (`numpy`, `pandas`, `scipy`, `matplotlib`, …) для воспроизводимости сборки.

**`polars-u64-idx`** — та же библиотека Polars с 64-битными индексами; для больших таблиц (co-visitation и т.п.) обычный wheel `polars` может упасть с сообщением вроде `Polars' maximum length reached`.

При несовпадении версии CUDA driver на хосте Vast и runtime в образе подберите другой тег на [Docker Hub PyTorch](https://hub.docker.com/r/pytorch/pytorch/tags).

Сборка и публикация (из корня репозитория):

```bash
docker build -t yourdockerhub/avito-cup-train:v1 -f docker/Dockerfile .
docker push yourdockerhub/avito-cup-train:v1
```

Для GHCR: логин в `ghcr.io`, тег вида `ghcr.io/<org>/<repo>/avito-cup-train:v1`.

В [`src/scripts/server/config.json`](src/scripts/server/config.json) укажите полный тег образа в поле `IMAGE` (например `docker.io/yourdockerhub/avito-cup-train:v1`). После смены `requirements-train.txt` пересоберите и запушьте образ, чтобы инстансы подтягивали актуальный слой.

## Конфиг

- `config.json` рядом со скриптами: пороги поиска офферов, `IMAGE`, `DISK_SIZE_GB`, опционально `EXPERIMENT_NAME`
- После `search` / `start` скриптами подставляются `BEST_SERVER_ID`, `INSTANCE_ID`, `SSH_URL` (не правьте вручную без необходимости)

## Код эксперимента

- Каталог: `src/experiments/<EXPERIMENT_NAME>/` (имя задаётся переменной окружения или полем `EXPERIMENT_NAME` в `config.json`)
- На сервер копируется в `~/avito_cup/run/<EXPERIMENT_NAME>/experiment/`

## Один полный прогон (локально)

```bash
cd src/scripts/server
export EXPERIMENT_NAME=baseline   # или задать в config.json
export PYTHON=python3             # или путь к venv
bash pipeline.sh
```

Шаги: `search` → `start` → rsync кода и скриптов → на сервере [`download_data_on_server.sh`](src/scripts/server/download_data_on_server.sh) (при необходимости ставит `curl`/`unzip` через `apt-get` или `apk`) → [`train_models.sh`](src/scripts/server/train_models.sh) → `artifacts-download` → `stop`.

Артефакты локально: `src/results/<YYYYMMDD_HHMMSS>/<EXPERIMENT_NAME>/`.

## Только Vast API (без pipeline)

```bash
cd src/scripts/server
python vast_ai.py search
python vast_ai.py start
python vast_ai.py artifacts-download   # нужен SSH_URL; EXPERIMENT_NAME в env или json
python vast_ai.py stop
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
