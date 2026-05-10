# Vast.ai: локальный запуск пайплайна

## Нужно

- Python 3 + venv, пакет `vastai`
- `ssh`, `rsync` (Git Bash / WSL / Linux)
- SSH-ключ зарегистрирован в Vast, вход без пароля
- Файл `secret.env`: строка `KEY="<api_key>"`

## Docker-образ для GPU (холодный старт без pip на инстансе)

Зависимости для обучения (`polars`, `lightgbm`, `catboost`, `scikit-learn`, `tqdm`) закреплены в [`docker/requirements-train.txt`](docker/requirements-train.txt); базовый образ — [`docker/Dockerfile`](docker/Dockerfile) (`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`). При несовпадении CUDA driver на хосте Vast и runtime в образе подберите другой тег на [Docker Hub PyTorch](https://hub.docker.com/r/pytorch/pytorch/tags).

Сборка и публикация (из корня репозитория):

```bash
docker build -t yourdockerhub/avito-cup-train:v1 -f docker/Dockerfile .
docker push yourdockerhub/avito-cup-train:v1
```

Для GHCR: логин в `ghcr.io`, тег вида `ghcr.io/<org>/<repo>/avito-cup-train:v1`.

В [`src/scripts/server/config.json`](src/scripts/server/config.json) укажите полный тег образа (например `docker.io/yourdockerhub/avito-cup-train:v1` или префикс GHCR). Замените `yourdockerhub` на свой логин или организацию.

## Конфиг

- `config.json` рядом со скриптами: пороги поиска, `IMAGE`, `DISK_SIZE_GB`, опционально `EXPERIMENT_NAME`
- После `search` / `start` подставляются `BEST_SERVER_ID`, `INSTANCE_ID`, `SSH_URL` (не править руками без нужды)

## Код эксперимента

- Каталог: `src/experiments/<EXPERIMENT_NAME>/` (имя = в env или в `config.json`)
- На сервер уходит в `~/avito_cup/run/<EXPERIMENT_NAME>/experiment/`

## Один полный прогон (локально)

```bash
cd src/scripts/server
export EXPERIMENT_NAME=baseline   # или задать в config.json
export PYTHON=python3             # или путь к venv
bash pipeline.sh
```

Шаги: `search` -> `start` -> rsync -> на сервере `download_data_on_server.sh` -> `train_models.sh` -> `artifacts-download` -> `stop`.

Артефакты: `src/results/<YYYYMMDD_HHMMSS>/<EXPERIMENT_NAME>/`.

## Только Vast API (без pipeline)

```bash
python vast_ai.py search
python vast_ai.py start
python vast_ai.py artifacts-download   # нужен SSH_URL; EXPERIMENT_NAME в env или json
python vast_ai.py stop
```

## Полезные пути на сервере

- Данные соревнования: `~/avito_cup/data` (скрипт `download_data_on_server.sh`)
- Прогон эксперимента: `~/avito_cup/run/<EXPERIMENT_NAME>/`

## Ошибки

- Нет офферов: см. stderr после `search` (разбивка фильтра)
- SSH/rsync: проверить ключ, `SSH_URL`, firewall
- Нет каталога эксперимента: создать `src/experiments/<EXPERIMENT_NAME>/`
