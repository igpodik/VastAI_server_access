#!/usr/bin/env bash
set -euo pipefail

# Оркестратор запускает этот скрипт из ~/avito_cup/run/baseline
# Сам код бейзлайна оркестратор копирует в подпапку experiment/

echo "Запускаем обучение и инференс CatBoost..."

python -u experiment/baseline.py \
    --data-dir ~/avito_cup/data \
    --out submission.csv \
    -v

echo "Готово! Сабмит сохранен."