#!/bin/bash

# Скрипт для запуска оптимизированного обучения XGBoost v3

echo "🚀 Запуск оптимизированного обучения XGBoost v3.0"
echo "=================================================="

# Активируем виртуальное окружение если есть
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# Очистка кэша для свежих данных
echo "🧹 Очистка кэша..."
rm -f cache/test_data.parquet

# Запуск с оптимизацией
echo "🎯 Запуск обучения с оптимизацией гиперпараметров..."
echo ""

python main.py \
    --test-mode \
    --optimize \
    --ensemble-size 5 \
    --task classification_binary

echo ""
echo "✅ Обучение завершено!"
echo ""
echo "📊 Результаты сохранены в папке logs/"
echo "📈 Для просмотра графиков откройте logs/xgboost_v3_*/plots/"