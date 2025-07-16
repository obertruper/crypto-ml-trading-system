#!/bin/bash
# Тестовый запуск XGBoost с отладкой

echo "🚀 Запуск XGBoost в режиме отладки..."
echo "Время начала: $(date)"

cd "/Users/ruslan/PycharmProjects/LLM TRANSFORM"

# Запускаем с debug флагом и тестовым режимом БЕЗ кеша
python train_xgboost_enhanced_v2.py \
    --task classification_binary \
    --ensemble_size 1 \
    --test_mode \
    --no-cache \
    --debug

echo "Время окончания: $(date)"
echo "✅ Завершено"