#!/bin/bash
# Финальный скрипт запуска обучения с унифицированной моделью

echo "🚀 Запуск обучения с UnifiedPatchTST (36 выходов)..."
echo "📊 Конфигурация:"
echo "  - Learning Rate: 0.001 (увеличен в 10 раз)"
echo "  - Batch Size: 128 (уменьшен для частых обновлений)"
echo "  - Scheduler: OneCycleLR"
echo "  - Model: UnifiedPatchTST с 36 выходами"

# Активация окружения
if [ -f "venv_crypto/bin/activate" ]; then
    source venv_crypto/bin/activate
fi

# Проверка checkpoint
CHECKPOINT="models_saved/best_model_20250701_120952.pth"
if [ -f "$CHECKPOINT" ]; then
    echo "✅ Найден checkpoint: $CHECKPOINT"
    read -p "Продолжить с checkpoint? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESUME="--resume $CHECKPOINT"
    else
        RESUME=""
    fi
else
    RESUME=""
fi

# Запуск обучения
echo "🏃 Запускаем обучение..."
python main.py --mode train \
    --config config/config.yaml \
    --log_every 50 \
    --save_every 1 \
    $RESUME

echo "✅ Готово!"
