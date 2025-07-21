#!/bin/bash

# Post-bash hook для ML Crypto Trading проекта
# Мониторит выполнение bash команд и предлагает оптимизации

COMMAND="$1"
EXIT_CODE="$2"
OUTPUT="$3"

LOG_FILE="/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading/logs/hooks.log"
echo "[$(date)] POST-BASH: Command=$COMMAND, ExitCode=$EXIT_CODE" >> "$LOG_FILE"

# Проверяем команды связанные с обучением
if [[ "$COMMAND" == *"python"*"train"* ]] || [[ "$COMMAND" == *"main.py"* ]]; then
    if [ "$EXIT_CODE" = "0" ]; then
        echo "🚀 Обучение запущено успешно!"
        echo "💡 Подсказки:"
        echo "   - Используйте monitor_training.py для мониторинга в реальном времени"
        echo "   - TensorBoard: tensorboard --logdir logs/"
        echo "   - Графики обновляются каждые 5 эпох в logs/training_*/plots/"
    else
        echo "❌ Ошибка при запуске обучения"
        echo "🔍 Проверьте:"
        echo "   - Доступность GPU (nvidia-smi)"
        echo "   - Наличие данных в БД"
        echo "   - Корректность config.yaml"
    fi
fi

# Проверяем команды работы с данными
if [[ "$COMMAND" == *"download_data"* ]] || [[ "$COMMAND" == *"prepare_dataset"* ]]; then
    echo "📊 Работа с данными"
    echo "💡 Полезные команды:"
    echo "   - python validate_futures_symbols.py - проверка символов"
    echo "   - python check_history_depth.py - проверка глубины данных"
fi

# Проверяем git команды
if [[ "$COMMAND" == "git status"* ]]; then
    echo "📝 Статус репозитория проверен"
    echo "💡 Важные файлы для коммита:"
    echo "   - models/patchtst*.py - архитектура модели"
    echo "   - config/config.yaml - конфигурация"
    echo "   - Не коммитьте: logs/, __pycache__, *.pth"
fi

# Мониторинг ресурсов при длительных операциях
if [[ "$COMMAND" == *"python"* ]] && [ "$EXIT_CODE" = "0" ]; then
    # Проверяем использование GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$GPU_USAGE" ]; then
            echo "🎮 GPU использование: $GPU_USAGE%"
        fi
    fi
fi

exit 0