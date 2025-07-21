#!/bin/bash

# Stop hook для ML Crypto Trading проекта
# Выполняет финальные проверки и рекомендации

LOG_FILE="/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading/logs/hooks.log"
echo "[$(date)] STOP-HOOK: Session completed" >> "$LOG_FILE"

echo "🏁 Сессия завершена"
echo ""

# Проверяем незакоммиченные изменения
cd "/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading" 2>/dev/null
if [ $? -eq 0 ]; then
    CHANGES=$(git status --porcelain 2>/dev/null | wc -l)
    if [ "$CHANGES" -gt 0 ]; then
        echo "📝 Обнаружены незакоммиченные изменения ($CHANGES файлов)"
        echo "   Используйте 'git status' для просмотра"
        echo ""
    fi
fi

# Проверяем запущенные процессы обучения
TRAINING_PIDS=$(pgrep -f "train.*\.py" 2>/dev/null)
if [ -n "$TRAINING_PIDS" ]; then
    echo "🚀 Обнаружены активные процессы обучения:"
    for pid in $TRAINING_PIDS; do
        echo "   PID: $pid - $(ps -p $pid -o comm= 2>/dev/null)"
    done
    echo "   Используйте 'monitor_training.py' для мониторинга"
    echo ""
fi

# Проверяем последние логи
LATEST_LOG=$(find "/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading/logs" -name "training_*" -type d 2>/dev/null | sort -r | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "📊 Последняя сессия обучения:"
    echo "   $LATEST_LOG"
    if [ -f "$LATEST_LOG/final_report.txt" ]; then
        echo "   ✅ Финальный отчет доступен"
    fi
    echo ""
fi

# Полезные напоминания
echo "💡 Полезные команды:"
echo "   - tensorboard --logdir logs/ - визуализация метрик"
echo "   - python monitor_training.py - мониторинг в реальном времени"
echo "   - ./start_metabase.sh - аналитика данных"
echo ""

echo "✨ Спасибо за использование ML Crypto Trading System!"

exit 0