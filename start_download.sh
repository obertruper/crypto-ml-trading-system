#!/bin/bash
cd /mnt/SSD/PYCHARMPRODJECT/LLM\ TRANSFORM
source venv/bin/activate

echo "🚀 Запуск загрузки данных с 25 потоками..."
echo "📅 Время старта: $(date)"

# Запускаем с выводом в лог
python download_data.py > download_full.log 2>&1 &
PID=$!

echo "✅ Процесс запущен с PID: $PID"
echo "📋 Логи: download_full.log"
echo ""
echo "Команды мониторинга:"
echo "  tail -f download_full.log    # смотреть лог в реальном времени"
echo "  ps aux | grep $PID          # проверить процесс"
echo "  kill $PID                    # остановить загрузку"

# Сохраняем PID
echo $PID > download.pid