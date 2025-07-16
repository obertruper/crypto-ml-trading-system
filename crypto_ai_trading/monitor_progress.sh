#!/bin/bash
# Мониторинг прогресса обучения в реальном времени

echo "📊 Мониторинг прогресса обучения на Vast.ai"
echo "============================================"

while true; do
    clear
    echo "📊 Прогресс обучения - $(date)"
    echo "============================================"
    
    # Получаем последние строки с прогрессом
    ssh -p 30197 root@ssh6.vast.ai "cd /root/crypto_ai_trading && tail -30 training_progress.log | grep -E 'Создание признаков:|Нормализация:|Epoch|loss|этап|Train:|Val:' | tail -20"
    
    echo ""
    echo "🖥️ Использование GPU:"
    ssh -p 30197 root@ssh6.vast.ai "nvidia-smi --query-gpu=gpu_name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf \"  %s: %s%% GPU, %s/%s MB\\n\", \$1, \$2, \$3, \$4}'"
    
    echo ""
    echo "Нажмите Ctrl+C для выхода. Обновление через 10 секунд..."
    sleep 10
done