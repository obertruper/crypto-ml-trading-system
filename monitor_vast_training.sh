#!/bin/bash

echo "📊 Мониторинг обучения на Vast.ai"
echo "================================="
echo ""

# Функция для получения последнего лога
get_latest_log() {
    ssh -p 27681 root@79.116.73.220 "ls -t /workspace/crypto_trading/logs/training_*/training.log 2>/dev/null | head -1"
}

# Основной цикл мониторинга
while true; do
    clear
    echo "📊 МОНИТОРИНГ ОБУЧЕНИЯ МОДЕЛИ"
    echo "============================="
    echo "Время: $(date)"
    echo ""
    
    # Получаем путь к последнему логу
    LATEST_LOG=$(get_latest_log)
    
    if [ -n "$LATEST_LOG" ]; then
        echo "📁 Лог файл: $LATEST_LOG"
        echo ""
        
        # Показываем последние строки лога
        echo "📋 Последние события:"
        echo "--------------------"
        ssh -p 27681 root@79.116.73.220 "tail -20 $LATEST_LOG" 2>/dev/null || echo "Ошибка чтения лога"
        
        echo ""
        echo "📈 Статистика:"
        echo "-------------"
        # Ищем ключевые метрики
        ssh -p 27681 root@79.116.73.220 "grep -E '(Загружено|записей|эпоха|loss|accuracy|MAE)' $LATEST_LOG | tail -10" 2>/dev/null || echo "Ждем метрики..."
        
    else
        echo "⚠️ Лог файл не найден. Проверяем tmux сессию..."
        ssh -p 27681 root@79.116.73.220 "tmux ls" 2>/dev/null || echo "Tmux сессия не найдена"
    fi
    
    echo ""
    echo "🔄 Обновление через 10 секунд... (Ctrl+C для выхода)"
    echo ""
    echo "💡 Команды:"
    echo "  - Подключиться к сессии: ssh -p 27681 root@79.116.73.220 'tmux attach -t ml_training'"
    echo "  - Проверить GPU: ssh -p 27681 root@79.116.73.220 'nvidia-smi'"
    
    sleep 10
done