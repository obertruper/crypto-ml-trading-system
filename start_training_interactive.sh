#!/bin/bash

echo "🚀 Интерактивный запуск обучения"
echo "================================"
echo ""

# Функция для очистки при выходе
cleanup() {
    echo ""
    echo "🔄 Завершение работы..."
    kill $SSH_PID 2>/dev/null
    exit 0
}

trap cleanup EXIT INT TERM

# Убиваем старые процессы
pkill -f "ssh.*5555:localhost:5555" 2>/dev/null
sleep 1

# SSH туннель
echo "📡 Создание SSH туннеля..."
ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N &
SSH_PID=$!
sleep 3

# Меню выбора
echo ""
echo "Выберите режим мониторинга:"
echo "1) Полный лог в реальном времени"
echo "2) Только ключевые метрики"
echo "3) Запустить в фоне и выйти"
echo ""
read -p "Ваш выбор (1-3): " choice

case $choice in
    1)
        # Полный лог
        ssh -t -p 27681 root@79.116.73.220 'bash -l -c "
            tmux kill-session -t ml_training 2>/dev/null
            tmux new-session -d -s ml_training \"cd /workspace/crypto_trading && source /workspace/venv/bin/activate && python train_universal_transformer.py 2>&1 | tee logs/training_realtime.log\"
            sleep 2
            echo \"📊 ПОЛНЫЙ ЛОГ ОБУЧЕНИЯ (Ctrl+C для выхода)\"
            echo \"===========================================\"
            echo \"\"
            tail -f /workspace/crypto_training/logs/training_realtime.log
        "'
        ;;
        
    2)
        # Только метрики
        ssh -t -p 27681 root@79.116.73.220 'bash -l -c "
            tmux kill-session -t ml_training 2>/dev/null
            tmux new-session -d -s ml_training \"cd /workspace/crypto_training && source /workspace/venv/bin/activate && python train_universal_transformer.py\"
            sleep 2
            echo \"📊 МОНИТОРИНГ МЕТРИК (обновление каждые 5 сек)\"
            echo \"=============================================\"
            echo \"\"
            
            # Получаем путь к последнему логу
            while true; do
                clear
                echo \"📊 СТАТУС ОБУЧЕНИЯ: $(date)\"
                echo \"=====================================\"
                
                # Ищем последний лог
                LATEST_LOG=$(ls -t /workspace/crypto_training/logs/training_*/training.log 2>/dev/null | head -1)
                
                if [ -n \"$LATEST_LOG\" ]; then
                    echo \"📁 Лог: $LATEST_LOG\"
                    echo \"\"
                    
                    # Показываем последние события
                    echo \"📈 Последние метрики:\"
                    grep -E \"(Загружено|записей|Эпоха|loss:|mae:|val_loss|val_mae|accuracy)\" \"$LATEST_LOG\" | tail -15
                    
                    echo \"\"
                    echo \"🔄 Прогресс:\"
                    tail -5 \"$LATEST_LOG\"
                else
                    echo \"⏳ Ожидание начала обучения...\"
                fi
                
                echo \"\"
                echo \"Обновление через 5 секунд... (Ctrl+C для выхода)\"
                sleep 5
            done
        "'
        ;;
        
    3)
        # В фоне
        ssh -p 27681 root@79.116.73.220 << 'EOF'
tmux kill-session -t ml_training 2>/dev/null
tmux new-session -d -s ml_training "cd /workspace/crypto_training && source /workspace/venv/bin/activate && python train_universal_transformer.py"
echo "✅ Обучение запущено в фоне"
echo ""
echo "Для мониторинга используйте:"
echo "ssh -p 27681 root@79.116.73.220 'tmux attach -t ml_training'"
EOF
        ;;
esac