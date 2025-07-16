#!/bin/bash

echo "🚀 Запуск стабильного обучения Enhanced TFT v2.1"
echo "================================================"

# Очистка старых процессов
pkill -f "ssh.*5555" 2>/dev/null
sleep 2

# Функция для проверки и восстановления туннеля
check_tunnel() {
    if ! ssh -p 27681 root@79.116.73.220 "timeout 2 nc -zv localhost 5555" 2>&1 | grep -q "open"; then
        echo "🔄 Восстанавливаем туннель..."
        pkill -f "ssh.*5555" 2>/dev/null
        ssh -f -N -p 27681 root@79.116.73.220 -R 5555:localhost:5555 \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes
        sleep 3
    fi
}

# Создаем начальный туннель
echo "🔄 Создание SSH туннеля..."
ssh -f -N -p 27681 root@79.116.73.220 -R 5555:localhost:5555 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes

sleep 3

# Проверка туннеля
check_tunnel

echo "✅ Туннель создан"

# Запуск обучения на сервере в tmux
echo "🚀 Запуск обучения в tmux..."
ssh -p 27681 root@79.116.73.220 << 'EOF'
# Убиваем старые сессии tmux
tmux kill-session -t training 2>/dev/null

# Создаем новую tmux сессию
tmux new-session -d -s training

# Отправляем команды в tmux
tmux send-keys -t training "cd /workspace/crypto_trading" C-m
tmux send-keys -t training "source /workspace/venv/bin/activate" C-m

# Запускаем обучение с сохранением вывода
tmux send-keys -t training "python train_universal_transformer_v2.py --task classification_binary --ensemble_size 1 2>&1 | tee training_output.log" C-m

echo "✅ Обучение запущено в tmux сессии 'training'"
echo ""
echo "📊 Для мониторинга:"
echo "   tmux attach -t training"
echo ""
echo "📥 Для просмотра логов:"
echo "   tail -f /workspace/crypto_trading/logs/training_*/training.log"
echo ""
echo "🔍 Проверка статуса:"
echo "   tmux ls"
EOF

echo ""
echo "✅ Готово! Обучение запущено в фоновом режиме."
echo ""
echo "🔗 Команды для управления:"
echo "   ssh -p 27681 root@79.116.73.220 'tmux attach -t training'  # Подключиться к сессии"
echo "   ssh -p 27681 root@79.116.73.220 'tmux ls'                  # Список сессий"
echo "   ./check_training_status.sh                                 # Проверить статус"