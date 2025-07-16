#!/bin/bash

echo "🚀 Запуск обучения с мониторингом в реальном времени"
echo "===================================================="
echo ""

# Убиваем старые процессы
pkill -f "ssh.*5555:localhost:5555" 2>/dev/null
tmux kill-session -t ml_training 2>/dev/null
sleep 1

# Создаем SSH туннель для БД
echo "🔄 Создание SSH туннеля для БД..."
ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N &
SSH_PID=$!
echo "✅ Туннель создан (PID: $SSH_PID)"
sleep 3

# Запускаем обучение в tmux и сразу подключаемся к логам
echo ""
echo "📺 Запуск обучения и мониторинг..."
echo "=================================="
echo ""

# Создаем tmux сессию и сразу следим за логами
ssh -p 27681 root@79.116.73.220 << 'EOF'
# Завершаем старую сессию если есть
tmux kill-session -t ml_training 2>/dev/null

# Создаем новую tmux сессию в фоне
tmux new-session -d -s ml_training "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && python train_universal_transformer.py 2>&1 | tee logs/training_live.log"

# Ждем пока создастся лог файл
sleep 2

# Показываем логи в реальном времени
echo "📊 МОНИТОРИНГ ОБУЧЕНИЯ (Ctrl+C для выхода)"
echo "=========================================="
echo ""
tail -f /workspace/crypto_trading/logs/training_live.log
EOF

# При выходе показываем инструкции
echo ""
echo "=================="
echo "📋 Полезные команды:"
echo "=================="
echo ""
echo "1️⃣ Подключиться к tmux сессии:"
echo "   ssh -p 27681 root@79.116.73.220"
echo "   tmux attach -t ml_training"
echo ""
echo "2️⃣ Посмотреть логи снова:"
echo "   ssh -p 27681 root@79.116.73.220 'tail -f /workspace/crypto_training/logs/training_live.log'"
echo ""
echo "3️⃣ Проверить последний лог обучения:"
echo "   ssh -p 27681 root@79.116.73.220 'tail -100 /workspace/crypto_training/logs/training_*/training.log | grep -E \"(эпоха|loss|accuracy|MAE)\"'"
echo ""
echo "⚠️  SSH туннель все еще работает (PID: $SSH_PID)"
echo "   Для остановки: kill $SSH_PID"