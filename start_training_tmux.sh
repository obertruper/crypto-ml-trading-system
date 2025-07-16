#!/bin/bash

echo "🚀 Запуск обучения через tmux (с защитой от разрыва)"
echo "===================================================="
echo ""

# Убиваем старый туннель если есть
pkill -f "ssh.*5555:localhost:5555"
sleep 1

# Создаем SSH туннель для БД
echo "🔄 Создание SSH туннеля для БД..."
ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N &
SSH_PID=$!
echo "✅ Туннель создан (PID: $SSH_PID)"
sleep 3

# Запускаем обучение через tmux
echo ""
echo "📺 Запуск обучения в tmux сессии..."
ssh -p 27681 root@79.116.73.220 << 'EOF'
# Завершаем старую сессию если есть
tmux kill-session -t ml_training 2>/dev/null

# Создаем новую tmux сессию
tmux new-session -d -s ml_training "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && echo '🚀 Начало обучения: '$(date) && python train_universal_transformer.py 2>&1 | tee logs/training_tmux.log"

echo ""
echo "✅ Обучение успешно запущено в tmux сессии 'ml_training'"
echo ""
echo "📋 Команды для управления:"
echo "   Подключиться к сессии: tmux attach -t ml_training"
echo "   Отключиться от сессии: Ctrl+B, затем D"
echo "   Список сессий: tmux ls"
echo "   Завершить сессию: tmux kill-session -t ml_training"
echo ""
echo "📊 Мониторинг прогресса:"
echo "   tail -f /workspace/crypto_training/logs/training_tmux.log"
echo ""
echo "🔍 Проверка статуса:"
tmux ls
EOF

echo ""
echo "=================="
echo "✅ Запуск завершен"
echo "=================="
echo ""
echo "⚠️  SSH туннель для БД работает в фоне (PID: $SSH_PID)"
echo ""
echo "💡 Если интернет пропадет:"
echo "   1. Обучение ПРОДОЛЖИТСЯ на сервере"
echo "   2. При восстановлении переподключите туннель:"
echo "      ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N"
echo "   3. Подключитесь к tmux сессии:"
echo "      ssh -p 27681 root@79.116.73.220"
echo "      tmux attach -t ml_training"
echo ""
echo "📊 Для просмотра логов прямо сейчас:"
echo "   ssh -p 27681 root@79.116.73.220 'tail -f /workspace/crypto_training/logs/training_tmux.log'"