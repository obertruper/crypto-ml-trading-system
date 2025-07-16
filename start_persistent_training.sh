#!/bin/bash

echo "🚀 Запуск обучения с защитой от разрыва соединения"
echo "=================================================="
echo ""

# Опции запуска
echo "Выберите способ запуска:"
echo "1) Screen (рекомендуется) - продолжит работу при разрыве SSH"
echo "2) Nohup - запустит в фоне с логированием"
echo "3) Обычный - для отладки"
echo ""
read -p "Ваш выбор (1-3): " choice

case $choice in
    1)
        echo ""
        echo "📺 Запуск через Screen..."
        echo ""
        
        # SSH туннель для БД
        echo "🔄 Создание SSH туннеля для БД..."
        ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N &
        SSH_PID=$!
        echo "✅ Туннель создан (PID: $SSH_PID)"
        sleep 3
        
        # Запуск в screen
        ssh -p 27681 root@79.116.73.220 << 'EOF'
# Завершаем старую сессию если есть
screen -S ml_training -X quit 2>/dev/null

# Создаем новую screen сессию
screen -dmS ml_training bash -c '
cd /workspace/crypto_trading
source /workspace/venv/bin/activate
echo "=================================================="
echo "🚀 Начало обучения: $(date)"
echo "=================================================="
python train_universal_transformer.py 2>&1 | tee logs/training_screen.log
echo "=================================================="
echo "✅ Обучение завершено: $(date)"
echo "=================================================="
'

echo "✅ Обучение запущено в screen сессии 'ml_training'"
echo ""
echo "📋 Команды для управления:"
echo "   Подключиться: screen -r ml_training"
echo "   Отключиться: Ctrl+A, затем D"
echo "   Проверить статус: screen -ls"
echo "   Завершить: screen -S ml_training -X quit"
echo ""
echo "📊 Просмотр логов:"
echo "   tail -f /workspace/crypto_training/logs/training_screen.log"
EOF
        
        echo ""
        echo "⚠️  SSH туннель работает (PID: $SSH_PID)"
        echo "   Для остановки: kill $SSH_PID"
        echo ""
        echo "💡 Если интернет пропадет:"
        echo "   1. Обучение продолжится на сервере"
        echo "   2. Переподключите туннель: ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N"
        echo "   3. Подключитесь к screen: ssh -p 27681 root@79.116.73.220 'screen -r ml_training'"
        ;;
        
    2)
        echo ""
        echo "🔄 Запуск через nohup..."
        
        # SSH туннель
        ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N &
        SSH_PID=$!
        echo "✅ Туннель создан (PID: $SSH_PID)"
        sleep 3
        
        # Запуск через nohup
        ssh -p 27681 root@79.116.73.220 << 'EOF'
cd /workspace/crypto_trading
source /workspace/venv/bin/activate
nohup python train_universal_transformer.py > logs/training_nohup.log 2>&1 &
echo "✅ Обучение запущено в фоне (PID: $!)"
echo "📊 Логи: tail -f /workspace/crypto_training/logs/training_nohup.log"
EOF
        ;;
        
    3)
        echo ""
        echo "🔄 Обычный запуск..."
        
        # SSH туннель
        ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N &
        SSH_PID=$!
        echo "✅ Туннель создан (PID: $SSH_PID)"
        sleep 3
        
        # Обычный запуск
        ssh -p 27681 root@79.116.73.220 "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && python train_universal_transformer.py"
        
        # Закрываем туннель
        kill $SSH_PID 2>/dev/null
        ;;
esac