#!/bin/bash

echo "📊 Проверка статуса обучения Enhanced TFT v2.1"
echo "=============================================="
echo ""

# Проверка туннеля
echo "🔍 Проверка SSH туннеля..."
if ssh -p 27681 root@79.116.73.220 "timeout 2 nc -zv localhost 5555" 2>&1 | grep -q "open"; then
    echo "✅ Туннель БД активен"
else
    echo "❌ Туннель БД не работает"
fi

echo ""
echo "🔍 Проверка процессов..."
ssh -p 27681 root@79.116.73.220 << 'EOF'
# Проверка tmux сессии
if tmux has-session -t training 2>/dev/null; then
    echo "✅ Tmux сессия 'training' активна"
else
    echo "❌ Tmux сессия 'training' не найдена"
fi

# Проверка процесса Python
if ps aux | grep -q "[p]ython train_universal_transformer_v2.py"; then
    echo "✅ Процесс обучения запущен"
    ps aux | grep "[p]ython train_universal_transformer_v2.py" | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "❌ Процесс обучения не найден"
fi

# Проверка GPU
echo ""
echo "🖥️ Использование GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "   %s: %s/%s MB (%s%% загрузка)\n", $1, $2, $3, $4}'

# Последние логи
echo ""
echo "📋 Последние строки лога:"
if [ -d "/workspace/crypto_trading/logs" ]; then
    LATEST_LOG=$(ls -t /workspace/crypto_trading/logs/training_*/training.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "   Файл: $LATEST_LOG"
        tail -5 "$LATEST_LOG" | sed 's/^/   /'
    else
        echo "   Логи не найдены"
    fi
fi

# Проверка выходного файла
echo ""
if [ -f "/workspace/crypto_trading/training_output.log" ]; then
    echo "📄 Последние строки вывода:"
    tail -5 /workspace/crypto_trading/training_output.log | sed 's/^/   /'
fi
EOF