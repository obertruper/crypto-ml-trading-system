#!/bin/bash

echo "🚀 Быстрый запуск обучения на Vast.ai"
echo ""

# Шаг 1: Создаем SSH туннель для БД
echo "📡 Создание SSH туннеля для БД (запустится в фоне)..."
ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N &
SSH_TUNNEL_PID=$!
echo "✅ Туннель запущен (PID: $SSH_TUNNEL_PID)"
sleep 3

# Шаг 2: Запуск обучения
echo ""
echo "🎯 Запуск обучения..."
ssh -p 27681 root@79.116.73.220 "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && python train_universal_transformer.py"

# Шаг 3: Завершение туннеля
echo ""
echo "🔄 Закрытие SSH туннеля..."
kill $SSH_TUNNEL_PID 2>/dev/null
echo "✅ Готово!"