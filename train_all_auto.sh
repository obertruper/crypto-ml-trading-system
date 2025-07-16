#!/bin/bash
# Автоматический запуск с проверками

echo "🚀 Автоматическое обучение всех моделей"
echo "========================================"

# Проверка БД
if ! pg_isready -p 5555 -h localhost > /dev/null 2>&1; then
    echo "❌ PostgreSQL не запущен!"
    exit 1
fi
echo "✅ БД доступна"

# Убиваем старые туннели
echo "🔄 Закрываю старые SSH туннели..."
pkill -f "ssh.*5555.*79.116.73.220" 2>/dev/null || true
sleep 2

# Запускаем обучение с защитой от разрыва соединения
echo ""
echo "📊 Запуск Regression моделей..."
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=720 -o TCPKeepAlive=yes \
    -p 27681 root@79.116.73.220 -R 5555:localhost:5555 \
    "cd /workspace/crypto_trading && \
     source /workspace/venv/bin/activate && \
     export TF_CPP_MIN_LOG_LEVEL=1 && \
     python train_universal_transformer.py --task regression --config remote_config.yaml" || {
    echo "❌ Ошибка при обучении regression моделей"
    exit 1
}

echo ""
echo "📊 Запуск Classification моделей..."
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=720 -o TCPKeepAlive=yes \
    -p 27681 root@79.116.73.220 -R 5555:localhost:5555 \
    "cd /workspace/crypto_trading && \
     source /workspace/venv/bin/activate && \
     export TF_CPP_MIN_LOG_LEVEL=1 && \
     python train_universal_transformer.py --task classification --config remote_config.yaml" || {
    echo "❌ Ошибка при обучении classification моделей"
    exit 1
}

echo ""
echo "✅ Все модели успешно обучены!"