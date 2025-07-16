#!/bin/bash

echo "🚀 Запуск обучения регрессионной модели на Vast.ai"
echo "=================================================="

# Проверка SSH туннеля для БД
echo "📡 Проверка SSH туннеля для БД..."
if ! nc -z localhost 5555 2>/dev/null; then
    echo "❌ SSH туннель для БД не активен!"
    echo "Запустите в отдельном терминале:"
    echo "ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N"
    exit 1
fi
echo "✅ SSH туннель активен"

# Подключение к серверу и запуск
echo ""
echo "🔄 Подключаюсь к серверу и запускаю обучение..."
ssh -p 27681 root@79.116.73.220 << 'ENDSSH'
echo "📂 Переход в рабочую директорию..."
cd /workspace/crypto_trading

echo "🐍 Активация виртуального окружения..."
source /workspace/venv/bin/activate

echo "📊 Проверка GPU..."
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}'); print(f'GPU доступен: {len(tf.config.list_physical_devices(\"GPU\"))} устройств')"

echo ""
echo "🚀 ЗАПУСК ОБУЧЕНИЯ РЕГРЕССИОННОЙ МОДЕЛИ"
echo "======================================="
echo "Модели: buy_return_predictor и sell_return_predictor"
echo "Задача: предсказание expected returns"
echo ""

# Запуск обучения
python train_universal_transformer.py --config config.yaml

echo ""
echo "✅ Обучение завершено!"
ENDSSH