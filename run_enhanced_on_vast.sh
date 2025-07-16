#!/bin/bash

echo "🚀 Запуск Enhanced TFT v2.1 на Vast.ai"
echo "========================================"
echo ""

# Синхронизация файлов
echo "📤 Синхронизация файлов на Vast.ai..."
./sync_to_vast.sh

# Очистка старых соединений
echo ""
echo "🧹 Очистка старых SSH соединений..."
pkill -f "ssh.*5555" 2>/dev/null
sleep 2

# SSH подключение для обучения
echo ""
echo "🔄 Создание SSH туннеля для БД..."
./setup_remote_db_tunnel.sh &
TUNNEL_PID=$!
echo "✅ Туннель создан (PID: $TUNNEL_PID)"

# Ждем установки соединения
sleep 5

# Проверяем туннель
echo "🔍 Проверка SSH туннеля..."
# Проверяем процесс туннеля
if ps -p $TUNNEL_PID > /dev/null 2>&1; then
    echo "✅ SSH процесс активен (PID: $TUNNEL_PID)"
    
    # Проверка работы туннеля на сервере
    echo "🔍 Проверка туннеля на сервере..."
    if ssh -p 27681 root@79.116.73.220 "timeout 3 nc -zv localhost 5555" 2>&1 | grep -q "open"; then
        echo "✅ Туннель работает на сервере - БД доступна!"
    else
        echo "❌ Туннель не работает на сервере!"
        kill $TUNNEL_PID 2>/dev/null
        exit 1
    fi
else
    echo "❌ SSH процесс туннеля не запущен!"
    exit 1
fi

# Выбор режима
echo ""
echo "Выберите режим обучения:"
echo "1) Бинарная классификация (рекомендуется)"
echo "2) Регрессия"
echo "3) Ансамбль классификации (3 модели)"
echo "4) 🧪 ТЕСТОВЫЙ РЕЖИМ (2 символа, 3 эпохи)"
echo -n "Ваш выбор (1-4): "
read choice

case $choice in
    1)
        TASK="classification_binary"
        ENSEMBLE=1
        TEST_MODE=""
        echo "✅ Выбрана бинарная классификация"
        ;;
    2)
        TASK="regression"
        ENSEMBLE=1
        TEST_MODE=""
        echo "✅ Выбрана регрессия"
        ;;
    3)
        TASK="classification_binary"
        ENSEMBLE=3
        echo "✅ Выбран ансамбль классификации (3 модели)"
        TEST_MODE=""
        ;;
    4)
        TASK="classification_binary"
        ENSEMBLE=1
        TEST_MODE="--test_mode"
        echo "✅ Выбран ТЕСТОВЫЙ режим"
        ;;
    *)
        echo "❌ Неверный выбор"
        kill $TUNNEL_PID 2>/dev/null
        exit 1
        ;;
esac

# Установка TEST_MODE для остальных режимов
if [ -z "$TEST_MODE" ] && [ "$choice" != "4" ]; then
    TEST_MODE=""
fi

echo ""
echo "📺 Запуск обучения Enhanced TFT v2.1..."
echo "=================================================="
echo ""

# SSH команда для запуска с параметрами стабильности
ssh -p 27681 root@79.116.73.220 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=10 \
    -o TCPKeepAlive=yes \
    -o ConnectTimeout=30 \
    << EOF
cd /workspace/crypto_trading

# Активация правильного venv
if [ -f "/workspace/venv/bin/activate" ]; then
    echo "✅ Найден venv в /workspace/venv"
    source /workspace/venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "✅ Найден venv в ./venv"
    source venv/bin/activate
else
    echo "❌ venv не найден, используем системный Python"
fi

echo "📊 Проверка окружения:"
which python
python --version
echo ""

echo "🔄 Запуск Enhanced TFT v2.1..."
echo "Task: $TASK"
echo "Ensemble size: $ENSEMBLE"
echo "Test mode: $TEST_MODE"
echo ""

# Проверка подключения к БД
echo "🔍 Проверка подключения к БД..."
if python -c "import psycopg2; conn = psycopg2.connect('postgresql://ruslan:ruslan@localhost:5555/crypto_trading'); print('✅ Подключение к БД успешно'); conn.close()" 2>/dev/null; then
    echo ""
    echo "🚀 Запуск обучения..."
    python train_universal_transformer_v2.py --task $TASK --ensemble_size $ENSEMBLE $TEST_MODE
else
    echo "❌ ОШИБКА: Не удается подключиться к БД!"
    echo "Проверьте SSH туннель"
    exit 1
fi
EOF

# Копирование результатов обратно
echo ""
echo "📥 Копирование результатов..."
scp -P 27681 root@79.116.73.220:/workspace/crypto_trading/trained_model/*_enhanced_v2.1_*.h5 ./trained_model/
scp -P 27681 root@79.116.73.220:/workspace/crypto_trading/trained_model/metadata_v2.1.json ./trained_model/
scp -P 27681 root@79.116.73.220:/workspace/crypto_trading/trained_model/feature_config_v2.1.json ./trained_model/
scp -P 27681 root@79.116.73.220:/workspace/crypto_trading/trained_model/scaler_v2.1.pkl ./trained_model/

# Копирование логов
LATEST_LOG=$(ssh -p 27681 root@79.116.73.220 "ls -t /workspace/crypto_trading/logs | grep enhanced_training | head -1")
if [ ! -z "$LATEST_LOG" ]; then
    echo "📥 Копирование логов из $LATEST_LOG..."
    scp -r -P 27681 root@79.116.73.220:/workspace/crypto_trading/logs/$LATEST_LOG ./logs/
fi

# Завершение туннеля
echo ""
echo "🔄 Закрытие SSH туннеля..."
kill $TUNNEL_PID 2>/dev/null

echo ""
echo "✅ Обучение Enhanced TFT v2.1 завершено!"
echo ""
echo "📊 Результаты скопированы в:"
echo "  - trained_model/*_enhanced_v2.1_*.h5"
echo "  - trained_model/metadata_v2.1.json"
echo "  - trained_model/feature_config_v2.1.json"
echo "  - logs/$LATEST_LOG/"