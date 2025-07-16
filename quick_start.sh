#!/bin/bash
# Быстрый запуск ML Crypto Trading System

echo "🚀 ML Crypto Trading System - Быстрый запуск"
echo "============================================"

# Определение пути к проекту
PROJECT_DIR="/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM"
cd "$PROJECT_DIR"

# Проверка PostgreSQL
echo "🔍 Проверка PostgreSQL..."
if pg_isready -h localhost -p 5555 > /dev/null 2>&1; then
    echo "✅ PostgreSQL работает на порту 5555"
else
    echo "❌ PostgreSQL не запущен. Запускаем..."
    echo "ilpnqw1234" | sudo -S systemctl start postgresql
    sleep 2
fi

# Активация виртуального окружения
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Виртуальное окружение активировано"
else
    echo "❌ Виртуальное окружение не найдено. Создайте его с помощью:"
    echo "   python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Проверка установки зависимостей
if python -c "import pandas, numpy, tensorflow" 2>/dev/null; then
    echo "✅ Python зависимости установлены"
else
    echo "⚠️  Некоторые зависимости отсутствуют. Установка..."
    pip install -r requirements.txt
fi

# Запуск MCP серверов (если настроены)
if [ -f ".mcp/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js" ]; then
    echo "🌐 Запуск MCP серверов..."
    ./start_mcp.sh &
    echo "✅ MCP серверы запущены"
fi

# Меню выбора действий
echo ""
echo "📋 Выберите действие:"
echo "1) Инициализация базы данных"
echo "2) Загрузка данных с Bybit"
echo "3) Подготовка датасета"
echo "4) Обучение модели (регрессия)"
echo "5) Обучение модели (классификация)"
echo "6) Мониторинг обучения"
echo "7) Запуск crypto_ai_trading проекта"
echo "8) Полный пайплайн (1-4)"
echo "0) Выход"

read -p "Ваш выбор: " choice

case $choice in
    1)
        echo "🗄️ Инициализация базы данных..."
        python init_database.py
        ;;
    2)
        echo "📥 Загрузка данных с Bybit..."
        python download_data.py
        ;;
    3)
        echo "🔧 Подготовка датасета..."
        python prepare_dataset.py
        ;;
    4)
        echo "🧠 Обучение модели (регрессия)..."
        python train_universal_transformer.py --task regression
        ;;
    5)
        echo "🧠 Обучение модели (классификация)..."
        python train_universal_transformer.py --task classification
        ;;
    6)
        echo "📊 Мониторинг обучения..."
        python monitor_training.py
        ;;
    7)
        echo "🚀 Запуск crypto_ai_trading..."
        cd crypto_ai_trading
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        python main.py --mode demo
        ;;
    8)
        echo "🔄 Запуск полного пайплайна..."
        python run_futures_pipeline.py
        ;;
    0)
        echo "👋 До свидания!"
        exit 0
        ;;
    *)
        echo "❌ Неверный выбор"
        exit 1
        ;;
esac