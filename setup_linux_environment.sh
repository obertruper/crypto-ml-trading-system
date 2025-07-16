#!/bin/bash
# Скрипт настройки окружения для Linux

echo "🚀 Настройка окружения для ML Crypto Trading System"

# Определение пути к проекту
PROJECT_DIR="/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM"
cd "$PROJECT_DIR"

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo "❌ Виртуальное окружение не найдено"
    exit 1
fi

# Активация виртуального окружения
source venv/bin/activate

# Проверка PostgreSQL
echo "🔍 Проверка PostgreSQL на порту 5555..."
if pg_isready -h localhost -p 5555; then
    echo "✅ PostgreSQL работает на порту 5555"
else
    echo "❌ PostgreSQL не отвечает на порту 5555"
    exit 1
fi

# Инициализация базы данных
echo "🗄️ Инициализация базы данных..."
python init_database.py

# Проверка подпроекта crypto_ai_trading
CRYPTO_PROJECT_DIR="$PROJECT_DIR/crypto_ai_trading"
if [ -d "$CRYPTO_PROJECT_DIR" ]; then
    echo "📁 Найден проект crypto_ai_trading"
    
    # Создание виртуального окружения для подпроекта
    cd "$CRYPTO_PROJECT_DIR"
    if [ ! -d "venv" ]; then
        echo "🔧 Создание виртуального окружения для crypto_ai_trading..."
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Запуск LSP сервера
    if [ -d "lsp_server" ]; then
        echo "🧠 Запуск LSP сервера..."
        cd lsp_server
        if [ -f "start_lsp_auto.sh" ]; then
            chmod +x start_lsp_auto.sh
            ./start_lsp_auto.sh &
            echo "✅ LSP сервер запущен"
        fi
        cd ..
    fi
fi

echo "✅ Окружение настроено успешно!"
echo ""
echo "📋 Доступные команды:"
echo "  - python main.py --mode demo         # Демо режим"
echo "  - python main.py --mode full         # Полное обучение"
echo "  - python monitor_training.py         # Мониторинг обучения"
echo "  - tensorboard --logdir logs/         # TensorBoard"
echo ""
echo "💡 Для работы с основным проектом:"
echo "  cd '$PROJECT_DIR'"
echo "  source venv/bin/activate"
echo ""
echo "💡 Для работы с crypto_ai_trading:"
echo "  cd '$CRYPTO_PROJECT_DIR'"
echo "  source venv/bin/activate"