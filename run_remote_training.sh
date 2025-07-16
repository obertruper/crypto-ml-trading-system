#!/bin/bash

echo "🚀 Запуск обучения XGBoost v3 на удаленном сервере"
echo "=================================================="

# Конфигурация сервера
SERVER_HOST="ssh1.vast.ai"
SERVER_PORT=18645
SERVER_USER="root"

# Проверяем существующие туннели
echo "🔍 Проверка существующих SSH туннелей..."
EXISTING_TUNNELS=$(ps aux | grep "ssh.*$SERVER_PORT.*5555" | grep -v grep | wc -l)

if [ $EXISTING_TUNNELS -eq 0 ]; then
    echo "📡 Создание туннеля для БД..."
    # Создаем обратный туннель для доступа к локальной БД с сервера
    ssh -f -N -p $SERVER_PORT -R 5555:localhost:5555 $SERVER_USER@$SERVER_HOST
    echo "✅ Туннель создан (удаленный порт 5555 -> локальная БД)"
    sleep 2
else
    echo "✅ Туннель уже существует"
fi

# Синхронизация кода
echo -e "\n📤 Синхронизация кода на сервер..."
rsync -avz --progress \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='logs/' \
  --exclude='cache/' \
  --exclude='*.pkl' \
  --exclude='*.csv' \
  --exclude='*.png' \
  xgboost_v3/ \
  -e "ssh -p $SERVER_PORT" \
  $SERVER_USER@$SERVER_HOST:~/xgboost_v3/

# Выбор режима обучения
echo -e "\n🎯 Выберите режим обучения:"
echo "1. 🧪 Тестовый (2 символа, быстро)"
echo "2. 🎯 Продакшн (все символы, оптимизация)"
echo "3. 🔧 Пользовательские параметры"
read -p "Ваш выбор (1-3): " MODE

case $MODE in
    1)
        COMMAND="cd ~/xgboost_v3 && python3 main.py"
        echo "🧪 Запуск в тестовом режиме..."
        ;;
    2)
        COMMAND="cd ~/xgboost_v3 && python3 main.py --mode production"
        echo "🎯 Запуск в продакшн режиме..."
        ;;
    3)
        echo "Введите дополнительные параметры (например: --gpu --epochs 2000):"
        read PARAMS
        COMMAND="cd ~/xgboost_v3 && python3 main.py $PARAMS"
        echo "🔧 Запуск с параметрами: $PARAMS"
        ;;
    *)
        echo "❌ Неверный выбор"
        exit 1
        ;;
esac

# Запуск обучения
echo -e "\n🚀 Запуск обучения на сервере..."
echo "Команда: $COMMAND"
echo "=================================================="

# Интерактивная SSH сессия для мониторинга
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST -t "$COMMAND"

echo -e "\n✅ Обучение завершено!"
echo "💾 Для скачивания результатов используйте:"
echo "   ./download_results.sh"