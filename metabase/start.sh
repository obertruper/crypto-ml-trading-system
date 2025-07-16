#!/bin/bash

echo "🚀 Запуск Metabase для анализа криптоданных..."

# Переходим в директорию metabase
cd "$(dirname "$0")"

# Проверяем Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен!"
    echo "📥 Установите Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker не запущен!"
    echo "🔧 Запустите Docker Desktop и попробуйте снова"
    exit 1
fi

# Создаем директорию для данных
mkdir -p metabase-data

# Останавливаем старые контейнеры
echo "🛑 Останавливаем существующие контейнеры..."
docker-compose down 2>/dev/null

# Запускаем Metabase
echo "🔄 Запускаем Metabase..."
docker-compose up -d

# Ждем запуска
echo "⏳ Ожидаем запуска Metabase..."
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null; then
        echo "✅ Metabase запущен!"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "✅ Metabase успешно запущен!"
echo ""
echo "📊 Откройте в браузере: http://localhost:3000"
echo ""
echo "🔧 При первом запуске:"
echo "1. Создайте аккаунт администратора"
echo "2. Подключите вашу БД PostgreSQL:"
echo "   • Тип: PostgreSQL"
echo "   • Имя: Crypto Trading"
echo "   • Хост: host.docker.internal (для Mac) или 172.17.0.1 (для Linux)"
echo "   • Порт: 5555"
echo "   • База данных: crypto_trading"
echo "   • Пользователь: ruslan"
echo "   • Пароль: (оставьте пустым)"
echo ""
echo "💡 Используйте готовые запросы из metabase_queries.sql"
echo ""
echo "🛑 Для остановки: ./stop.sh"