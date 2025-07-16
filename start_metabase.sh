#!/bin/bash

echo "🚀 Запуск Metabase для анализа криптоданных..."

# Проверяем, установлен ли Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Пожалуйста, установите Docker Desktop"
    exit 1
fi

# Проверяем, запущен ли Docker
if ! docker info &> /dev/null; then
    echo "❌ Docker не запущен. Запустите Docker Desktop"
    exit 1
fi

# Останавливаем существующие контейнеры
echo "🛑 Останавливаем существующие контейнеры..."
docker-compose down

# Запускаем Metabase
echo "🔄 Запускаем Metabase..."
docker-compose up -d

# Ждем запуска
echo "⏳ Ожидаем запуска Metabase (30 секунд)..."
sleep 30

# Проверяем статус
if docker ps | grep -q metabase_crypto; then
    echo "✅ Metabase успешно запущен!"
    echo "📊 Откройте в браузере: http://localhost:3000"
    echo ""
    echo "🔧 Настройка подключения к вашей БД:"
    echo "   Тип: PostgreSQL"
    echo "   Хост: host.docker.internal"
    echo "   Порт: 5555"
    echo "   База данных: crypto_trading"
    echo "   Пользователь: ruslan"
    echo ""
    echo "💡 Совет: Используйте host.docker.internal для подключения к локальной PostgreSQL"
else
    echo "❌ Ошибка запуска Metabase"
    docker-compose logs
fi