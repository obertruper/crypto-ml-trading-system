#!/bin/bash

echo "🚀 Запуск Metabase для проекта crypto_ai_trading..."

# Проверяем Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен"
    echo "Установите Docker:"
    echo "  sudo apt update"
    echo "  sudo apt install docker.io docker-compose"
    echo "  sudo usermod -aG docker $USER"
    echo "  (затем перелогиньтесь)"
    exit 1
fi

# Проверяем docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose не установлен"
    echo "  sudo apt install docker-compose"
    exit 1
fi

# Проверяем, может ли пользователь запускать docker без sudo
if ! docker ps &> /dev/null; then
    echo "⚠️  Docker требует sudo. Пробуем с sudo..."
    DOCKER_CMD="sudo docker"
    COMPOSE_CMD="sudo docker-compose"
else
    DOCKER_CMD="docker"
    COMPOSE_CMD="docker-compose"
fi

# Создаем директорию для данных Metabase
mkdir -p metabase-data

# Останавливаем существующий контейнер
echo "🛑 Останавливаем существующие контейнеры..."
$COMPOSE_CMD -f docker-compose-metabase.yml down 2>/dev/null

# Запускаем Metabase
echo "🔄 Запускаем Metabase..."
$COMPOSE_CMD -f docker-compose-metabase.yml up -d

# Ждем запуска
echo "⏳ Ожидаем запуска Metabase (30 секунд)..."
for i in {1..30}; do
    if $DOCKER_CMD ps | grep -q metabase_crypto_ai && curl -s http://localhost:3333 > /dev/null 2>&1; then
        echo ""
        echo "✅ Metabase успешно запущен!"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Проверяем статус
if $DOCKER_CMD ps | grep -q metabase_crypto_ai; then
    echo "📊 Откройте в браузере: http://localhost:3333"
    echo ""
    echo "🔧 Настройка подключения к вашей БД:"
    echo "   Тип: PostgreSQL"
    echo "   Хост: host.docker.internal (или 172.17.0.1)"
    echo "   Порт: 5555"
    echo "   База данных: crypto_trading"
    echo "   Пользователь: ruslan"
    echo "   Пароль: ruslan"
    echo ""
    echo "💡 Если host.docker.internal не работает, используйте:"
    echo "   - 172.17.0.1 (Docker bridge IP)"
    echo "   - IP вашей машины (ip addr show)"
    echo ""
    echo "📋 Просмотр логов: $COMPOSE_CMD -f docker-compose-metabase.yml logs -f"
else
    echo "❌ Ошибка запуска Metabase"
    $COMPOSE_CMD -f docker-compose-metabase.yml logs
fi