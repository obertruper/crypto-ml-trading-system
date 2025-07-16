#!/bin/bash
# Скрипт для настройки SSH туннеля к локальной БД

echo "🔗 Настройка SSH туннеля для доступа к локальной PostgreSQL..."
echo "==========================================================="

# Проверяем, что PostgreSQL запущен локально
if ! pg_isready -p 5555 -h localhost > /dev/null 2>&1; then
    echo "❌ PostgreSQL не запущен на порту 5555!"
    echo "Запустите PostgreSQL командой: pg_ctl start -D /usr/local/var/postgres"
    exit 1
fi

echo "✅ PostgreSQL работает на порту 5555"

# Создаем обратный SSH туннель
# Это позволит серверу Vast.ai подключаться к вашей локальной БД
echo ""
echo "📡 Создаю обратный SSH туннель..."
echo "Сервер сможет подключаться к БД через localhost:5555"
echo ""
echo "Нажмите Ctrl+C для остановки туннеля"
echo ""

ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N -v -o ServerAliveInterval=30 -o ServerAliveCountMax=3 2>&1 | grep -E "remote forward|forwarding|success"