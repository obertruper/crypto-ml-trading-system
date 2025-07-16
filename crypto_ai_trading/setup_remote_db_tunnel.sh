#!/bin/bash

echo "🔧 Настройка SSH туннеля для PostgreSQL"
echo "="*60

# Проверяем, запущен ли PostgreSQL локально
echo "🔍 Проверка локальной PostgreSQL..."
if pg_isready -h localhost -p 5555 >/dev/null 2>&1; then
    echo "✅ PostgreSQL работает на порту 5555"
else
    echo "❌ PostgreSQL не доступна на порту 5555"
    echo "Запустите базу данных командой:"
    echo "  brew services start postgresql@14"
    echo "или"
    echo "  pg_ctl -D /usr/local/var/postgresql@14 start"
    exit 1
fi

# Создаем обратный SSH туннель
echo ""
echo "🚇 Создание обратного SSH туннеля..."
echo "Локальный порт 5555 -> Удаленный порт 5555"

# Убиваем старые туннели
pkill -f "5555:localhost:5555"

# Создаем новый туннель
ssh -o StrictHostKeyChecking=no -f -N -R 5555:localhost:5555 root@ssh1.vast.ai -p 30421

if [ $? -eq 0 ]; then
    echo "✅ Туннель создан успешно!"
    
    # Проверяем подключение с сервера
    echo ""
    echo "🔍 Проверка подключения с сервера..."
    ssh -o StrictHostKeyChecking=no root@ssh1.vast.ai -p 30421 "
        python3 -c \"
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5555,
        database='crypto_trading',
        user='ruslan',
        password='ruslan'
    )
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM raw_market_data')
    count = cur.fetchone()[0]
    print(f'✅ Подключение успешно! Записей в БД: {count:,}')
    conn.close()
except Exception as e:
    print(f'❌ Ошибка подключения: {e}')
        \"
    "
    
    echo ""
    echo "📋 Туннель активен. Команды управления:"
    echo "  • Проверить: ps aux | grep '5555:localhost:5555'"
    echo "  • Остановить: pkill -f '5555:localhost:5555'"
    echo ""
    echo "🚀 Теперь можно запускать обучение на сервере!"
    
else
    echo "❌ Не удалось создать туннель"
    exit 1
fi