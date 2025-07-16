#!/bin/bash

echo "====================================================="
echo "🐘 Установка и настройка PostgreSQL для crypto_ai_trading"
echo "====================================================="

# Проверка наличия Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker найден. Используем Docker для запуска PostgreSQL..."
    
    # Запуск PostgreSQL через docker-compose
    echo "🚀 Запуск PostgreSQL контейнера..."
    docker-compose -f docker-compose-postgres.yml up -d
    
    # Ожидание запуска
    echo "⏳ Ожидание запуска PostgreSQL..."
    sleep 10
    
    echo "✅ PostgreSQL запущен на порту 5555"
    echo "   User: ruslan"
    echo "   Password: ruslan"
    echo "   Database: crypto_trading"
    
else
    echo "❌ Docker не найден. Инструкция по установке PostgreSQL:"
    echo ""
    echo "1. Установите PostgreSQL:"
    echo "   sudo apt update"
    echo "   sudo apt install postgresql postgresql-contrib"
    echo ""
    echo "2. Запустите PostgreSQL:"
    echo "   sudo systemctl start postgresql"
    echo ""
    echo "3. Создайте пользователя и базу данных:"
    echo "   sudo -u postgres psql << EOF"
    echo "   CREATE USER ruslan WITH PASSWORD 'ruslan';"
    echo "   CREATE DATABASE crypto_trading OWNER ruslan;"
    echo "   GRANT ALL PRIVILEGES ON DATABASE crypto_trading TO ruslan;"
    echo "   EOF"
    echo ""
    echo "4. Измените порт на 5555:"
    echo "   sudo nano /etc/postgresql/*/main/postgresql.conf"
    echo "   Найдите строку 'port = 5432' и измените на 'port = 5555'"
    echo ""
    echo "5. Перезапустите PostgreSQL:"
    echo "   sudo systemctl restart postgresql"
    echo ""
    echo "Альтернатива: Установите Docker и запустите этот скрипт снова"
fi

echo ""
echo "====================================================="
echo "📝 Следующие шаги:"
echo "1. Запустите инициализацию БД: python init_database.py"
echo "2. Загрузите данные: python download_data.py"
echo "3. Подготовьте датасет: python prepare_dataset.py"
echo "4. Запустите обучение: python main.py --mode demo"
echo "====================================================="