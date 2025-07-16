#!/bin/bash

echo "🧹 Полная очистка SSH соединений..."

# Убиваем все SSH процессы
pkill -f "ssh.*79\.116"
pkill -f "ssh.*vast"
pkill -f "ssh.*27681"
pkill -f "ssh.*17171"

# Ждем
sleep 2

echo "🔑 Проверка SSH ключа..."
if [ -f ~/.ssh/id_rsa ]; then
    echo "✅ SSH ключ найден"
else
    echo "❌ SSH ключ не найден!"
    exit 1
fi

echo ""
echo "📡 Подключение к Vast.ai (прямое)..."
echo "Server: 79.116.73.220"
echo "Port: 27681"
echo ""

# Пробуем прямое подключение с явными параметрами
ssh -v \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i ~/.ssh/id_rsa \
    -p 27681 \
    root@79.116.73.220

# Если не удалось, пробуем через прокси
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ Прямое подключение не удалось"
    echo "📡 Пробуем через прокси ssh5.vast.ai:17171..."
    echo ""
    
    ssh -v \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o TCPKeepAlive=yes \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -i ~/.ssh/id_rsa \
        -p 17171 \
        root@ssh5.vast.ai
fi