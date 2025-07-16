#!/bin/bash

echo "🔄 Мониторинг SSH туннеля для БД"
echo "================================="

while true; do
    # Проверяем туннель
    if ! ssh -p 27681 root@79.116.73.220 "timeout 2 nc -zv localhost 5555" 2>&1 | grep -q "open"; then
        echo "❌ $(date): Туннель не работает, восстанавливаем..."
        
        # Убиваем старые процессы
        pkill -f "ssh.*5555"
        sleep 2
        
        # Создаем новый туннель
        ssh -f -N -p 27681 root@79.116.73.220 -R 5555:localhost:5555 \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes
        
        echo "✅ $(date): Туннель восстановлен"
    else
        echo "✅ $(date): Туннель работает"
    fi
    
    # Проверяем каждые 60 секунд
    sleep 60
done