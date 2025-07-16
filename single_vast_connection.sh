#!/bin/bash

echo "🔄 Закрываем все существующие SSH соединения..."
pkill -f "ssh.*79.116.73.220" 2>/dev/null
pkill -f "ssh.*27681" 2>/dev/null
pkill -f "ssh.*vast.ai" 2>/dev/null
sleep 2

echo "📡 Создаем единственное SSH подключение к Vast.ai..."
echo "Server: 79.116.73.220:27681"
echo ""

# Добавляем ключ в SSH агент если нужно
if ! ssh-add -l | grep -q "id_rsa"; then
    echo "🔑 Добавляем SSH ключ в агент..."
    ssh-add ~/.ssh/id_rsa
fi

# Подключаемся с интерактивной оболочкой, явно указывая ключ
ssh -i ~/.ssh/id_rsa -p 27681 root@79.116.73.220