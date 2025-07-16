#!/bin/bash

echo "🔄 Закрываем все существующие SSH соединения..."
pkill -f "ssh.*vast.ai" 2>/dev/null
pkill -f "ssh.*79.116.73.220" 2>/dev/null
sleep 2

echo "📡 Подключение через SSH прокси Vast.ai..."
echo "Proxy: ssh5.vast.ai:17171"
echo "Instance: 20927170"
echo ""

# Подключение через прокси
ssh -i ~/.ssh/id_rsa -p 17171 root@ssh5.vast.ai