#!/bin/bash

echo "🔍 Проверка доступа к Linux Home серверу"
echo "======================================="

# Цвета
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "\n1️⃣ Локальное подключение (порт 2222):"
if ssh -o ConnectTimeout=5 linux-home-local "echo 'Connected'" 2>/dev/null; then
    echo -e "${GREEN}✅ РАБОТАЕТ${NC}"
    echo "Команда: ssh linux-home-local"
else
    echo -e "${RED}❌ НЕ РАБОТАЕТ${NC}"
fi

echo -e "\n2️⃣ Прямое подключение (localhost:2222):"
if ssh -o ConnectTimeout=5 -p 2222 obertruper@localhost "echo 'Connected'" 2>/dev/null; then
    echo -e "${GREEN}✅ РАБОТАЕТ${NC}"
    echo "Команда: ssh -p 2222 obertruper@localhost"
else
    echo -e "${RED}❌ НЕ РАБОТАЕТ${NC}"
fi

echo -e "\n3️⃣ Cloudflare туннель:"
echo -e "${YELLOW}⚠️  Требуется актуальный URL туннеля${NC}"
echo "Текущий статус Cloudflared:"
if pgrep -f cloudflared > /dev/null; then
    echo -e "${GREEN}✅ Процесс запущен${NC}"
    # Попробуем получить URL из логов
    if [ -f ~/cloudflared.log ]; then
        URL=$(grep -o 'https://.*\.trycloudflare\.com' ~/cloudflared.log | tail -1)
        if [ -n "$URL" ]; then
            echo "Последний известный URL: $URL"
        fi
    fi
else
    echo -e "${RED}❌ Процесс не запущен${NC}"
    echo -e "\nДля создания нового туннеля выполните:"
    echo -e "${GREEN}cloudflared tunnel --url tcp://localhost:2222${NC}"
fi

echo -e "\n📋 Сводка SSH алиасов:"
echo "- linux-home-local   → localhost:2222"
echo "- linux-home-cf      → через Cloudflare туннель"
echo "- linux-home-direct  → 192.168.10.101:22"

echo -e "\n✨ Рекомендация:"
echo "Используйте: ssh linux-home-local"