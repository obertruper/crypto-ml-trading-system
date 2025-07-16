#!/bin/bash

echo "=== ВАРИАНТЫ ПОДКЛЮЧЕНИЯ К MAC ==="
echo ""
echo "1. TAILSCALE (САМЫЙ ПРОСТОЙ):"
echo "   На Linux: sudo tailscale up"
echo "   На Mac: установить Tailscale и подключиться"
echo "   Потом: ssh obertruper@[tailscale-ip]"
echo ""
echo "2. NGROK SSH ТУННЕЛЬ:"
echo "   На Mac: brew install ngrok"
echo "   На Mac: ngrok tcp 22"
echo "   На Linux: ssh -p [port] obertruper@[ngrok-url]"
echo ""
echo "3. ZEROTIER (VPN):"
echo "   Установить на обеих машинах"
echo "   Создать сеть на zerotier.com"
echo "   Подключиться к одной сети"
echo ""
echo "4. ПЕРЕКЛЮЧИТЬ MAC В СЕТЬ 192.168.10.x:"
echo "   Настройки WiFi -> Дополнительно -> TCP/IP"
echo "   Вручную: 192.168.10.200"
echo ""
echo "5. SSH REVERSE TUNNEL:"
echo "   Если есть общий сервер (VPS)"
echo ""

# Проверка Tailscale
if command -v tailscale &> /dev/null; then
    echo "✅ Tailscale установлен на Linux!"
    echo "Статус:"
    sudo tailscale status
fi