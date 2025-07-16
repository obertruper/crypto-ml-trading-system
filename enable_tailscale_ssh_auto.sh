#!/bin/bash
# Автоматическое включение Tailscale SSH

echo "🔐 Включение Tailscale SSH..."
echo "ilpnqw1234" | sudo -S tailscale up --ssh

if [ $? -eq 0 ]; then
    echo "✅ Tailscale SSH успешно включен!"
    echo ""
    echo "Теперь с Mac можно подключаться:"
    echo "ssh obertruper@100.118.184.106"
else
    echo "❌ Ошибка при включении Tailscale SSH"
fi