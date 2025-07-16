#!/bin/bash

echo "=== SSH Connection Test Script ==="
echo "Тестирование SSH подключения к Linux системе"
echo ""

# Информация о системе
echo "🖥️  Информация о системе:"
echo "IP адрес: 192.168.10.101"
echo "Пользователь: obertruper"
echo "SSH порт: 22"
echo ""

# Проверка SSH сервиса
echo "🔍 Проверка SSH сервиса:"
systemctl status ssh | grep Active
echo ""

# Проверка прослушивания порта
echo "🔌 Проверка порта 22:"
ss -tlnp | grep :22
echo ""

# Проверка firewall
echo "🔥 Статус Firewall:"
sudo ufw status
echo ""

# Проверка authorized_keys
echo "🔑 SSH ключи в authorized_keys:"
cat ~/.ssh/authorized_keys
echo ""

echo "📋 Инструкции для подключения с Mac:"
echo "1. На Mac выполните: ssh obertruper@192.168.10.101"
echo "2. Если не работает, проверьте:"
echo "   - ping 192.168.10.101"
echo "   - ssh -v obertruper@192.168.10.101 (для отладки)"
echo ""

echo "💡 Совет: Создайте alias на Mac в ~/.ssh/config:"
echo "Host linux-ml"
echo "    HostName 192.168.10.101"
echo "    User obertruper"
echo "    Port 22"
echo ""
echo "После этого можно подключаться: ssh linux-ml"