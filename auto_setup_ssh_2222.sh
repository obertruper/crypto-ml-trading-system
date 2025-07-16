#!/bin/bash
# Автоматический скрипт настройки SSH на порту 2222

echo "=== Автоматическая настройка SSH для порта 2222 ==="
echo "Этот скрипт требует прав sudo"
echo ""

# Создаем конфигурационный файл для порта 2222
CONFIG_FILE="/tmp/port2222.conf"
cat > $CONFIG_FILE << 'EOF'
# Дополнительный порт SSH
Port 22
Port 2222

# Рекомендуемые настройки безопасности
PermitRootLogin no
PasswordAuthentication yes
PubkeyAuthentication yes
EOF

echo "1. Создан временный конфигурационный файл: $CONFIG_FILE"
echo ""
echo "Для завершения настройки выполните:"
echo ""
echo "sudo cp $CONFIG_FILE /etc/ssh/sshd_config.d/port2222.conf"
echo "sudo sshd -t"
echo "sudo ufw allow 2222/tcp"
echo "sudo systemctl restart ssh"
echo ""
echo "После перезапуска можете подключаться:"
echo "ssh -p 2222 $USER@$(hostname -I | awk '{print $1}')"