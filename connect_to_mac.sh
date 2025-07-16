#!/bin/bash
# Скрипт для создания reverse SSH туннеля к Mac

MAC_IP="192.168.88.15"
MAC_USER="ruslan"
LINUX_USER="obertruper"
LINUX_IP="192.168.10.101"

echo "=== Настройка reverse SSH туннеля к Mac ==="
echo ""
echo "Параметры подключения:"
echo "Linux: $LINUX_USER@$LINUX_IP"
echo "Mac: $MAC_USER@$MAC_IP"
echo ""

# Создаем reverse туннель
echo "Создаю reverse SSH туннель..."
echo "Команда: ssh -R 2222:localhost:22 $MAC_USER@$MAC_IP -o StrictHostKeyChecking=no"
echo ""

# Показываем инструкции
echo "1. Убедитесь, что на Mac включен SSH:"
echo "   Системные настройки → Общий доступ → Удаленный вход"
echo ""
echo "2. Выполните команду для создания туннеля:"
echo "   ssh -R 2222:localhost:22 $MAC_USER@$MAC_IP -o StrictHostKeyChecking=no"
echo ""
echo "3. После успешного подключения, на Mac можно будет подключиться к Linux:"
echo "   ssh -p 2222 $LINUX_USER@localhost"
echo ""
echo "4. Для постоянного туннеля в фоне:"
echo "   ssh -R 2222:localhost:22 $MAC_USER@$MAC_IP -f -N -o StrictHostKeyChecking=no"
echo ""

# Проверяем доступность Mac
echo "Проверяю доступность Mac..."
ping -c 1 $MAC_IP > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Mac доступен по сети"
else
    echo "❌ Mac недоступен. Проверьте сетевое подключение"
fi

echo ""
echo "Для автоматического подключения запустите:"
echo "./connect_to_mac.sh auto"