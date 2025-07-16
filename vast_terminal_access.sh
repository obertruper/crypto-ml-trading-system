#!/bin/bash

echo "🔧 Решение проблем с SSH подключением к Vast.ai"
echo "================================================"

# Очистка
echo -e "\n1. Очистка старых соединений..."
killall ssh 2>/dev/null
sleep 1

# Проверка ключа
echo -e "\n2. Проверка SSH ключа..."
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "❌ Ошибка: SSH ключ не найден!"
    exit 1
fi

# Метод 1: Прямое подключение
echo -e "\n3. Попытка 1: Прямое подключение..."
echo "   ssh -p 27681 root@79.116.73.220"
timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -p 27681 root@79.116.73.220 "echo 'SUCCESS: Direct connection works!'"

if [ $? -eq 0 ]; then
    echo "✅ Прямое подключение работает!"
    ssh -i ~/.ssh/id_rsa -p 27681 root@79.116.73.220
    exit 0
fi

# Метод 2: Через прокси
echo -e "\n4. Попытка 2: Через прокси ssh5.vast.ai..."
echo "   ssh -p 17171 root@ssh5.vast.ai"
timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -p 17171 root@ssh5.vast.ai "echo 'SUCCESS: Proxy connection works!'"

if [ $? -eq 0 ]; then
    echo "✅ Подключение через прокси работает!"
    ssh -i ~/.ssh/id_rsa -p 17171 root@ssh5.vast.ai
    exit 0
fi

# Метод 3: С instance ID
echo -e "\n5. Попытка 3: Используя instance ID..."
INSTANCE_ID="20927170"
echo "   ssh -p 27681 root@$INSTANCE_ID"
timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa -p 27681 root@$INSTANCE_ID "echo 'SUCCESS: Instance ID connection works!'"

if [ $? -eq 0 ]; then
    echo "✅ Подключение через instance ID работает!"
    ssh -i ~/.ssh/id_rsa -p 27681 root@$INSTANCE_ID
    exit 0
fi

# Если ничего не работает
echo -e "\n❌ Все методы подключения не сработали!"
echo -e "\n📋 Возможные решения:"
echo "1. Проверьте, что instance активен на vast.ai"
echo "2. Убедитесь, что SSH ключ загружен в vast.ai аккаунт"
echo "3. Попробуйте пересоздать SSH ключ:"
echo "   ssh-keygen -t rsa -b 4096 -f ~/.ssh/vast_ai_key"
echo "   cat ~/.ssh/vast_ai_key.pub"
echo "   # Скопируйте и вставьте в настройки vast.ai"
echo ""
echo "4. Проверьте instance на сайте vast.ai и используйте команду оттуда"
echo ""
echo "5. Попробуйте подключиться через веб-консоль и проверить authorized_keys:"
echo "   cat ~/.ssh/authorized_keys"