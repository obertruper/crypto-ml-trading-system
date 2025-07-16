#!/bin/bash
# Скрипт для подключения к Vast.ai серверу
# Новый сервер
# IP: 184.98.25.179 (прямой) или ssh8.vast.ai (proxy)
# SSH Port: 41575 (прямой) или 13641 (proxy)

# Параметры по умолчанию
SSH_KEY="id_rsa"

# Обработка параметров
if [ "$1" == "--key" ] && [ -n "$2" ]; then
    SSH_KEY="$2"
fi

echo "╔════════════════════════════════════════════╗"
echo "║      Подключение к Vast.ai сервер         ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "IP адрес: 184.98.25.179 (прямой) / ssh8.vast.ai (proxy)"
echo "SSH порт: 41575 (прямой) / 13641 (proxy)"
echo "Локальный туннель: 8080 -> localhost:8080"
echo ""

# Проверка SSH ключа
echo "🔑 Проверка SSH ключа: ~/.ssh/$SSH_KEY"
if [ -f ~/.ssh/$SSH_KEY ]; then
    echo "✓ SSH ключ найден"
    echo "Fingerprint: $(ssh-keygen -lf ~/.ssh/$SSH_KEY.pub 2>/dev/null | awk '{print $2}')"
else
    echo "✗ SSH ключ не найден в ~/.ssh/$SSH_KEY"
    exit 1
fi

# Прямое подключение
echo ""
echo "🚀 Попытка прямого подключения..."
echo "Команда: ssh -p 41575 root@184.98.25.179"
echo ""

ssh -i ~/.ssh/$SSH_KEY \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p 41575 \
    root@184.98.25.179 \
    -L 8080:localhost:8080 \
    -L 6006:localhost:6006 \
    -L 8384:localhost:8384 || {
    
    echo ""
    echo "❌ Прямое подключение не удалось"
    echo ""
    echo "🔄 Попытка подключения через прокси..."
    echo "Команда: ssh -p 13641 root@ssh8.vast.ai"
    echo ""
    
    ssh -i ~/.ssh/$SSH_KEY \
        -o ConnectTimeout=10 \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -p 13641 \
        root@ssh8.vast.ai \
        -L 8080:localhost:8080 \
        -L 6006:localhost:6006 \
        -L 8384:localhost:8384 || {
        
        echo ""
        echo "❌ Подключение через прокси тоже не удалось"
        echo ""
        echo "═══════════════════════════════════════════════"
        echo "⚠️  ДИАГНОСТИКА ПРОБЛЕМЫ"
        echo "═══════════════════════════════════════════════"
        echo ""
        
        # Проверка доступных SSH ключей
        echo "📁 Доступные SSH ключи:"
        ls -la ~/.ssh/*.pub | awk '{print "   - " $9}'
        echo ""
        
        echo "🔑 Fingerprint текущего ключа id_rsa:"
        ssh-keygen -lf ~/.ssh/id_rsa.pub
        echo ""
        
        echo "💡 Возможные решения:"
        echo ""
        echo "1. Убедитесь, что в Vast.ai загружен правильный ключ:"
        echo "   • Ключ уже скопирован в буфер обмена"
        echo "   • Откройте: https://cloud.vast.ai/console/keys"
        echo "   • Вставьте ПОЛНЫЙ ключ (включая ssh-rsa и user@host)"
        echo "   • Нажмите Save/Update"
        echo "   • Подождите 2-3 минуты для применения"
        echo ""
        echo "2. Попробуйте альтернативные ключи:"
        if [ -f ~/.ssh/vast_ai_key ]; then
            echo "   ./connect_vast.sh --key vast_ai_key"
        fi
        if [ -f ~/.ssh/vast_key ]; then
            echo "   ./connect_vast.sh --key vast_key"
        fi
        echo ""
        echo "3. Проверьте статус инстанса:"
        echo "   • https://vast.ai/console/instances/"
        echo "   • Проверьте новый Instance ID"
        echo ""
        echo "4. Используйте команду из консоли Vast.ai"
        echo "   (она может содержать специальный proxy jump)"
        echo ""
        echo "📊 Информация о сервере:"
        echo "   IP: 184.98.25.179 (прямой) / ssh8.vast.ai (proxy)"
        echo "   SSH порт: 41575 (прямой) / 13641 (proxy)"
        echo "   Локальные туннели:"
        echo "   - localhost:6006 -> TensorBoard"
        echo "   - localhost:8080 -> Web UI"
        echo "   - localhost:8384 -> Syncthing"
        echo ""
        
        # Дополнительная диагностика
        echo "🔍 Диагностика SSH:"
        echo "   • Права доступа SSH директории: $(ls -ld ~/.ssh | awk '{print $1}')"
        echo "   • Права доступа приватного ключа: $(ls -l ~/.ssh/$SSH_KEY 2>/dev/null | awk '{print $1}')"
        echo ""
        
        # Проверка прав доступа
        if [ -f ~/.ssh/$SSH_KEY ]; then
            PERMS=$(stat -f %A ~/.ssh/$SSH_KEY 2>/dev/null || stat -c %a ~/.ssh/$SSH_KEY 2>/dev/null)
            if [ "$PERMS" != "600" ]; then
                echo "⚠️  ВНИМАНИЕ: Неправильные права доступа к ключу!"
                echo "   Выполните: chmod 600 ~/.ssh/$SSH_KEY"
                echo ""
            fi
        fi
        
        # Копируем ключ в буфер
        cat ~/.ssh/$SSH_KEY.pub | pbcopy
        echo "✅ SSH ключ скопирован в буфер обмена!"
        echo ""
        echo "📌 Прямая ссылка на управление ключами:"
        echo "   https://cloud.vast.ai/console/keys"
    }
}