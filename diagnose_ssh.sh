#!/bin/bash
# Диагностика SSH подключения к Vast.ai

echo "🔍 Диагностика SSH подключения к Vast.ai"
echo "========================================"

# Параметры
HOST="79.116.73.220"
PORT="27681"
USER="root"

echo ""
echo "📋 Проверка SSH ключей:"
echo "-----------------------"

# Список потенциальных ключей
SSH_KEYS=(
    "$HOME/.ssh/id_rsa"
    "$HOME/.ssh/id_ed25519"
    "$HOME/.ssh/vast_ai_key"
    "./vast_ssh_key_fixed.txt"
)

for key in "${SSH_KEYS[@]}"; do
    if [ -f "$key" ]; then
        echo "✅ Найден: $key"
        ls -la "$key"
        
        # Проверяем публичный ключ
        if [ -f "${key}.pub" ]; then
            echo "   Публичный ключ:"
            head -c 50 "${key}.pub"
            echo "..."
        fi
    else
        echo "❌ Не найден: $key"
    fi
    echo ""
done

echo ""
echo "📋 SSH Config:"
echo "--------------"
if [ -f ~/.ssh/config ]; then
    grep -A10 -B2 "vast\|79.116" ~/.ssh/config || echo "Нет записей для Vast.ai"
else
    echo "❌ Файл ~/.ssh/config не найден"
fi

echo ""
echo "📋 Проверка подключения с разными опциями:"
echo "----------------------------------------"

# Пробуем подключиться с задержкой
echo ""
echo "1. Пробую с задержкой 5 секунд..."
sleep 5

ssh -v -o ConnectTimeout=30 -o StrictHostKeyChecking=no -p "$PORT" "$USER@$HOST" "echo 'TEST OK'" 2>&1 | grep -E "Offering|Trying|Authentication|Permission denied" | head -20

echo ""
echo "💡 Рекомендации:"
echo "----------------"
echo "1. Проверьте, что сервер Vast.ai запущен: https://vast.ai/console/instances/"
echo "2. Проверьте SSH ключи в аккаунте: https://vast.ai/console/account/"
echo "3. Скопируйте актуальную команду подключения из консоли Vast.ai"
echo ""
echo "Если сервер запущен, попробуйте:"
echo "- Перезапустить инстанс"
echo "- Добавить SSH ключ заново"
echo "- Использовать команду из консоли Vast.ai"