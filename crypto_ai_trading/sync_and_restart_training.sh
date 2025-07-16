#!/bin/bash

echo "🔄 Синхронизация исправлений с сервером Vast.ai..."
echo "="*60

# Проверяем доступные порты
echo "🔍 Проверка подключения к серверу..."

# Пробуем разные комбинации
SERVERS=(
    "root@109.198.107.223 -p 48937"
    "root@ssh1.vast.ai -p 30421"
    "root@184.98.25.179 -p 41575"
    "root@ssh8.vast.ai -p 13641"
)

SSH_CMD=""
for server in "${SERVERS[@]}"; do
    echo "Пробую: ssh $server"
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $server "echo 'OK'" 2>/dev/null; then
        SSH_CMD="ssh -o StrictHostKeyChecking=no $server"
        echo "✅ Подключение установлено!"
        break
    fi
done

if [ -z "$SSH_CMD" ]; then
    echo "❌ Не удалось подключиться к серверу!"
    echo "Проверьте:"
    echo "1. VPN подключение"
    echo "2. SSH ключи"
    echo "3. Доступность сервера"
    exit 1
fi

echo ""
echo "📤 Загрузка исправленных файлов..."

# Создаем список измененных файлов
FILES_TO_SYNC=(
    "data/feature_engineering.py"
    "data/dataset.py"
    "config/config.yaml"
)

# Получаем параметры SSH
SSH_PARAMS="${SSH_CMD#ssh -o StrictHostKeyChecking=no }"

# Синхронизируем файлы
for file in "${FILES_TO_SYNC[@]}"; do
    echo "  • Загружаю $file..."
    scp -o StrictHostKeyChecking=no -P 48937 "$file" root@109.198.107.223:/root/crypto_ai_trading/$file
done

echo ""
echo "🛑 Останавливаем текущее обучение..."
$SSH_CMD "pkill -f train_model.py || pkill -f main.py || echo 'Процессы остановлены'"

echo ""
echo "🧹 Очистка старых данных..."
$SSH_CMD "cd /root/crypto_ai_trading && rm -rf processed_data/*.pkl 2>/dev/null || true"

echo ""
echo "🚀 Запуск обучения с исправленной логикой..."

# Создаем скрипт запуска на сервере
$SSH_CMD "cat > /root/crypto_ai_trading/restart_training.sh << 'EOF'
#!/bin/bash
cd /root/crypto_ai_trading

echo '📊 Пересоздание датасета с исправленной логикой...'
python main.py --mode data

echo ''
echo '🚀 Запуск обучения...'
python main.py --mode train 2>&1 | tee training_fixed_\$(date +%Y%m%d_%H%M%S).log
EOF"

$SSH_CMD "chmod +x /root/crypto_ai_trading/restart_training.sh"

echo ""
echo "🎯 Запуск обучения в screen сессии..."
$SSH_CMD "cd /root/crypto_ai_trading && screen -dmS training bash restart_training.sh"

echo ""
echo "✅ Обучение запущено с исправленной логикой!"
echo ""
echo "📋 Команды для мониторинга:"
echo "  • Подключение к screen: ssh -o StrictHostKeyChecking=no root@109.198.107.223 -p 48937 'screen -r training'"
echo "  • Просмотр логов: ssh -o StrictHostKeyChecking=no root@109.198.107.223 -p 48937 'tail -f /root/crypto_ai_trading/training_fixed_*.log'"
echo "  • TensorBoard: ssh -L 6006:localhost:6006 root@109.198.107.223 -p 48937"
echo ""
echo "="*60