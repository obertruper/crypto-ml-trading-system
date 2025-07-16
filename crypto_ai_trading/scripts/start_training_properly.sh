#!/bin/bash
# Правильный запуск обучения на GPU

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Запуск обучения на GPU сервере${NC}"

# Параметры
HOST="ssh1.vast.ai"
PORT="30421"
REMOTE_PATH="/root/crypto_ai_trading"

# 1. Копируем данные если их нет
echo -e "${YELLOW}📤 Проверка данных на сервере...${NC}"
ssh -p $PORT root@$HOST "test -f $REMOTE_PATH/cache/features_cache.pkl" || {
    echo -e "${YELLOW}Копирование данных на сервер (это может занять время)...${NC}"
    scp -P $PORT cache/features_cache.pkl root@$HOST:$REMOTE_PATH/cache/
    echo -e "${GREEN}✅ Данные скопированы${NC}"
}

# 2. Устанавливаем зависимости и запускаем обучение
echo -e "${YELLOW}🔧 Подготовка окружения...${NC}"

ssh -p $PORT root@$HOST << 'ENDSSH'
cd /root/crypto_ai_trading

# Создаем необходимые директории
mkdir -p experiments/runs logs models_saved cache

# Проверяем Python и устанавливаем базовые пакеты
echo "📦 Установка зависимостей..."
python3 -m pip install --quiet torch pandas numpy scikit-learn tensorboard PyYAML

# Проверяем GPU
echo "🖥️ Проверка GPU..."
python3 -c "import torch; print(f'GPU доступна: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Нет\"}')"

# Останавливаем старый TensorBoard
pkill -f tensorboard

# Запускаем TensorBoard
echo "📊 Запуск TensorBoard..."
nohup tensorboard --logdir=experiments/runs --port=6006 --host=0.0.0.0 > logs/tensorboard.log 2>&1 &
sleep 2

# Запускаем обучение в tmux
echo "🧠 Запуск обучения..."
tmux kill-session -t training 2>/dev/null
tmux new-session -d -s training "cd /root/crypto_ai_trading && python3 run_full_pipeline.py --mode demo 2>&1 | tee logs/training.log"

echo "✅ Все запущено!"
ENDSSH

# 3. Устанавливаем SSH туннель
echo -e "${YELLOW}🔗 Установка туннеля для мониторинга...${NC}"
pkill -f "ssh.*6006.*$HOST"
ssh -f -N -L 6006:localhost:6006 -p $PORT root@$HOST

echo -e "${GREEN}✅ Готово!${NC}"
echo ""
echo -e "${BLUE}📊 Мониторинг:${NC}"
echo "   • TensorBoard: http://localhost:6006"
echo "   • Подключение к обучению: ssh -p $PORT root@$HOST 'tmux attach -t training'"
echo "   • Логи: ssh -p $PORT root@$HOST 'tail -f /root/crypto_ai_trading/logs/training.log'"
echo ""
echo -e "${YELLOW}💡 TensorBoard не требует логина/пароля${NC}"