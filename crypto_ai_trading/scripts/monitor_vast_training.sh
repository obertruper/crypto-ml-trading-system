#!/bin/bash
# Мониторинг обучения на Vast.ai

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}📊 Мониторинг обучения на Vast.ai${NC}"
echo "====================================="

# Выбор сервера
echo -e "\n${YELLOW}Выберите сервер:${NC}"
echo "1) Новый сервер (ssh1.vast.ai:30421) - РЕКОМЕНДУЕТСЯ"
echo "2) Старый сервер (109.198.107.223:48937)"
read -p "Выбор (1/2) [по умолчанию 1]: " choice

# Если пустой ввод - используем новый сервер
if [ -z "$choice" ] || [ "$choice" = "1" ]; then
    SSH_HOST="ssh1.vast.ai"
    SSH_PORT="30421"
else
    SSH_HOST="109.198.107.223"
    SSH_PORT="48937"
fi

# SSH команда
SSH_CMD="ssh -p $SSH_PORT root@$SSH_HOST"

echo -e "\n${GREEN}📡 Подключение к $SSH_HOST:$SSH_PORT${NC}"

# Функция для выполнения команд на удаленном сервере
remote_exec() {
    $SSH_CMD "$1" 2>/dev/null
}

# Проверка статуса обучения
echo -e "\n${BLUE}🔍 Проверка процесса обучения:${NC}"
remote_exec "ps aux | grep -E 'python.*(train|main)' | grep -v grep" | head -5

# Последние логи
echo -e "\n${BLUE}📄 Последние записи из логов:${NC}"
remote_exec "cd /workspace/crypto_ai_trading && tail -n 20 logs/training_*/training.log 2>/dev/null | grep -E '(Эпоха|loss|Learning|Checkpoint|Метрики)'" || \
remote_exec "cd /workspace && find . -name 'training.log' -type f -exec tail -n 20 {} \; 2>/dev/null | grep -E '(Эпоха|loss|Learning|Checkpoint|Метрики)'"

# Использование GPU
echo -e "\n${BLUE}🎮 Использование GPU:${NC}"
remote_exec "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader" | while read line; do
    echo "   GPU: $line"
done

# Проверка моделей
echo -e "\n${BLUE}💾 Сохраненные модели:${NC}"
remote_exec "cd /workspace/crypto_ai_trading && ls -lah models_saved/*.pth 2>/dev/null | tail -5" || \
remote_exec "cd /workspace && find . -name '*.pth' -type f -ls 2>/dev/null | tail -5"

# Место на диске
echo -e "\n${BLUE}💿 Использование диска:${NC}"
remote_exec "df -h /workspace | tail -1"

# Опции мониторинга
echo -e "\n${YELLOW}Дополнительные опции:${NC}"
echo "1) Непрерывный мониторинг логов"
echo "2) Открыть TensorBoard (http://localhost:6007)"
echo "3) Интерактивная SSH сессия"
echo "4) Выход"
read -p "Выбор (1-4): " option

case $option in
    1)
        echo -e "${GREEN}📊 Непрерывный мониторинг (Ctrl+C для выхода)${NC}"
        $SSH_CMD "cd /workspace/crypto_ai_trading && tail -f logs/training_*/training.log | grep -E '(Эпоха|loss|Learning|Checkpoint|Метрики)'"
        ;;
    2)
        echo -e "${GREEN}🌐 Открываю TensorBoard...${NC}"
        echo "Проброс портов..."
        ssh -p $SSH_PORT -L 6007:localhost:6007 root@$SSH_HOST "cd /workspace/crypto_ai_trading && tensorboard --logdir logs/ --port 6007 --bind_all" &
        sleep 3
        open http://localhost:6007 || xdg-open http://localhost:6007
        ;;
    3)
        echo -e "${GREEN}🔌 Интерактивная сессия${NC}"
        $SSH_CMD
        ;;
    4)
        echo -e "${GREEN}👋 Выход${NC}"
        exit 0
        ;;
esac