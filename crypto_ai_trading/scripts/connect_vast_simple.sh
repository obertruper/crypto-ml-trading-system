#!/bin/bash
# Простое подключение к Vast.ai серверу

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Подключение к Vast.ai GPU сервере${NC}"

# Получаем SSH алиас из конфига или переменной окружения
SSH_ALIAS=${VAST_SSH_ALIAS:-vast-current}

# Проверяем подключение и получаем информацию о GPU
GPU_INFO=$(ssh $SSH_ALIAS "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null | head -2)
if [ -n "$GPU_INFO" ]; then
    echo -e "${YELLOW}GPU: ${NC}"
    echo "$GPU_INFO" | while read gpu; do
        echo "  • $gpu"
    done
else
    echo -e "${YELLOW}Сервер: $SSH_ALIAS${NC}"
fi
echo ""

# Опции подключения
echo -e "${YELLOW}Выберите действие:${NC}"
echo "1) Интерактивная SSH сессия"
echo "2) Мониторинг GPU (nvidia-smi)"
echo "3) Просмотр логов обучения"
echo "4) Подключение к tmux сессии"
echo "5) Запуск TensorBoard туннеля"
echo -n "Выбор (1-5): "
read choice

case $choice in
    1)
        echo -e "\n${GREEN}Подключение к серверу...${NC}"
        echo -e "${YELLOW}Для выхода используйте: exit${NC}"
        ssh $SSH_ALIAS
        ;;
    2)
        echo -e "\n${GREEN}Мониторинг GPU...${NC}"
        echo -e "${YELLOW}Для выхода нажмите Ctrl+C${NC}"
        ssh $SSH_ALIAS "watch -n 1 nvidia-smi"
        ;;
    3)
        echo -e "\n${GREEN}Просмотр логов...${NC}"
        echo -e "${YELLOW}Для выхода нажмите Ctrl+C${NC}"
        ssh $SSH_ALIAS "tail -f /root/crypto_ai_trading/logs/training_gpu.log"
        ;;
    4)
        echo -e "\n${GREEN}Подключение к tmux...${NC}"
        echo -e "${YELLOW}Для отключения используйте: Ctrl+B, затем D${NC}"
        ssh -t $SSH_ALIAS "tmux attach -t training || tmux new -s training"
        ;;
    5)
        echo -e "\n${GREEN}Запуск TensorBoard туннеля...${NC}"
        echo -e "${YELLOW}TensorBoard будет доступен на: http://localhost:6006${NC}"
        echo -e "${YELLOW}Для остановки нажмите Ctrl+C${NC}"
        echo -e "${YELLOW}Примечание: Порты уже пробрасываются автоматически через SSH config${NC}"
        ssh $SSH_ALIAS "cd /root/crypto_ai_trading && tensorboard --logdir logs/ --host 0.0.0.0 --port 6006"
        ;;
    *)
        echo -e "${RED}Неверный выбор${NC}"
        exit 1
        ;;
esac