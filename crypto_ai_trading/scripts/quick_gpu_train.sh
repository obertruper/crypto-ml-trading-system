#!/bin/bash
# Быстрый запуск GPU обучения с проверками

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Быстрый запуск GPU обучения${NC}"

# 1. Проверка системы
echo -e "\n${YELLOW}Проверка готовности...${NC}"
if ! ./scripts/check_gpu_setup.sh; then
    echo -e "${RED}❌ Система не готова${NC}"
    exit 1
fi

# 2. Синхронизация если нужно
echo -e "\n${YELLOW}Проверка синхронизации...${NC}"
if ! ssh -p 30421 -i ~/.ssh/id_rsa root@ssh1.vast.ai "test -f /root/crypto_ai_trading/cache/features_cache.pkl" &>/dev/null; then
    echo -e "${YELLOW}📤 Синхронизация проекта и кэша...${NC}"
    ./scripts/sync_to_vast.sh
    
    # Копируем кэш
    echo -e "${YELLOW}📦 Копирование кэша...${NC}"
    scp -P 30421 -i ~/.ssh/id_rsa cache/features_cache.pkl root@ssh1.vast.ai:/root/crypto_ai_trading/cache/
fi

# 3. Запуск обучения
echo -e "\n${GREEN}🧠 Запуск обучения (5 эпох)...${NC}"
export USE_CACHE_ONLY=1
export GPU_TRAINING_MODE=1
export GPU_TRAINING_EPOCHS=5

./scripts/run_on_vast.sh

# 4. Автоматический запуск мониторинга
echo -e "\n${YELLOW}📊 Запуск мониторинга...${NC}"
sleep 3

# Устанавливаем туннель
pkill -f "ssh.*6006"
ssh -f -N -L 6006:localhost:6006 -p 48937 -i ~/.ssh/vast_ai_key root@109.198.107.223

echo -e "\n${GREEN}✅ Готово!${NC}"
echo -e "${BLUE}📊 TensorBoard: http://localhost:6006${NC}"
echo -e "${YELLOW}💡 Для проверки логов:${NC}"
echo "   ssh -p 48937 -i ~/.ssh/vast_ai_key root@109.198.107.223 'tmux attach -t training'"

# Открываем браузер
if command -v open &> /dev/null; then
    sleep 2
    open http://localhost:6006
fi