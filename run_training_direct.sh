#!/bin/bash

# Скрипт для прямого запуска обучения на сервере

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Настройки сервера
SERVER_HOST="84.68.60.115"
SERVER_PORT="42244"
SERVER_USER="root"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Прямой запуск обучения Transformer v3           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo

# Меню выбора
echo -e "${YELLOW}Выберите режим обучения:${NC}"
echo "[1] 🚀 Регрессия (expected returns)"
echo "[2] 🎯 Классификация (profit/loss)"
echo "[3] 🧪 Тестовый режим"
echo "[4] 📊 TensorBoard"
echo "[0] ❌ Выход"
echo

read -p "Выбор: " choice

case $choice in
    1)
        echo -e "${GREEN}🚀 Запуск обучения регрессии...${NC}"
        ssh -t -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "cd /workspace && source venv/bin/activate && python transformer_v3/main.py --task regression"
        ;;
    2)
        echo -e "${GREEN}🎯 Запуск обучения классификации...${NC}"
        ssh -t -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "cd /workspace && source venv/bin/activate && python transformer_v3/main.py --task classification_binary"
        ;;
    3)
        echo -e "${YELLOW}🧪 Запуск тестового обучения...${NC}"
        ssh -t -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "cd /workspace && source venv/bin/activate && python transformer_v3/main.py --test-mode --task regression"
        ;;
    4)
        echo -e "${YELLOW}📊 Открытие TensorBoard...${NC}"
        echo -e "${GREEN}TensorBoard будет доступен на http://localhost:6006${NC}"
        echo "Нажмите Ctrl+C для закрытия"
        ssh -p $SERVER_PORT -L 6006:localhost:6006 $SERVER_USER@$SERVER_HOST "cd /workspace && source venv/bin/activate && tensorboard --logdir logs/ --host 0.0.0.0"
        ;;
    0)
        echo -e "${GREEN}До свидания!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Неверный выбор!${NC}"
        exit 1
        ;;
esac