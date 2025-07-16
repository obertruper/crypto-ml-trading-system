#!/bin/bash
# Синхронизация проекта с Vast.ai сервером

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔄 Синхронизация проекта с Vast.ai${NC}"

# Параметры из конфига (можно переопределить)
if [ "$VAST_CONNECTION_MODE" = "1" ] || [ -z "$VAST_CONNECTION_MODE" ]; then
    # Новый сервер (по умолчанию)
    HOST="ssh1.vast.ai"
    PORT="30421"
else
    # Старый сервер
    HOST="109.198.107.223"
    PORT="48937"
fi

REMOTE_PATH="/root/crypto_ai_trading"
# Используем стандартный ключ id_rsa, так как он работает
KEY_PATH="$HOME/.ssh/id_rsa"

# Проверка ключа
if [ ! -f "$KEY_PATH" ]; then
    echo -e "${RED}❌ SSH ключ не найден: $KEY_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Используется SSH ключ: $KEY_PATH${NC}"

# Исключения для rsync
EXCLUDES=(
    "--exclude=.git/"
    "--exclude=__pycache__/"
    "--exclude=*.pyc"
    "--exclude=.DS_Store"
    "--exclude=cache/"
    "--exclude=logs/"
    "--exclude=experiments/runs/"
    "--exclude=models_saved/*.pth"
    "--exclude=.venv/"
    "--exclude=venv/"
)

echo -e "${YELLOW}📤 Загрузка файлов...${NC}"

# Создание директории на сервере
ssh -p $PORT -i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$HOST "mkdir -p $REMOTE_PATH"

# Синхронизация
rsync -avzP \
    -e "ssh -p $PORT -i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    "${EXCLUDES[@]}" \
    ./ root@$HOST:$REMOTE_PATH/

echo -e "${GREEN}✅ Синхронизация завершена!${NC}"

# Опционально: установка зависимостей
echo -e "\n${YELLOW}Установить зависимости на сервере? (y/n)${NC}"
read -n 1 install_deps
echo

if [ "$install_deps" = "y" ]; then
    echo -e "${YELLOW}📦 Установка зависимостей...${NC}"
    ssh -p $PORT -i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$HOST "cd $REMOTE_PATH && pip install -r requirements.txt"
    echo -e "${GREEN}✅ Зависимости установлены!${NC}"
fi