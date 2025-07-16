#!/bin/bash

# Скрипт для настройки окружения на Vast.ai

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Настройки сервера
SERVER_HOST="84.68.60.115"
SERVER_PORT="42244"
PROXY_HOST="ssh1.vast.ai"
PROXY_PORT="18645"
SERVER_USER="root"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Настройка окружения на Vast.ai сервере           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo

# Проверяем подключение
echo -e "${YELLOW}🔍 Проверка подключения...${NC}"
if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Прямое подключение доступно${NC}"
    SSH_CMD="ssh -t -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
else
    echo -e "${YELLOW}⚠️  Используем прокси подключение${NC}"
    SSH_CMD="ssh -t -p $PROXY_PORT $SERVER_USER@$PROXY_HOST"
fi

# Выполняем настройку на сервере
$SSH_CMD << 'ENDSSH'

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}📦 Проверка системы...${NC}"

# Проверяем ОС
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "ОС: $NAME $VERSION"
fi

# Проверяем наличие Python
echo -e "\n${YELLOW}🐍 Проверка Python...${NC}"
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✅ Python3 найден:${NC} $(python3 --version)"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    echo -e "${GREEN}✅ Python найден:${NC} $(python --version)"
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python не найден!${NC}"
    echo -e "${YELLOW}Установка Python...${NC}"
    
    # Определяем пакетный менеджер
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y python3 python3-pip python3-venv
    elif command -v yum &> /dev/null; then
        yum install -y python3 python3-pip
    else
        echo -e "${RED}❌ Не удалось определить пакетный менеджер${NC}"
        exit 1
    fi
    
    PYTHON_CMD="python3"
fi

# Создаем символическую ссылку если нужно
if [ "$PYTHON_CMD" = "python3" ] && ! command -v python &> /dev/null; then
    ln -sf $(which python3) /usr/bin/python
    echo -e "${GREEN}✅ Создана ссылка python -> python3${NC}"
fi

# Проверяем pip
echo -e "\n${YELLOW}📦 Проверка pip...${NC}"
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}Установка pip...${NC}"
    $PYTHON_CMD -m ensurepip --upgrade || curl https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD
fi

# Проверяем GPU
echo -e "\n${YELLOW}🖥️ Проверка GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}❌ nvidia-smi не найден${NC}"
fi

# Проверяем CUDA
echo -e "\n${YELLOW}🔧 Проверка CUDA...${NC}"
if [ -d /usr/local/cuda ]; then
    echo -e "${GREEN}✅ CUDA найдена:${NC}"
    ls -la /usr/local/ | grep cuda
    if [ -f /usr/local/cuda/version.txt ]; then
        cat /usr/local/cuda/version.txt
    fi
else
    echo -e "${YELLOW}⚠️  CUDA не найдена в стандартном месте${NC}"
fi

# Создаем виртуальное окружение
echo -e "\n${YELLOW}🔧 Настройка виртуального окружения...${NC}"
cd /workspace

if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    $PYTHON_CMD -m venv venv
fi

# Активируем окружение
source venv/bin/activate

# Обновляем pip
pip install --upgrade pip

# Проверяем требования transformer_v3
if [ -f "transformer_v3/requirements.txt" ]; then
    echo -e "\n${YELLOW}📦 Установка зависимостей transformer_v3...${NC}"
    pip install -r transformer_v3/requirements.txt
else
    echo -e "\n${YELLOW}📦 Установка базовых зависимостей...${NC}"
    pip install \
        tensorflow \
        pandas \
        numpy \
        scikit-learn \
        matplotlib \
        seaborn \
        psycopg2-binary \
        pyyaml \
        tqdm
fi

# Проверяем TensorFlow GPU
echo -e "\n${YELLOW}🔍 Проверка TensorFlow GPU...${NC}"
python -c "
import tensorflow as tf
print(f'TensorFlow версия: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✅ Найдено GPU: {len(gpus)}')
    for gpu in gpus:
        print(f'   {gpu}')
else:
    print('❌ GPU не обнаружен TensorFlow')
"

echo -e "\n${GREEN}✅ Настройка завершена!${NC}"
echo -e "${YELLOW}Для активации окружения используйте:${NC}"
echo "cd /workspace && source venv/bin/activate"

ENDSSH

echo -e "\n${GREEN}✅ Настройка сервера завершена!${NC}"