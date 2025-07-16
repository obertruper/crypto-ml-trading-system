#!/bin/bash

# Скрипт синхронизации проекта с Linux сервером
# Автор: Claude AI
# Дата: 2025-07-16

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Конфигурация
REMOTE_USER="obertruper"
REMOTE_HOST="localhost"
REMOTE_DIR="/mnt/SSD/PYCHARMPRODJECT/BOT_Trading_v3"
LOCAL_DIR="."

# Актуальный Cloudflare туннель
CLOUDFLARE_TUNNEL="lb-others-hunting-rec.trycloudflare.com"

# Функция для проверки доступности SSH
check_ssh_connection() {
    local host=$1
    local port=$2
    local proxy_command=$3
    
    echo -ne "${YELLOW}Проверка подключения к $host:$port...${NC} "
    
    if [ -n "$proxy_command" ]; then
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
            -o ProxyCommand="$proxy_command" \
            $REMOTE_USER@$host "exit" 2>/dev/null
    else
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
            -p $port $REMOTE_USER@$host "exit" 2>/dev/null
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Успешно${NC}"
        return 0
    else
        echo -e "${RED}✗ Недоступен${NC}"
        return 1
    fi
}

# Функция синхронизации
sync_project() {
    local ssh_command=$1
    
    echo -e "\n${BLUE}=== Начинаю синхронизацию проекта ===${NC}"
    echo -e "${YELLOW}Источник:${NC} $LOCAL_DIR"
    echo -e "${YELLOW}Назначение:${NC} $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
    echo -e "${YELLOW}SSH команда:${NC} $ssh_command"
    
    # Создаем директорию на удаленном сервере если её нет
    eval "$ssh_command \"mkdir -p '$REMOTE_DIR'\""
    
    # Выполняем синхронизацию
    rsync -avzP \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='venv' \
        --exclude='env' \
        --exclude='*.log' \
        --exclude='*.tmp' \
        --exclude='*.cache' \
        --exclude='.DS_Store' \
        --exclude='*.swp' \
        --exclude='*.swo' \
        --exclude='build/' \
        --exclude='dist/' \
        --exclude='*.egg-info' \
        --exclude='.pytest_cache' \
        --exclude='.coverage' \
        --exclude='htmlcov' \
        --exclude='.idea' \
        --exclude='.vscode' \
        --max-size=50M \
        -e "$ssh_command" \
        "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Синхронизация завершена успешно!${NC}"
        
        # Показываем статистику
        echo -e "\n${BLUE}=== Статистика синхронизации ===${NC}"
        eval "$ssh_command \"cd '$REMOTE_DIR' && echo -e '\\n${YELLOW}Количество файлов:${NC}' && find . -type f | wc -l && echo -e '\\n${YELLOW}Размер проекта:${NC}' && du -sh . | cut -f1\""
        
        return 0
    else
        echo -e "\n${RED}✗ Ошибка при синхронизации!${NC}"
        return 1
    fi
}

# Основная логика
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Синхронизация проекта с сервером     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"

# Проверяем различные способы подключения
echo -e "${YELLOW}Проверка доступных способов подключения...${NC}\n"

# 1. Проверка через Cloudflare туннель
PROXY_CMD="cloudflared access tcp --hostname $CLOUDFLARE_TUNNEL"
if check_ssh_connection "$REMOTE_HOST" "22" "$PROXY_CMD"; then
    SSH_COMMAND="ssh -o ProxyCommand=\"$PROXY_CMD\""
    sync_project "$SSH_COMMAND"
    exit 0
fi

# 2. Проверка локального SSH на порту 2222
if check_ssh_connection "$REMOTE_HOST" "2222" ""; then
    SSH_COMMAND="ssh -p 2222"
    sync_project "$SSH_COMMAND"
    exit 0
fi

# 3. Проверка стандартного SSH на порту 22
if check_ssh_connection "$REMOTE_HOST" "22" ""; then
    SSH_COMMAND="ssh"
    sync_project "$SSH_COMMAND"
    exit 0
fi

# Если ни один способ не работает
echo -e "\n${RED}✗ Не удалось установить SSH соединение!${NC}"
echo -e "\n${YELLOW}Возможные решения:${NC}"
echo -e "1. Проверьте, что SSH сервер запущен на удаленной машине"
echo -e "2. Проверьте актуальность Cloudflare туннеля: $CLOUDFLARE_TUNNEL"
echo -e "3. Убедитесь, что у вас есть доступ к серверу"
echo -e "4. Проверьте настройки файрвола\n"

echo -e "${YELLOW}Для создания нового туннеля выполните на сервере:${NC}"
echo -e "${GREEN}cloudflared tunnel --url tcp://localhost:2222${NC}\n"

exit 1