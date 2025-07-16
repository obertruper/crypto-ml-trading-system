#!/bin/bash

# Автоматический запуск LSP сервера для crypto_ai_trading

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Определяем директорию скрипта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Проверяем, не запущен ли уже LSP сервер
if pgrep -f "enhanced_lsp_server.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  LSP сервер уже запущен${NC}"
    echo -e "${GREEN}✅ LSP контекст доступен${NC}"
    exit 0
fi

echo -e "${GREEN}🚀 Запуск LSP сервера для crypto_ai_trading...${NC}"

# Активируем виртуальное окружение
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${RED}❌ Виртуальное окружение не найдено. Создаем...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Запускаем LSP сервер в фоновом режиме
nohup python enhanced_lsp_server.py --stdio > lsp_service.log 2>&1 &
LSP_PID=$!

# Проверяем, что сервер запустился
sleep 2
if ps -p $LSP_PID > /dev/null; then
    echo -e "${GREEN}✅ LSP сервер запущен (PID: $LSP_PID)${NC}"
    echo $LSP_PID > lsp_server.pid
    echo -e "${GREEN}📋 Логи: $SCRIPT_DIR/lsp_service.log${NC}"
    echo -e "${GREEN}🔥 Теперь доступны:${NC}"
    echo -e "   - Локальный контекст через LSP"
    echo -e "   - Актуальная документация через Context7"
    echo -e "   - Глубокий анализ через Sequential Thinking"
else
    echo -e "${RED}❌ Ошибка запуска LSP сервера${NC}"
    tail -n 20 lsp_service.log
    exit 1
fi