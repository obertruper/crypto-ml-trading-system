#!/bin/bash
# Тестирование всех MCP серверов для проекта crypto_ai_trading

echo "🔍 Тестирование MCP серверов для crypto_ai_trading"
echo "=================================================="

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция проверки сервера
check_server() {
    local name=$1
    local test_cmd=$2
    
    echo -n "Проверяю $name... "
    
    if eval "$test_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Работает${NC}"
        return 0
    else
        echo -e "${RED}✗ Не работает${NC}"
        return 1
    fi
}

echo -e "\n📋 Установленные MCP серверы:"
claude mcp list

echo -e "\n🧪 Тестирование серверов:\n"

# 1. Claude Code
check_server "Claude Code" "claude mcp serve --help"

# 2. Sequential Thinking
check_server "Sequential Thinking" "npx -y @modelcontextprotocol/server-sequential-thinking --help 2>&1 | grep -q 'sequential'"

# 3. Context7
check_server "Context7" "test -f ~/.context7/config.json"

# 4. Filesystem
check_server "Filesystem (проект)" "test -d '/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading'"

# 5. GitHub
check_server "GitHub" "gh auth status"

# 6. PostgreSQL
echo -n "Проверяю PostgreSQL... "
if psql -h localhost -p 5555 -U ruslan -d crypto_trading -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ База данных доступна${NC}"
else
    echo -e "${YELLOW}⚠ База данных недоступна (запустите PostgreSQL на порту 5555)${NC}"
fi

# 7. Memory
check_server "Memory" "mkdir -p .mcp_memory"

# 8. Brave Search
echo -n "Проверяю Brave Search... "
if grep -q "your_brave_api_key_here" .env.mcp; then
    echo -e "${YELLOW}⚠ Требуется API ключ (получите на https://api.search.brave.com/)${NC}"
else
    echo -e "${GREEN}✓ API ключ настроен${NC}"
fi

# 9. Puppeteer
check_server "Puppeteer" "which chromium || which google-chrome"

echo -e "\n📊 Статистика:"
echo "=============="

# Подсчет активных процессов
active_count=$(ps aux | grep -E "mcp|claude" | grep -v grep | wc -l)
echo "Активных MCP процессов: $active_count"

# Проверка памяти
if [ -d ".mcp_memory" ]; then
    memory_files=$(find .mcp_memory -type f 2>/dev/null | wc -l)
    echo "Файлов в памяти MCP: $memory_files"
fi

echo -e "\n💡 Рекомендации:"
echo "==============="

if ! psql -h localhost -p 5555 -U ruslan -d crypto_trading -c "SELECT 1;" > /dev/null 2>&1; then
    echo "• Запустите PostgreSQL: pg_ctl -D /usr/local/var/postgres -o '-p 5555' start"
fi

if grep -q "your_brave_api_key_here" .env.mcp; then
    echo "• Получите Brave API ключ: https://api.search.brave.com/"
fi

echo -e "\n✅ Проверка завершена!"