#!/bin/bash
# Запуск MCP серверов

echo "🚀 Запуск MCP серверов..."

# Загрузка переменных окружения
if [ -f .mcp/.env ]; then
    export $(cat .mcp/.env | grep -v '^#' | xargs)
fi

# Проверка установки
if [ ! -d ".mcp/node_modules" ]; then
    echo "❌ MCP серверы не установлены. Запустите ./setup_mcp_local.sh"
    exit 1
fi

# Запуск filesystem сервера
echo "Starting filesystem server..."
node .mcp/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js "$MCP_PROJECT_ROOT" &
PID1=$!

# Запуск PostgreSQL сервера
echo "Starting PostgreSQL server..."
DATABASE_URL="$MCP_DB_CONNECTION" node .mcp/node_modules/@modelcontextprotocol/server-postgres/dist/index.js &
PID2=$!

echo "✅ MCP серверы запущены"
echo "   Filesystem server PID: $PID1"
echo "   PostgreSQL server PID: $PID2"
echo ""
echo "Для остановки используйте: kill $PID1 $PID2"

# Сохраняем PID'ы
echo $PID1 > .mcp/filesystem.pid
echo $PID2 > .mcp/postgres.pid

# Ждем завершения
wait
