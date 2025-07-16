#!/bin/bash
# Остановка MCP серверов

echo "🛑 Остановка MCP серверов..."

if [ -f .mcp/filesystem.pid ]; then
    kill $(cat .mcp/filesystem.pid) 2>/dev/null
    rm .mcp/filesystem.pid
fi

if [ -f .mcp/postgres.pid ]; then
    kill $(cat .mcp/postgres.pid) 2>/dev/null
    rm .mcp/postgres.pid
fi

echo "✅ MCP серверы остановлены"
