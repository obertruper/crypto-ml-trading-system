#!/bin/bash

# Скрипт проверки настройки MCP для Claude Desktop
echo "🔍 Проверка настройки MCP для Claude Desktop"
echo "============================================="

# Проверка Node.js
echo -e "\n📦 Проверка Node.js:"
if command -v node &> /dev/null; then
    echo "✅ Node.js установлен: $(node --version)"
else
    echo "❌ Node.js не установлен! Установите через: brew install node"
    exit 1
fi

if command -v npm &> /dev/null; then
    echo "✅ npm установлен: $(npm --version)"
else
    echo "❌ npm не установлен!"
    exit 1
fi

# Проверка конфигурационного файла
CONFIG_FILE="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
echo -e "\n📄 Проверка конфигурационного файла:"
if [ -f "$CONFIG_FILE" ]; then
    echo "✅ Конфигурационный файл найден"
    
    # Проверка синтаксиса JSON
    if python3 -m json.tool "$CONFIG_FILE" > /dev/null 2>&1; then
        echo "✅ JSON синтаксис корректный"
    else
        echo "❌ Ошибка в JSON синтаксисе!"
        python3 -m json.tool "$CONFIG_FILE"
    fi
    
    # Показать установленные MCP серверы
    echo -e "\n🔧 Установленные MCP серверы:"
    python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
    if 'mcpServers' in config:
        for server in config['mcpServers']:
            print(f'  - {server}')
    else:
        print('  ❌ Секция mcpServers не найдена!')
"
else
    echo "❌ Конфигурационный файл не найден!"
    echo "   Путь: $CONFIG_FILE"
fi

# Проверка логов
LOG_DIR="$HOME/Library/Logs/Claude"
echo -e "\n📊 Проверка директории логов:"
if [ -d "$LOG_DIR" ]; then
    echo "✅ Директория логов существует"
    echo "   Последние логи MCP:"
    ls -la "$LOG_DIR"/mcp*.log 2>/dev/null | tail -5 || echo "   Логи MCP пока не созданы"
else
    echo "⚠️  Директория логов не найдена"
fi

# Проверка процессов Claude
echo -e "\n🏃 Проверка процессов Claude:"
if pgrep -x "Claude" > /dev/null; then
    echo "✅ Claude Desktop запущен"
else
    echo "⚠️  Claude Desktop не запущен"
fi

# Тест установки MCP серверов
echo -e "\n🧪 Тест установки базовых MCP серверов:"
echo "   (это может занять некоторое время...)"

# Тест fetch сервера
echo -n "   - @modelcontextprotocol/server-fetch: "
if npx -y @modelcontextprotocol/server-fetch --help > /dev/null 2>&1; then
    echo "✅"
else
    echo "❌"
fi

# Тест filesystem сервера
echo -n "   - @modelcontextprotocol/server-filesystem: "
if npx -y @modelcontextprotocol/server-filesystem --help > /dev/null 2>&1; then
    echo "✅"
else
    echo "❌"
fi

echo -e "\n✨ Проверка завершена!"
echo -e "\n💡 Подсказки:"
echo "   - Перезапустите Claude Desktop после изменения конфигурации"
echo "   - Логи MCP можно посмотреть: tail -f ~/Library/Logs/Claude/mcp*.log"
echo "   - Документация: https://modelcontextprotocol.io/"