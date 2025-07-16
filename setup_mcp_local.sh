#!/bin/bash
# Установка MCP (Model Context Protocol) инструментов локально

echo "🚀 Установка MCP инструментов для ML Crypto Trading проекта..."
echo "=============================================="

# Проверка наличия Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js не установлен. Установите Node.js версии 16 или выше"
    echo "   Скачать можно с: https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js найден: $(node --version)"

# Создание директории для MCP если не существует
mkdir -p .mcp
cd .mcp

# Инициализация npm проекта
echo "📦 Инициализация npm проекта..."
npm init -y

# Установка MCP серверов локально
echo ""
echo "📦 Установка MCP серверов..."

# 1. Filesystem сервер - для работы с файлами проекта
echo "1️⃣ Установка filesystem сервера..."
npm install @modelcontextprotocol/server-filesystem

# 2. PostgreSQL сервер - для работы с базой данных
echo "2️⃣ Установка PostgreSQL сервера..."
npm install @modelcontextprotocol/server-postgres

# 3. GitHub сервер - для интеграции с репозиторием
echo "3️⃣ Установка GitHub сервера..."
npm install @modelcontextprotocol/server-github

# 4. Memory сервер - для контекстной памяти
echo "4️⃣ Установка Memory сервера..."
npm install @modelcontextprotocol/server-memory

# Возвращаемся в корневую директорию
cd ..

# Создание файла переменных окружения
echo ""
echo "📝 Создание файла окружения..."
cat > .mcp/.env << EOF
# MCP Environment Variables
MCP_PROJECT_ROOT=/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM
MCP_DB_CONNECTION=postgres://ruslan:your_secure_password_here@localhost:5555/crypto_trading
MCP_LOG_LEVEL=info

# GitHub Token (добавьте свой токен)
GITHUB_PERSONAL_ACCESS_TOKEN=

# API Keys (если нужны)
BYBIT_API_KEY=
BYBIT_API_SECRET=
EOF

# Создание конфигурации MCP
echo ""
echo "📝 Создание конфигурации MCP..."
cat > .mcp/mcp_config.json << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "command": "node",
      "args": [
        ".mcp/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js",
        "/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM"
      ]
    },
    "postgres": {
      "command": "node",
      "args": [
        ".mcp/node_modules/@modelcontextprotocol/server-postgres/dist/index.js"
      ],
      "env": {
        "DATABASE_URL": "postgres://ruslan:your_secure_password_here@localhost:5555/crypto_trading"
      }
    },
    "github": {
      "command": "node",
      "args": [
        ".mcp/node_modules/@modelcontextprotocol/server-github/dist/index.js"
      ],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    },
    "memory": {
      "command": "node",
      "args": [
        ".mcp/node_modules/@modelcontextprotocol/server-memory/dist/index.js"
      ]
    }
  }
}
EOF

# Создание скрипта запуска MCP
echo ""
echo "📝 Создание скрипта запуска..."
cat > start_mcp.sh << 'EOF'
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
EOF

chmod +x start_mcp.sh

# Создание скрипта остановки
cat > stop_mcp.sh << 'EOF'
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
EOF

chmod +x stop_mcp.sh

echo ""
echo "✅ Установка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo "1. Добавьте GitHub токен в .mcp/.env (если нужно)"
echo "2. Запустите MCP серверы: ./start_mcp.sh"
echo "3. Остановка серверов: ./stop_mcp.sh"
echo ""
echo "💡 MCP серверы установлены локально в .mcp/node_modules"