#!/bin/bash
# Установка MCP (Model Context Protocol) инструментов для проекта

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

# Установка MCP серверов
echo ""
echo "📦 Установка MCP серверов..."

# 1. Filesystem сервер - для работы с файлами проекта
echo "1️⃣ Установка filesystem сервера..."
npm install -g @modelcontextprotocol/server-filesystem

# 2. PostgreSQL сервер - для работы с базой данных
echo "2️⃣ Установка PostgreSQL сервера..."
npm install -g @modelcontextprotocol/server-postgres

# 3. GitHub сервер - для интеграции с репозиторием
echo "3️⃣ Установка GitHub сервера..."
npm install -g @modelcontextprotocol/server-github

# 4. Memory сервер - для контекстной памяти
echo "4️⃣ Установка Memory сервера..."
npm install -g @modelcontextprotocol/server-memory

# 5. Fetch сервер - для работы с внешними API
echo "5️⃣ Установка Fetch сервера..."
npm install -g @modelcontextprotocol/server-fetch

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

# Запуск filesystem сервера
echo "Starting filesystem server..."
npx @modelcontextprotocol/server-filesystem "$MCP_PROJECT_ROOT" &

# Запуск PostgreSQL сервера
echo "Starting PostgreSQL server..."
npx @modelcontextprotocol/server-postgres "$MCP_DB_CONNECTION" &

echo "✅ MCP серверы запущены"
echo "Для остановки используйте: pkill -f modelcontextprotocol"
EOF

chmod +x start_mcp.sh

# Создание документации
echo ""
echo "📚 Создание документации..."
cat > .mcp/README.md << EOF
# MCP (Model Context Protocol) для ML Crypto Trading

## Установленные компоненты

1. **Filesystem Server** - доступ к файлам проекта
2. **PostgreSQL Server** - работа с базой данных
3. **GitHub Server** - интеграция с Git
4. **Memory Server** - контекстная память
5. **Fetch Server** - работа с внешними API

## Использование

### Запуск серверов
\`\`\`bash
./start_mcp.sh
\`\`\`

### Остановка серверов
\`\`\`bash
pkill -f modelcontextprotocol
\`\`\`

## Конфигурация

- Основная конфигурация: \`.mcp/config.json\`
- Переменные окружения: \`.mcp/.env\`

## Интеграция с Claude

MCP серверы автоматически предоставляют контекст о:
- Структуре проекта
- Схеме базы данных
- Истории изменений
- Внешних API

Это позволяет Claude лучше понимать проект и давать более точные рекомендации.
EOF

echo ""
echo "✅ Установка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo "1. Добавьте GitHub токен в .mcp/.env (если нужно)"
echo "2. Запустите MCP серверы: ./start_mcp.sh"
echo "3. Серверы будут доступны для Claude автоматически"
echo ""
echo "💡 Подсказка: MCP позволяет Claude лучше понимать контекст вашего проекта"