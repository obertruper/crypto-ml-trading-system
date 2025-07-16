#!/bin/bash
# Скрипт для синхронизации проекта с Vast.ai сервером

echo "🚀 Синхронизация проекта с Vast.ai сервером..."
echo "============================================="

# Параметры подключения
REMOTE_HOST="184.98.25.179"
REMOTE_PORT="41575"
REMOTE_USER="root"
LOCAL_PROJECT="/Users/ruslan/PycharmProjects/LLM TRANSFORM"
REMOTE_PROJECT="/workspace/crypto_trading"

# Исключаемые файлы и папки
EXCLUDE_PATTERNS=(
    "*.pyc"
    "__pycache__/"
    ".git/"
    ".idea/"
    "venv/"
    "logs/"
    "*.log"
    ".DS_Store"
    "trained_model/"
    "plots/"
)

# Формируем параметры исключения для rsync
EXCLUDE_ARGS=""
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude='$pattern'"
done

# Создаем директорию на сервере если её нет
ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_PROJECT"

# Синхронизация файлов
echo "📁 Синхронизирую файлы..."
rsync -avz --progress \
    -e "ssh -p $REMOTE_PORT" \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='.idea/' \
    --exclude='venv/' \
    --exclude='logs/' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    --exclude='trained_model/' \
    --exclude='plots/' \
    "$LOCAL_PROJECT/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT/"

if [ $? -eq 0 ]; then
    echo "✅ Синхронизация завершена успешно!"
    
    # Показываем структуру проекта на сервере
    echo ""
    echo "📂 Структура проекта на сервере:"
    ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PROJECT && ls -la"
else
    echo "❌ Ошибка при синхронизации!"
    exit 1
fi

echo ""
echo "💡 Подсказки:"
echo "  - Для подключения к серверу: ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST"
echo "  - Рабочая директория: $REMOTE_PROJECT"
echo "  - Активация venv: source /workspace/venv/bin/activate"