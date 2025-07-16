#!/bin/bash
# Скрипт для настройки удаленного сервера

# Загрузка конфигурации из config.yaml
CONFIG_FILE="config/config.yaml"

# Функция для чтения значений из YAML
get_config() {
    python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    remote = config.get('remote_server', {})
    print(remote.get('$1', ''))
"
}

# Получение параметров
HOST=$(get_config "host")
PORT=$(get_config "port")
USER=$(get_config "user")
KEY_PATH=$(get_config "key_path")
REMOTE_PATH=$(get_config "remote_path")

# Проверка настроек
if [ -z "$HOST" ]; then
    echo "❌ Ошибка: Не указан хост сервера в config.yaml"
    echo "Добавьте настройки в секцию remote_server:"
    echo "  host: YOUR_SERVER_IP"
    exit 1
fi

echo "🚀 Настройка удаленного сервера"
echo "📍 Сервер: $USER@$HOST:$PORT"
echo "📁 Путь: $REMOTE_PATH"

# Расширение пути к ключу
KEY_PATH="${KEY_PATH/#\~/$HOME}"

# SSH команда
SSH_CMD="ssh -p $PORT"
if [ -f "$KEY_PATH" ]; then
    SSH_CMD="$SSH_CMD -i $KEY_PATH"
fi
SSH_CMD="$SSH_CMD $USER@$HOST"

# Создание директории на сервере
echo "📁 Создание директории проекта..."
$SSH_CMD "mkdir -p $REMOTE_PATH"

# Установка зависимостей
echo "📦 Проверка Python окружения..."
$SSH_CMD "cd $REMOTE_PATH && if [ ! -d 'venv' ]; then python3 -m venv venv; fi"

# Создание структуры директорий
echo "📂 Создание структуры проекта..."
$SSH_CMD "cd $REMOTE_PATH && mkdir -p cache logs experiments/runs experiments/logs models_saved results"

echo "✅ Сервер готов к работе!"