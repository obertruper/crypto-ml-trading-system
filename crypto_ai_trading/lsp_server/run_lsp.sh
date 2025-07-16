#!/bin/bash
# Скрипт запуска Enhanced Python LSP Server

# Переход в директорию сервера
cd "$(dirname "$0")"

# Активация виртуального окружения
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Виртуальное окружение не найдено. Создаем..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Устанавливаем зависимости..."
    pip install -r requirements.txt
fi

# Запуск сервера
echo "Запуск Enhanced Python LSP Server..."
python enhanced_lsp_server.py "$@"
