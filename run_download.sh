#!/bin/bash
echo "🚀 Запуск загрузки данных..."
echo "📁 Рабочая директория: $(pwd)"

# Активируем виртуальное окружение
echo "🔧 Активация виртуального окружения..."
source /mnt/SSD/PYCHARMPRODJECT/LLM\ TRANSFORM/venv/bin/activate

# Проверяем Python
echo "🐍 Python: $(which python)"
echo "📦 Проверка модулей..."
python -c "import pandas; print('✅ pandas OK')"
python -c "import psycopg2; print('✅ psycopg2 OK')"

# Запускаем скрипт
echo ""
echo "▶️  Запуск download_data.py..."
python download_data.py