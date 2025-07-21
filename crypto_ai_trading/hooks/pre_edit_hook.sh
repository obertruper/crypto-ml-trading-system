#!/bin/bash

# Pre-edit hook для ML Crypto Trading проекта
# Проверяет важные файлы перед редактированием

# Получаем параметры от Claude Code
TOOL_NAME="$1"
FILE_PATH="$2"

# Логирование
LOG_FILE="/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading/logs/hooks.log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "[$(date)] PRE-EDIT: Tool=$TOOL_NAME, File=$FILE_PATH" >> "$LOG_FILE"

# Список важных файлов, требующих особого внимания
IMPORTANT_FILES=(
    "models/patchtst.py"
    "models/patchtst_unified.py"
    "train_universal_transformer.py"
    "config/config.yaml"
    "data/feature_engineering.py"
    "trading/signals.py"
    "main.py"
)

# Проверяем, является ли файл важным
IS_IMPORTANT=false
for important_file in "${IMPORTANT_FILES[@]}"; do
    if [[ "$FILE_PATH" == *"$important_file"* ]]; then
        IS_IMPORTANT=true
        break
    fi
done

if [ "$IS_IMPORTANT" = true ]; then
    # Для важных файлов выводим предупреждение
    echo "⚠️  ВНИМАНИЕ: Вы редактируете важный файл проекта!"
    echo "📍 Файл: $FILE_PATH"
    echo "🔍 Рекомендуется:"
    echo "   1. Проверить текущее состояние файла"
    echo "   2. Использовать Sequential Thinking для анализа"
    echo "   3. Создать резервную копию при необходимости"
    
    # Проверяем, есть ли незакоммиченные изменения
    cd "/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading"
    if git status --porcelain "$FILE_PATH" 2>/dev/null | grep -q "^"; then
        echo "⚠️  В файле есть незакоммиченные изменения!"
    fi
fi

# Всегда разрешаем редактирование (exit 0)
exit 0