#!/bin/bash

# Post-edit hook для ML Crypto Trading проекта
# Выполняет проверки после редактирования файлов

TOOL_NAME="$1"
FILE_PATH="$2"
EXIT_CODE="$3"

LOG_FILE="/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading/logs/hooks.log"
echo "[$(date)] POST-EDIT: Tool=$TOOL_NAME, File=$FILE_PATH, ExitCode=$EXIT_CODE" >> "$LOG_FILE"

# Если редактирование не удалось, выходим
if [ "$EXIT_CODE" != "0" ]; then
    exit 0
fi

# Определяем тип файла и выполняем соответствующие проверки
if [[ "$FILE_PATH" == *.py ]]; then
    # Для Python файлов проверяем синтаксис
    echo "🔍 Проверка синтаксиса Python файла..."
    
    cd "/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading"
    
    # Проверяем синтаксис
    python -m py_compile "$FILE_PATH" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "❌ ОШИБКА: Синтаксическая ошибка в файле $FILE_PATH"
        echo "💡 Рекомендуется исправить синтаксические ошибки"
    fi
    
    # Если это файл модели, напоминаем о необходимости тестирования
    if [[ "$FILE_PATH" == *"models/"* ]] || [[ "$FILE_PATH" == *"train"* ]]; then
        echo "📊 Файл модели изменен. Рекомендуется:"
        echo "   - Проверить совместимость с существующими чекпоинтами"
        echo "   - Запустить тесты модели"
        echo "   - Проверить конфигурацию в config.yaml"
    fi
    
elif [[ "$FILE_PATH" == *.yaml ]] || [[ "$FILE_PATH" == *.yml ]]; then
    # Для YAML файлов проверяем валидность
    echo "🔍 Проверка YAML конфигурации..."
    
    python -c "import yaml; yaml.safe_load(open('$FILE_PATH'))" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "❌ ОШИБКА: Невалидный YAML в файле $FILE_PATH"
    else
        echo "✅ YAML конфигурация валидна"
    fi
fi

# Для важных файлов выводим напоминание о коммите
IMPORTANT_FILES=("models/" "train" "config/" "trading/signals.py" "main.py")
for pattern in "${IMPORTANT_FILES[@]}"; do
    if [[ "$FILE_PATH" == *"$pattern"* ]]; then
        echo "📝 Важный файл изменен. Не забудьте закоммитить изменения после проверки!"
        break
    fi
done

exit 0