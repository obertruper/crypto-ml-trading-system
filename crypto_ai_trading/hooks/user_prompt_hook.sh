#!/bin/bash

# User prompt hook для ML Crypto Trading проекта
# Анализирует запросы пользователя и предоставляет контекст

USER_PROMPT="$1"
CWD="$2"

LOG_FILE="/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading/logs/hooks.log"
echo "[$(date)] USER-PROMPT: Prompt='$USER_PROMPT', CWD=$CWD" >> "$LOG_FILE"

# Функция для вывода JSON
output_json() {
    local decision="$1"
    local message="$2"
    cat <<EOF
{
    "decision": "$decision",
    "message": "$message"
}
EOF
}

# Проверяем ключевые слова в запросе
if [[ "$USER_PROMPT" =~ (обуч|train|модел|model) ]]; then
    echo "🧠 Контекст: Работа с ML моделями"
    echo "📍 Основные файлы:"
    echo "   - train_universal_transformer.py - главный файл обучения"
    echo "   - models/patchtst_unified.py - архитектура PatchTST"
    echo "   - config/config.yaml - конфигурация"
    echo ""
fi

if [[ "$USER_PROMPT" =~ (данн|data|загруз|download) ]]; then
    echo "📊 Контекст: Работа с данными"
    echo "🔧 Утилиты:"
    echo "   - download_data.py - загрузка с Bybit"
    echo "   - prepare_dataset.py - подготовка признаков"
    echo "   - validate_futures_symbols.py - проверка символов"
    echo ""
fi

if [[ "$USER_PROMPT" =~ (торг|trad|стратег|signal) ]]; then
    echo "📈 Контекст: Торговые стратегии"
    echo "📍 Файлы стратегий:"
    echo "   - trading/signals.py - сигналы и стратегии"
    echo "   - trading/portfolio.py - управление портфелем"
    echo "   - utils/risk_management.py - риск-менеджмент"
    echo ""
fi

if [[ "$USER_PROMPT" =~ (test|тест|провер|check) ]]; then
    echo "🧪 Контекст: Тестирование"
    echo "🔍 Проверки:"
    echo "   - test_model_performance.py - тест производительности"
    echo "   - Запустите pytest для unit-тестов"
    echo "   - monitor_training.py - мониторинг обучения"
    echo ""
fi

# Проверяем, находимся ли мы в директории проекта
if [[ "$CWD" == *"crypto_ai_trading"* ]]; then
    # Проверяем статус важных сервисов
    echo "🔧 Статус системы:"
    
    # PostgreSQL
    if pg_isready -h localhost -p 5555 &>/dev/null; then
        echo "   ✅ PostgreSQL (порт 5555) - активна"
    else
        echo "   ❌ PostgreSQL (порт 5555) - не доступна"
    fi
    
    # GPU
    if command -v nvidia-smi &> /dev/null && nvidia-smi &>/dev/null; then
        echo "   ✅ GPU - доступен"
    else
        echo "   ⚠️  GPU - не обнаружен"
    fi
    
    # Python окружение
    if [[ -d "venv_crypto" ]]; then
        echo "   ✅ Виртуальное окружение - найдено"
    else
        echo "   ⚠️  Виртуальное окружение - не найдено"
    fi
fi

# Всегда разрешаем выполнение
output_json "continue" "Контекстная информация предоставлена"