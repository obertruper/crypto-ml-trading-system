#!/bin/bash

echo "🚀 Запуск Enhanced XGBoost v2.0 локально"
echo "========================================"
echo ""

# Проверка PostgreSQL
echo "🔍 Проверка PostgreSQL..."
if pg_isready -h localhost -p 5555 > /dev/null 2>&1; then
    echo "✅ PostgreSQL работает на порту 5555"
else
    echo "❌ PostgreSQL не запущен на порту 5555!"
    echo "Запустите БД командой: docker-compose up -d"
    exit 1
fi

# Активация виртуального окружения
echo ""
echo "🐍 Активация виртуального окружения..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Виртуальное окружение активировано"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Виртуальное окружение активировано"
else
    echo "⚠️ Виртуальное окружение не найдено, используем системный Python"
fi

# Выбор режима
echo ""
echo "Выберите режим обучения:"
echo "1) 🎯 Бинарная классификация (рекомендуется)"
echo "2) 📊 Регрессия" 
echo "3) 🎲 Мультиклассовая классификация (5 классов)"
echo "4) 🧪 ТЕСТОВЫЙ РЕЖИМ (2 символа, быстрое обучение)"
echo -n "Ваш выбор (1-4): "
read choice

case $choice in
    1)
        TASK="classification_binary"
        ENSEMBLE=3
        TEST_MODE=""
        echo "✅ Выбрана бинарная классификация с ансамблем из 3 моделей"
        ;;
    2)
        TASK="regression"
        ENSEMBLE=3
        TEST_MODE=""
        echo "✅ Выбрана регрессия с ансамблем из 3 моделей"
        ;;
    3)
        TASK="classification_multiclass"
        ENSEMBLE=1
        TEST_MODE=""
        echo "✅ Выбрана мультиклассовая классификация"
        ;;
    4)
        TASK="classification_binary"
        ENSEMBLE=1
        TEST_MODE="--test_mode"
        echo "✅ Выбран ТЕСТОВЫЙ режим"
        ;;
    *)
        echo "❌ Неверный выбор"
        exit 1
        ;;
esac

# Проверка зависимостей
echo ""
echo "📦 Проверка зависимостей..."
if ! python -c "import xgboost" 2>/dev/null; then
    echo "❌ XGBoost не установлен!"
    echo "Установите командой: pip install xgboost"
    exit 1
fi

echo "✅ Все зависимости установлены"

# Запуск обучения
echo ""
echo "🚀 Запуск обучения Enhanced XGBoost v2.0..."
echo "=================================================="
echo "Task: $TASK"
echo "Ensemble size: $ENSEMBLE"
echo "Test mode: $TEST_MODE"
echo ""

# Запуск с отображением прогресса
python train_xgboost_enhanced_v2.py \
    --task $TASK \
    --ensemble_size $ENSEMBLE \
    $TEST_MODE

# Проверка результатов
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Обучение успешно завершено!"
    echo ""
    echo "📊 Результаты сохранены в:"
    echo "  - trained_model/*_xgboost_v2_*.pkl"
    echo "  - trained_model/metadata_xgboost_v2.json"
    echo "  - trained_model/scaler_xgboost_v2.pkl"
    echo ""
    echo "📈 Графики и отчеты:"
    LATEST_LOG=$(ls -t logs | grep xgboost_training | head -1)
    echo "  - logs/$LATEST_LOG/plots/"
    echo "  - logs/$LATEST_LOG/final_report.txt"
    echo ""
    echo "💡 Для просмотра графиков:"
    echo "  open logs/$LATEST_LOG/plots/*.png"
else
    echo ""
    echo "❌ Ошибка при обучении!"
    echo "Проверьте логи для деталей."
fi