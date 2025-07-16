#!/bin/bash

echo "🚀 Запуск ПОЛНОГО обучения Enhanced XGBoost v2.0"
echo "================================================"
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

# Проверка зависимостей
echo ""
echo "📦 Проверка зависимостей..."
if ! python -c "import xgboost" 2>/dev/null; then
    echo "❌ XGBoost не установлен!"
    echo "Установите командой: pip install xgboost"
    exit 1
fi

echo "✅ Все зависимости установлены"

# Параметры для полного обучения
echo ""
echo "📊 Параметры обучения:"
echo "  - Режим: Бинарная классификация"
echo "  - Ансамбль: 3 модели"
echo "  - Данные: ВСЕ доступные символы"
echo "  - Признаки: Расширенный набор с техническими индикаторами"
echo ""

# Подтверждение
echo "⚠️ ВНИМАНИЕ: Полное обучение может занять 10-30 минут!"
echo -n "Продолжить? (y/n): "
read confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Отменено пользователем"
    exit 0
fi

# Запуск обучения
echo ""
echo "🚀 Запуск полного обучения Enhanced XGBoost v2.0..."
echo "=================================================="
echo ""

# Запуск с полными параметрами
python train_xgboost_enhanced_v2.py \
    --task classification_binary \
    --ensemble_size 3 \
    --config config.yaml

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
    echo ""
    echo "📊 Для просмотра финального отчета:"
    echo "  cat logs/$LATEST_LOG/final_report.txt"
else
    echo ""
    echo "❌ Ошибка при обучении!"
    echo "Проверьте логи для деталей."
fi