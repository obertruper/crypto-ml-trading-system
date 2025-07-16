#!/bin/bash

# Скрипт для быстрого запуска обучения Direction модели
# с enhanced features и всеми улучшениями

echo "🚀 Запуск обучения Direction модели для прибыльной торговли"
echo "=================================================="

# Проверка окружения
if ! command -v python &> /dev/null; then
    echo "❌ Python не найден!"
    exit 1
fi

# Активация виртуального окружения если есть
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi

# Создание директорий
mkdir -p logs
mkdir -p models_saved
mkdir -p data/enhanced_datasets

# Шаг 1: Подготовка enhanced датасета (если не существует)
echo ""
echo "📊 Шаг 1: Проверка данных..."
echo "-----------------------------------------"

# Проверяем processed данные
if [ -f "data/processed/train_data.parquet" ]; then
    echo "✅ Найдены обработанные данные в data/processed/"
    echo "   Будут использованы для обучения"
else
    LATEST_DATASET=$(ls -t data/enhanced_datasets/enhanced_dataset_*.pkl 2>/dev/null | head -1)
    
    if [ -z "$LATEST_DATASET" ]; then
        echo "Enhanced датасет не найден, создаем новый..."
        python prepare_enhanced_dataset.py \
            --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT \
            --start-date 2024-01-01 \
            || { echo "⚠️ Не удалось создать enhanced датасет, попробуем стандартную загрузку"; }
    else
        echo "✅ Найден enhanced датасет: $LATEST_DATASET"
    fi
fi

# Шаг 2: Обучение Direction модели
echo ""
echo "🧠 Шаг 2: Обучение Direction модели..."
echo "-------------------------------------"

# Используем меньше эпох для начала
python train_direction_model.py \
    --config configs/direction_only.yaml \
    --epochs 10 \
    || { echo "❌ Ошибка обучения"; exit 1; }

# Поиск последнего checkpoint
LATEST_CHECKPOINT=$(ls -t models_saved/best_direction_model_*.pth 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ Checkpoint не найден после обучения"
    exit 1
fi

echo "✅ Модель обучена: $LATEST_CHECKPOINT"

# Шаг 3: Оценка модели
echo ""
echo "📈 Шаг 3: Оценка точности модели..."
echo "-----------------------------------"

python evaluate_direction_model.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --dataset test \
    || { echo "❌ Ошибка оценки"; exit 1; }

# Шаг 4: Бэктест стратегии
echo ""
echo "💰 Шаг 4: Бэктест торговой стратегии..."
echo "---------------------------------------"

python backtest_direction_strategy.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --timeframe 4h \
    --initial-capital 10000 \
    || { echo "❌ Ошибка бэктеста"; exit 1; }

# Результаты
echo ""
echo "✅ ВСЕ ШАГИ ВЫПОЛНЕНЫ УСПЕШНО!"
echo "=============================="
echo ""
echo "📁 Результаты сохранены в:"
echo "  - Модель: $LATEST_CHECKPOINT"
echo "  - Логи обучения: logs/direction_training_*"
echo "  - Оценка: logs/evaluation_*"
echo "  - Бэктест: logs/backtest_*"
echo ""
echo "🎯 Следующие шаги:"
echo "  1. Проверьте результаты в папках логов"
echo "  2. Если Directional Accuracy > 55% - модель готова"
echo "  3. Если нет - увеличьте epochs или добавьте больше данных"
echo ""
echo "💡 Для production используйте:"
echo "  python main.py --mode production --model $LATEST_CHECKPOINT"