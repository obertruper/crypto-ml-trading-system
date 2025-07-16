#!/bin/bash

echo "🚀 Запуск Enhanced TFT v2.1"
echo "========================================"
echo ""

# Проверка аргументов
if [ "$1" == "regression" ]; then
    TASK="regression"
    echo "📊 Режим: Регрессия (предсказание expected returns)"
elif [ "$1" == "classification" ] || [ "$1" == "binary" ]; then
    TASK="classification_binary"
    echo "📊 Режим: Бинарная классификация (порог > 0.3%)"
else
    echo "Использование: $0 [regression|classification] [ensemble_size]"
    echo ""
    echo "Примеры:"
    echo "  $0 regression      # Регрессия с 1 моделью"
    echo "  $0 classification  # Классификация с 1 моделью"
    echo "  $0 classification 3 # Классификация с ансамблем из 3 моделей"
    exit 1
fi

# Размер ансамбля
ENSEMBLE_SIZE=${2:-1}
echo "🎯 Размер ансамбля: $ENSEMBLE_SIZE"
echo ""

# Создание лог директории
LOG_DIR="logs/enhanced_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# Запуск обучения
echo "🔄 Запуск обучения Enhanced TFT v2.1..."
echo "📁 Логи сохраняются в: $LOG_DIR"
echo ""

python train_universal_transformer_v2.py \
    --task $TASK \
    --ensemble_size $ENSEMBLE_SIZE

echo ""
echo "✅ Обучение завершено!"
echo ""
echo "📊 Результаты:"
echo "  - Модели: trained_model/*_enhanced_v2.1_*.h5"
echo "  - Метаданные: trained_model/metadata_v2.1.json"
echo "  - Конфигурация признаков: trained_model/feature_config_v2.1.json"
echo "  - Отчет: $LOG_DIR/final_report_v2.1.txt"
echo ""

# Показать краткую статистику если файл существует
if [ -f "$LOG_DIR/final_report_v2.txt" ]; then
    echo "📈 Краткие результаты:"
    tail -n 20 "$LOG_DIR/final_report_v2.txt"
fi