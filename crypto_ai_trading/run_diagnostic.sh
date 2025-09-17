#!/bin/bash

echo "=========================================="
echo "🔍 ЗАПУСК ДИАГНОСТИКИ ОБУЧЕНИЯ"
echo "=========================================="
echo ""

# Создание директории для логов
mkdir -p diagnostic_logs

# Очистка старых логов (опционально)
# rm -rf diagnostic_logs/*

echo "📝 Логи будут сохранены в diagnostic_logs/"
echo ""

# Запуск диагностики
echo "🚀 Запуск диагностического менеджера..."
python training_diagnostic_manager.py

# Проверка результата
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Диагностика завершена успешно!"
    echo ""
    echo "📊 Результаты:"
    ls -lah diagnostic_logs/*.log 2>/dev/null
    ls -lah diagnostic_logs/*.json 2>/dev/null
    echo ""
    echo "Для просмотра логов:"
    echo "  cat diagnostic_logs/diagnostic_*.log"
    echo ""
    echo "Для анализа JSON:"
    echo "  python -m json.tool diagnostic_logs/diagnostic_*.json | less"
else
    echo ""
    echo "❌ Диагностика завершилась с ошибкой!"
    echo "Проверьте последний лог файл в diagnostic_logs/"
fi