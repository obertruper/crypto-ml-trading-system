#!/bin/bash
# Тестовый запуск улучшенной модели XGBoost v3

echo "🚀 Запуск улучшенной модели XGBoost v3"
echo "Основные изменения:"
echo "  ✅ Порог классификации: 0.3% (было 0.5%)"
echo "  ✅ Упрощенная модель: max_depth=4, learning_rate=0.01"
echo "  ✅ Добавлены простые ценовые признаки"
echo "  ✅ Добавлен базовый час как технический признак"
echo "  ✅ Отключен SMOTE, используется scale_pos_weight"
echo "  ✅ Увеличены попытки Optuna до 50"
echo ""

cd /Users/ruslan/PycharmProjects/LLM\ TRANSFORM

# Активация виртуального окружения если нужно
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Запуск обучения
python xgboost_v3/main.py --test-mode --optimize --ensemble-size 5

echo ""
echo "✅ Обучение завершено. Проверьте результаты в logs/"