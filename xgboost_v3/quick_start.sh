#!/bin/bash

# 🚀 БЫСТРЫЙ СТАРТ ML TRADING SYSTEM v3.0
# Решение проблемы ROC-AUC 0.5

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     🚀 ML TRADING SYSTEM v3.0                 ║"
echo "║                                                                ║"
echo "║  Решение проблемы ROC-AUC 0.5 через:                         ║"
echo "║  • Адаптивные пороги на основе волатильности                  ║"
echo "║  • Confidence-based предсказания                              ║"
echo "║  • Ансамбль стратегий                                         ║"
echo "║  • Walk-forward анализ                                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Проверяем наличие config.yaml
if [ ! -f "config.yaml" ]; then
    echo "⚠️ Файл config.yaml не найден. Копирую из logs..."
    if [ -f "logs/xgboost_v3_20250614_161646/config.yaml" ]; then
        cp logs/xgboost_v3_20250614_161646/config.yaml .
        echo "✅ config.yaml скопирован"
    else
        echo "❌ Не найден config.yaml ни в корне, ни в logs/"
        exit 1
    fi
fi

echo "Выберите режим запуска:"
echo "1) 🧪 Тест (быстро, 1 символ, 50k записей)"
echo "2) 🚀 Полное обучение (медленно, несколько символов, все данные)"
echo "3) 📊 Только анализ данных (без обучения)"
echo "4) ❓ Справка по командам"

read -p "Ваш выбор [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "🧪 ЗАПУСК ТЕСТОВОГО РЕЖИМА..."
        echo "Это займет 5-10 минут"
        python run_ml_trading.py --mode test
        ;;
    2)
        echo ""
        echo "Выберите символы:"
        echo "1) BTCUSDT ETHUSDT (базовый набор)"
        echo "2) BTCUSDT ETHUSDT BNBUSDT XRPUSDT (расширенный)"
        echo "3) Ввести вручную"
        
        read -p "Ваш выбор [1-3]: " symbol_choice
        
        case $symbol_choice in
            1)
                symbols="BTCUSDT ETHUSDT"
                ;;
            2)
                symbols="BTCUSDT ETHUSDT BNBUSDT XRPUSDT"
                ;;
            3)
                read -p "Введите символы через пробел: " symbols
                ;;
            *)
                symbols="BTCUSDT ETHUSDT"
                ;;
        esac
        
        echo ""
        echo "🚀 ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ..."
        echo "Символы: $symbols"
        echo "Это может занять 30-60 минут"
        python run_ml_trading.py --mode full --symbols $symbols
        ;;
    3)
        echo ""
        echo "📊 АНАЛИЗ ДАННЫХ..."
        
        # Проверяем таблицы
        echo "Проверка таблиц в базе данных:"
        psql -U ruslan -d crypto_trading -p 5555 -c "
        SELECT 
            'raw_market_data' as table_name,
            COUNT(*) as records,
            COUNT(DISTINCT symbol) as symbols,
            MIN(timestamp) as start_date,
            MAX(timestamp) as end_date
        FROM raw_market_data
        UNION ALL
        SELECT 
            'processed_market_data',
            COUNT(*),
            COUNT(DISTINCT symbol),
            MIN(timestamp::text),
            MAX(timestamp::text)
        FROM processed_market_data
        UNION ALL
        SELECT 
            'simple_targets',
            COUNT(*),
            COUNT(DISTINCT symbol),
            MIN(timestamp::text),
            MAX(timestamp::text)
        FROM simple_targets
        WHERE EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'simple_targets')
        UNION ALL
        SELECT 
            'advanced_targets',
            COUNT(*),
            COUNT(DISTINCT symbol),
            MIN(timestamp::text),
            MAX(timestamp::text)
        FROM advanced_targets
        WHERE EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'advanced_targets');
        "
        ;;
    4)
        echo ""
        echo "📖 СПРАВКА ПО КОМАНДАМ:"
        echo ""
        echo "ОСНОВНЫЕ КОМАНДЫ:"
        echo "# Быстрый тест"
        echo "python run_ml_trading.py --mode test"
        echo ""
        echo "# Полное обучение"
        echo "python run_ml_trading.py --mode full --symbols BTCUSDT ETHUSDT"
        echo ""
        echo "# Разные временные горизонты"
        echo "python run_ml_trading.py --mode test --horizon 4hour"
        echo ""
        echo "# Больше CV folds"
        echo "python run_ml_trading.py --mode full --cv-splits 10"
        echo ""
        echo "ОТДЕЛЬНЫЕ МОДУЛИ:"
        echo "# Создание простых целевых переменных"
        echo "python init_simple_targets.py --test"
        echo ""
        echo "# Создание продвинутых целевых переменных"  
        echo "python advanced_trading_system.py --test"
        echo ""
        echo "# Обучение продвинутых моделей"
        echo "python train_advanced_models.py --symbols BTCUSDT"
        echo ""
        echo "АНАЛИЗ РЕЗУЛЬТАТОВ:"
        echo "# Проверка таблиц в БД"
        echo "psql -U ruslan -d crypto_trading -p 5555 -c \"SELECT COUNT(*) FROM advanced_targets;\""
        echo ""
        echo "# Просмотр результатов модели"
        echo "ls -la ml_models_*/"
        echo "cat ml_models_*/final_report.txt"
        ;;
    *)
        echo "❌ Неверный выбор. Запустите скрипт снова."
        exit 1
        ;;
esac

echo ""
echo "✅ Выполнение завершено!"
echo ""
echo "📝 СЛЕДУЮЩИЕ ШАГИ:"
echo "1. Проверьте результаты в папке ml_models_*/"
echo "2. Откройте final_report.txt для анализа"
echo "3. Если ROC-AUC > 0.55 - модель улучшена!"
echo "4. Если результаты хорошие - интегрируйте в торговую стратегию"
echo ""
echo "🔗 Полезные команды:"
echo "ls -la ml_models_*/"
echo "cat ml_models_*/final_report.txt"
echo "python -c \"import joblib; model = joblib.load('ml_models_*/ensemble_model.pkl'); print('Модель загружена успешно')\""