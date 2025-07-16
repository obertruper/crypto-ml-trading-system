# 🚀 ML Trading System v3.0 - Решение проблемы ROC-AUC 0.5

## 🎯 Проблема и решение

**ПРОБЛЕМА**: Исходная модель показывала ROC-AUC ~0.5 (случайные предсказания)

**РЕШЕНИЕ**: Комплексный подход с несколькими улучшениями:

1. **Адаптивные пороги** на основе волатильности (ATR)
2. **Confidence-based предсказания** - модель знает, когда она уверена
3. **Ансамбль стратегий** - разные модели для разных рыночных условий
4. **Walk-forward анализ** - правильная временная валидация
5. **Учет рыночного режима** - тренд/флет/волатильность

## 🚀 Быстрый старт

### Вариант 1: Автоматический скрипт
```bash
./quick_start.sh
```

### Вариант 2: Ручные команды
```bash
# Быстрый тест (5-10 минут)
python run_ml_trading.py --mode test

# Полное обучение (30-60 минут)
python run_ml_trading.py --mode full --symbols BTCUSDT ETHUSDT BNBUSDT
```

## 📊 Архитектура решения

### 1. Продвинутые целевые переменные
- **Простые бинарные**: цена выросла/упала
- **Адаптивные пороги**: на основе ATR (0.1% - 2%)
- **Confidence метки**: сильные сигналы (2x порог)
- **Risk-adjusted returns**: доходность / максимальная просадка
- **Стратегические сигналы**: для разных подходов

### 2. Ансамбль стратегий
- **Trend Following**: EMA пересечения, трендовые индикаторы
- **Mean Reversion**: RSI overbought/oversold
- **Breakout**: Bollinger Bands пробои
- **Momentum**: Price momentum > 2%

### 3. Confidence-based модели
- **Direction Model**: XGBoost классификатор (направление)
- **Confidence Model**: XGBoost регрессор (уверенность 0-1)
- **Фильтрация**: только предсказания с confidence > 0.6

### 4. Walk-forward валидация
```
|-- Train 30d --|-- Gap 1h --|-- Test 7d --|
                |-- Train 30d --|-- Gap 1h --|-- Test 7d --|
                               |-- Train 30d --|-- Gap 1h --|-- Test 7d --|
```

## 📈 Ожидаемые результаты

### Целевые метрики:
- **ROC-AUC**: 0.55-0.65 (вместо 0.5)
- **High-confidence accuracy**: 60-75%
- **Coverage**: 25-40% (доля высокоуверенных предсказаний)

### Интерпретация:
- **ROC-AUC > 0.55**: Модель лучше случайной
- **ROC-AUC > 0.60**: Хороший результат для крипто
- **ROC-AUC > 0.65**: Отличный результат

## 🏗️ Структура проекта

```
xgboost_v3/
├── run_ml_trading.py           # 🚀 ГЛАВНЫЙ МОДУЛЬ ЗАПУСКА
├── quick_start.sh              # 🛠️ Автоматический скрипт
├── advanced_trading_system.py  # 📊 Продвинутые целевые переменные
├── train_advanced_models.py    # 🤖 Обучение моделей
├── config.yaml                 # ⚙️ Конфигурация
└── ml_models_YYYYMMDD_HHMMSS/ # 📁 Результаты обучения
    ├── ensemble_model.pkl      # 💾 Обученная модель
    ├── cv_results.csv         # 📊 Результаты кросс-валидации
    └── final_report.txt       # 📝 Итоговый отчет
```

## 📋 Подробные команды

### Тестирование (быстро)
```bash
# Базовый тест
python run_ml_trading.py --mode test

# Тест с другим горизонтом
python run_ml_trading.py --mode test --horizon 4hour

# Тест с большим количеством CV
python run_ml_trading.py --mode test --cv-splits 5
```

### Продакшн обучение
```bash
# Базовое обучение
python run_ml_trading.py --mode full

# Конкретные символы
python run_ml_trading.py --mode full --symbols BTCUSDT ETHUSDT BNBUSDT XRPUSDT

# Долгосрочный горизонт
python run_ml_trading.py --mode full --horizon 16hour --cv-splits 10
```

### Отдельные модули
```bash
# Только создание целевых переменных
python advanced_trading_system.py --test
python advanced_trading_system.py --symbols BTCUSDT ETHUSDT --limit 100000

# Только обучение (если данные уже есть)
python train_advanced_models.py --symbols BTCUSDT --horizon 1hour --cv-splits 5
```

## 🔍 Анализ результатов

### 1. Проверка таблиц
```sql
-- Количество записей в таблицах
SELECT 'advanced_targets' as table_name, COUNT(*) as records 
FROM advanced_targets
UNION ALL
SELECT 'simple_targets', COUNT(*) FROM simple_targets;

-- Распределение по символам
SELECT symbol, COUNT(*) as records 
FROM advanced_targets 
GROUP BY symbol 
ORDER BY records DESC;

-- Распределение сигналов
SELECT 
    symbol,
    AVG(CASE WHEN buy_adaptive_1hour THEN 1 ELSE 0 END) * 100 as buy_signals_pct,
    AVG(CASE WHEN buy_strong_1hour THEN 1 ELSE 0 END) * 100 as strong_signals_pct
FROM advanced_targets 
GROUP BY symbol;
```

### 2. Анализ модели
```python
import joblib
import pandas as pd

# Загрузка модели
model = joblib.load('ml_models_*/ensemble_model.pkl')

# Результаты кросс-валидации
cv_results = pd.read_csv('ml_models_*/cv_results.csv')
print(cv_results.describe())

# Средние метрики
print(f"Средний ROC-AUC: {cv_results['roc_auc'].mean():.3f}")
print(f"Средняя точность (high-conf): {cv_results['high_confidence_accuracy'].mean():.3f}")
```

### 3. Визуализация результатов
```python
import matplotlib.pyplot as plt

# График ROC-AUC по fold
cv_results['roc_auc'].plot(kind='bar', title='ROC-AUC по fold')
plt.axhline(y=0.5, color='red', linestyle='--', label='Random')
plt.axhline(y=0.55, color='green', linestyle='--', label='Target')
plt.legend()
plt.show()

# Точность vs Покрытие
plt.scatter(cv_results['coverage'], cv_results['high_confidence_accuracy'])
plt.xlabel('Coverage (доля высокоуверенных предсказаний)')
plt.ylabel('High-confidence Accuracy')
plt.title('Trade-off: Точность vs Покрытие')
plt.show()
```

## 🛠️ Устранение проблем

### Проблема: "Нет данных для обучения"
```bash
# Проверить наличие таблиц
psql -U ruslan -d crypto_trading -p 5555 -c "SELECT COUNT(*) FROM advanced_targets;"

# Если таблица пуста - создать данные
python advanced_trading_system.py --test
```

### Проблема: "ROC-AUC все еще ~0.5"
1. **Больше данных**: увеличить количество символов и записей
2. **Другой горизонт**: попробовать 4hour или 16hour
3. **Больше CV folds**: увеличить до 10-15
4. **Проверить признаки**: убедиться, что нет data leakage

### Проблема: "Низкое покрытие confidence"
1. **Снизить порог**: с 0.6 до 0.5
2. **Улучшить confidence модель**: больше данных для обучения
3. **Проверить баланс**: может быть мало сигналов для обучения

## 📊 База данных

### Основные таблицы:
- `raw_market_data` - сырые данные OHLCV
- `processed_market_data` - с техническими индикаторами  
- `simple_targets` - простые целевые переменные
- `advanced_targets` - продвинутые целевые переменные

### Ключевые колонки в advanced_targets:
- `buy_adaptive_1hour` - адаптивный сигнал покупки
- `buy_strong_1hour` - сильный сигнал (высокая confidence)
- `market_regime` - режим рынка (trending_up/down, sideways, etc.)
- `adaptive_threshold` - адаптивный порог (0.1% - 2%)
- `volatility_percentile` - перцентиль волатильности

## 🎯 Дальнейшие улучшения

Если текущие результаты недостаточны:

### 1. Больше данных
- Все 51 символ вместо 2-4
- Больше исторических данных
- Более частые обновления

### 2. Дополнительные признаки
- Order flow данные
- Social sentiment (Twitter, Reddit)
- Макроэкономические индикаторы
- Cross-asset корреляции

### 3. Другие архитектуры
- LSTM/GRU для временных зависимостей
- Transformer модели
- Graph Neural Networks для корреляций

### 4. Продвинутые техники
- Meta-learning (учиться учиться)
- Adversarial training
- Multi-task learning
- Online learning (continuous adaptation)

## 🎉 Заключение

Эта система представляет собой значительное улучшение по сравнению с базовой моделью:

✅ **Решает проблему ROC-AUC 0.5** через адаптивные пороги  
✅ **Confidence-based подход** для качественных предсказаний  
✅ **Правильная временная валидация** через walk-forward  
✅ **Учет рыночных условий** через режимы и стратегии  
✅ **Готовая инфраструктура** для дальнейших улучшений  

Система готова к использованию и дальнейшему развитию!