# 🎯 Модель предсказания направления движения цены

## Описание

Новый подход к ML трейдингу, основанный на простом предсказании направления движения цены вместо сложных expected returns.

### Преимущества нового подхода:
- ✅ **Простая целевая переменная**: цена выросла/упала через 1 час
- ✅ **Walk-Forward анализ**: золотой стандарт валидации для трейдинга
- ✅ **Несколько временных горизонтов**: 5 мин, 15 мин, 1 час, 4 часа
- ✅ **Адаптивный отбор признаков**: топ-50 признаков на каждом периоде
- ✅ **Реалистичная валидация**: обучение на 30 днях, тест на следующих 7

## 🚀 Быстрый старт

### 1. Инициализация простых целевых переменных

```bash
# Тестовый запуск (2 символа, 100k записей)
python init_simple_targets.py --test

# Полная инициализация
python init_simple_targets.py --symbols BTCUSDT ETHUSDT BNBUSDT XRPUSDT ADAUSDT

# Все символы с порогом 0.2%
python init_simple_targets.py --threshold 0.2
```

### 2. Запуск обучения модели

```bash
# Базовое обучение (2 символа, 5 splits)
python train_direction_model.py

# Расширенное обучение
python train_direction_model.py \
    --symbols BTCUSDT ETHUSDT BNBUSDT XRPUSDT \
    --n-splits 10 \
    --target-type buy_signal_threshold_1hour
```

### 3. Проверка результатов

Результаты сохраняются в `direction_model_results/YYYYMMDD_HHMMSS/`:
- `final_report.txt` - итоговый отчет с метриками
- `buy_metrics.json` / `sell_metrics.json` - детальные метрики
- `walk_forward_results.png` - график производительности по периодам
- `feature_importance_comparison.png` - важность признаков

## 📊 Целевые переменные

### Простые бинарные сигналы
- `buy_signal_1hour` - цена выросла через 1 час (любое движение вверх)
- `sell_signal_1hour` - цена упала через 1 час (любое движение вниз)

### Сигналы с порогом
- `buy_signal_threshold_1hour` - цена выросла более чем на 0.1%
- `sell_signal_threshold_1hour` - цена упала более чем на 0.1%

### Мультиклассовая классификация
- 0: Сильное падение (< -1%)
- 1: Слабое падение (-1% до -0.1%)
- 2: Нейтрально (-0.1% до 0.1%)
- 3: Слабый рост (0.1% до 1%)
- 4: Сильный рост (> 1%)

## 🔍 Walk-Forward анализ

Модель использует скользящее окно для обучения и тестирования:

```
|-- Train 30d --|-- Gap 1h --|-- Test 7d --|
                |-- Train 30d --|-- Gap 1h --|-- Test 7d --|
                               |-- Train 30d --|-- Gap 1h --|-- Test 7d --|
```

- **Train**: 30 дней исторических данных
- **Gap**: 1 час зазор (избегаем data leakage)
- **Test**: 7 дней для оценки производительности
- **Переобучение**: каждые 7 дней на новых данных

## 📈 Ожидаемые результаты

На основе исследований для криптовалют:
- **ROC-AUC**: 0.55-0.65 (лучше случайного 0.5)
- **Accuracy**: 52-59% (со временем может улучшаться)
- **Sharpe Ratio**: 1.5-3.0 (при правильной стратегии)

## 🛠️ Настройка параметров

### В `init_simple_targets.py`:
```python
min_movement_threshold = 0.1  # Минимальное движение для сигнала
horizons = {
    '5min': 0.33,    # Краткосрочные сигналы
    '15min': 1,      # Основной таймфрейм
    '1hour': 4,      # Среднесрочные
    '4hours': 16     # Долгосрочные
}
```

### В `train_direction_model.py`:
```python
# Walk-forward параметры
train_window_days = 30  # Окно обучения
test_window_days = 7    # Окно тестирования
n_splits = 10          # Количество периодов

# XGBoost параметры
max_depth = 6
learning_rate = 0.05
n_estimators = 300
```

## 🔧 Структура БД

Новая таблица `simple_targets`:
```sql
CREATE TABLE simple_targets (
    timestamp TIMESTAMP,
    symbol VARCHAR(20),
    close_price DECIMAL(20, 8),
    
    -- Будущие цены
    price_1hour DECIMAL(20, 8),
    
    -- Процентные изменения
    change_1hour DECIMAL(10, 4),
    
    -- Бинарные сигналы
    buy_signal_1hour BOOLEAN,
    sell_signal_1hour BOOLEAN,
    
    -- Сигналы с порогом
    buy_signal_threshold_1hour BOOLEAN,
    sell_signal_threshold_1hour BOOLEAN
);
```

## 📝 Примеры SQL запросов

```sql
-- Проверка распределения сигналов
SELECT 
    symbol,
    COUNT(*) as total,
    AVG(CASE WHEN buy_signal_threshold_1hour THEN 1 ELSE 0 END) * 100 as buy_pct,
    AVG(CASE WHEN sell_signal_threshold_1hour THEN 1 ELSE 0 END) * 100 as sell_pct
FROM simple_targets
WHERE change_1hour IS NOT NULL
GROUP BY symbol
ORDER BY total DESC;

-- Статистика изменений
SELECT 
    symbol,
    AVG(change_1hour) as avg_change,
    STDDEV(change_1hour) as std_change,
    MIN(change_1hour) as min_change,
    MAX(change_1hour) as max_change
FROM simple_targets
WHERE change_1hour IS NOT NULL
GROUP BY symbol;
```

## ⚡ Оптимизация производительности

1. **Используйте подмножество символов** для первых экспериментов
2. **Начните с простых сигналов** без порога
3. **Увеличивайте n_splits постепенно** (5 → 10 → 20)
4. **Мониторьте стабильность** метрик между splits

## 🎯 Следующие шаги

После успешного обучения базовой модели:

1. **Добавить больше данных**: 51 символ вместо 2-4
2. **Оптимизировать пороги**: найти оптимальный threshold
3. **Ансамблирование**: объединить модели разных горизонтов
4. **Feature engineering**: добавить order flow, volume profile
5. **Интеграция с трейдингом**: бэктестинг стратегий

## 🚨 Важные замечания

- Модель предсказывает **вероятность** движения, не гарантию
- Требуется правильный **risk management** при торговле
- Производительность может меняться в разных рыночных условиях
- Регулярное переобучение критически важно