# ML система прогнозирования криптовалютного рынка

Система машинного обучения для прогнозирования движения цен криптовалют на основе технических индикаторов и исторических данных.

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка PostgreSQL

Убедитесь, что PostgreSQL запущен на порту 5430 (или измените порт в `config.yaml`).

Создайте базу данных и таблицы:
```bash
python init_database.py
```

### 3. Настройка конфигурации

Отредактируйте `config.yaml`:
- Измените пароль PostgreSQL в секции `database`
- Проверьте риск-профиль в секции `risk_profile`
- Список монет уже настроен (51 криптовалюта)

### 4. Запуск системы

#### Шаг 1: Загрузка исторических данных
```bash
python download_data.py
```
⏱️ Время выполнения: ~2-3 часа для всех монет

Для тестирования можно загрузить только Bitcoin:
```python
# В download_data.py раскомментируйте строку:
results = {'BTCUSDT': downloader.download_historical_data('BTCUSDT', interval, days)}
```

#### Шаг 2: Подготовка датасета
```bash
python prepare_dataset.py
```
⏱️ Время выполнения: ~30-60 минут

#### Шаг 3: Обучение модели
```bash
python train_model_postgres.py
```
⏱️ Время выполнения: ~4-8 часов (зависит от GPU)

## 📊 Архитектура системы

### Модели
Система обучает 4 модели для прогнозирования:
- **buy_profit_model** - вероятность достижения +5.8% до -1.1%
- **buy_loss_model** - вероятность достижения -1.1% до +5.8%
- **sell_profit_model** - вероятность достижения -5.8% до +1.1%
- **sell_loss_model** - вероятность достижения +1.1% до -5.8%

### Технические индикаторы (60+)
- Трендовые: EMA, ADX, MACD, Ichimoku, SAR
- Осцилляторы: RSI, Stochastic, CCI, Williams %R
- Объемные: OBV, CMF, MFI
- Волатильность: ATR, Bollinger Bands, Donchian

### База данных PostgreSQL
```
crypto_trading/
├── raw_market_data         # Сырые OHLCV данные
├── processed_market_data   # Данные с индикаторами
├── model_metadata         # Метаданные моделей
├── model_predictions      # Предсказания
└── training_sequences     # Обучающие последовательности
```

## 🎯 Результаты

После обучения модель предсказывает:
```python
{
    "buy_profit_probability": 0.78,   # 78% шанс профита для BUY
    "buy_loss_probability": 0.22,     # 22% шанс убытка для BUY
    "sell_profit_probability": 0.65,  # 65% шанс профита для SELL
    "sell_loss_probability": 0.35     # 35% шанс убытка для SELL
}
```

## 📁 Структура проекта
```
project/
├── config.yaml              # Конфигурация
├── init_database.py         # Создание БД
├── download_data.py         # Загрузка данных
├── prepare_dataset.py       # Подготовка датасета
├── train_model_postgres.py  # Обучение модели
├── requirements.txt         # Зависимости
├── trained_model/          # Сохраненные модели
│   ├── buy_profit_model.h5
│   ├── buy_loss_model.h5
│   ├── sell_profit_model.h5
│   ├── sell_loss_model.h5
│   ├── scaler.pkl
│   └── feature_columns.json
└── plots/                  # Графики обучения
```

## ⚙️ Требования
- Python 3.8+
- PostgreSQL 12+
- 16GB+ RAM
- GPU (опционально, но рекомендуется)
- 50GB+ свободного места

## 🔧 Дополнительные настройки

### Изменение риск-профиля
В `config.yaml` измените параметры:
```yaml
risk_profile:
  stop_loss_pct_buy: 0.989    # Ваш SL для BUY
  take_profit_pct_buy: 1.058  # Ваш TP для BUY
  stop_loss_pct_sell: 1.011   # Ваш SL для SELL
  take_profit_pct_sell: 0.942 # Ваш TP для SELL
```

### Добавление новых монет
Добавьте символы в `config.yaml`:
```yaml
data_download:
  symbols:
    - NEWUSDT
```

## 📈 Мониторинг производительности
Проверка статистики БД:
```sql
SELECT symbol, COUNT(*) FROM raw_market_data GROUP BY symbol;
SELECT * FROM model_metadata ORDER BY created_at DESC;
```

## 🚨 Решение проблем

### Ошибка подключения к PostgreSQL
```bash
# Проверьте что PostgreSQL запущен
pg_ctl status

# Проверьте порт в config.yaml (должен быть 5430)
```

### Недостаточно памяти при обучении
Уменьшите batch_size в `config.yaml`:
```yaml
model:
  batch_size: 16  # вместо 32
```

### Медленная загрузка данных
Загрузите только несколько монет для начала, изменив список в `config.yaml`.

## 📞 Поддержка
При возникновении проблем создайте issue с описанием ошибки и логами.