# 📚 Структура проекта ML для криптотрейдинга

## 📁 Структура файлов

```
LLM TRANSFORM/
├── config.yaml                 # Главный конфигурационный файл
├── init_database.py           # Инициализация БД PostgreSQL
├── download_data.py           # Загрузка исторических данных с Bybit
├── prepare_dataset.py         # Подготовка датасета с учетом частичных закрытий
├── train_universal_transformer.py  # Универсальная TFT модель (регрессия/классификация)
├── train_transformer_model.py     # Оригинальная TFT модель для регрессии
├── train_advanced.py             # LSTM с Attention для классификации
├── monitor_training.py           # Мониторинг процесса обучения
├── requirements.txt           # Зависимости Python
├── README.md                  # Инструкция по запуску
├── PROJECT_STRUCTURE.md       # Этот файл - документация структуры
├── CLAUDE.md                  # Правила для Claude AI
└── trained_model/            # Папка с обученными моделями (создается автоматически)
    ├── buy_return_predictor.h5
    ├── sell_return_predictor.h5
    ├── scaler.pkl
    └── metadata.json
```

## 📄 Описание файлов

### 1. config.yaml
**Назначение**: Централизованная конфигурация всего проекта

**Структура**:
```yaml
database:           # Настройки PostgreSQL
  host: localhost
  port: 5555
  database: crypto_trading
  user: ruslan
  password:

risk_profile:       # Параметры риск-менеджмента
  default_balance: 500
  risk_per_trade: 0.02
  leverage: 10
  stop_loss_pct_buy: 0.989    # -1.1%
  stop_loss_pct_sell: 1.011   # +1.1%
  take_profit_pct_buy: 1.058  # +5.8%
  take_profit_pct_sell: 0.942 # -5.8%

model:              # Параметры модели
  sequence_length: 60         # 15 часов для 15m таймфрейма
  prediction_horizon: 100     # 25 часов прогноз
  batch_size: 32
  epochs: 100
  learning_rate: 0.001

data_download:      # Параметры загрузки данных
  interval: '15'              # 15-минутные свечи
  days: 1095                  # 3 года данных
  symbols: [...]              # Список 51 криптовалюты
```

### 2. init_database.py
**Назначение**: Создание и инициализация структуры БД PostgreSQL

**Основные функции**:
- `load_config()` - загрузка конфигурации из YAML
- `create_database()` - создание базы данных если не существует
- `init_tables()` - создание всех необходимых таблиц

**Создаваемые таблицы**:
1. **raw_market_data** - сырые OHLCV данные с биржи
2. **processed_market_data** - данные с рассчитанными индикаторами
3. **model_metadata** - метаданные обученных моделей
4. **training_sequences** - подготовленные последовательности для обучения
5. **model_predictions** - логирование предсказаний модели

### 3. download_data.py
**Назначение**: Загрузка исторических данных с Bybit API в PostgreSQL

**Классы**:
- `PostgreSQLManager` - управление подключением к БД
- `BybitDataDownloader` - загрузка данных с биржи

**Основные методы**:
- `get_klines()` - получение свечей с Bybit API
- `insert_raw_data_batch()` - пакетная вставка данных в БД
- `download_historical_data()` - загрузка данных для одного символа
- `download_multiple_symbols()` - загрузка для списка символов

**Параметры**:
- Интервал: 15 минут
- Период: 1095 дней (3 года)
- Batch size: 1000 свечей за запрос
- Поддержка: futures и spot рынки

### 4. prepare_dataset.py
**Назначение**: Расчет технических индикаторов и создание меток для обучения

**Классы**:
- `PostgreSQLManager` - работа с БД
- `MarketDatasetPreparator` - подготовка датасета

**Основные методы**:
- `calculate_technical_indicators()` - расчет 49 технических индикаторов
- `create_labels_based_on_risk_profile()` - создание целевых меток
- `_calculate_enhanced_result()` - расчет с учетом частичных закрытий и защиты прибыли
- `save_processed_data()` - сохранение обработанных данных в БД

**Рассчитываемые индикаторы** (49 групп):
1. **Трендовые**: EMA, ADX, MACD, Ichimoku, SAR, Aroon, DPO
2. **Осцилляторы**: RSI, Stochastic, CCI, Williams %R, ROC, Ultimate
3. **Волатильность**: ATR, Bollinger Bands, Donchian Channel
4. **Объемные**: OBV, CMF, MFI
5. **Паттерны**: Vortex, ценовые изменения, волатильность

**Создание меток**:
- Анализирует будущие 100 баров (25 часов)
- Рассчитывает ожидаемую доходность (expected returns) для регрессии
- Учитывает частичные закрытия позиций (20% на +1.2%, 30% на +2.4%, 30% на +3.5%)
- Применяет защиту прибыли (breakeven, profit locking)
- Создает бинарные метки для классификации

### 5. train_universal_transformer.py ⭐ ГЛАВНЫЙ ФАЙЛ ДЛЯ ОБУЧЕНИЯ
**Назначение**: Универсальная модель для регрессии и классификации

**⚠️ ВАЖНО**: Все доработки и улучшения теперь вносятся ТОЛЬКО в этот файл!

**Архитектура - Temporal Fusion Transformer (TFT)**:
- Variable Selection Network (VSN) - выбор важных признаков
- Gated Residual Networks (GRN) - обработка информации
- LSTM encoder - захват локальных временных зависимостей
- Positional Encoding - кодирование позиций во времени
- Transformer Blocks (4) - self-attention механизм
- Interpretable Multi-Head Attention - интерпретируемое внимание
- Output layer - регрессия или классификация

**Параметры архитектуры**:
- `d_model`: 128 (размерность модели)
- `num_heads`: 8 (количество голов внимания)
- `num_transformer_blocks`: 4
- `dropout_rate`: 0.2
- `sequence_length`: 60 (15 часов истории)

**Режимы работы**:
1. **Регрессия**: предсказание ожидаемой доходности (%)
   - Loss: Huber Loss (устойчива к выбросам)
   - Метрики: MAE, RMSE, R², Direction Accuracy
   
2. **Классификация**: предсказание вероятности profit/loss
   - Loss: Focal Loss (для дисбаланса классов)
   - Метрики: Accuracy, Precision, Recall, F1-Score

**Особенности**:
- Поддержка GPU с автоматическим определением
- Продвинутая визуализация процесса обучения
- TensorBoard интеграция
- Автоматическая балансировка классов
- Early stopping и адаптивный learning rate

### 📊 ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ОБУЧЕНИЯ

**Да, процесс обучения визуализируется в реальном времени!**

#### Что отображается во время обучения:

1. **Автоматические графики (обновляются каждые 5 эпох)**:
   - График Loss (train/validation)
   - График основной метрики (MAE для регрессии, Accuracy для классификации)
   - График Learning Rate Schedule
   - Текущая статистика обучения

2. **Сохраняемые файлы**:
   - `logs/training_YYYYMMDD_HHMMSS/plots/training_progress.png` - основной график прогресса
   - `logs/training_YYYYMMDD_HHMMSS/plots/epoch_XXX.png` - снимки на каждой 5-й эпохе
   - `logs/training_YYYYMMDD_HHMMSS/MODEL_NAME_metrics.csv` - все метрики в CSV

3. **После обучения создаются графики оценки**:
   - Scatter plot предсказаний vs истинных значений
   - Распределение ошибок
   - Временная визуализация примеров предсказаний
   - Статистика по диапазонам значений

4. **TensorBoard визуализация**:
   ```bash
   # Запустить в отдельном терминале:
   tensorboard --logdir logs/training_YYYYMMDD_HHMMSS/tensorboard/
   ```
   Показывает:
   - Графики всех метрик
   - Гистограммы весов
   - Архитектуру модели
   - Градиенты

### 6. monitor_training.py
**Назначение**: Мониторинг процесса обучения в реальном времени

**Функционал**:
- Отображение прогресса обучения
- Визуализация метрик
- Оценка времени завершения
- Сохранение графиков

## 🗄️ Структура базы данных PostgreSQL

### Таблица: raw_market_data
```sql
id BIGSERIAL PRIMARY KEY
symbol VARCHAR(20)           # Символ (BTCUSDT)
timestamp BIGINT            # Unix timestamp
datetime TIMESTAMP          # Человекочитаемая дата
open DECIMAL(20, 8)        # Цена открытия
high DECIMAL(20, 8)        # Максимум
low DECIMAL(20, 8)         # Минимум
close DECIMAL(20, 8)       # Цена закрытия
volume DECIMAL(20, 8)      # Объем
turnover DECIMAL(20, 8)    # Оборот
interval_minutes INTEGER    # Интервал (15)
market_type VARCHAR(20)     # Тип рынка (futures/spot)
created_at TIMESTAMP       # Дата создания записи
```

### Таблица: processed_market_data
```sql
id BIGSERIAL PRIMARY KEY
raw_data_id BIGINT         # Ссылка на raw_market_data
symbol VARCHAR(20)
timestamp BIGINT
datetime TIMESTAMP
open, high, low, close, volume DECIMAL(20, 8)
technical_indicators JSONB  # Все индикаторы + expected returns
buy_profit_target INTEGER   # Метка для buy profit (0/1)
buy_loss_target INTEGER     # Метка для buy loss (0/1)
sell_profit_target INTEGER  # Метка для sell profit (0/1)
sell_loss_target INTEGER    # Метка для sell loss (0/1)
processing_version VARCHAR(10)
created_at TIMESTAMP
updated_at TIMESTAMP
```

**technical_indicators JSONB содержит**:
- 49 технических индикаторов
- buy_expected_return - ожидаемая доходность для BUY
- sell_expected_return - ожидаемая доходность для SELL

### Таблица: model_metadata
```sql
id SERIAL PRIMARY KEY
model_name VARCHAR(100)     # Имя модели
model_type VARCHAR(50)      # Тип (universal_transformer)
version VARCHAR(20)         # Версия
feature_columns JSONB       # Список признаков
training_config JSONB       # Конфигурация обучения
performance_metrics JSONB   # Метрики производительности
file_path VARCHAR(500)      # Путь к файлу модели
created_at TIMESTAMP
is_active BOOLEAN
```

## 📊 Метрики и результаты

### Метрики для регрессии:
- **MAE** (Mean Absolute Error) - средняя абсолютная ошибка в %
- **RMSE** (Root Mean Square Error) - корень среднеквадратичной ошибки
- **R²** - коэффициент детерминации
- **Direction Accuracy** - точность предсказания направления движения

### Метрики для классификации:
- **Accuracy** - общая точность
- **Precision** - точность положительных предсказаний
- **Recall** - полнота (процент найденных положительных)
- **F1 Score** - гармоническое среднее Precision и Recall
- **AUC** - площадь под ROC кривой

### Ожидаемые результаты:
**Регрессия**:
- MAE: 1.5-2.5%
- Direction Accuracy: 55-65%

**Классификация**:
- Accuracy: 65-75%
- F1 Score: 0.6-0.7

## 🔄 Процесс обновления

### При изменении структуры:
1. Обновить этот файл PROJECT_STRUCTURE.md
2. Обновить config.yaml если добавлены новые параметры
3. Обновить requirements.txt при добавлении библиотек
4. Запустить init_database.py если изменена структура БД

### Версионирование моделей:
- Каждая новая модель сохраняется с timestamp
- Метаданные в БД позволяют отслеживать версии
- Старые модели архивируются в trained_model/archive/

## 🚀 Последовательность запуска

### Полный пайплайн:
```bash
python run_futures_pipeline.py
```

### По шагам:
1. `python init_database.py` - создание БД
2. `python validate_futures_symbols.py` - проверка символов
3. `python download_data.py` - загрузка данных
4. `python prepare_dataset.py` - подготовка датасета с расчетом expected returns
5. **`python train_universal_transformer.py --task regression`** - ⭐ ОСНОВНАЯ КОМАНДА ДЛЯ ОБУЧЕНИЯ

### Запуск обучения с визуализацией:
```bash
# Для регрессии (предсказание ожидаемой доходности)
python train_universal_transformer.py --task regression

# Для классификации (предсказание profit/loss)
python train_universal_transformer.py --task classification
```

### Мониторинг в реальном времени:
```bash
# В отдельном терминале для просмотра прогресса
python monitor_training.py

# Или через TensorBoard (более детально)
tensorboard --logdir logs/training_YYYYMMDD_HHMMSS/tensorboard/
```

### Где смотреть результаты:
- **Графики**: `logs/training_YYYYMMDD_HHMMSS/plots/`
- **Метрики**: `logs/training_YYYYMMDD_HHMMSS/*_metrics.csv`
- **Отчет**: `logs/training_YYYYMMDD_HHMMSS/final_report.txt`
- **Модели**: `trained_model/`

## 📝 Примечания

- Все скрипты используют config.yaml для настроек
- Логирование ведется в папку logs/ с timestamp
- Ошибки обрабатываются с подробными сообщениями
- Прогресс отображается через tqdm progress bars
- Визуализация сохраняется в plots/
- Поддержка GPU автоматическая
- Работаем только с FUTURES данными