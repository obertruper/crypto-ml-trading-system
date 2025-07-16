# Transformer v3.0 - ML Crypto Trading System

## 🏗️ Архитектура

Transformer v3.0 представляет собой адаптацию успешной архитектуры XGBoost v3.0 для **Temporal Fusion Transformer (TFT)** модели. Проект использует временные последовательности для предсказания ожидаемой доходности в криптотрейдинге.

### 🔑 Ключевые особенности

- **Temporal Fusion Transformer**: Современная архитектура для временных рядов с attention механизмом
- **Модульная структура**: Переиспользование 70% кода из XGBoost v3.0
- **Иерархический отбор признаков**: 120 лучших признаков из 89 групп
- **Временные последовательности**: Анализ 100 свечей (25 часов) для предсказания
- **Dual-mode**: Поддержка регрессии и бинарной классификации
- **GPU оптимизация**: Mixed precision, memory growth

## 📁 Структура проекта

```
transformer_v3/
├── config/                  # Конфигурация
│   ├── settings.py         # Dataclass конфигурация для TFT
│   ├── features_config.py  # Группы признаков (89 типов)
│   └── constants.py        # Константы проекта
├── data/                   # Работа с данными
│   ├── loader.py          # Загрузка из PostgreSQL
│   ├── preprocessor.py    # Предобработка для последовательностей
│   ├── feature_engineer.py # Создание 120+ признаков
│   ├── sequence_creator.py # Создание временных последовательностей
│   ├── cacher.py         # Кэширование данных
│   └── btc_data_loader.py # BTC корреляционные данные
├── models/                 # Модели TFT
│   ├── tft_model.py       # Temporal Fusion Transformer архитектура
│   ├── tft_trainer.py     # Trainer с визуализацией
│   ├── ensemble.py        # Ансамблевые методы
│   └── optimizer.py       # Optuna оптимизация
├── utils/                  # Вспомогательные модули
│   ├── feature_selector.py # Иерархический отбор (60/20/10/10)
│   ├── metrics.py         # Расчет метрик
│   ├── visualization.py   # Графики и визуализация
│   ├── report_generator.py # Генерация отчетов
│   └── logging_manager.py # Логирование
└── main.py                # Основная точка входа
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
cd transformer_v3
pip install -r requirements.txt
```

### 2. Запуск обучения

```bash
# Регрессия (предсказание ожидаемой доходности в %)
python main.py --task regression

# Классификация (предсказание profit/loss)
python main.py --task classification_binary

# Тестовый режим (2 символа, быстрое обучение)
python main.py --task regression --test-mode

# С кастомными параметрами
python main.py --task regression --sequence-length 150 --batch-size 64 --epochs 50
```

### 3. Мониторинг обучения

```bash
# TensorBoard (если включен)
tensorboard --logdir logs/transformer_v3_*/tensorboard/

# Графики в реальном времени сохраняются в:
# logs/transformer_v3_*/plots/
```

## ⚙️ Конфигурация

### Основные параметры (config_default.yaml)

```yaml
model:
  hidden_size: 160          # Размер скрытого слоя
  sequence_length: 100      # Длина последовательности (25 часов)
  num_heads: 4             # Attention heads
  batch_size: 32           # Размер батча
  learning_rate: 0.001     # Скорость обучения
  epochs: 100              # Количество эпох

training:
  task_type: "regression"   # regression | classification_binary
  top_k_features: 120      # Количество отобранных признаков
  use_data_augmentation: true  # Аугментация временных рядов
```

## 🧠 Архитектура TFT

### Компоненты модели

1. **Feature Selection Network**: Отбор релевантных признаков
2. **LSTM Encoder-Decoder**: Обработка временных зависимостей  
3. **Self-Attention**: Механизм внимания для важных моментов времени
4. **Gated Residual Networks (GRN)**: Нелинейная обработка с gate механизмом
5. **Output Layer**: Финальные предсказания

### Временная структура

- **Входные данные**: [batch_size, 100, features] 
- **Последовательность**: 100 свечей по 15 минут = 25 часов истории
- **Предсказание**: Следующая ожидаемая доходность
- **Аугментация**: Шум + временные сдвиги для увеличения данных

## 📊 Признаки (Feature Engineering)

### Иерархическое распределение (120 признаков)

- **80% Технические индикаторы**: RSI, MACD, BB, ADX, ATR, Stochastic, Williams %R, MFI, CCI, CMF, OBV, Ichimoku, SAR, Aroon
- **10% BTC корреляции**: Корреляции с BTC на разных таймфреймах
- **5% Временные**: Циклические признаки (час, день недели)
- **5% Прочие**: Символы, паттерны свечей, дивергенции

### Группы признаков

1. **Technical Indicators** (67 признаков)
2. **Market Features** (13 признаков) 
3. **OHLC Features** (16 признаков)
4. **Symbol Features** (14 признаков)
5. **Binary Features** (7 признаков)
6. **Engineered Features** (5 признаков)
7. **Divergence Features** (3 признака)
8. **Candle Patterns** (2 признака)
9. **Volume Profile** (1 признак)

## 📈 Метрики и результаты

### Регрессия
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **R²**: Coefficient of determination
- **Directional Accuracy**: Процент правильных направлений

### Классификация
- **Accuracy**: Общая точность
- **Precision/Recall**: Точность и полнота
- **F1-Score**: Гармоническое среднее
- **ROC-AUC**: Area Under Curve

## 🔧 Продвинутые возможности

### 1. Аугментация данных
```python
# Автоматическая аугментация временных рядов
use_data_augmentation: true
augmentation_noise_level: 0.01    # 1% шум
augmentation_shift_range: 2       # ±2 timestep сдвиги
```

### 2. Mixed Precision (GPU)
```python
# Ускорение обучения на GPU
use_mixed_precision: true
memory_growth: true
```

### 3. Optuna оптимизация
```python
# Автоматический подбор гиперпараметров
use_optuna: true
optuna_trials: 50
```

### 4. Визуализация в реальном времени
- Графики loss/accuracy обновляются каждые 5 эпох
- TensorBoard интеграция
- Attention weights для интерпретации

## 🎯 Сравнение с XGBoost v3.0

| Компонент | XGBoost v3.0 | Transformer v3.0 |
|-----------|--------------|------------------|
| **Модель** | Gradient Boosting | Temporal Fusion Transformer |
| **Входные данные** | Табличные | Временные последовательности |
| **Память** | ~2GB | ~8GB (GPU) |
| **Время обучения** | 20-30 мин | 2-4 часа |
| **Интерпретируемость** | Feature importance | Attention weights |
| **Качество** | Высокое | Потенциально выше |

## 📝 Логирование

### Структура логов
```
logs/transformer_v3_YYYYMMDD_HHMMSS/
├── training.log              # Полный лог обучения
├── config.yaml               # Сохраненная конфигурация
├── buy_models/               # Модели для buy направления
│   ├── buy_predictor/        # SavedModel формат
│   ├── buy_predictor_weights.h5  # Веса модели
│   └── buy_predictor_metadata.json  # Метаданные
├── sell_models/              # Модели для sell направления
├── plots/                    # Графики и визуализация
│   ├── *_training_history.png    # История обучения
│   ├── *_progress_epoch_*.png     # Прогресс по эпохам
│   └── sequence_info.json         # Информация о последовательностях
├── tensorboard/              # TensorBoard логи
└── final_report.txt          # Итоговый отчет
```

## 🐛 Отладка и устранение неисправностей

### Частые проблемы

1. **Out of Memory (GPU)**
```python
# Уменьшите batch_size
batch_size: 16  # вместо 32

# Или уменьшите sequence_length
sequence_length: 50  # вместо 100
```

2. **Медленное обучение**
```python
# Включите mixed precision
use_mixed_precision: true

# Увеличьте batch_size (если позволяет память)
batch_size: 64
```

3. **Переобучение**
```python
# Увеличьте dropout
dropout_rate: 0.2  # вместо 0.1

# Добавьте регуляризацию
l2_regularization: 0.02
```

## 🔄 Интеграция с XGBoost v3.0

Transformer v3.0 полностью совместим с pipeline XGBoost v3.0:

1. **Общая база данных**: PostgreSQL на порту 5555
2. **Одинаковые признаки**: Feature engineering переиспользован
3. **Сравнимые результаты**: Единая система метрик  
4. **Ансамблирование**: Можно комбинировать XGBoost + TFT предсказания

## 📚 Дополнительные ресурсы

- [Temporal Fusion Transformer Paper](https://arxiv.org/abs/1912.09363)
- [TensorFlow Time Series Guide](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [XGBoost v3.0 Documentation](../xgboost_v3/README.md)

## 🤝 Вклад в проект

1. Fork репозиторий
2. Создайте feature branch: `git checkout -b feature/amazing-feature`
3. Commit изменения: `git commit -m 'Add amazing feature'`
4. Push в branch: `git push origin feature/amazing-feature`
5. Создайте Pull Request

## 📄 Лицензия

Distributed under the MIT License. See `LICENSE` for more information.