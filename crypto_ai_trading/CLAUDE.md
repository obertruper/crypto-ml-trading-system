# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 Проект: Crypto AI Trading System

Система алгоритмической торговли криптовалютами на основе PatchTST архитектуры с многозадачным обучением для прогнозирования 20 целевых переменных.

## 🚀 Основные команды

```bash
# Полный цикл: данные + обучение
python main.py --mode full

# Только обучение (если данные готовы)
python main.py --mode train

# Только подготовка данных
python main.py --mode data

# Демо режим (10 символов)
python main.py --mode demo

# Бэктестинг
python main.py --mode backtest

# Интерактивное меню
python main.py --mode interactive

# Мониторинг обучения
tensorboard --logdir logs/
# или
python monitor_training.py

# Проверка GPU
nvidia-smi
```

## 🏗️ Архитектура системы

### Ключевые компоненты:
- **UnifiedPatchTST** (`models/patchtst_unified.py`) - основная модель с 20 выходами
- **OptimizedTrainer** (`training/optimized_trainer.py`) - оптимизированное обучение для RTX 5090
- **PrecomputedDataset** (`data/precomputed_dataset.py`) - кэширование окон данных с custom_collate_fn
- **FeatureEngineering** (`data/feature_engineering.py`) - 171 технический индикатор

### Критические файлы (требуют особого внимания):
1. `config/config.yaml` - глобальная конфигурация, влияет на всю систему
2. `models/patchtst_unified.py` - архитектура модели с 20 целевыми переменными
3. `data/feature_engineering.py` - расчет всех индикаторов и целей
4. `training/optimized_trainer.py` - процесс обучения с GPU оптимизациями

### База данных:
- **PostgreSQL**: порт 5555, база crypto_trading, пользователь ruslan
- **Таблицы**: raw_market_data, processed_market_data, model_metadata
- **Загрузка**: оптимизирована через предвычисленные окна

## ⚙️ Технические особенности

### GPU оптимизации (RTX 5090):
- Batch size: 8192 с gradient accumulation 2 (эффективно 16384)
- Mixed Precision (AMP): float16 для ускорения
- TF32: включено для матричных операций
- Pin memory: исправлено через custom_collate_fn
- Скорость: 60,000+ samples/s на батчах
- torch.compile: отключено (не поддерживается sm_120)

### Модель и обучение:
- **20 целевых переменных**: возвраты, направления, уровни TP/SL, риск-метрики
- **171 индикатор**: трендовые, осцилляторы, волатильность, объем, микроструктура
- **Context window**: 96 (24 часа истории на 15-минутных свечах)
- **Целевые метрики**: Win Rate 50-52%, F1 Score > 0.42, Precision > 0.30

### 🎯 Оптимальная конфигурация для калибровки (2025-09-06):
```yaml
# Финальная калибровка для точной торговли
loss:
  class_weights: [1.0, 1.0, 1.0]  # Нейтральные веса для естественного распределения
  use_weighted_sampling: true      # Рандомизация батчей
  label_smoothing: 0.1             # Минимальное размытие классов

model:
  learning_rate: 0.00002          # 2e-5 для тонкой настройки
  temperature: 1.2                # Калибровка уверенности
  dropout: 0.25                   # Регуляризация
  gradient_accumulation_steps: 2  # Стабильность при малом LR
  mixup_alpha: 0.0               # Отключен для точности
  
  # Инициализация direction head
  direction_head_init:
    bias_init: balanced           # С учетом дисбаланса классов
    weight_scale: 1.0            # Нормальный масштаб весов

scheduler:
  patience: 6                    # Больше терпения для малого LR
  
trading:
  confidence_thresholds:         # Строгая фильтрация сигналов
    15m: 0.60
    1h: 0.60  
    4h: 0.65
    12h: 0.65
```

### Важные ограничения индикаторов:
- RSI, Stochastic, ADX: диапазон [0, 100] - НЕ нормализовать!
- Объемные индикаторы: логарифмическая нормализация
- Микроструктура: z-score нормализация

## 🛠️ Разработка и отладка

### При проблемах с обучением:
```bash
# Проверка данных
python check_data_quality.py

# Анализ предсказаний
python analyze_model_predictions.py

# Диагностика переобучения
python analyze_overfitting.py

# Проверка чекпоинта
python check_checkpoint.py
```

### Известные проблемы и решения:
- **CUDA pin_memory error**: исправлено через custom_collate_fn в PrecomputedDataset
- **RTX 5090 torch.compile**: отключено, не поддерживается sm_120
- **Большие батчи OOM**: используется gradient accumulation
- **Низкая энтропия предсказаний**: увеличен dropout, добавлен label smoothing
- **Дисбаланс классов**: взвешенная loss функция

## 📊 Структура данных

### Целевые переменные (20 штук):
1. **Возвраты**: future_return_15m, 1h, 4h, 12h
2. **Направления**: direction_15m, 1h, 4h, 12h (LONG/SHORT/FLAT)  
3. **LONG уровни**: will_reach_1%, 2%, 3%, 5% за 4h/12h
4. **SHORT уровни**: аналогично для коротких позиций
5. **Риск**: max_drawdown, max_rally за 1h/4h
6. **Сигналы**: best_action, signal_strength, risk_reward_ratio

### Входные признаки (171):
- Трендовые: EMA, SMA, ADX, MACD, Ichimoku, SAR, Aroon
- Осцилляторы: RSI, Stochastic, CCI, Williams %R, MFI
- Волатильность: ATR, Bollinger Bands, Donchian, Keltner
- Объемные: OBV, CMF, VWAP, Volume Profile
- Паттерны: свечные паттерны, уровни поддержки/сопротивления
- Микроструктура: bid/ask спред, дисбаланс объемов, токсичность потока

## 🔧 Правила работы

### При изменении кода:
1. НЕ создавать новые файлы без явной просьбы
2. Редактировать только существующие файлы
3. НЕ создавать файлы с префиксами fix_, fixed_, new_, temp_
4. Все комментарии и логи на русском языке
5. Использовать данные ФЬЮЧЕРСНОГО рынка (не спот)
6. Исключать TESTUSDT и тестовые символы

### Приоритеты оптимизации:
1. Максимальная утилизация GPU (целевая 95%+)
2. Стабильность обучения (избегать NaN/Inf)
3. Качество предсказаний (F1 > 0.40)
4. Скорость сходимости (early stopping)

## 📈 Текущий статус и метрики

### Оптимальные показатели:
- Train Loss: 1.41-1.42 (стабильный)
- Val Loss: 1.34-1.35 (ниже train = хорошее обобщение)
- Win Rate: 46-47% (оптимально для крипто)
- F1 Score: > 0.40 (основная метрика)
- GPU утилизация: 90-98%
- Скорость: 28-30k samples/s на эпоху

### Активные задачи:
- Обучение на полном датасете (50+ символов)
- Борьба с низкой энтропией предсказаний
- Балансировка классов LONG/SHORT/FLAT
- Оптимизация hyperparameters через Optuna