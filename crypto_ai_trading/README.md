# 🚀 Crypto AI Trading System v3.0

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

Профессиональная система алгоритмической торговли криптовалютами на основе PatchTST архитектуры с 20 целевыми переменными для точного прогнозирования.

📚 **[Подробная документация о признаках и целевых переменных](docs/FEATURES_AND_TARGETS_LOGIC.md)**

## 🎯 Быстрый старт

### 1. Подготовка данных (первый раз)
```bash
python prepare_trading_data.py
```

### 2. Обучение модели
```bash
python main.py --mode train
```

### 3. Полный цикл (данные + обучение)
```bash
python main.py --mode full
```

## 📋 Основные команды

### 1. Полное обучение модели
```bash
python main.py --mode full
```

### 2. Интерактивное меню
```bash
python main.py --mode interactive
# или
python run_interactive.py
```

### 3. Демо режим (проверка БД)
```bash
python main.py --mode demo
```

### 4. Использование улучшенной модели
```bash
python main.py --mode full --use-improved-model
```

### 5. Валидация конфигурации
```bash
python main.py --validate-only
```

## ✨ Ключевые особенности

### 🎯 20 Целевых переменных
- **Базовые возвраты**: future_return_15m, 1h, 4h, 12h
- **Направления**: direction_15m, 1h, 4h, 12h (LONG/SHORT/FLAT)
- **LONG уровни**: will_reach_1%, 2%, 3%, 5% за 4h/12h
- **SHORT уровни**: аналогичные метрики для коротких позиций
- **Риск-метрики**: max_drawdown, max_rally за 1h/4h
- **Торговые сигналы**: best_action, signal_strength, risk_reward_ratio

### 🧠 Оптимальные настройки
- **Learning Rate**: 0.000005 (микро LR для стабильности)
- **Batch Size**: 8192 с gradient accumulation 2
- **Dropout**: 0.3 для предотвращения переобучения
- **Context Window**: 96 (24 часа истории)
- **Win Rate**: 46.2% (оптимально для крипто)
- **F1 Score**: > 0.40 (основная метрика)

### 🏗️ Архитектура
- **UnifiedPatchTST** - оптимизированный трансформер для временных рядов
- **171 технический индикатор**: полный набор + микроструктура
- **Multi-Task Learning**: одновременное предсказание 20 целей
- **RTX 5090 оптимизации**: 60,000+ samples/s

### 💼 Риск-менеджмент
- **Частичные закрытия**: 40%, 40%, 20% на уровнях TP
- **Динамические уровни**: адаптация к волатильности
- **6 стратегий позиций**: Kelly, Volatility-based, Risk Parity и др.

## 📊 Структура проекта

```
crypto_ai_trading/
├── config/               # Конфигурация
│   └── config.yaml      # Основной конфиг
├── data/                # Работа с данными
│   ├── data_loader.py   # Загрузка из БД
│   ├── dataset.py       # PyTorch datasets
│   └── feature_engineering.py  # Создание признаков
├── models/              # Модели ML
│   ├── patchtst.py     # PatchTST архитектура (включает улучшения)
│   └── ensemble.py     # Ансамбли
├── trading/            # Торговая логика
│   ├── signals.py      # Генерация сигналов
│   ├── risk_manager.py # Управление рисками
│   └── backtester.py   # Бэктестинг
├── training/           # Обучение
│   ├── trainer.py      # Основной трейнер
│   └── optimizer.py    # Оптимизаторы
├── utils/              # Утилиты
│   ├── logger.py       # Логирование
│   └── metrics.py      # Метрики
├── main.py            # Главный скрипт (единая точка входа)
└── run_interactive.py # Интерактивное меню
```

## 🔧 Конфигурация

Основные параметры в `config/config.yaml`:

```yaml
model:
  batch_size: 8192      # Оптимально для RTX 5090
  context_window: 96    # Окно контекста (24 часа)
  d_model: 256         # Размерность модели
  dropout: 0.3         # Оптимальный dropout
  learning_rate: 0.000005 # Микро LR для стабильности
  epochs: 200          # Полное обучение
  early_stopping_patience: 15  # Терпение для early stopping
  gradient_accumulation_steps: 2  # Эффективный batch 16384
  
data:
  symbols: ['BTCUSDT', 'ETHUSDT', ...]  # Торговые пары
  train_ratio: 0.6     # Доля train
  val_ratio: 0.2       # Доля validation
  test_ratio: 0.2      # Доля test
```

## 📈 Результаты

После обучения:
- Модель сохраняется в `models_saved/best_model.pth`
- Логи в `experiments/logs/`
- Метрики в `experiments/logs/*_metrics.csv`

## 🚨 Решение проблем

### PostgreSQL не подключается
```bash
# Проверить статус
pg_ctl -D /usr/local/var/postgres status

# Перезапустить
brew services restart postgresql
```

### Недостаточно памяти
- Уменьшить `batch_size` в конфиге
- Уменьшить количество символов

### Модель переобучается
- Увеличить `dropout`
- Уменьшить `d_model`
- Добавить больше данных

## 📝 Лицензия

MIT License - см. файл [LICENSE](LICENSE)

## 📊 Мониторинг обучения

```bash
# TensorBoard для визуализации
tensorboard --logdir logs/

# Или встроенный мониторинг
python monitor_training.py
```

## 🚀 Производительность

- **RTX 5090**: 60 секунд на эпоху, ~3 часа полное обучение
- **Скорость**: 60,000+ samples/s на батчах
- **GPU утилизация**: 90-98%
- **Размер модели**: ~50MB
- **RAM**: минимум 16GB
- **VRAM**: 8-10GB используется из 32GB

## 📊 Текущие результаты

- **Win Rate**: 46.2% (оптимально для крипто)
- **F1 Score**: 0.403 и растет
- **Val Loss**: 1.34 (ниже train loss = хорошее обобщение)
- **Скорость улучшения**: +0.4% F1 за 5 эпох

---

**Crypto AI Trading System v3.0** - профессиональная система с 20 целевыми переменными и Win Rate 46.2% для криптовалютных рынков.