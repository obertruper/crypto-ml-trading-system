# XGBoost v3.0 - ML Crypto Trading System

## 🚀 Новые возможности v3.0

XGBoost v3.0 - это полностью переработанная версия системы машинного обучения для криптотрейдинга с улучшенной архитектурой и оптимизацией.

### ✅ Главные улучшения:
- **Optuna оптимизация**: автоматический подбор гиперпараметров (100+ попыток)
- **Feature Selection**: выбор 60 лучших признаков из 112+ 
- **ADASYN балансировка**: адаптивная синтетическая выборка
- **Повышенный порог**: 1.5% вместо 0.5% для снижения шума
- **Улучшенные параметры**: max_depth=8, learning_rate=0.03, регуляризация
- **Централизованные константы**: все параметры в config/constants.py
- **Исправлены критические ошибки**: decimal типы, SMOTE индексы, выбор признаков

### 📊 Ожидаемые результаты:
- ROC-AUC: >0.75 (было 0.60)
- Precision: >40% (было 26%)
- Технические индикаторы в топ-10 важных признаков

## 📁 Структура проекта

```
xgboost_v3/
├── config/              # Конфигурация
│   ├── settings.py      # Основные настройки
│   └── features_config.py # Конфигурация признаков
├── data/               # Работа с данными
│   ├── loader.py       # Загрузка из БД
│   ├── preprocessor.py # Предобработка
│   └── feature_engineer.py # Создание признаков
├── models/             # Модели
│   ├── xgboost_trainer.py # Обучение XGBoost
│   ├── ensemble.py     # Ансамблевые методы
│   └── optimizer.py    # Optuna оптимизация
├── utils/              # Утилиты
│   ├── metrics.py      # Расчет метрик
│   ├── balancing.py    # Балансировка классов
│   ├── visualization.py # Визуализация
│   └── cache.py        # Кэширование
└── main.py             # Главный скрипт

```

## 🛠️ Установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Создание requirements.txt если его нет
pip install xgboost pandas numpy scikit-learn optuna matplotlib seaborn psycopg2-binary pyyaml tqdm joblib imblearn
```

## 🎯 Быстрый старт

### 1. Оптимизированный запуск (рекомендуется):
```bash
# Интерактивное меню с 7 режимами
python run_xgboost_v3.py

# Режимы 1 и 2 - оба с ПОЛНОЙ оптимизацией:
# 1 - Оптимизированный тест (подробный вывод)
# 2 - Быстрый тест (те же настройки, краткий вывод)

# Прямой запуск с оптимизацией
python run_xgboost_v3.py --test-mode --optimize --ensemble-size 5
```

### 2. Базовые команды:
```bash
# Быстрый тест с оптимизацией
python xgboost_v3/main.py --test-mode --optimize

# Полное обучение на всех символах
python xgboost_v3/main.py --optimize --ensemble-size 5

# Регрессия
python xgboost_v3/main.py --task regression --optimize
```

### 3. Просмотр результатов:
```bash
# Показать последние результаты
python show_xgboost_results.py

# Сравнить последние 5 запусков
python show_xgboost_results.py --compare
```

### Параметры командной строки:

- `--task`: Тип задачи (classification_binary, regression)
- `--test-mode`: Быстрый тест на 2 символах (BTCUSDT, ETHUSDT)
- `--optimize`: Запустить Optuna оптимизацию гиперпараметров
- `--ensemble-size N`: Размер ансамбля (по умолчанию 5)
- `--no-cache`: Не использовать кэш данных
- `--config`: Путь к YAML файлу конфигурации

### ⚡ Новые настройки v3.0:
- **Порог классификации**: 1.5% (увеличен с 0.5%)
- **Балансировка**: ADASYN (вместо SMOTE)
- **Feature selection**: топ-60 признаков
- **Оптимизация**: включена для test-mode автоматически

## ⚙️ Конфигурация

### Создание конфигурационного файла:

```yaml
# config.yaml
database:
  host: localhost
  port: 5555
  database: crypto_trading
  user: ruslan

model:
  objective: binary:logistic
  max_depth: 6
  learning_rate: 0.01
  n_estimators: 1000

training:
  task_type: classification_binary
  test_mode: false
  ensemble_size: 3
  balance_method: smote
  classification_threshold: 0.5
```

Использование:
```bash
python xgboost_v3/main.py --config config.yaml
```

## 📊 Признаки (Features)

Система использует 112+ признаков, разделенных на группы:

1. **Технические индикаторы (49)**
   - Трендовые: EMA, ADX, MACD, Ichimoku, SAR, Aroon
   - Осцилляторы: RSI, Stochastic, CCI, Williams %R
   - Объемные: OBV, CMF, MFI
   - Волатильность: ATR, Bollinger Bands, Keltner, Donchian

2. **Рыночные признаки (13)**
   - Корреляции с BTC
   - Относительная сила
   - Рыночные режимы
   - Циклические временные признаки

3. **OHLC признаки (13)**
   - Нормализованные цены
   - Размеры свечей
   - Расстояния до MA

4. **Инженерные признаки**
   - Взвешенные комбинации
   - Дивергенции
   - Паттерны свечей
   - Volume profile

## 📈 Результаты

После обучения создаются:
- 📁 `logs/xgboost_v3_YYYYMMDD_HHMMSS/` - основная директория
- 📄 `training.log` - полный лог обучения
- 📊 `plots/` - все графики и визуализации
- 💾 `models/` - сохраненные модели
- ⚙️ `config.yaml` - использованная конфигурация

### Визуализации включают:
- История обучения
- Feature importance
- ROC кривые
- Confusion matrix
- Распределение предсказаний
- Оптимизация Optuna

## 🔧 Продвинутое использование

### Программный запуск:

```python
from xgboost_v3 import Config, DataLoader, XGBoostTrainer

# Создание конфигурации
config = Config()
config.training.task_type = "regression"
config.training.test_mode = True

# Загрузка данных
loader = DataLoader(config)
loader.connect()
df = loader.load_data()
loader.disconnect()

# Обучение модели
trainer = XGBoostTrainer(config)
# ... продолжение
```

### Кастомная балансировка:

```python
from xgboost_v3.utils import BalanceStrategy

balancer = BalanceStrategy(config)
X_balanced, y_balanced = balancer.balance_data(X, y)
```

## 🐛 Отладка

### Проверка кэша:
```python
from xgboost_v3.utils import CacheManager

cache = CacheManager()
info = cache.get_cache_info()
print(f"Кэш: {info['total_items']} файлов, {info['total_size_mb']:.1f} MB")

# Очистка старого кэша
cache.cleanup_old_cache(days=7)
```

### Логирование:
Все операции логируются в `training.log`. Уровень логирования можно изменить в `main.py`.

## 📝 TODO

- [ ] Добавить поддержку MLflow для tracking экспериментов
- [ ] Реализовать online learning
- [ ] Добавить A/B тестирование моделей
- [ ] Интеграция с торговым ботом
- [ ] Веб-интерфейс для мониторинга

## 🤝 Вклад

Приветствуются улучшения! Создавайте issues и pull requests.

## 📄 Лицензия

MIT License