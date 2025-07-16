# 🧠 Правила для Claude AI - ML Crypto Trading System с Thinking LSP

## 🔴 КРИТИЧЕСКИ ВАЖНО - THINKING LSP ИНТЕГРАЦИЯ

### ПЕРЕД ЛЮБОЙ ОПЕРАЦИЕЙ С ВАЖНЫМИ ФАЙЛАМИ:
1. **ВСЕГДА** запрашивать контекст через `mcp_lsp_bridge.get_file_context()`
2. **АНАЛИЗИРОВАТЬ** с помощью Sequential Thinking для важных файлов
3. **ПРОВЕРЯТЬ** историю изменений через `mcp_get_recent_changes()`
4. **УЧИТЫВАТЬ** рекомендации из 5-шагового анализа

### ПРОЦЕСС МЫШЛЕНИЯ (обязательно для важных файлов):
```python
# Перед работой с файлом:
from lsp_server.thinking_lsp_integration import analyze_with_thinking
analysis = await analyze_with_thinking(file_path)

# Или через MCP Bridge:
from lsp_server.mcp_lsp_bridge import get_bridge
bridge = get_bridge()
context = bridge.get_file_context(file_path)
```

### ВАЖНЫЕ ФАЙЛЫ (требуют обязательного анализа):
- `models/patchtst.py` - архитектура модели
- `train_universal_transformer.py` - главный файл обучения  
- `config/config.yaml` - конфигурация системы
- `data/feature_engineering.py` - инженерия признаков
- `trading/signals.py` - торговые стратегии

### СТАТУС THINKING LSP:
- ✅ Sequential Thinking серверы активны (PID 4701, 4693)
- ✅ LSP сервер работает (PID 86614)
- ⚠️ Некоторые методы Sequential Thinking не реализованы
- ✅ Контекстный анализ и история изменений работают

## 🌐 Язык общения
- **ВСЕГДА** отвечать на русском языке
- Комментарии в коде писать на русском
- Документацию писать на русском
- Логи и сообщения в коде на русском

## 📁 Работа с файлами
- **НЕ СОЗДАВАТЬ** новые файлы без явной просьбы
- **РЕДАКТИРОВАТЬ** только существующие файлы
- **НЕ СОЗДАВАТЬ** файлы с префиксами fix_, fixed_, new_, temp_
- При необходимости изменений - вносить их в текущий файл

## 🚫 Запреты
- НЕ создавать дубликаты файлов!!
- НЕ создавать временные файлы!!
- НЕ создавать файлы для исправлений - править оригинал!!
- НЕ удалять файлы без явного запроса!!!

## ✅ Рекомендации
- Перед созданием файла спрашивать разрешение
- Показывать, какие изменения будут внесены
- Использовать Edit tool для изменения существующих файлов
- Группировать связанные изменения в одном файле

## 🎯 Приоритеты проекта
- Использовать данные с ФЬЮЧЕРСНОГО рынка (не спот)
- Исключить TESTUSDT и другие тестовые символы
- Фокус на ML модели для криптотрейдинга
- PostgreSQL база данных на порту 5555
- Expected returns как основная метрика для обучения
- Частичные закрытия позиций (20%, 30%, 30% на разных уровнях)
- Защита прибыли (breakeven, profit locking)

## 📊 Структура проекта ML Crypto Trading

### Основные файлы:
- `init_database.py` - инициализация БД PostgreSQL
- `download_data.py` - загрузка данных с Bybit (поддержка фьючерсов)
- `prepare_dataset.py` - расчет 49 технических индикаторов + expected returns
- **`train_universal_transformer.py`** ⭐ - ГЛАВНЫЙ ФАЙЛ ДЛЯ ОБУЧЕНИЯ (Temporal Fusion Transformer)
- `train_advanced.py` - LSTM модель (устаревшая, не изменять)
- `train_transformer_model.py` - оригинальная TFT (устаревшая, не изменять)
- `monitor_training.py` - мониторинг обучения в реальном времени

### Вспомогательные файлы:
- `validate_futures_symbols.py` - проверка доступности символов на фьючерсах
- `monitor_training.py` - мониторинг процесса обучения
- `check_history_depth.py` - проверка глубины исторических данных

### Конфигурация:
- `config.yaml` - основная конфигурация
- `requirements.txt` - Python зависимости
- `CLAUDE.md` - этот файл с правилами

## 🔧 Технические детали

### База данных:
- PostgreSQL на порту 5555
- Пользователь: ruslan
- База: crypto_trading
- Таблицы: raw_market_data, processed_market_data, model_metadata

### Модели ML:
1. **Temporal Fusion Transformer (TFT)** - основная архитектура в `train_universal_transformer.py`
   - Поддерживает регрессию (expected returns) и классификацию
   - Визуализация в реальном времени
   - Все улучшения вносятся ТОЛЬКО в этот файл

### Технические индикаторы (49 штук):
- Трендовые: EMA, ADX, MACD, Ichimoku, SAR, Aroon
- Осцилляторы: RSI, Stochastic, CCI, Williams %R
- Волатильность: ATR, Bollinger Bands, Donchian
- Объемные: OBV, CMF, MFI
- Производные и временные признаки

### Риск-профиль:
- Stop Loss: ±1.1%
- Take Profit: ±5.8%
- Анализ 100 свечей вперед (25 часов)
- Частичные закрытия: 20% на +1.2%, 30% на +2.4%, 30% на +3.5%
- Защита прибыли: breakeven на +1.2%, profit locking на разных уровнях

## 📝 Логирование

### Структура логов:
```
logs/training_YYYYMMDD_HHMMSS/
├── training.log              # Полный лог
├── model_name_metrics.csv    # Метрики по эпохам
├── final_report.txt          # Итоговый отчет
├── plots/                    # Графики визуализации
│   ├── training_progress.png # Обновляется каждые 5 эпох
│   ├── epoch_XXX.png        # Снимки прогресса
│   └── *_evaluation.png     # Финальная оценка
└── tensorboard/             # Данные для TensorBoard
```

### Визуализация (ДА, ЕСТЬ В РЕАЛЬНОМ ВРЕМЕНИ!):
- **Автоматические графики обновляются каждые 5 эпох**
- Отображается: Loss, MAE/Accuracy, Learning Rate, статистика
- TensorBoard: `tensorboard --logdir logs/training_*/tensorboard/`
- Мониторинг через `monitor_training.py`

## 🚀 Рабочий процесс

### Полный пайплайн:
```bash
# Основная команда для обучения
python main.py --mode train
```

### По шагам:
1. `python main.py --mode data` - загрузка и подготовка данных
2. `python main.py --mode train` - обучение модели
3. `python main.py --mode backtest` - бэктестинг стратегий
5. **`python train_universal_transformer.py --task regression`** ⭐ - ОСНОВНАЯ КОМАНДА

### ⚠️ ВАЖНО - ВСЕ ДОРАБОТКИ ТОЛЬКО В `train_universal_transformer.py`:
```bash
# Для регрессии (предсказание ожидаемой доходности в %)
python train_universal_transformer.py --task regression

# Для классификации (предсказание profit/loss)
python train_universal_transformer.py --task classification
```

### Мониторинг и визуализация:
```bash
# В отдельном терминале для просмотра прогресса
python monitor_training.py

# Или через TensorBoard (более детально)
tensorboard --logdir logs/training_YYYYMMDD_HHMMSS/tensorboard/
```

### Где смотреть результаты:
- **Графики**: `logs/training_*/plots/training_progress.png`
- **Метрики**: `logs/training_*/*_metrics.csv`
- **Отчет**: `logs/training_*/final_report.txt`

## ⚠️ Важные моменты
- Всегда работаем с ФЬЮЧЕРСНЫМИ данными (market_type='futures')
- Исключаем TESTUSDT и подобные тестовые символы
- Используем 15-минутный таймфрейм
- Для регрессии: обучаем 2 модели (buy_return_predictor, sell_return_predictor)
- Для классификации: обучаем 4 модели (buy_profit, buy_loss, sell_profit, sell_loss)
- Полное логирование каждого шага обучения
- **ВСЕ ИЗМЕНЕНИЯ ТОЛЬКО В `train_universal_transformer.py`**

## 🖥️ Интеграции и инструменты

### 1. IDE интеграция через MCP (Model Context Protocol)
- Автоматическое понимание структуры проекта
- Контекстная навигация по коду
- Встроенная диагностика через `mcp__ide__getDiagnostics`

### 2. Vast.ai интеграция для GPU обучения
- **Сервер**: 184.98.25.179:41575 или ssh8.vast.ai:13641
- **Подключение**: `./connect_vast.sh` (создает туннели для Web UI, TensorBoard)
- **БД туннель**: `./setup_remote_db_tunnel.sh` (локальная БД → удаленный сервер)
- **Синхронизация**: `./sync_to_vast.sh` (rsync проекта)
- **Обучение**: `./run_training_on_vast.sh` (меню выбора задач)
- **Мониторинг**: TensorBoard на http://localhost:6006

### 3. Metabase для анализа данных
- **Запуск**: `./start_metabase.sh` или `docker-compose up -d` в папке metabase/
- **Доступ**: http://localhost:3333
- **БД**: автоматически подключается к PostgreSQL:5555
- **Дашборды**: статистика по символам, индикаторам, результатам обучения

### 4. Визуализация и мониторинг
- **Realtime графики**: обновляются каждые 5 эпох во время обучения
- **TensorBoard**: детальная статистика и метрики
- **Matplotlib**: графики Loss, Accuracy, предсказаний
- **CSV логи**: все метрики сохраняются для анализа

## 📈 Статус проекта
- ✅ База данных PostgreSQL инициализирована
- ✅ Загрузка фьючерсных данных с Bybit
- ✅ Расчет 49 технических индикаторов
- ✅ Expected returns с учетом частичных закрытий
- ✅ TFT архитектура в `train_universal_transformer.py`
- ✅ Визуализация процесса обучения
- ✅ Интеграция с Vast.ai для GPU
- ⚠️ Требуется обучение на полном датасете (51 символ)

## 💡 Текущие задачи
1. Загрузить и подготовить данные: `python main.py --mode data`
2. Обучить модель: `python main.py --mode train`
3. Проанализировать результаты в Metabase
4. Оптимизировать гиперпараметры на основе метрик

## 🚨 ОСНОВНОЙ ПРОЕКТ - crypto_ai_trading

### Локация: `/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/`
### GitHub: https://github.com/obertruper/crypto_ai_trading

### 📋 Описание проекта:
Полноценная система для алгоритмической торговли криптовалютными фьючерсами с использованием современной архитектуры **PatchTST** (Patch Time Series Transformer). Проект включает полный цикл от загрузки данных до бэктестинга торговых стратегий.

### 🏗️ Структура проекта:
```
crypto_ai_trading/
├── config/               # Конфигурация системы
├── data/                 # Загрузка и обработка данных
├── models/               # ML модели (PatchTST, ансамбли)
├── trading/              # Торговые стратегии и риск-менеджмент
├── training/             # Обучение и валидация моделей
├── utils/                # Утилиты и визуализация
├── notebooks/            # Jupyter notebooks для анализа
└── main.py              # Главная точка входа
```

### 🔑 Ключевые особенности:
1. **PatchTST архитектура** - передовая модель для временных рядов
2. **100+ технических индикаторов** - полный набор для анализа
3. **Многозадачное обучение** - одновременное предсказание цены и вероятностей TP/SL
4. **6 стратегий управления позициями** - Kelly, Volatility-based, Risk Parity и др.
5. **Продвинутый риск-менеджмент** - частичные закрытия, защита прибыли
6. **Ансамблирование моделей** - Voting, Stacking, Dynamic ensembles
7. **Полный бэктестинг** - с учетом комиссий и проскальзывания
8. **PostgreSQL интеграция** - надежное хранение данных

### 🚀 Запуск проекта:
```bash
cd /Users/ruslan/PycharmProjects/LLM\ TRANSFORM/crypto_ai_trading/

# Установка зависимостей
pip install -r requirements.txt

# Настройка БД
python setup.py

# Демо режим
python main.py --mode demo

# Полное обучение
python main.py --mode full

# Только подготовка данных
python main.py --mode data

# Только обучение (если данные готовы)
python main.py --mode train

# Бэктестинг
python main.py --mode backtest
```

### 📊 Работа с данными:
- **Источник**: Bybit фьючерсы
- **Таймфрейм**: 15 минут
- **Символы**: 50+ криптовалютных пар
- **Период**: с 2022 года
- **База данных**: PostgreSQL (порт 5555)

### 🧪 Jupyter Notebooks:
1. **01_data_exploration.ipynb** - исследование данных
2. **02_feature_analysis.ipynb** - анализ признаков
3. **03_model_evaluation.ipynb** - оценка моделей

### 📈 Мониторинг обучения:
```bash
# TensorBoard
tensorboard --logdir logs/

# Или встроенный мониторинг
python monitor_training.py
```

### 🔧 Конфигурация:
Все настройки в `config/config.yaml`:
- Параметры модели
- Риск-менеджмент
- Параметры обучения
- Настройки БД

### ⚡ Важные команды:
```bash
# Проверка данных в БД
python -c "from data.data_loader import CryptoDataLoader; loader = CryptoDataLoader(config); print(loader.get_data_stats())"

# Быстрая валидация модели
python -c "from training.validator import validate_checkpoint; validate_checkpoint('models_saved/best_model.pth', test_loader, config)"

# Генерация отчета
python utils/generate_report.py
```

### 🛠️ Разработка:
- **ВСЕ изменения в архитектуре** - только в `models/patchtst.py`
- **Новые признаки** - в `data/feature_engineering.py`
- **Стратегии** - в `trading/signals.py`
- **Метрики** - в `utils/metrics.py`

### 📝 Соглашения:
- Комментарии и логи на русском языке
- Использовать type hints
- Документировать все функции
- Следовать структуре проекта

### 🎯 Приоритеты разработки:
1. Улучшение качества предсказаний
2. Оптимизация риск-менеджмента
3. Добавление новых источников данных
4. Расширение набора признаков
5. A/B тестирование стратегий