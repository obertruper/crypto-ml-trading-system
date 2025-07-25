# 📁 Скрипты для GPU обучения

## 🚀 Быстрый старт

### 1. Проверка готовности системы
```bash
./scripts/check_gpu_setup.sh
```
Проверяет:
- ✅ SSH ключ
- ✅ Подключение к серверу
- ✅ Наличие кэша
- ✅ Проект на сервере
- ✅ Доступность GPU

### 2. Быстрый запуск обучения
```bash
./scripts/quick_gpu_train.sh
```
Автоматически:
- Проверяет систему
- Синхронизирует проект
- Копирует кэш
- Запускает обучение
- Открывает TensorBoard

## 📋 Описание скриптов

### run_on_vast.sh
Основной скрипт запуска обучения на сервере.
- Поддерживает передачу параметров через переменные окружения
- Создает tmux сессию для обучения
- Настраивает использование кэша

### connect_vast.sh
Подключение к серверу с пробросом портов.
- Автоматический выбор режима через `VAST_CONNECTION_MODE`
- Проброс портов для TensorBoard (6006)
- Проверка SSH ключа

### sync_to_vast.sh
Синхронизация проекта с сервером.
- Исключает большие файлы и кэши
- Опционально устанавливает зависимости

## 🔧 Переменные окружения

- `USE_CACHE_ONLY=1` - использовать кэш вместо БД
- `GPU_TRAINING_MODE` - режим обучения (1=demo, 2=full, 3=custom)
- `GPU_TRAINING_EPOCHS` - количество эпох
- `VAST_CONNECTION_MODE` - автовыбор подключения (1=прямое, 2=прокси)

## 🐛 Устранение проблем

### SSH ключ не найден
```bash
cp ~/.ssh/id_rsa ~/.ssh/vast_ai_key
chmod 600 ~/.ssh/vast_ai_key
```

### Проект не синхронизирован
```bash
./scripts/sync_to_vast.sh
```

### Кэш не найден на сервере
```bash
scp -P 41575 -i ~/.ssh/vast_ai_key cache/features_cache.pkl root@184.98.25.179:/root/crypto_ai_trading/cache/
```

### TensorBoard не открывается
```bash
# Проверить туннель
lsof -i :6006

# Переустановить туннель
pkill -f "ssh.*6006"
ssh -f -N -L 6006:localhost:6006 -p 41575 -i ~/.ssh/vast_ai_key root@184.98.25.179
```