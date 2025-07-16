# 🚀 Инструкция по запуску обучения на Vast.ai

## ✅ Что уже настроено:

1. **SSH подключение** к серверу Vast.ai работает
2. **Проект синхронизирован** на сервер в `/workspace/crypto_trading`
3. **Python окружение** с TensorFlow 2.19.0 и поддержкой GPU (RTX 4090)
4. **Конфигурация** для работы с локальной БД через SSH туннель

## 📋 Пошаговая инструкция:

### Шаг 1: Запустите локальную PostgreSQL
```bash
# Проверьте, что БД запущена на порту 5555
pg_isready -p 5555 -h localhost

# Если не запущена, запустите:
pg_ctl start -D /usr/local/var/postgres
```

### Шаг 2: Создайте SSH туннель к БД
**В отдельном терминале:**
```bash
./setup_remote_db_tunnel.sh
```
Оставьте этот терминал открытым!

### Шаг 3: Запустите обучение
**В основном терминале:**
```bash
./run_training_on_vast.sh
```
Выберите:
- 1 - для regression (предсказание доходности)
- 2 - для classification (предсказание profit/loss)

### Альтернативный способ (ручной):

#### 1. Подключитесь к серверу:
```bash
ssh -p 27681 root@79.116.73.220
```

#### 2. В отдельном локальном терминале создайте туннель:
```bash
ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 -N
```

#### 3. На сервере запустите обучение:
```bash
cd /workspace/crypto_trading
source /workspace/venv/bin/activate

# Для regression
python train_universal_transformer.py --task regression --config remote_config.yaml

# Для classification
python train_universal_transformer.py --task classification --config remote_config.yaml
```

## 📊 Мониторинг:

### Вариант 1: TensorBoard (рекомендуется)
**Локально:**
```bash
# Создайте SSH туннель для TensorBoard
ssh -p 27681 root@79.116.73.220 -L 6006:localhost:16006 -N
```
Откройте в браузере: http://localhost:6006

### Вариант 2: Monitor script
```bash
ssh -p 27681 root@79.116.73.220
cd /workspace/crypto_trading
source /workspace/venv/bin/activate
python monitor_training.py
```

### Вариант 3: Jupyter Notebook
Доступен по адресу из логов Vast.ai (порт 8080)

## 🔧 Полезные команды:

### Проверка GPU:
```bash
ssh -p 27681 root@79.116.73.220 nvidia-smi
```

### Просмотр логов:
```bash
ssh -p 27681 root@79.116.73.220 "cd /workspace/crypto_trading && tail -f logs/training_*/training.log"
```

### Синхронизация изменений:
```bash
./sync_to_vast.sh
```

### Скачивание результатов:
```bash
# Модели
rsync -avz -e "ssh -p 27681" root@79.116.73.220:/workspace/crypto_trading/trained_model/ ./trained_model/

# Графики
rsync -avz -e "ssh -p 27681" root@79.116.73.220:/workspace/crypto_trading/logs/training_*/plots/ ./plots/
```

## ⚠️ Важно:

1. **SSH туннель должен быть активен** все время обучения
2. **Локальная БД должна быть доступна** на порту 5555
3. **Не закрывайте терминал с туннелем** до завершения обучения

## 🎯 Быстрый старт:

```bash
# Терминал 1
./setup_remote_db_tunnel.sh

# Терминал 2
./run_training_on_vast.sh
# Выберите 1 или 2

# Терминал 3 (опционально для мониторинга)
ssh -p 27681 root@79.116.73.220 -L 6006:localhost:16006 -N
# Откройте http://localhost:6006
```

Готово! 🚀