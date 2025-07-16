#!/bin/bash
# Скрипт для настройки удаленного сервера и запуска обучения модели

set -e  # Остановка при ошибках

echo "🚀 Начинаем настройку сервера для обучения модели..."

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Функция для вывода статуса
log_status() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ⚠️  $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ❌ $1"
}

# 1. Проверка системы
log_status "Проверка системы..."
echo "========================="
echo "Система: $(uname -a)"
echo "Python: $(python3 --version)"
echo "Память:"
free -h
echo "CPU:"
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core"
echo "========================="

# Проверка GPU
if command -v nvidia-smi &> /dev/null; then
    log_status "Обнаружен GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    log_warning "GPU не обнаружен, будет использоваться CPU"
fi

# 2. Создание директории проекта
PROJECT_DIR="/workspace/crypto_trading"
log_status "Создание директории проекта: $PROJECT_DIR"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# 3. Создание виртуального окружения
log_status "Создание виртуального окружения..."
python3 -m venv venv
source venv/bin/activate

# Обновление pip
pip install --upgrade pip

# 4. Создание requirements.txt
log_status "Создание requirements.txt..."
cat > requirements.txt << 'EOF'
# Основные зависимости
numpy==1.24.3
pandas==2.0.3
tensorflow==2.13.0
scikit-learn==1.3.0
scipy==1.11.1
psycopg2-binary==2.9.9
pyyaml==6.0.1
tqdm==4.66.1
matplotlib==3.7.2
seaborn==0.12.2
ta==0.10.2

# Дополнительные для анализа
jupyter==1.0.0
tensorboard==2.13.0
EOF

# 5. Установка зависимостей
log_status "Установка Python пакетов..."
pip install -r requirements.txt

# 6. Создание конфигурации для подключения к БД
log_status "Создание config.yaml..."
cat > config.yaml << 'EOF'
# Конфигурация для обучения на удаленном сервере

# База данных - ИЗМЕНИТЕ НА СВОИ ПАРАМЕТРЫ!
database:
  host: "localhost"  # Будет использоваться SSH туннель
  port: 5432        # Локальный порт после туннеля
  database: "crypto_trading"
  user: "ruslan"
  password: ""

# Параметры модели
model:
  sequence_length: 30
  batch_size: 64     # Увеличено для сервера
  epochs: 100
  learning_rate: 0.0001

# Риск-профиль (из вашей конфигурации)
risk_profile:
  stop_loss_pct_buy: 0.989    # -1.1%
  take_profit_pct_buy: 1.058  # +5.8%
  stop_loss_pct_sell: 1.011   # +1.1%
  take_profit_pct_sell: 0.942 # -5.8%

# Данные
data_download:
  market_type: "futures"
  interval: "15"
  days: 1095
  max_workers: 10
  symbols: []  # Будут загружены из БД
EOF

# 7. Создание скрипта для SSH туннеля
log_status "Создание скрипта для SSH туннеля..."
cat > setup_tunnel.sh << 'EOF'
#!/bin/bash
# Скрипт для создания SSH туннеля к локальному PostgreSQL

# ИЗМЕНИТЕ ЭТИ ПАРАМЕТРЫ!
LOCAL_HOST="your_local_ip"  # IP вашего локального компьютера
LOCAL_USER="your_user"       # Пользователь на локальном компьютере
LOCAL_PG_PORT=5555          # Порт PostgreSQL на локальной машине
TUNNEL_PORT=5432            # Порт на сервере

echo "🔗 Создание SSH туннеля к PostgreSQL..."
echo "Локальный PostgreSQL: $LOCAL_HOST:$LOCAL_PG_PORT"
echo "Туннель на сервере: localhost:$TUNNEL_PORT"

# Создаем обратный SSH туннель
ssh -f -N -L $TUNNEL_PORT:localhost:$LOCAL_PG_PORT $LOCAL_USER@$LOCAL_HOST

echo "✅ Туннель создан!"
echo "Проверка подключения..."
nc -zv localhost $TUNNEL_PORT
EOF
chmod +x setup_tunnel.sh

# 8. Скачивание основного скрипта обучения
log_status "Копирование train_universal_transformer.py..."
# Здесь нужно скопировать файл с локальной машины
# Временно создаем заглушку
cat > copy_training_script.sh << 'EOF'
#!/bin/bash
# Скопируйте train_universal_transformer.py с локальной машины:
# scp /path/to/train_universal_transformer.py root@server:/workspace/crypto_trading/
echo "⚠️  Скопируйте train_universal_transformer.py с локальной машины!"
echo "Команда: scp /Users/ruslan/PycharmProjects/LLM\\ TRANSFORM/train_universal_transformer.py root@$(hostname -I | awk '{print $1}'):/workspace/crypto_trading/"
EOF
chmod +x copy_training_script.sh

# 9. Создание скрипта мониторинга
log_status "Создание скрипта мониторинга..."
cat > monitor_training.py << 'EOF'
#!/usr/bin/env python3
"""Мониторинг процесса обучения"""

import time
import subprocess
import psutil
import os
from datetime import datetime

def get_gpu_stats():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
            return f"GPU: {gpu_util}% | VRAM: {mem_used}/{mem_total}MB"
    except:
        pass
    return "GPU: N/A"

def monitor():
    print("📊 Мониторинг обучения...")
    print("-" * 60)
    
    while True:
        # CPU и память
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU
        gpu_stats = get_gpu_stats()
        
        # Время
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Вывод статистики
        print(f"\r[{current_time}] CPU: {cpu_percent}% | RAM: {memory.percent}% | {gpu_stats}", end='', flush=True)
        
        time.sleep(5)

if __name__ == "__main__":
    monitor()
EOF
chmod +x monitor_training.py

# 10. Создание скрипта запуска обучения
log_status "Создание скрипта запуска обучения..."
cat > start_training.sh << 'EOF'
#!/bin/bash
# Скрипт для запуска обучения модели

source /workspace/crypto_trading/venv/bin/activate

echo "🚀 Запуск обучения модели..."
echo "Используйте screen или tmux для фоновой работы:"
echo "  screen -S training"
echo "  ./start_training.sh"
echo ""

# Проверка наличия основного файла
if [ ! -f "train_universal_transformer.py" ]; then
    echo "❌ Файл train_universal_transformer.py не найден!"
    echo "Скопируйте его командой:"
    echo "scp /path/to/train_universal_transformer.py root@server:/workspace/crypto_trading/"
    exit 1
fi

# Проверка подключения к БД
echo "🔍 Проверка подключения к PostgreSQL..."
python3 -c "import psycopg2, yaml; 
config = yaml.safe_load(open('config.yaml')); 
try: 
    conn = psycopg2.connect(**config['database']); 
    print('✅ Подключение к БД успешно!'); 
    conn.close()
except Exception as e: 
    print(f'❌ Ошибка подключения к БД: {e}')"

# Запуск обучения
echo ""
echo "Запуск обучения..."
echo "================================="
python train_universal_transformer.py --task regression
EOF
chmod +x start_training.sh

# 11. Создание README
log_status "Создание README..."
cat > README_SERVER.md << 'EOF'
# 🚀 Обучение модели на удаленном сервере

## Быстрый старт

1. **Настройте SSH туннель к вашему PostgreSQL:**
   ```bash
   # Отредактируйте setup_tunnel.sh и запустите:
   ./setup_tunnel.sh
   ```

2. **Скопируйте основной файл обучения:**
   ```bash
   scp /path/to/train_universal_transformer.py root@server:/workspace/crypto_trading/
   ```

3. **Запустите обучение:**
   ```bash
   screen -S training
   ./start_training.sh
   ```

4. **Мониторинг (в отдельном терминале):**
   ```bash
   ./monitor_training.py
   ```

## Структура файлов

- `config.yaml` - конфигурация (отредактируйте параметры БД!)
- `requirements.txt` - Python зависимости
- `start_training.sh` - запуск обучения
- `monitor_training.py` - мониторинг ресурсов
- `setup_tunnel.sh` - настройка SSH туннеля

## Логи

Логи сохраняются в `logs/training_YYYYMMDD_HHMMSS/`

## Результаты

Обученные модели сохраняются в `trained_model/`
EOF

# Финальное сообщение
echo ""
echo "================================="
log_status "✅ Настройка завершена!"
echo "================================="
echo ""
echo "📋 Следующие шаги:"
echo ""
echo "1. Настройте SSH туннель к вашему PostgreSQL:"
echo "   - Отредактируйте: $PROJECT_DIR/setup_tunnel.sh"
echo "   - Запустите: ./setup_tunnel.sh"
echo ""
echo "2. Скопируйте train_universal_transformer.py:"
echo "   scp /Users/ruslan/PycharmProjects/LLM\ TRANSFORM/train_universal_transformer.py root@$(hostname -I | awk '{print $1}'):$PROJECT_DIR/"
echo ""
echo "3. Отредактируйте config.yaml при необходимости"
echo ""
echo "4. Запустите обучение:"
echo "   cd $PROJECT_DIR"
echo "   screen -S training"
echo "   ./start_training.sh"
echo ""
echo "🎯 Проект готов к запуску!"