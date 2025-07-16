#!/bin/bash
# Скрипт для развертывания Transformer v3 на GPU сервере

echo "🚀 Развертывание Temporal Fusion Transformer v3 на сервере..."

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Параметры сервера
SERVER_USER="root"
SERVER_HOST="79.116.73.220"
SERVER_PORT="27681"
REMOTE_DIR="/workspace/LLM_TRANSFORM"

echo -e "${YELLOW}📦 Создание архива с необходимыми файлами...${NC}"

# Создаем временную директорию
TEMP_DIR=$(mktemp -d)
echo "Временная директория: $TEMP_DIR"

# Копируем основные файлы трансформера
mkdir -p "$TEMP_DIR/transformer_v3"
cp train_transformer_v3_server.py "$TEMP_DIR/"

# Копируем необходимые модули из xgboost_v3
mkdir -p "$TEMP_DIR/xgboost_v3/data"
mkdir -p "$TEMP_DIR/xgboost_v3/utils"
mkdir -p "$TEMP_DIR/xgboost_v3/config"

# Создаем упрощенные версии необходимых модулей
cat > "$TEMP_DIR/xgboost_v3/__init__.py" << 'EOF'
# XGBoost v3 modules
EOF

cat > "$TEMP_DIR/xgboost_v3/data/__init__.py" << 'EOF'
# Data modules
EOF

cat > "$TEMP_DIR/xgboost_v3/utils/__init__.py" << 'EOF'
# Utils modules
EOF

# Создаем requirements для сервера
cat > "$TEMP_DIR/requirements_transformer.txt" << 'EOF'
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
psycopg2-binary>=2.9.0
joblib>=1.3.0
tqdm>=4.65.0
EOF

# Создаем скрипт запуска
cat > "$TEMP_DIR/run_transformer_server.sh" << 'EOF'
#!/bin/bash
# Скрипт запуска обучения на сервере

echo "🚀 Запуск Temporal Fusion Transformer v3"

# Активация виртуального окружения если есть
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Проверка GPU
python3 -c "import torch; print(f'GPU доступен: {torch.cuda.is_available()}'); print(f'Количество GPU: {torch.cuda.device_count()}')"

# Параметры по умолчанию
TASK=${1:-classification}
TARGET_TYPE=${2:-threshold_binary}
EPOCHS=${3:-100}
BATCH_SIZE=${4:-512}
LIMIT=${5:-500000}

echo "Параметры запуска:"
echo "  Задача: $TASK"
echo "  Тип таргета: $TARGET_TYPE"
echo "  Эпох: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Лимит данных: $LIMIT"

# Запуск обучения
python3 train_transformer_v3_server.py \
    --task $TASK \
    --target-type $TARGET_TYPE \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --limit $LIMIT \
    --hidden-dim 256 \
    --num-workers 4

echo "✅ Обучение завершено!"
EOF

# Создаем скрипт мониторинга
cat > "$TEMP_DIR/monitor_training.sh" << 'EOF'
#!/bin/bash
# Мониторинг процесса обучения

echo "📊 Мониторинг обучения Transformer v3"

# Находим последнюю директорию с логами
LOG_DIR=$(ls -td logs/transformer_v3_* 2>/dev/null | head -1)

if [ -z "$LOG_DIR" ]; then
    echo "❌ Директория с логами не найдена"
    exit 1
fi

echo "📁 Директория логов: $LOG_DIR"

# Функция для показа последних строк лога
show_log() {
    if [ -f "transformer_v3_server.log" ]; then
        echo -e "\n📋 Последние записи лога:"
        tail -n 20 transformer_v3_server.log
    fi
}

# Функция для показа метрик
show_metrics() {
    if [ -f "$LOG_DIR/training_summary.json" ]; then
        echo -e "\n📊 Текущие метрики:"
        python3 -c "
import json
with open('$LOG_DIR/training_summary.json', 'r') as f:
    data = json.load(f)
    print(f\"  Эпох обучено: {data.get('total_epochs', 'N/A')}\"
    print(f\"  Лучший Val Loss: {data.get('best_val_loss', 'N/A'):.4f}\"
    if 'final_val_metrics' in data:
        for k, v in data['final_val_metrics'].items():
            print(f\"  {k}: {v:.4f}\")"
    fi
}

# Функция для мониторинга GPU
monitor_gpu() {
    echo -e "\n🎮 Использование GPU:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read line; do
        echo "  $line"
    done
}

# Основной цикл мониторинга
while true; do
    clear
    echo "📊 МОНИТОРИНГ TRANSFORMER V3 - $(date)"
    echo "=" * 60
    
    show_log
    show_metrics
    monitor_gpu
    
    echo -e "\n🔄 Обновление через 10 секунд... (Ctrl+C для выхода)"
    sleep 10
done
EOF

# Делаем скрипты исполняемыми
chmod +x "$TEMP_DIR/run_transformer_server.sh"
chmod +x "$TEMP_DIR/monitor_training.sh"

# Создаем архив
cd "$TEMP_DIR"
tar -czf transformer_v3_deploy.tar.gz *
cd -

echo -e "${YELLOW}📤 Загрузка файлов на сервер...${NC}"

# Копируем архив на сервер
scp -P $SERVER_PORT "$TEMP_DIR/transformer_v3_deploy.tar.gz" $SERVER_USER@$SERVER_HOST:$REMOTE_DIR/

# Распаковываем и настраиваем на сервере
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST << EOF
cd $REMOTE_DIR
echo "📦 Распаковка файлов..."
tar -xzf transformer_v3_deploy.tar.gz
rm transformer_v3_deploy.tar.gz

echo "📝 Проверка зависимостей..."
pip install -r requirements_transformer.txt

echo "✅ Развертывание завершено!"
echo ""
echo "Для запуска обучения выполните:"
echo "  ./run_transformer_server.sh [task] [target_type] [epochs] [batch_size] [limit]"
echo ""
echo "Примеры:"
echo "  ./run_transformer_server.sh classification threshold_binary 100 512 500000"
echo "  ./run_transformer_server.sh regression simple_regression 50 256 1000000"
echo ""
echo "Для мониторинга:"
echo "  ./monitor_training.sh"
EOF

# Очистка
rm -rf "$TEMP_DIR"

echo -e "${GREEN}✅ Развертывание завершено успешно!${NC}"
echo ""
echo "Подключение к серверу:"
echo "  ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
echo ""
echo "Запуск обучения:"
echo "  cd $REMOTE_DIR"
echo "  ./run_transformer_server.sh"