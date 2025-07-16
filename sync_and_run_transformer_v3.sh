#!/bin/bash

# Скрипт для синхронизации transformer_v3 с сервером и запуска

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Настройки сервера
# Прямое подключение
SERVER_HOST="84.68.60.115"
SERVER_PORT="42244"
# Альтернативное подключение через прокси
PROXY_HOST="ssh1.vast.ai"
PROXY_PORT="18645"
SERVER_USER="root"
REMOTE_DIR="/workspace/transformer_v3"
LOCAL_DIR="transformer_v3"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Синхронизация и запуск Transformer v3 на сервере    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo

# Проверка наличия папки transformer_v3
if [ ! -d "$LOCAL_DIR" ]; then
    echo -e "${RED}❌ Папка $LOCAL_DIR не найдена!${NC}"
    exit 1
fi

# Функция для синхронизации
sync_to_server() {
    echo -e "${YELLOW}📤 Синхронизация файлов с сервером...${NC}"
    
    # Проверяем какое подключение использовать
    if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Используем прямое подключение для rsync${NC}"
        SSH_CMD="ssh -p $SERVER_PORT"
        REMOTE_HOST=$SERVER_HOST
    else
        echo -e "${YELLOW}⚠️  Используем прокси подключение для rsync${NC}"
        SSH_CMD="ssh -p $PROXY_PORT"
        REMOTE_HOST=$PROXY_HOST
    fi
    
    # Создаем папку на сервере если её нет
    $SSH_CMD $SERVER_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR"
    
    # Синхронизируем transformer_v3
    rsync -avz --progress \
        -e "$SSH_CMD" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.DS_Store' \
        --exclude='logs/' \
        --exclude='cache/' \
        --exclude='*.h5' \
        --exclude='*.pkl' \
        --delete \
        $LOCAL_DIR/ $SERVER_USER@$REMOTE_HOST:$REMOTE_DIR/
    
    # Синхронизируем также config.yaml из основной папки
    if [ -f "config.yaml" ]; then
        rsync -avz --progress \
            -e "$SSH_CMD" \
            config.yaml $SERVER_USER@$REMOTE_HOST:/workspace/
    fi
    
    # Синхронизируем run_transformer_v3.py
    if [ -f "run_transformer_v3.py" ]; then
        rsync -avz --progress \
            -e "$SSH_CMD" \
            run_transformer_v3.py $SERVER_USER@$REMOTE_HOST:/workspace/
    fi
    
    # Синхронизируем run_cache_data.py
    if [ -f "run_cache_data.py" ]; then
        rsync -avz --progress \
            -e "$SSH_CMD" \
            run_cache_data.py $SERVER_USER@$REMOTE_HOST:/workspace/
    fi
    
    echo -e "${GREEN}✅ Синхронизация завершена${NC}"
}

# Функция для настройки туннеля к БД
setup_db_tunnel() {
    echo -e "${YELLOW}🔗 Настройка туннеля к локальной БД...${NC}"
    
    # Проверяем подключение к серверу и создаем туннель
    if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Прямое подключение к серверу${NC}"
        # Убиваем старые туннели
        ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "pkill -f 'ssh.*5555' || true"
        # Создаем обратный туннель: локальный 5555 -> удаленный 5555
        ssh -p $SERVER_PORT -fNR 5555:localhost:5555 $SERVER_USER@$SERVER_HOST
        SSH_HOST=$SERVER_HOST
        SSH_PORT=$SERVER_PORT
    else
        echo -e "${YELLOW}⚠️  Используем прокси подключение${NC}"
        # Убиваем старые туннели
        ssh -p $PROXY_PORT $SERVER_USER@$PROXY_HOST "pkill -f 'ssh.*5555' || true"
        # Создаем обратный туннель через прокси
        ssh -p $PROXY_PORT -fNR 5555:localhost:5555 $SERVER_USER@$PROXY_HOST
        SSH_HOST=$PROXY_HOST
        SSH_PORT=$PROXY_PORT
    fi
    
    # Ждем установки туннеля
    sleep 2
    echo -e "${GREEN}✅ Туннель создан (локальный 5555 -> удаленный 5555)${NC}"
}

# Функция для запуска на сервере
run_on_server() {
    echo -e "${YELLOW}🚀 Запуск на сервере...${NC}"
    echo
    
    # Проверяем доступность прямого подключения
    echo -e "${YELLOW}Проверка прямого подключения...${NC}"
    if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Используем прямое подключение${NC}"
        SSH_CMD="ssh -t -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
    else
        echo -e "${YELLOW}⚠️  Прямое подключение недоступно, используем прокси${NC}"
        SSH_CMD="ssh -t -p $PROXY_PORT $SERVER_USER@$PROXY_HOST"
    fi
    
    # SSH команда с интерактивным режимом
    $SSH_CMD "cd /workspace && source venv/bin/activate && export TERM=xterm-256color && python run_transformer_v3.py"
}

# Функция для остановки процессов
stop_training() {
    echo -e "${YELLOW}🛑 Остановка текущего обучения...${NC}"
    
    if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
        SSH_CMD="ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
    else
        SSH_CMD="ssh -p $PROXY_PORT $SERVER_USER@$PROXY_HOST"
    fi
    
    # Останавливаем все процессы Python связанные с transformer_v3
    $SSH_CMD "pkill -f 'transformer_v3/main.py' || true"
    $SSH_CMD "pkill -f 'run_transformer_v3.py' || true"
    
    echo -e "${GREEN}✅ Процессы остановлены${NC}"
}

# Функция для запуска в фоновом режиме с оптимизацией GPU
run_background_optimized() {
    echo -e "${YELLOW}🚀 Запуск обучения в фоновом режиме с оптимизацией GPU...${NC}"
    
    if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
        SSH_CMD="ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
    else
        SSH_CMD="ssh -p $PROXY_PORT $SERVER_USER@$PROXY_HOST"
    fi
    
    # Создаем скрипт запуска с оптимизацией на сервере
    $SSH_CMD "cat > /workspace/start_optimized_training.sh << 'EOF'
#!/bin/bash
cd /workspace
source venv/bin/activate

# Настройки окружения для оптимизации GPU
export TF_CPP_MIN_LOG_LEVEL=2
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
export TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1
export TF_ENABLE_WINOGRAD_NONFUSED=1
export TF_SYNC_ON_FINISH=0
export TF_AUTOTUNE_THRESHOLD=2
export TF_CUDNN_USE_AUTOTUNE=1
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=1
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=1
export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=1

# XLA оптимизации
export TF_XLA_FLAGS='--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

# CUDA настройки
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648
export CUDA_LAUNCH_BLOCKING=0

# NCCL настройки для multi-GPU
export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0

# Убиваем старые процессы
pkill -f 'transformer_v3/main.py' || true

# Запускаем обучение
echo 'Запуск обучения с оптимизацией GPU...'
nohup python transformer_v3/main.py --task regression --batch-size 512 > logs/training_optimized_\$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo 'Обучение запущено в фоне. PID:' \$!
echo \$! > /workspace/training.pid
EOF"
    
    # Делаем скрипт исполняемым
    $SSH_CMD "chmod +x /workspace/start_optimized_training.sh"
    
    # Запускаем скрипт
    $SSH_CMD "/workspace/start_optimized_training.sh"
    
    echo -e "${GREEN}✅ Обучение запущено в фоновом режиме${NC}"
    echo -e "${BLUE}📊 Для мониторинга используйте: tail -f /workspace/logs/training_optimized_*.log${NC}"
}

# Функция для проверки статуса GPU
check_gpu_status() {
    echo -e "${YELLOW}🖥️ Проверка статуса GPU...${NC}"
    
    if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
        SSH_CMD="ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
    else
        SSH_CMD="ssh -p $PROXY_PORT $SERVER_USER@$PROXY_HOST"
    fi
    
    $SSH_CMD "nvidia-smi"
}

# Главное меню
echo -e "${YELLOW}Выберите действие:${NC}"
echo "[1] 🔄 Только синхронизация"
echo "[2] 🚀 Синхронизация и запуск меню"
echo "[3] 🧪 Синхронизация и тестовый запуск"
echo "[4] 🛑 Остановить текущее обучение"
echo "[5] 🚀 Запустить обучение с оптимизацией GPU (фоновый режим)"
echo "[6] 🖥️ Проверить статус GPU"
echo "[7] 📊 Открыть TensorBoard туннель"
echo "[8] 📈 Мониторинг текущего обучения"
echo "[9] 💾 Кэширование данных (тестовые символы)"
echo "[10] 🌍 Кэширование данных (все символы)"
echo "[0] ❌ Выход"
echo

read -p "Выбор: " choice

case $choice in
    1)
        sync_to_server
        setup_db_tunnel
        ;;
    2)
        sync_to_server
        setup_db_tunnel
        run_on_server
        ;;
    3)
        sync_to_server
        setup_db_tunnel
        echo -e "${YELLOW}🧪 Запуск в тестовом режиме...${NC}"
        if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
            ssh -t -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "cd /workspace && source venv/bin/activate && python transformer_v3/main.py --test-mode"
        else
            ssh -t -p $PROXY_PORT $SERVER_USER@$PROXY_HOST "cd /workspace && source venv/bin/activate && python transformer_v3/main.py --test-mode"
        fi
        ;;
    4)
        stop_training
        ;;
    5)
        sync_to_server
        setup_db_tunnel
        stop_training
        run_background_optimized
        ;;
    6)
        check_gpu_status
        ;;
    7)
        echo -e "${YELLOW}📊 Создание туннеля для TensorBoard...${NC}"
        if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
            ssh -p $SERVER_PORT -L 6006:localhost:6006 $SERVER_USER@$SERVER_HOST -N &
        else
            ssh -p $PROXY_PORT -L 6006:localhost:6006 $SERVER_USER@$PROXY_HOST -N &
        fi
        echo -e "${GREEN}✅ TensorBoard доступен на http://localhost:6006${NC}"
        echo "Нажмите Ctrl+C для закрытия туннеля"
        wait
        ;;
    8)
        echo -e "${YELLOW}📈 Мониторинг текущего обучения...${NC}"
        if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
            ssh -t -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "cd /workspace && tail -f logs/training_optimized_*.log"
        else
            ssh -t -p $PROXY_PORT $SERVER_USER@$PROXY_HOST "cd /workspace && tail -f logs/training_optimized_*.log"
        fi
        ;;
    9)
        sync_to_server
        setup_db_tunnel
        echo -e "${YELLOW}💾 Запуск кэширования данных (тестовые символы)...${NC}"
        if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
            ssh -t -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "cd /workspace && source venv/bin/activate && python transformer_v3/cache_all_data.py --test-symbols BTCUSDT ETHUSDT"
        else
            ssh -t -p $PROXY_PORT $SERVER_USER@$PROXY_HOST "cd /workspace && source venv/bin/activate && python transformer_v3/cache_all_data.py --test-symbols BTCUSDT ETHUSDT"
        fi
        ;;
    10)
        sync_to_server
        setup_db_tunnel
        echo -e "${YELLOW}🌍 Запуск кэширования данных (все символы)...${NC}"
        echo -e "${RED}⚠️  ВНИМАНИЕ: Это займет много времени и места!${NC}"
        read -p "Продолжить? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            if ssh -o ConnectTimeout=5 -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'OK'" >/dev/null 2>&1; then
                ssh -t -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "cd /workspace && source venv/bin/activate && python transformer_v3/cache_all_data.py --all-symbols"
            else
                ssh -t -p $PROXY_PORT $SERVER_USER@$PROXY_HOST "cd /workspace && source venv/bin/activate && python transformer_v3/cache_all_data.py --all-symbols"
            fi
        else
            echo -e "${YELLOW}Отменено пользователем${NC}"
        fi
        ;;
    0)
        echo -e "${GREEN}До свидания!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Неверный выбор!${NC}"
        exit 1
        ;;
esac