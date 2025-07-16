#!/bin/bash
# Запуск обучения на Vast.ai сервере с гибкими настройками

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Запуск обучения на Vast.ai GPU${NC}"

# Получаем настройки из конфига или переменных окружения
CONFIG_FILE="config/config.yaml"

# Извлекаем SSH алиас из конфига
SSH_ALIAS=$(python3 -c "
import yaml
import os
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
rs = config['remote_server']
# Поддержка переменных окружения в конфиге
ssh_alias = rs['ssh_alias']
if ssh_alias.startswith('\${'):
    # Извлекаем имя переменной и дефолтное значение
    var_name = ssh_alias.split(':')[0].replace('\${', '')
    default_val = ssh_alias.split(':')[1].rstrip('}')
    ssh_alias = os.environ.get(var_name, default_val)
print(ssh_alias)
")

REMOTE_PATH=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(config['remote_server']['remote_path'])
")

echo -e "\n${YELLOW}📋 Используемые настройки:${NC}"
echo "   • SSH алиас: $SSH_ALIAS"
echo "   • Remote path: $REMOTE_PATH"

# Проверка подключения
echo -e "\n${YELLOW}🔗 Проверка подключения...${NC}"
if ssh $SSH_ALIAS "echo 'Подключение успешно'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Подключение установлено${NC}"
    
    # Показываем информацию о GPU
    GPU_INFO=$(ssh $SSH_ALIAS "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null)
    if [ -n "$GPU_INFO" ]; then
        echo -e "\n${YELLOW}🎮 Доступные GPU:${NC}"
        echo "$GPU_INFO" | while IFS=, read -r gpu_name gpu_memory; do
            echo "   • $gpu_name ($gpu_memory)"
        done
    fi
else
    echo -e "${RED}❌ Не удалось подключиться к серверу${NC}"
    echo -e "${YELLOW}Проверьте:${NC}"
    echo "   1. SSH алиас '$SSH_ALIAS' в ~/.ssh/config"
    echo "   2. Или установите переменную: export VAST_SSH_ALIAS=your-alias"
    exit 1
fi

# SSH команда
SSH_CMD="ssh $SSH_ALIAS"

# Проверяем, передан ли режим через переменную окружения
if [ -n "$GPU_TRAINING_MODE" ]; then
    # Режим передан из интерактивного меню
    choice=$GPU_TRAINING_MODE
    EPOCHS=${GPU_TRAINING_EPOCHS:-5}
    
    case $choice in
        1)
            MODE="train"  # Изменено с "demo" на "train" для запуска обучения
            ;;
        2)
            MODE="train"
            ;;
        3)
            MODE="train"
            ;;
        *)
            MODE="train"  # Изменено с "demo" на "train"
            ;;
    esac
else
    # Запуск напрямую из консоли - показываем меню
    echo -e "\n${YELLOW}Выберите режим запуска:${NC}"
    echo "1) Демо обучение (5 эпох)"
    echo "2) Полное обучение (100 эпох)"
    echo "3) Пользовательские настройки"
    echo -n "Выбор (1-3): "
    read choice

    case $choice in
        1)
            MODE="train"  # Изменено с "demo" на "train" для запуска обучения
            EPOCHS=5
            ;;
        2)
            MODE="train"
            EPOCHS=100
            ;;
        3)
            echo -n "Количество эпох: "
            read EPOCHS
            MODE="train"
            ;;
        *)
            echo -e "${RED}❌ Неверный выбор${NC}"
            exit 1
            ;;
    esac
fi

echo -e "\n${YELLOW}📊 Параметры обучения:${NC}"
echo "   • Режим: $MODE"
echo "   • Эпох: $EPOCHS"
echo "   • Batch size: 32"

# Создание скрипта запуска на сервере
echo -e "\n${YELLOW}🔧 Подготовка запуска...${NC}"

cat > /tmp/run_training.sh << SCRIPT
#!/bin/bash
cd $REMOTE_PATH

# Переданные переменные
export USE_CACHE_ONLY='${USE_CACHE_ONLY:-0}'
export MODE='${MODE}'
export EPOCHS=${EPOCHS}

# Активация окружения если есть
if [ -d 'venv' ]; then
    source venv/bin/activate
fi

# Установка переменных окружения
export CUDA_VISIBLE_DEVICES=0,1  # Используем обе GPU
export PYTHONUNBUFFERED=1

# Изменение количества эпох в конфиге
python -c "
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model']['epochs'] = \$EPOCHS
config['performance']['device'] = 'cuda'
with open('config/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"

# Создаем директорию для логов
mkdir -p logs

# Запуск обучения
echo '🚀 Запуск обучения на GPU...'
echo 'Режим: '\$MODE
echo 'Эпохи: '\$EPOCHS
echo 'Использовать кэш: '\$USE_CACHE_ONLY

# Проверяем доступность БД
if nc -z localhost 5555 2>/dev/null; then
    echo '✅ PostgreSQL доступен на порту 5555'
    echo '📊 Используется подключение к БД'
else
    echo '⚠️  PostgreSQL недоступен на порту 5555'
    echo '📊 Переключаемся на использование кэша'
    export USE_CACHE_ONLY=1
fi

# Запуск обучения
python3 run_full_pipeline.py --mode \$MODE 2>&1 | tee logs/training_gpu.log
SCRIPT

# Копируем скрипт на сервер
scp /tmp/run_training.sh $SSH_ALIAS:$REMOTE_PATH/run_training.sh

# Запуск в tmux
echo -e "\n${YELLOW}🖥️  Запуск обучения...${NC}"

$SSH_CMD "
chmod +x $REMOTE_PATH/run_training.sh
# Проверяем наличие tmux
if command -v tmux &> /dev/null; then
    echo 'Запуск в tmux сессии...'
    tmux new-session -d -s training 'cd $REMOTE_PATH && bash run_training.sh'
    echo '✅ Обучение запущено в tmux сессии \"training\"'
else
    echo 'tmux не установлен, запуск в screen...'
    if command -v screen &> /dev/null; then
        screen -dmS training bash -c 'cd $REMOTE_PATH && bash run_training.sh'
        echo '✅ Обучение запущено в screen сессии \"training\"'
    else
        echo 'Запуск в фоновом режиме с nohup...'
        cd $REMOTE_PATH && nohup bash run_training.sh > logs/training_gpu.log 2>&1 &
        echo '✅ Обучение запущено в фоне'
    fi
fi
"

echo -e "\n${GREEN}✅ Обучение запущено!${NC}"
echo -e "\n${YELLOW}📌 Полезные команды:${NC}"
echo "   • Подключиться к серверу: ssh $SSH_ALIAS"
echo "   • Подключиться к сессии: tmux attach -t training"
echo "   • Отключиться от сессии: Ctrl+B, затем D"
echo "   • Мониторинг GPU: nvidia-smi -l 1"
echo "   • Просмотр логов: tail -f $REMOTE_PATH/logs/training_gpu.log"
echo ""
echo -e "${BLUE}💡 Для мониторинга TensorBoard:${NC}"
echo "   ssh -L 6006:localhost:6006 $SSH_ALIAS"
echo "   Затем откройте http://localhost:6006"

# Явный выход из скрипта
exit 0