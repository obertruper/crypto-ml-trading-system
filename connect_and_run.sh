#!/bin/bash
# Скрипт для подключения к Vast.ai и запуска команд

# Параметры подключения
REMOTE_HOST="79.116.73.220"
REMOTE_PORT="27681"
REMOTE_USER="root"
REMOTE_PROJECT="/workspace/crypto_trading"

# Функция для выполнения команд на сервере
run_remote() {
    ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PROJECT && source /workspace/venv/bin/activate && $1"
}

# Меню выбора действий
echo "🚀 Управление проектом на Vast.ai"
echo "================================="
echo "1) Подключиться к серверу (интерактивная сессия)"
echo "2) Синхронизировать проект"
echo "3) Запустить обучение модели (regression)"
echo "4) Запустить обучение модели (classification)"
echo "5) Мониторинг обучения"
echo "6) Просмотр логов"
echo "7) Проверка GPU"
echo "8) Установить зависимости"
echo "9) TensorBoard (локально)"
echo ""
read -p "Выберите действие (1-9): " choice

case $choice in
    1)
        echo "🔗 Подключаюсь к серверу..."
        ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST -t "cd $REMOTE_PROJECT && source /workspace/venv/bin/activate && exec bash"
        ;;
    2)
        echo "📁 Синхронизирую проект..."
        ./sync_to_vast.sh
        ;;
    3)
        echo "🧠 Запускаю обучение (regression)..."
        run_remote "python train_universal_transformer.py --task regression"
        ;;
    4)
        echo "🧠 Запускаю обучение (classification)..."
        run_remote "python train_universal_transformer.py --task classification"
        ;;
    5)
        echo "📊 Запускаю мониторинг..."
        run_remote "python monitor_training.py"
        ;;
    6)
        echo "📜 Показываю последние логи..."
        run_remote "ls -la logs/training_*/training.log | tail -5"
        echo ""
        run_remote "tail -n 50 logs/training_*/training.log | tail -1"
        ;;
    7)
        echo "🎮 Информация о GPU..."
        ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "nvidia-smi"
        ;;
    8)
        echo "📦 Устанавливаю зависимости..."
        run_remote "pip install -r requirements.txt"
        ;;
    9)
        echo "📈 Открываю TensorBoard..."
        echo "Подключитесь к: http://localhost:6006"
        ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST -L 6006:localhost:16006 -N
        ;;
    *)
        echo "❌ Неверный выбор!"
        exit 1
        ;;
esac