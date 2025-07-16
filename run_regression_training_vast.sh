#!/bin/bash
# Скрипт для запуска обучения регрессионной модели на Vast.ai

echo "🚀 Запуск обучения регрессионной модели на Vast.ai"
echo "=================================================="

# Проверяем подключение к БД
echo "🔍 Проверка подключения к локальной БД..."
if ! pg_isready -p 5555 -h localhost > /dev/null 2>&1; then
    echo "❌ PostgreSQL не запущен локально на порту 5555!"
    echo "Запустите БД командой:"
    echo "  pg_ctl start -D /usr/local/var/postgres"
    exit 1
fi
echo "✅ Локальная БД доступна"

# Меню выбора
echo ""
echo "Выберите действие:"
echo "1) Запустить обучение"
echo "2) Синхронизировать файлы на сервер"
echo "3) Проверить статус GPU"
echo "4) Мониторинг обучения"
echo "5) Скачать результаты"
echo ""
read -p "Ваш выбор (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🧠 Запускаю обучение моделей регрессии..."
        echo ""
        echo "⚠️  ВАЖНО: Откройте НОВЫЙ терминал и выполните:"
        echo "    ./setup_remote_db_tunnel.sh"
        echo ""
        echo "Это создаст SSH туннель для доступа к локальной БД"
        echo ""
        read -p "Нажмите Enter когда туннель будет создан..."
        
        echo ""
        echo "📡 Подключаюсь к серверу и запускаю обучение..."
        ssh -p 27681 root@79.116.73.220 -t "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=1 && python train_universal_transformer.py --config remote_config.yaml"
        ;;
    
    2)
        echo ""
        echo "📤 Синхронизация проекта на сервер..."
        ./sync_to_vast.sh
        ;;
    
    3)
        echo ""
        echo "🖥️ Проверка статуса GPU..."
        ssh -p 27681 root@79.116.73.220 nvidia-smi
        ;;
    
    4)
        echo ""
        echo "📈 Мониторинг обучения..."
        echo "Выберите метод:"
        echo "1) Просмотр логов в реальном времени"
        echo "2) TensorBoard (требует туннель)"
        echo "3) Monitor script"
        read -p "Ваш выбор (1-3): " monitor_choice
        
        case $monitor_choice in
            1)
                ssh -p 27681 root@79.116.73.220 "cd /workspace/crypto_trading && tail -f logs/training_*/training.log"
                ;;
            2)
                echo ""
                echo "📊 Создаю туннель для TensorBoard..."
                echo "Откройте в браузере: http://localhost:6006"
                echo ""
                ssh -p 27681 root@79.116.73.220 -L 6006:localhost:16006 -N
                ;;
            3)
                ssh -p 27681 root@79.116.73.220 -t "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && python monitor_training.py"
                ;;
        esac
        ;;
    
    5)
        echo ""
        echo "📥 Скачивание результатов..."
        echo ""
        
        # Создаем локальные директории если их нет
        mkdir -p trained_model
        mkdir -p plots
        mkdir -p logs
        
        # Скачиваем модели
        echo "📦 Скачиваю обученные модели..."
        rsync -avz -e "ssh -p 27681" root@79.116.73.220:/workspace/crypto_trading/trained_model/ ./trained_model/
        
        # Скачиваем графики
        echo "📊 Скачиваю графики..."
        rsync -avz -e "ssh -p 27681" root@79.116.73.220:/workspace/crypto_trading/logs/training_*/plots/ ./plots/
        
        # Скачиваем логи
        echo "📝 Скачиваю логи..."
        rsync -avz -e "ssh -p 27681" root@79.116.73.220:/workspace/crypto_trading/logs/training_*/ ./logs/
        
        echo ""
        echo "✅ Результаты успешно скачаны!"
        ;;
    
    *)
        echo "❌ Неверный выбор!"
        exit 1
        ;;
esac