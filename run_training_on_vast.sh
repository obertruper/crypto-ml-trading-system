#!/bin/bash
# Скрипт для запуска обучения на Vast.ai с локальной БД

echo "🚀 Запуск обучения модели на Vast.ai"
echo "===================================="

# Проверяем подключение к БД
echo "🔍 Проверка подключения к локальной БД..."
if ! pg_isready -p 5555 -h localhost > /dev/null 2>&1; then
    echo "❌ PostgreSQL не запущен локально на порту 5555!"
    echo "Запустите БД и попробуйте снова."
    exit 1
fi
echo "✅ Локальная БД доступна"

# Меню выбора
echo ""
echo "Выберите режим обучения:"
echo "1) Regression (предсказание доходности)"
echo "2) Classification (предсказание profit/loss)"
echo "3) ВСЁ - обучить обе модели последовательно ⭐"
echo "4) Проверка датасета"
echo "5) Мониторинг обучения"
echo ""
read -p "Ваш выбор (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🧠 Запускаю обучение в режиме REGRESSION..."
        echo "Откройте в новом терминале: ./setup_remote_db_tunnel.sh"
        echo "для создания туннеля к БД"
        echo ""
        read -p "Нажмите Enter когда туннель будет создан..."
        
        ssh -p 27681 root@79.116.73.220 -t "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=1 && python train_universal_transformer.py --config remote_config.yaml"
        ;;
    2)
        echo ""
        echo "❌ Режим Classification больше не поддерживается"
        echo "Модель теперь работает только в режиме регрессии"
        echo "Используйте опцию 1 для запуска обучения"
        ;;
    3)
        echo ""
        echo "🚀 Запускаю обучение моделей регрессии..."
        echo "Откройте в новом терминале: ./setup_remote_db_tunnel.sh"
        echo "для создания туннеля к БД"
        echo ""
        read -p "Нажмите Enter когда туннель будет создан..."
        
        echo "🧠 Запускаю обучение моделей регрессии..."
        ssh -p 27681 root@79.116.73.220 -t "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=1 && python train_universal_transformer.py --config remote_config.yaml"
        
        echo ""
        echo "✅ Полное обучение завершено!"
        ;;
    4)
        echo ""
        echo "📊 Проверка датасета..."
        ssh -p 27681 root@79.116.73.220 -t "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && python check_dataset_status.py"
        ;;
    5)
        echo ""
        echo "📈 Запуск мониторинга..."
        ssh -p 27681 root@79.116.73.220 -t "cd /workspace/crypto_trading && source /workspace/venv/bin/activate && python monitor_training.py"
        ;;
    *)
        echo "❌ Неверный выбор!"
        exit 1
        ;;
esac