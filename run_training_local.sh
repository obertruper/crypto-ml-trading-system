#!/bin/bash
# Скрипт для локального запуска обучения

echo "🚀 Запуск обучения регрессионной модели локально"
echo "==============================================="

# Проверяем подключение к БД
echo "🔍 Проверка подключения к PostgreSQL..."
if ! pg_isready -p 5555 -h localhost > /dev/null 2>&1; then
    echo "❌ PostgreSQL не запущен на порту 5555!"
    echo ""
    echo "Запустите БД одним из способов:"
    echo "1) pg_ctl start -D /usr/local/var/postgres"
    echo "2) brew services start postgresql"
    echo ""
    exit 1
fi
echo "✅ PostgreSQL доступен"

# Проверяем Python окружение
echo ""
echo "🐍 Проверка Python окружения..."
python_version=$(python3 --version 2>&1)
echo "   $python_version"

# Проверяем TensorFlow
echo ""
echo "🤖 Проверка TensorFlow..."
python3 -c "import tensorflow as tf; print(f'   TensorFlow {tf.__version__}')" 2>/dev/null || {
    echo "❌ TensorFlow не установлен!"
    echo "   Установите: pip install tensorflow"
    exit 1
}

# Меню
echo ""
echo "Выберите действие:"
echo "1) Запустить обучение"
echo "2) Проверить данные в БД"
echo "3) Мониторинг последнего обучения"
echo "4) Открыть TensorBoard"
echo ""
read -p "Ваш выбор (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🧠 Запуск обучения..."
        echo "Это может занять несколько часов в зависимости от объема данных"
        echo ""
        
        # Запускаем обучение
        python3 train_universal_transformer.py --config config.yaml
        
        echo ""
        echo "✅ Обучение завершено!"
        echo "📊 Результаты сохранены в:"
        echo "   - Модели: trained_model/"
        echo "   - Логи: logs/training_*/"
        echo "   - Графики: logs/training_*/plots/"
        ;;
    
    2)
        echo ""
        echo "📊 Проверка данных в БД..."
        python3 check_dataset_status.py
        ;;
    
    3)
        echo ""
        echo "📈 Мониторинг последнего обучения..."
        
        # Находим последнюю директорию с логами
        latest_log=$(ls -td logs/training_* 2>/dev/null | head -1)
        
        if [ -z "$latest_log" ]; then
            echo "❌ Логи обучения не найдены"
            exit 1
        fi
        
        echo "Просмотр логов из: $latest_log"
        echo ""
        
        # Показываем последние строки лога
        if [ -f "$latest_log/training.log" ]; then
            tail -f "$latest_log/training.log"
        else
            echo "❌ Файл лога не найден"
        fi
        ;;
    
    4)
        echo ""
        echo "📊 Запуск TensorBoard..."
        
        # Находим последнюю директорию с логами
        latest_log=$(ls -td logs/training_* 2>/dev/null | head -1)
        
        if [ -z "$latest_log" ]; then
            echo "❌ Логи обучения не найдены"
            exit 1
        fi
        
        echo "Открываю TensorBoard для: $latest_log"
        echo "Откройте в браузере: http://localhost:6006"
        echo ""
        echo "Нажмите Ctrl+C для остановки"
        
        tensorboard --logdir "$latest_log/tensorboard" --host localhost --port 6006
        ;;
    
    *)
        echo "❌ Неверный выбор!"
        exit 1
        ;;
esac