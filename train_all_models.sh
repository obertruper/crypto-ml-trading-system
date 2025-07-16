#!/bin/bash
# Скрипт для автоматического обучения всех моделей

echo "🚀 Автоматический запуск полного обучения"
echo "========================================"

# Проверяем БД
if ! pg_isready -p 5555 -h localhost > /dev/null 2>&1; then
    echo "❌ PostgreSQL не запущен на порту 5555!"
    echo "Запустите: pg_ctl start -D /usr/local/var/postgres"
    exit 1
fi

echo "✅ БД доступна"
echo ""
echo "⚠️  ВАЖНО: Сейчас будет создан SSH туннель к БД"
echo "   Это окно должно оставаться открытым всё время обучения!"
echo ""
echo "   Рекомендуется использовать tmux или screen:"
echo "   tmux new -s training"
echo ""
read -p "Нажмите Enter для продолжения..."

# Функция для обучения с туннелем
train_with_tunnel() {
    local task=$1
    echo ""
    echo "🔄 Запуск обучения: $task"
    
    # Создаем туннель и запускаем обучение в фоне
    ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 \
        "cd /workspace/crypto_trading && \
         source /workspace/venv/bin/activate && \
         export TF_CPP_MIN_LOG_LEVEL=1 && \
         python train_universal_transformer.py --task $task --config remote_config.yaml"
}

# Обучаем обе модели
echo "📊 Этап 1/2: Regression модели (предсказание доходности)..."
train_with_tunnel "regression"

echo ""
echo "📊 Этап 2/2: Classification модели (предсказание profit/loss)..."
train_with_tunnel "classification"

echo ""
echo "✅ Полное обучение завершено!"
echo ""
echo "📁 Результаты сохранены на сервере в:"
echo "   - /workspace/crypto_trading/trained_model/"
echo "   - /workspace/crypto_trading/logs/"
echo ""
echo "Для скачивания используйте:"
echo "rsync -avz -e 'ssh -p 27681' root@79.116.73.220:/workspace/crypto_trading/trained_model/ ./trained_model/"