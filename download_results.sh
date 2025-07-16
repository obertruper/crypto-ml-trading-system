#!/bin/bash

echo "📥 Скачивание результатов обучения с сервера"
echo "==========================================="

# Конфигурация сервера
SERVER_HOST="ssh1.vast.ai"
SERVER_PORT=18645
SERVER_USER="root"

# Находим последнюю папку с логами на сервере
echo "🔍 Поиск последних результатов на сервере..."
LATEST_LOG=$(ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "cd ~/xgboost_v3 && ls -td logs/xgboost_v3_* 2>/dev/null | head -1")

if [ -z "$LATEST_LOG" ]; then
    echo "❌ Результаты не найдены на сервере"
    exit 1
fi

echo "✅ Найдены результаты: $LATEST_LOG"

# Создаем локальную директорию
LOCAL_DIR="logs_from_gpu/$(basename $LATEST_LOG)"
mkdir -p "$LOCAL_DIR"

# Скачиваем результаты
echo -e "\n📥 Скачивание файлов..."
rsync -avz --progress \
  -e "ssh -p $SERVER_PORT" \
  $SERVER_USER@$SERVER_HOST:~/xgboost_v3/$LATEST_LOG/ \
  "$LOCAL_DIR/"

echo -e "\n✅ Результаты скачаны в: $LOCAL_DIR"

# Показываем краткую статистику
echo -e "\n📊 Содержимое:"
ls -la "$LOCAL_DIR/" | head -20

# Проверяем наличие ключевых файлов
echo -e "\n🔍 Проверка ключевых файлов:"
for file in "config.yaml" "metrics.json" "final_report.txt"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file не найден"
    fi
done

# Если есть metrics.json, показываем результаты
if [ -f "$LOCAL_DIR/metrics.json" ]; then
    echo -e "\n📈 Результаты обучения:"
    python3 -c "
import json
with open('$LOCAL_DIR/metrics.json', 'r') as f:
    metrics = json.load(f)
    if 'buy' in metrics:
        print(f\"Buy - Accuracy: {metrics['buy'].get('accuracy', 'N/A'):.3f}, Precision: {metrics['buy'].get('precision', 'N/A'):.3f}\")
    if 'sell' in metrics:
        print(f\"Sell - Accuracy: {metrics['sell'].get('accuracy', 'N/A'):.3f}, Precision: {metrics['sell'].get('precision', 'N/A'):.3f}\")
    "
fi

echo -e "\n✅ Готово!"