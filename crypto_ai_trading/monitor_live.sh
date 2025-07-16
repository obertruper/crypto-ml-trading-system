#\!/bin/bash
# Мониторинг логов обучения в реальном времени

echo "🔍 Подключение к серверу для мониторинга..."
ssh -i ~/.ssh/id_rsa -p 48937 root@109.198.107.223 "tail -f /root/crypto_ai_trading/training.log | grep -E \"Epoch|loss=|Validation|Best model\""
