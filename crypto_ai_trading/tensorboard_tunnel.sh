#\!/bin/bash
# Создание SSH туннеля для TensorBoard

echo "🚇 Создание туннеля для TensorBoard..."
echo "После подключения откройте: http://localhost:6006"
echo ""
echo "Нажмите Ctrl+C для выхода"

ssh -L 6006:localhost:6006 -i ~/.ssh/id_rsa -p 30421 root@ssh1.vast.ai "cd /root/crypto_ai_trading && tensorboard --logdir experiments/logs/tensorboard --port 6006 --host 0.0.0.0"
