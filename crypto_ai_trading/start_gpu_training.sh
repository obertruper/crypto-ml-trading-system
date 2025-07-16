#!/bin/bash
# Простой скрипт для запуска обучения на GPU

echo "🚀 Запуск обучения на GPU сервере..."

# Синхронизация данных
echo "📤 Синхронизация данных..."
scp -P 48937 cache/features_cache.pkl root@109.198.107.223:/root/crypto_ai_trading/cache/

# Запуск обучения на сервере
echo "🧠 Запуск обучения..."
ssh -p 48937 root@109.198.107.223 "cd /root/crypto_ai_trading && \
    mkdir -p experiments/runs logs models_saved && \
    pip install -q torch torchvision tensorboard pandas numpy scikit-learn && \
    python3 -c 'import torch; print(f\"GPU доступна: {torch.cuda.is_available()}\")' && \
    tensorboard --logdir=experiments/runs --port=6006 --host=0.0.0.0 &>/dev/null & \
    python3 run_full_pipeline.py --mode demo"

echo "✅ Готово! TensorBoard доступен на http://localhost:6006"