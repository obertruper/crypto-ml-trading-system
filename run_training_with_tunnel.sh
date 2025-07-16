#!/bin/bash

echo "🚀 Запуск Enhanced TFT v2.1 с интегрированным туннелем"
echo "======================================================="

# Запускаем SSH с туннелем и командой в одной сессии
ssh -p 27681 root@79.116.73.220 -R 5555:localhost:5555 << 'EOF'
cd /workspace/crypto_trading
source /workspace/venv/bin/activate

echo "🔍 Проверка туннеля БД..."
python -c "import psycopg2; conn = psycopg2.connect('postgresql://ruslan:ruslan@localhost:5555/crypto_trading'); print('✅ БД доступна'); conn.close()" || exit 1

echo "🚀 Запуск обучения..."
python train_universal_transformer_v2.py --task classification_binary --ensemble_size 1
EOF

echo "✅ Обучение завершено!"