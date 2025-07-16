#!/bin/bash
cd "/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM"
source venv/bin/activate

echo "🚀 Запуск init_database.py..."
python init_database.py

echo "📊 Проверка таблиц в БД..."
PGPASSWORD=your_secure_password_here psql -h localhost -p 5555 -U ruslan -d crypto_trading -c "\dt"

echo "✅ Готово!"