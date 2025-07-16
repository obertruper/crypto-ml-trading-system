#!/usr/bin/env python3
"""
Очистка неправильно рассчитанных данных для SELL
"""

import psycopg2
import yaml

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

print("🧹 Очистка неправильных данных...")

# Очищаем все данные с новыми метками (они рассчитаны неправильно)
cursor.execute("""
    UPDATE processed_market_data 
    SET technical_indicators = technical_indicators - 'buy_expected_return' - 'sell_expected_return'
    WHERE technical_indicators->>'sell_expected_return' IS NOT NULL
""")

affected_rows = cursor.rowcount
print(f"✅ Очищено {affected_rows:,} записей")

conn.commit()
cursor.close()
conn.close()

print("🔄 Теперь запустите prepare_dataset.py заново для правильного расчета")