#!/usr/bin/env python3
"""
Проверка уникальности expected_return в базе данных
"""

import psycopg2
import yaml
import numpy as np
import pandas as pd

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

print("🔍 Проверка уникальности expected_return")
print("="*60)

# Подключение к БД
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Запрос данных
query = """
SELECT 
    symbol,
    buy_expected_return,
    sell_expected_return,
    datetime
FROM processed_market_data
WHERE buy_expected_return IS NOT NULL
  AND sell_expected_return IS NOT NULL
ORDER BY symbol, timestamp
LIMIT 100000
"""

print("📊 Загрузка данных...")
df = pd.read_sql_query(query, conn)
print(f"✅ Загружено {len(df)} записей")

# Анализ уникальности
print("\n📈 Анализ buy_expected_return:")
buy_unique = df['buy_expected_return'].nunique()
buy_total = len(df)
buy_pct = (buy_unique / buy_total) * 100
print(f"   Уникальных значений: {buy_unique:,} из {buy_total:,} ({buy_pct:.1f}%)")
print(f"   Топ-10 частых значений:")
print(df['buy_expected_return'].value_counts().head(10))

print("\n📈 Анализ sell_expected_return:")
sell_unique = df['sell_expected_return'].nunique()
sell_total = len(df)
sell_pct = (sell_unique / sell_total) * 100
print(f"   Уникальных значений: {sell_unique:,} из {sell_total:,} ({sell_pct:.1f}%)")
print(f"   Топ-10 частых значений:")
print(df['sell_expected_return'].value_counts().head(10))

# Проверка по символам
print("\n📊 Уникальность по символам:")
for symbol in df['symbol'].unique()[:5]:
    symbol_df = df[df['symbol'] == symbol]
    buy_u = symbol_df['buy_expected_return'].nunique()
    sell_u = symbol_df['sell_expected_return'].nunique()
    total = len(symbol_df)
    print(f"   {symbol}: buy={buy_u}/{total} ({buy_u/total*100:.1f}%), sell={sell_u}/{total} ({sell_u/total*100:.1f}%)")

# Проверка диапазонов
print("\n📊 Диапазоны значений:")
print(f"   Buy min/max: {df['buy_expected_return'].min():.4f} / {df['buy_expected_return'].max():.4f}")
print(f"   Sell min/max: {df['sell_expected_return'].min():.4f} / {df['sell_expected_return'].max():.4f}")

# Проверка нулевых значений
buy_zeros = (df['buy_expected_return'] == 0).sum()
sell_zeros = (df['sell_expected_return'] == 0).sum()
print(f"\n⚠️  Нулевые значения:")
print(f"   Buy: {buy_zeros} ({buy_zeros/len(df)*100:.1f}%)")
print(f"   Sell: {sell_zeros} ({sell_zeros/len(df)*100:.1f}%)")

if buy_pct < 20 or sell_pct < 20:
    print("\n❌ КРИТИЧЕСКИ НИЗКАЯ УНИКАЛЬНОСТЬ!")
    print("   Необходимо пересчитать данные с помощью prepare_dataset.py")
    print("   или recalculate_realistic_data.py")
else:
    print("\n✅ Уникальность в норме")

cursor.close()
conn.close()