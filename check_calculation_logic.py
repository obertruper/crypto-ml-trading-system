#!/usr/bin/env python3
"""
Детальная проверка логики расчета expected_return
"""

import psycopg2
import yaml
import pandas as pd

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

print("="*80)
print("🔍 ДЕТАЛЬНАЯ ПРОВЕРКА ЛОГИКИ РАСЧЕТА")
print("="*80)

# 1. Берем символ с нормальными ценами
cursor.execute("""
    SELECT DISTINCT symbol 
    FROM processed_market_data 
    WHERE symbol IN ('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT')
    LIMIT 1
""")
result = cursor.fetchone()

if not result:
    # Если нет популярных, берем любой кроме PEPE
    cursor.execute("""
        SELECT DISTINCT symbol 
        FROM processed_market_data 
        WHERE symbol NOT LIKE '%PEPE%'
        LIMIT 1
    """)
    result = cursor.fetchone()

if not result:
    print("⚠️ Нет данных в таблице!")
    conn.close()
    exit()

symbol = result[0]
print(f"\n📊 Анализируем символ: {symbol}")

# 2. Проверяем последовательные записи
query = """
WITH numbered_data AS (
    SELECT 
        p.symbol,
        p.datetime,
        r.close as entry_price,
        r.high,
        r.low,
        p.buy_expected_return,
        p.sell_expected_return,
        ROW_NUMBER() OVER (ORDER BY p.datetime) as row_num
    FROM processed_market_data p
    JOIN raw_market_data r ON p.symbol = r.symbol AND p.datetime = r.datetime
    WHERE p.symbol = %s
    ORDER BY p.datetime
    LIMIT 20
)
SELECT * FROM numbered_data
"""

df = pd.read_sql(query, conn, params=(symbol,))

print(f"\n📋 Анализ последовательных точек входа:")
print("-"*100)
print(f"{'#':>3} {'Время':<20} {'Цена':>10} {'High':>10} {'Low':>10} {'Buy Ret':>10} {'Sell Ret':>10} {'Изменение':>15}")
print("-"*100)

prev_buy = None
prev_sell = None

for i, row in df.iterrows():
    change = ""
    if prev_buy is not None:
        if row['buy_expected_return'] == prev_buy and row['sell_expected_return'] == prev_sell:
            change = "⚠️ SAME!"
        else:
            change = "✅ Different"
    
    print(f"{row['row_num']:>3} {str(row['datetime']):<20} {row['entry_price']:>10.4f} {row['high']:>10.4f} {row['low']:>10.4f} {row['buy_expected_return']:>10.3f}% {row['sell_expected_return']:>10.3f}% {change:>15}")
    
    prev_buy = row['buy_expected_return']
    prev_sell = row['sell_expected_return']

# 3. Проверяем уникальность значений
print(f"\n\n📊 СТАТИСТИКА УНИКАЛЬНОСТИ:")
unique_buy = df['buy_expected_return'].nunique()
unique_sell = df['sell_expected_return'].nunique()
total = len(df)

print(f"Всего записей: {total}")
print(f"Уникальных buy_expected_return: {unique_buy} ({unique_buy/total*100:.1f}%)")
print(f"Уникальных sell_expected_return: {unique_sell} ({unique_sell/total*100:.1f}%)")

# 4. Группируем по значениям
print(f"\n\n📊 ГРУППИРОВКА ПО ЗНАЧЕНИЯМ:")
grouped = df.groupby(['buy_expected_return', 'sell_expected_return']).size().reset_index(name='count')
grouped = grouped.sort_values('count', ascending=False)

print(f"\n{'Buy Return':>12} {'Sell Return':>12} {'Количество':>12}")
print("-"*40)
for _, row in grouped.head(10).iterrows():
    print(f"{row['buy_expected_return']:>12.3f}% {row['sell_expected_return']:>12.3f}% {row['count']:>12}")

# 5. Проверяем распределение значений
print(f"\n\n📊 РАСПРЕДЕЛЕНИЕ ЗНАЧЕНИЙ:")
buy_values = df['buy_expected_return'].value_counts()
print(f"\nТоп-5 значений buy_expected_return:")
for val, count in buy_values.head().items():
    print(f"  {val:>8.3f}%: {count} раз")

# 6. Проверяем корреляцию с волатильностью
print(f"\n\n🔍 ПРОВЕРКА КОРРЕЛЯЦИИ С ДВИЖЕНИЕМ ЦЕНЫ:")
df['price_change'] = ((df['high'] - df['low']) / df['entry_price'] * 100).round(3)
df['has_movement'] = df['price_change'] > 0.5  # Движение больше 0.5%

static_returns = df[df['price_change'] < 0.1]  # Малое движение
volatile_returns = df[df['price_change'] > 1.0]  # Большое движение

print(f"\nПри малом движении цены (<0.1%):")
if len(static_returns) > 0:
    print(f"  Среднее buy_return: {static_returns['buy_expected_return'].mean():.3f}%")
    print(f"  Уникальных значений: {static_returns['buy_expected_return'].nunique()}")

print(f"\nПри большом движении цены (>1%):")
if len(volatile_returns) > 0:
    print(f"  Среднее buy_return: {volatile_returns['buy_expected_return'].mean():.3f}%")
    print(f"  Уникальных значений: {volatile_returns['buy_expected_return'].nunique()}")

conn.close()

print("\n\n💡 ВЫВОДЫ:")
print("1. Если много одинаковых значений - возможна проблема в расчете")
print("2. Expected return должен меняться для каждой новой точки входа")
print("3. Значения должны коррелировать с волатильностью рынка")