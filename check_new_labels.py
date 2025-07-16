#!/usr/bin/env python3
"""
Проверка новых меток после исправления логики расчета
"""

import psycopg2
import yaml
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

print("="*80)
print("🔍 ПРОВЕРКА ДАННЫХ ПОСЛЕ ИСПРАВЛЕНИЯ ЛОГИКИ")
print("="*80)

# 1. Проверяем наличие данных
cursor.execute("SELECT COUNT(*) FROM processed_market_data")
total_count = cursor.fetchone()[0]

if total_count == 0:
    print("⚠️ Таблица processed_market_data пуста!")
    print("💡 Запустите: python prepare_dataset.py")
    conn.close()
    exit()

print(f"\n✅ Найдено {total_count:,} записей в таблице")

# 2. Проверяем структуру данных
query = """
SELECT 
    symbol,
    datetime,
    buy_expected_return,
    sell_expected_return,
    buy_profit_target,
    buy_loss_target,
    sell_profit_target,
    sell_loss_target
FROM processed_market_data
LIMIT 10
"""

df = pd.read_sql(query, conn)
print(f"\n📊 Примеры данных:")
print(df.to_string(index=False))

# 3. Статистика по expected returns
query = """
SELECT 
    AVG(buy_expected_return) as avg_buy,
    AVG(sell_expected_return) as avg_sell,
    STDDEV(buy_expected_return) as std_buy,
    STDDEV(sell_expected_return) as std_sell,
    MIN(buy_expected_return) as min_buy,
    MAX(buy_expected_return) as max_buy,
    MIN(sell_expected_return) as min_sell,
    MAX(sell_expected_return) as max_sell,
    COUNT(CASE WHEN buy_expected_return > 0 THEN 1 END) as buy_profits,
    COUNT(CASE WHEN buy_expected_return < 0 THEN 1 END) as buy_losses,
    COUNT(CASE WHEN sell_expected_return > 0 THEN 1 END) as sell_profits,
    COUNT(CASE WHEN sell_expected_return < 0 THEN 1 END) as sell_losses
FROM processed_market_data
"""

stats = pd.read_sql(query, conn).iloc[0]

print(f"\n📈 СТАТИСТИКА EXPECTED RETURNS:")
print(f"\n🟢 BUY:")
print(f"   Среднее: {stats['avg_buy']:.3f}%")
print(f"   Std Dev: {stats['std_buy']:.3f}%")
print(f"   Min/Max: {stats['min_buy']:.3f}% / {stats['max_buy']:.3f}%")
print(f"   Win Rate: {stats['buy_profits']/(stats['buy_profits']+stats['buy_losses'])*100:.1f}%")
print(f"   Прибыльных: {stats['buy_profits']:,} | Убыточных: {stats['buy_losses']:,}")

print(f"\n🔴 SELL:")
print(f"   Среднее: {stats['avg_sell']:.3f}%")
print(f"   Std Dev: {stats['std_sell']:.3f}%")
print(f"   Min/Max: {stats['min_sell']:.3f}% / {stats['max_sell']:.3f}%")
print(f"   Win Rate: {stats['sell_profits']/(stats['sell_profits']+stats['sell_losses'])*100:.1f}%")
print(f"   Прибыльных: {stats['sell_profits']:,} | Убыточных: {stats['sell_losses']:,}")

# 4. Распределение значений
print(f"\n📊 РАСПРЕДЕЛЕНИЕ BUY EXPECTED RETURNS:")
query = """
SELECT 
    CASE 
        WHEN buy_expected_return <= -1.1 THEN 'Stop Loss (-1.1%)'
        WHEN buy_expected_return > -1.1 AND buy_expected_return < 0 THEN 'Небольшой убыток'
        WHEN buy_expected_return >= 0 AND buy_expected_return < 1 THEN 'Малая прибыль (0-1%)'
        WHEN buy_expected_return >= 1 AND buy_expected_return < 2 THEN 'Средняя прибыль (1-2%)'
        WHEN buy_expected_return >= 2 AND buy_expected_return < 3.5 THEN 'Хорошая прибыль (2-3.5%)'
        WHEN buy_expected_return >= 3.5 THEN 'Отличная прибыль (3.5%+)'
    END as category,
    COUNT(*) as count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM processed_market_data) as percentage
FROM processed_market_data
GROUP BY category
ORDER BY 
    CASE category
        WHEN 'Stop Loss (-1.1%)' THEN 1
        WHEN 'Небольшой убыток' THEN 2
        WHEN 'Малая прибыль (0-1%)' THEN 3
        WHEN 'Средняя прибыль (1-2%)' THEN 4
        WHEN 'Хорошая прибыль (2-3.5%)' THEN 5
        WHEN 'Отличная прибыль (3.5%+)' THEN 6
    END
"""

distribution = pd.read_sql(query, conn)
for _, row in distribution.iterrows():
    print(f"   {row['category']:<30} {row['count']:>10,} ({row['percentage']:>6.2f}%)")

# 5. Проверка экстремальных значений
print(f"\n🎯 ПРИМЕРЫ МАКСИМАЛЬНЫХ РЕЗУЛЬТАТОВ:")
query = """
SELECT symbol, datetime, buy_expected_return, sell_expected_return
FROM processed_market_data
WHERE buy_expected_return >= 3
ORDER BY buy_expected_return DESC
LIMIT 5
"""

extreme_df = pd.read_sql(query, conn)
for _, row in extreme_df.iterrows():
    print(f"   {row['symbol']} @ {row['datetime']}: BUY={row['buy_expected_return']:.2f}%, SELL={row['sell_expected_return']:.2f}%")

# 6. Статистика по символам
print(f"\n📊 СТАТИСТИКА ПО СИМВОЛАМ:")
query = """
SELECT 
    symbol,
    COUNT(*) as records,
    AVG(buy_expected_return) as avg_buy,
    AVG(sell_expected_return) as avg_sell,
    COUNT(CASE WHEN buy_expected_return > 0 THEN 1 END) * 100.0 / COUNT(*) as buy_wr,
    COUNT(CASE WHEN sell_expected_return > 0 THEN 1 END) * 100.0 / COUNT(*) as sell_wr
FROM processed_market_data
GROUP BY symbol
ORDER BY records DESC
LIMIT 10
"""

symbols_df = pd.read_sql(query, conn)
print(f"\n{'Символ':<15} {'Записей':>10} {'Buy Avg':>10} {'Buy WR':>8} {'Sell Avg':>10} {'Sell WR':>8}")
print("-" * 75)
for _, row in symbols_df.iterrows():
    print(f"{row['symbol']:<15} {row['records']:>10,} {row['avg_buy']:>10.3f}% {row['buy_wr']:>7.1f}% {row['avg_sell']:>10.3f}% {row['sell_wr']:>7.1f}%")

conn.close()

print("\n✅ Анализ завершен!")