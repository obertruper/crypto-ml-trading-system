#!/usr/bin/env python3
"""
Валидация expected_return значений перед обучением
"""

import psycopg2
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

conn = psycopg2.connect(**db_config)

print("="*80)
print("🔍 ВАЛИДАЦИЯ EXPECTED_RETURN ЗНАЧЕНИЙ")
print("="*80)

# 1. Общая статистика
query = """
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT buy_expected_return) as unique_buy,
    COUNT(DISTINCT sell_expected_return) as unique_sell,
    AVG(buy_expected_return) as avg_buy,
    AVG(sell_expected_return) as avg_sell,
    STDDEV(buy_expected_return) as std_buy,
    STDDEV(sell_expected_return) as std_sell,
    MIN(buy_expected_return) as min_buy,
    MAX(buy_expected_return) as max_buy,
    MIN(sell_expected_return) as min_sell,
    MAX(sell_expected_return) as max_sell
FROM processed_market_data
"""

result = conn.cursor()
result.execute(query)
stats = result.fetchone()

total, unique_buy, unique_sell, avg_buy, avg_sell, std_buy, std_sell, min_buy, max_buy, min_sell, max_sell = stats

print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
print(f"Всего записей: {total:,}")
print(f"\nBuy Expected Return:")
print(f"  Уникальных значений: {unique_buy:,} ({unique_buy/total*100:.1f}%)")
print(f"  Среднее: {avg_buy:.3f}%")
print(f"  Std: {std_buy:.3f}%")
print(f"  Диапазон: [{min_buy:.3f}%, {max_buy:.3f}%]")

print(f"\nSell Expected Return:")
print(f"  Уникальных значений: {unique_sell:,} ({unique_sell/total*100:.1f}%)")
print(f"  Среднее: {avg_sell:.3f}%")
print(f"  Std: {std_sell:.3f}%")
print(f"  Диапазон: [{min_sell:.3f}%, {max_sell:.3f}%]")

# Проверка уникальности
buy_uniqueness = unique_buy / total * 100
sell_uniqueness = unique_sell / total * 100

if buy_uniqueness < 50 or sell_uniqueness < 50:
    print("\n⚠️ ПРЕДУПРЕЖДЕНИЕ: Низкая уникальность значений!")
    print("   Модель может плохо обучаться на таких данных.")
    print("   Рекомендуется пересчитать данные с улучшенной логикой.")
else:
    print("\n✅ Уникальность значений в норме.")

# 2. Статистика по символам
print("\n\n📊 СТАТИСТИКА ПО СИМВОЛАМ:")
symbol_query = """
SELECT 
    symbol,
    COUNT(*) as records,
    COUNT(DISTINCT buy_expected_return) as unique_buy,
    COUNT(DISTINCT sell_expected_return) as unique_sell,
    AVG(buy_expected_return) as avg_buy,
    AVG(sell_expected_return) as avg_sell
FROM processed_market_data
GROUP BY symbol
ORDER BY symbol
"""

df_symbols = pd.read_sql(symbol_query, conn)

print(f"\n{'Символ':<15} {'Записей':>10} {'Уник.Buy':>10} {'Уник.Sell':>10} {'Avg Buy':>10} {'Avg Sell':>10}")
print("-"*75)
for _, row in df_symbols.iterrows():
    buy_pct = row['unique_buy'] / row['records'] * 100
    sell_pct = row['unique_sell'] / row['records'] * 100
    status = "⚠️" if buy_pct < 50 or sell_pct < 50 else "✅"
    print(f"{row['symbol']:<15} {row['records']:>10,} {row['unique_buy']:>10} {row['unique_sell']:>10} {row['avg_buy']:>10.3f}% {row['avg_sell']:>10.3f}% {status}")

# 3. Распределение значений
print("\n\n📊 РАСПРЕДЕЛЕНИЕ ЗНАЧЕНИЙ:")

# Получаем выборку данных для визуализации
sample_query = """
SELECT buy_expected_return, sell_expected_return
FROM processed_market_data
WHERE buy_expected_return IS NOT NULL 
  AND sell_expected_return IS NOT NULL
LIMIT 100000
"""

df_sample = pd.read_sql(sample_query, conn)

# Создаем графики
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Expected Return Distribution Analysis', fontsize=16)

# График 1: Гистограмма buy_expected_return
axes[0, 0].hist(df_sample['buy_expected_return'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Buy Expected Return Distribution')
axes[0, 0].set_xlabel('Return (%)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

# График 2: Гистограмма sell_expected_return
axes[0, 1].hist(df_sample['sell_expected_return'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Sell Expected Return Distribution')
axes[0, 1].set_xlabel('Return (%)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

# График 3: Scatter plot
axes[1, 0].scatter(df_sample['buy_expected_return'], df_sample['sell_expected_return'], 
                   alpha=0.1, s=1)
axes[1, 0].set_title('Buy vs Sell Expected Return')
axes[1, 0].set_xlabel('Buy Return (%)')
axes[1, 0].set_ylabel('Sell Return (%)')
axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)

# График 4: Статистика
axes[1, 1].axis('off')
stats_text = f"""
Validation Summary:

Buy Expected Return:
  Unique values: {unique_buy:,} ({buy_uniqueness:.1f}%)
  Mean: {avg_buy:.3f}%
  Std: {std_buy:.3f}%
  Range: [{min_buy:.3f}%, {max_buy:.3f}%]

Sell Expected Return:
  Unique values: {unique_sell:,} ({sell_uniqueness:.1f}%)
  Mean: {avg_sell:.3f}%
  Std: {std_sell:.3f}%
  Range: [{min_sell:.3f}%, {max_sell:.3f}%]

Status: {'⚠️ LOW UNIQUENESS' if buy_uniqueness < 50 or sell_uniqueness < 50 else '✅ GOOD'}
"""
axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
               verticalalignment='center', family='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))

plt.tight_layout()
plt.savefig('expected_return_validation.png', dpi=150, bbox_inches='tight')
print("\n📊 График сохранен: expected_return_validation.png")

# 4. Проверка корреляции с волатильностью
print("\n\n🔍 ПРОВЕРКА КОРРЕЛЯЦИИ С ВОЛАТИЛЬНОСТЬЮ:")
volatility_query = """
WITH volatility_data AS (
    SELECT 
        symbol,
        datetime,
        buy_expected_return,
        sell_expected_return,
        (high - low) / close * 100 as price_range_pct
    FROM processed_market_data p
    JOIN raw_market_data r ON p.symbol = r.symbol AND p.datetime = r.datetime
    WHERE p.buy_expected_return IS NOT NULL
    LIMIT 10000
)
SELECT 
    CASE 
        WHEN price_range_pct < 0.5 THEN 'Low (<0.5%)'
        WHEN price_range_pct < 1.0 THEN 'Medium (0.5-1%)'
        WHEN price_range_pct < 2.0 THEN 'High (1-2%)'
        ELSE 'Very High (>2%)'
    END as volatility_group,
    COUNT(*) as count,
    COUNT(DISTINCT buy_expected_return) as unique_buy,
    COUNT(DISTINCT sell_expected_return) as unique_sell,
    AVG(buy_expected_return) as avg_buy,
    AVG(sell_expected_return) as avg_sell
FROM volatility_data
GROUP BY volatility_group
ORDER BY 
    CASE volatility_group
        WHEN 'Low (<0.5%)' THEN 1
        WHEN 'Medium (0.5-1%)' THEN 2
        WHEN 'High (1-2%)' THEN 3
        ELSE 4
    END
"""

df_volatility = pd.read_sql(volatility_query, conn)

print(f"\n{'Волатильность':<20} {'Записей':>10} {'Уник.Buy':>10} {'Уник.Sell':>10} {'Avg Buy':>10} {'Avg Sell':>10}")
print("-"*80)
for _, row in df_volatility.iterrows():
    print(f"{row['volatility_group']:<20} {row['count']:>10,} {row['unique_buy']:>10} {row['unique_sell']:>10} {row['avg_buy']:>10.3f}% {row['avg_sell']:>10.3f}%")

conn.close()

print("\n\n💡 РЕКОМЕНДАЦИИ:")
if buy_uniqueness < 50 or sell_uniqueness < 50:
    print("1. ❌ Необходимо пересчитать данные с улучшенной логикой")
    print("2. Проверить, что расчет учитывает реальные рыночные цены")
    print("3. Добавить больше вариативности в расчеты")
else:
    print("1. ✅ Данные готовы для обучения")
    print("2. Рекомендуется использовать регрессию для предсказания expected_return")
    print("3. Следить за метриками MAE и R² во время обучения")