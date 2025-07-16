#!/usr/bin/env python3
"""
Анализ и визуализация реалистичных точек входа
"""

import psycopg2
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

print("="*80)
print("📊 АНАЛИЗ РЕАЛИСТИЧНЫХ ТОЧЕК ВХОДА")
print("="*80)

# Подключение к БД
conn = psycopg2.connect(**db_config)

# 1. Общая статистика
query = """
SELECT 
    COUNT(*) as total_bars,
    SUM(CASE WHEN is_long_entry THEN 1 ELSE 0 END) as long_entries,
    SUM(CASE WHEN is_short_entry THEN 1 ELSE 0 END) as short_entries,
    AVG(CASE WHEN is_long_entry THEN buy_expected_return END) as avg_long_return,
    AVG(CASE WHEN is_short_entry THEN sell_expected_return END) as avg_short_return,
    STDDEV(CASE WHEN is_long_entry THEN buy_expected_return END) as std_long_return,
    STDDEV(CASE WHEN is_short_entry THEN sell_expected_return END) as std_short_return
FROM processed_market_data
"""

df_stats = pd.read_sql(query, conn)
print("\n📊 Общая статистика:")
print(f"   Всего баров: {df_stats['total_bars'][0]:,}")
print(f"   LONG входов: {df_stats['long_entries'][0]:,} ({df_stats['long_entries'][0]/df_stats['total_bars'][0]*100:.2f}%)")
print(f"   SHORT входов: {df_stats['short_entries'][0]:,} ({df_stats['short_entries'][0]/df_stats['total_bars'][0]*100:.2f}%)")
print(f"   Средний return LONG: {df_stats['avg_long_return'][0]:.2f}% ± {df_stats['std_long_return'][0]:.2f}%")
print(f"   Средний return SHORT: {df_stats['avg_short_return'][0]:.2f}% ± {df_stats['std_short_return'][0]:.2f}%")

# 2. Анализ по типам входов
query_types = """
SELECT 
    direction,
    entry_type,
    COUNT(*) as count,
    AVG(expected_return) as avg_return,
    STDDEV(expected_return) as std_return,
    MIN(expected_return) as min_return,
    MAX(expected_return) as max_return,
    SUM(CASE WHEN expected_return > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100 as win_rate
FROM (
    SELECT 
        'LONG' as direction,
        long_entry_type as entry_type,
        buy_expected_return as expected_return
    FROM processed_market_data
    WHERE is_long_entry = TRUE
    UNION ALL
    SELECT 
        'SHORT' as direction,
        short_entry_type as entry_type,
        sell_expected_return as expected_return
    FROM processed_market_data
    WHERE is_short_entry = TRUE
) t
GROUP BY direction, entry_type
ORDER BY direction, avg_return DESC
"""

df_types = pd.read_sql(query_types, conn)
print("\n📊 Анализ по типам входов:")
for _, row in df_types.iterrows():
    print(f"\n   {row['direction']} - {row['entry_type']}:")
    print(f"      Количество: {row['count']}")
    print(f"      Средний return: {row['avg_return']:.2f}% ± {row['std_return']:.2f}%")
    print(f"      Win rate: {row['win_rate']:.1f}%")
    print(f"      Диапазон: [{row['min_return']:.2f}%, {row['max_return']:.2f}%]")

# 3. Анализ по символам
query_symbols = """
SELECT 
    symbol,
    COUNT(CASE WHEN is_long_entry THEN 1 END) as long_entries,
    COUNT(CASE WHEN is_short_entry THEN 1 END) as short_entries,
    AVG(CASE WHEN is_long_entry THEN buy_expected_return END) as avg_long_return,
    AVG(CASE WHEN is_short_entry THEN sell_expected_return END) as avg_short_return
FROM processed_market_data
GROUP BY symbol
ORDER BY (long_entries + short_entries) DESC
LIMIT 10
"""

df_symbols = pd.read_sql(query_symbols, conn)
print("\n📊 Топ-10 символов по количеству входов:")
for _, row in df_symbols.iterrows():
    print(f"   {row['symbol']}: LONG={row['long_entries']}, SHORT={row['short_entries']}, "
          f"Avg LONG={row['avg_long_return']:.2f}%, Avg SHORT={row['avg_short_return']:.2f}%")

# 4. Создание визуализаций
print("\n📈 Создание графиков...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Анализ реалистичных точек входа', fontsize=16)

# График 1: Распределение returns по типам входов (LONG)
ax1 = axes[0, 0]
query_dist = """
SELECT long_entry_type, buy_expected_return
FROM processed_market_data
WHERE is_long_entry = TRUE
"""
df_dist = pd.read_sql(query_dist, conn)
for entry_type in ['good', 'bad', 'random']:
    data = df_dist[df_dist['long_entry_type'] == entry_type]['buy_expected_return']
    if len(data) > 0:
        ax1.hist(data, bins=50, alpha=0.6, label=f'{entry_type} (n={len(data)})', density=True)
ax1.set_xlabel('Expected Return (%)')
ax1.set_ylabel('Density')
ax1.set_title('LONG: Распределение returns по типам входов')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Распределение returns по типам входов (SHORT)
ax2 = axes[0, 1]
query_dist = """
SELECT short_entry_type, sell_expected_return
FROM processed_market_data
WHERE is_short_entry = TRUE
"""
df_dist = pd.read_sql(query_dist, conn)
for entry_type in ['good', 'bad', 'random']:
    data = df_dist[df_dist['short_entry_type'] == entry_type]['sell_expected_return']
    if len(data) > 0:
        ax2.hist(data, bins=50, alpha=0.6, label=f'{entry_type} (n={len(data)})', density=True)
ax2.set_xlabel('Expected Return (%)')
ax2.set_ylabel('Density')
ax2.set_title('SHORT: Распределение returns по типам входов')
ax2.legend()
ax2.grid(True, alpha=0.3)

# График 3: Win rate по типам входов
ax3 = axes[0, 2]
pivot_data = df_types.pivot(index='entry_type', columns='direction', values='win_rate')
pivot_data.plot(kind='bar', ax=ax3)
ax3.set_ylabel('Win Rate (%)')
ax3.set_title('Win Rate по типам входов')
ax3.set_xlabel('Тип входа')
ax3.legend(title='Направление')
ax3.grid(True, alpha=0.3)

# График 4: Средний return по типам входов
ax4 = axes[1, 0]
pivot_data = df_types.pivot(index='entry_type', columns='direction', values='avg_return')
pivot_data.plot(kind='bar', ax=ax4)
ax4.set_ylabel('Average Return (%)')
ax4.set_title('Средний return по типам входов')
ax4.set_xlabel('Тип входа')
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4.legend(title='Направление')
ax4.grid(True, alpha=0.3)

# График 5: Количество входов по типам
ax5 = axes[1, 1]
pivot_data = df_types.pivot(index='entry_type', columns='direction', values='count')
pivot_data.plot(kind='bar', ax=ax5)
ax5.set_ylabel('Количество входов')
ax5.set_title('Распределение входов по типам')
ax5.set_xlabel('Тип входа')
ax5.legend(title='Направление')
ax5.grid(True, alpha=0.3)

# График 6: Временной анализ
ax6 = axes[1, 2]
query_time = """
SELECT 
    DATE_TRUNC('month', datetime) as month,
    COUNT(CASE WHEN is_long_entry THEN 1 END) as long_entries,
    COUNT(CASE WHEN is_short_entry THEN 1 END) as short_entries
FROM processed_market_data
WHERE datetime IS NOT NULL
GROUP BY month
ORDER BY month
"""
df_time = pd.read_sql(query_time, conn)
if not df_time.empty:
    df_time['month'] = pd.to_datetime(df_time['month'])
    ax6.plot(df_time['month'], df_time['long_entries'], 'b-', label='LONG entries')
    ax6.plot(df_time['month'], df_time['short_entries'], 'r-', label='SHORT entries')
    ax6.set_xlabel('Месяц')
    ax6.set_ylabel('Количество входов')
    ax6.set_title('Распределение входов по времени')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('realistic_entries_analysis.png', dpi=150, bbox_inches='tight')
print("✅ График сохранен: realistic_entries_analysis.png")

# 5. Проверка корреляции с индикаторами
print("\n🔍 Анализ корреляции с техническими индикаторами...")

query_indicators = """
SELECT 
    technical_indicators->>'rsi_val' as rsi,
    technical_indicators->>'adx_val' as adx,
    technical_indicators->>'bb_position' as bb_position,
    technical_indicators->>'volume_ratio' as volume_ratio,
    CASE 
        WHEN is_long_entry THEN buy_expected_return
        WHEN is_short_entry THEN sell_expected_return
        ELSE NULL
    END as expected_return,
    CASE 
        WHEN is_long_entry THEN 'LONG'
        WHEN is_short_entry THEN 'SHORT'
        ELSE NULL
    END as direction
FROM processed_market_data
WHERE (is_long_entry OR is_short_entry)
AND technical_indicators IS NOT NULL
LIMIT 10000
"""

df_indicators = pd.read_sql(query_indicators, conn)

# Преобразуем в числа
for col in ['rsi', 'adx', 'bb_position', 'volume_ratio', 'expected_return']:
    df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')

# Корреляция для LONG
df_long = df_indicators[df_indicators['direction'] == 'LONG']
if len(df_long) > 0:
    print("\n📊 Корреляция индикаторов с LONG returns:")
    for indicator in ['rsi', 'adx', 'bb_position', 'volume_ratio']:
        corr = df_long[indicator].corr(df_long['expected_return'])
        if not np.isnan(corr):
            print(f"   {indicator}: {corr:.3f}")

# Корреляция для SHORT
df_short = df_indicators[df_indicators['direction'] == 'SHORT']
if len(df_short) > 0:
    print("\n📊 Корреляция индикаторов с SHORT returns:")
    for indicator in ['rsi', 'adx', 'bb_position', 'volume_ratio']:
        corr = df_short[indicator].corr(df_short['expected_return'])
        if not np.isnan(corr):
            print(f"   {indicator}: {corr:.3f}")

# 6. Итоговые выводы
print("\n" + "="*80)
print("📌 ВЫВОДЫ:")
print("="*80)

# Проверяем качество данных
if df_stats['long_entries'][0] > 0 and df_stats['short_entries'][0] > 0:
    long_entry_rate = df_stats['long_entries'][0] / df_stats['total_bars'][0] * 100
    short_entry_rate = df_stats['short_entries'][0] / df_stats['total_bars'][0] * 100
    
    print(f"\n✅ Реалистичная частота входов:")
    print(f"   - LONG: {long_entry_rate:.2f}% (целевое: ~2%)")
    print(f"   - SHORT: {short_entry_rate:.2f}% (целевое: ~2%)")
    
    # Проверяем разницу между типами входов
    good_long = df_types[(df_types['direction'] == 'LONG') & (df_types['entry_type'] == 'good')]
    bad_long = df_types[(df_types['direction'] == 'LONG') & (df_types['entry_type'] == 'bad')]
    
    if len(good_long) > 0 and len(bad_long) > 0:
        good_return = good_long['avg_return'].values[0]
        bad_return = bad_long['avg_return'].values[0]
        
        if good_return > bad_return:
            print(f"\n✅ Логика типов входов работает корректно:")
            print(f"   - 'Good' LONG входы лучше 'bad' на {good_return - bad_return:.2f}%")
        else:
            print(f"\n⚠️ Проблема с логикой типов входов:")
            print(f"   - 'Bad' входы показывают лучший результат!")
    
    print(f"\n💡 Рекомендации для обучения:")
    print(f"   1. Используйте только бары с is_long_entry=True или is_short_entry=True")
    print(f"   2. Модель должна предсказывать expected_return")
    print(f"   3. Взвешивайте loss по типам входов (bad входы важнее для обучения)")
    print(f"   4. Минимальный порог для сигнала: ~0.5%")

conn.close()
print("\n" + "="*80)