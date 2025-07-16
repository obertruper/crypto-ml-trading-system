#!/usr/bin/env python3
"""
Анализ и визуализация случайных точек входа
Показывает реальную статистику случайной торговли
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
print("🎲 АНАЛИЗ СЛУЧАЙНЫХ ТОЧЕК ВХОДА")
print("="*80)

# Подключение к БД
conn = psycopg2.connect(**db_config)

# 1. Общая статистика
query = """
SELECT 
    COUNT(*) as total_bars,
    SUM(CASE WHEN is_long_entry THEN 1 ELSE 0 END) as long_entries,
    SUM(CASE WHEN is_short_entry THEN 1 ELSE 0 END) as short_entries,
    AVG(buy_expected_return) as avg_all_buy,
    AVG(sell_expected_return) as avg_all_sell,
    AVG(CASE WHEN is_long_entry THEN buy_expected_return END) as avg_random_long,
    AVG(CASE WHEN is_short_entry THEN sell_expected_return END) as avg_random_short,
    COUNT(DISTINCT buy_expected_return) as unique_buy,
    COUNT(DISTINCT sell_expected_return) as unique_sell
FROM processed_market_data
"""

df_stats = pd.read_sql(query, conn)
print("\n📊 Общая статистика:")
print(f"   Всего баров: {df_stats['total_bars'][0]:,}")
print(f"   Уникальность buy_expected_return: {df_stats['unique_buy'][0]/df_stats['total_bars'][0]*100:.1f}%")
print(f"   Уникальность sell_expected_return: {df_stats['unique_sell'][0]/df_stats['total_bars'][0]*100:.1f}%")

print(f"\n🎲 Случайные входы:")
print(f"   LONG входов: {df_stats['long_entries'][0]:,} ({df_stats['long_entries'][0]/df_stats['total_bars'][0]*100:.1f}%)")
print(f"   SHORT входов: {df_stats['short_entries'][0]:,} ({df_stats['short_entries'][0]/df_stats['total_bars'][0]*100:.1f}%)")

print(f"\n📈 Средние expected returns:")
print(f"   Все бары BUY: {df_stats['avg_all_buy'][0]:.3f}%")
print(f"   Все бары SELL: {df_stats['avg_all_sell'][0]:.3f}%")
print(f"   Случайные LONG: {df_stats['avg_random_long'][0]:.3f}%")
print(f"   Случайные SHORT: {df_stats['avg_random_short'][0]:.3f}%")

# 2. Win rate анализ
query_winrate = """
SELECT 
    -- Все бары
    SUM(CASE WHEN buy_expected_return > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100 as all_buy_winrate,
    SUM(CASE WHEN sell_expected_return > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100 as all_sell_winrate,
    -- Случайные входы
    SUM(CASE WHEN is_long_entry AND buy_expected_return > 0 THEN 1 ELSE 0 END)::FLOAT / 
        NULLIF(SUM(CASE WHEN is_long_entry THEN 1 ELSE 0 END), 0) * 100 as random_long_winrate,
    SUM(CASE WHEN is_short_entry AND sell_expected_return > 0 THEN 1 ELSE 0 END)::FLOAT / 
        NULLIF(SUM(CASE WHEN is_short_entry THEN 1 ELSE 0 END), 0) * 100 as random_short_winrate
FROM processed_market_data
"""

df_wr = pd.read_sql(query_winrate, conn)
print(f"\n🎯 Win Rate:")
print(f"   Все бары BUY: {df_wr['all_buy_winrate'][0]:.1f}%")
print(f"   Все бары SELL: {df_wr['all_sell_winrate'][0]:.1f}%")
print(f"   Случайные LONG: {df_wr['random_long_winrate'][0]:.1f}%")
print(f"   Случайные SHORT: {df_wr['random_short_winrate'][0]:.1f}%")

# 3. Распределение returns
query_dist = """
SELECT 
    buy_expected_return,
    sell_expected_return,
    is_long_entry,
    is_short_entry
FROM processed_market_data
WHERE buy_expected_return IS NOT NULL
LIMIT 100000
"""

df_dist = pd.read_sql(query_dist, conn)

# 4. Анализ по типам выхода
query_exits = """
WITH exit_analysis AS (
    SELECT 
        CASE 
            WHEN is_long_entry AND buy_expected_return <= -1.0 THEN 'LONG Stop Loss'
            WHEN is_long_entry AND buy_expected_return >= 5.0 THEN 'LONG Take Profit'
            WHEN is_long_entry AND buy_expected_return > 0 THEN 'LONG Partial/Timeout Profit'
            WHEN is_long_entry THEN 'LONG Timeout Loss'
            WHEN is_short_entry AND sell_expected_return <= -1.0 THEN 'SHORT Stop Loss'
            WHEN is_short_entry AND sell_expected_return >= 5.0 THEN 'SHORT Take Profit'
            WHEN is_short_entry AND sell_expected_return > 0 THEN 'SHORT Partial/Timeout Profit'
            WHEN is_short_entry THEN 'SHORT Timeout Loss'
        END as exit_type,
        CASE 
            WHEN is_long_entry THEN buy_expected_return
            WHEN is_short_entry THEN sell_expected_return
        END as return_pct
    FROM processed_market_data
    WHERE is_long_entry OR is_short_entry
)
SELECT 
    exit_type,
    COUNT(*) as count,
    AVG(return_pct) as avg_return
FROM exit_analysis
WHERE exit_type IS NOT NULL
GROUP BY exit_type
ORDER BY exit_type
"""

df_exits = pd.read_sql(query_exits, conn)
print("\n📊 Анализ выходов из случайных позиций:")
for _, row in df_exits.iterrows():
    print(f"   {row['exit_type']}: {row['count']} ({row['avg_return']:.2f}%)")

# 5. Создание визуализаций
print("\n📈 Создание графиков...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Анализ случайных точек входа', fontsize=16)

# График 1: Распределение всех returns vs случайных
ax1 = axes[0, 0]
# Все бары
ax1.hist(df_dist['buy_expected_return'], bins=100, alpha=0.4, label='Все бары', density=True, color='blue')
# Случайные входы
random_long = df_dist[df_dist['is_long_entry']]['buy_expected_return']
ax1.hist(random_long, bins=50, alpha=0.7, label=f'Случайные LONG (n={len(random_long)})', density=True, color='red')
ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax1.set_xlabel('Buy Expected Return (%)')
ax1.set_ylabel('Density')
ax1.set_title('Распределение BUY returns')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2, 6)

# График 2: То же для SELL
ax2 = axes[0, 1]
ax2.hist(df_dist['sell_expected_return'], bins=100, alpha=0.4, label='Все бары', density=True, color='blue')
random_short = df_dist[df_dist['is_short_entry']]['sell_expected_return']
ax2.hist(random_short, bins=50, alpha=0.7, label=f'Случайные SHORT (n={len(random_short)})', density=True, color='red')
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Sell Expected Return (%)')
ax2.set_ylabel('Density')
ax2.set_title('Распределение SELL returns')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2, 6)

# График 3: Win Rate сравнение
ax3 = axes[0, 2]
categories = ['Все\nBUY', 'Случайные\nLONG', 'Все\nSELL', 'Случайные\nSHORT']
win_rates = [
    df_wr['all_buy_winrate'][0],
    df_wr['random_long_winrate'][0],
    df_wr['all_sell_winrate'][0],
    df_wr['random_short_winrate'][0]
]
colors = ['lightblue', 'darkblue', 'lightcoral', 'darkred']
bars = ax3.bar(categories, win_rates, color=colors)
ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
ax3.set_ylabel('Win Rate (%)')
ax3.set_title('Win Rate: Все бары vs Случайные входы')
ax3.grid(True, alpha=0.3, axis='y')

# Добавляем значения на бары
for bar, wr in zip(bars, win_rates):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{wr:.1f}%', ha='center', va='bottom')

# График 4: Распределение по диапазонам returns
ax4 = axes[1, 0]
ranges = [(-100, -1), (-1, 0), (0, 1), (1, 3), (3, 10)]
range_labels = ['< -1%', '-1% to 0%', '0% to 1%', '1% to 3%', '> 3%']

# Для случайных LONG
long_counts = []
for low, high in ranges:
    count = len(random_long[(random_long >= low) & (random_long < high)])
    long_counts.append(count)

# Для случайных SHORT
short_counts = []
for low, high in ranges:
    count = len(random_short[(random_short >= low) & (random_short < high)])
    short_counts.append(count)

x = np.arange(len(range_labels))
width = 0.35

bars1 = ax4.bar(x - width/2, long_counts, width, label='LONG', color='blue')
bars2 = ax4.bar(x + width/2, short_counts, width, label='SHORT', color='red')

ax4.set_xlabel('Return Range')
ax4.set_ylabel('Count')
ax4.set_title('Распределение случайных входов по диапазонам return')
ax4.set_xticks(x)
ax4.set_xticklabels(range_labels)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# График 5: Кумулятивный return
ax5 = axes[1, 1]
# Сортируем по времени
query_cumulative = """
SELECT 
    timestamp,
    CASE WHEN is_long_entry THEN buy_expected_return ELSE 0 END as long_return,
    CASE WHEN is_short_entry THEN sell_expected_return ELSE 0 END as short_return
FROM processed_market_data
WHERE is_long_entry OR is_short_entry
ORDER BY timestamp
LIMIT 10000
"""
df_cum = pd.read_sql(query_cumulative, conn)
df_cum['cum_long'] = df_cum['long_return'].cumsum()
df_cum['cum_short'] = df_cum['short_return'].cumsum()
df_cum['cum_total'] = (df_cum['long_return'] + df_cum['short_return']).cumsum()

ax5.plot(df_cum.index, df_cum['cum_long'], label='LONG', color='blue', alpha=0.7)
ax5.plot(df_cum.index, df_cum['cum_short'], label='SHORT', color='red', alpha=0.7)
ax5.plot(df_cum.index, df_cum['cum_total'], label='Total', color='green', linewidth=2)
ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax5.set_xlabel('Trade Number')
ax5.set_ylabel('Cumulative Return (%)')
ax5.set_title('Кумулятивный return случайных входов (первые 10k)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# График 6: Статистика по символам
ax6 = axes[1, 2]
query_symbols = """
SELECT 
    symbol,
    COUNT(CASE WHEN is_long_entry THEN 1 END) as long_entries,
    COUNT(CASE WHEN is_short_entry THEN 1 END) as short_entries,
    AVG(CASE WHEN is_long_entry THEN buy_expected_return END) as avg_long,
    AVG(CASE WHEN is_short_entry THEN sell_expected_return END) as avg_short
FROM processed_market_data
GROUP BY symbol
HAVING COUNT(CASE WHEN is_long_entry THEN 1 END) > 100
ORDER BY (avg_long + avg_short) / 2 DESC
LIMIT 10
"""
df_symbols = pd.read_sql(query_symbols, conn)

symbols = df_symbols['symbol'].tolist()
avg_returns = ((df_symbols['avg_long'] + df_symbols['avg_short']) / 2).tolist()

bars = ax6.barh(symbols, avg_returns)
ax6.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax6.set_xlabel('Average Return (%)')
ax6.set_title('Топ-10 символов по среднему return случайных входов')
ax6.grid(True, alpha=0.3, axis='x')

# Раскрашиваем бары
for bar, ret in zip(bars, avg_returns):
    bar.set_color('green' if ret > 0 else 'red')

plt.tight_layout()
plt.savefig('random_entries_analysis.png', dpi=150, bbox_inches='tight')
print("✅ График сохранен: random_entries_analysis.png")

# 7. Итоговые выводы
print("\n" + "="*80)
print("📌 ВЫВОДЫ:")
print("="*80)

avg_random_return = (df_stats['avg_random_long'][0] + df_stats['avg_random_short'][0]) / 2
avg_random_wr = (df_wr['random_long_winrate'][0] + df_wr['random_short_winrate'][0]) / 2

print(f"\n🎲 Случайная торговля:")
print(f"   - Средний return: {avg_random_return:.3f}%")
print(f"   - Средний win rate: {avg_random_wr:.1f}%")

if avg_random_wr < 40:
    print(f"\n✅ Реалистичные данные!")
    print(f"   - Win rate случайных входов ~{avg_random_wr:.0f}% (ожидаемое: 30-40%)")
    print(f"   - Это подтверждает, что случайные входы убыточны")
    print(f"   - Модель должна научиться улучшить эти результаты")
else:
    print(f"\n⚠️ Данные слишком оптимистичные!")
    print(f"   - Win rate {avg_random_wr:.0f}% слишком высокий для случайных входов")
    print(f"   - Проверьте параметры риск-профиля")

print(f"\n💡 Задача для модели:")
print(f"   1. Научиться предсказывать expected_return по индикаторам")
print(f"   2. Входить только когда predicted_return > 1.5%")
print(f"   3. Целевой win rate > 60% (сейчас случайный: {avg_random_wr:.1f}%)")
print(f"   4. Целевой средний return > 1% (сейчас случайный: {avg_random_return:.3f}%)")

# Проверка качества данных для обучения
if df_stats['unique_buy'][0]/df_stats['total_bars'][0] > 0.8 and \
   df_stats['unique_sell'][0]/df_stats['total_bars'][0] > 0.8:
    print(f"\n✅ Данные готовы для обучения!")
    print(f"   - Высокая уникальность expected_returns")
    print(f"   - Модель будет обучаться на всех барах")
    print(f"   - Случайные входы служат для валидации")
else:
    print(f"\n⚠️ Низкая уникальность данных!")
    print(f"   - Требуется пересчет с исправленной логикой")

conn.close()
print("\n" + "="*80)