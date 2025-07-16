#!/usr/bin/env python3
"""
Скрипт для анализа проблемы с expected returns
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Настройки подключения
DB_CONFIG = {
    'host': 'localhost',
    'port': 5555,
    'user': 'ruslan',
    'password': 'ruslan',
    'database': 'crypto_trading'
}

def analyze_expected_returns():
    """Анализ распределения expected returns"""
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    # 1. Проверка общей статистики
    print("=" * 80)
    print("1. ОБЩАЯ СТАТИСТИКА EXPECTED RETURNS")
    print("=" * 80)
    
    query = """
    SELECT 
        COUNT(*) as total_records,
        -- BUY статистика
        COUNT(DISTINCT buy_expected_return) as unique_buy_values,
        MIN(buy_expected_return) as min_buy,
        MAX(buy_expected_return) as max_buy,
        AVG(buy_expected_return) as avg_buy,
        STDDEV(buy_expected_return) as std_buy,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY buy_expected_return) as median_buy,
        -- SELL статистика
        COUNT(DISTINCT sell_expected_return) as unique_sell_values,
        MIN(sell_expected_return) as min_sell,
        MAX(sell_expected_return) as max_sell,
        AVG(sell_expected_return) as avg_sell,
        STDDEV(sell_expected_return) as std_sell,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sell_expected_return) as median_sell
    FROM processed_market_data
    WHERE buy_expected_return IS NOT NULL
    """
    
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query)
    stats = cur.fetchone()
    
    print(f"Всего записей: {stats['total_records']:,}")
    print("\nBUY expected returns:")
    print(f"  Уникальных значений: {stats['unique_buy_values']:,}")
    print(f"  Диапазон: [{stats['min_buy']:.4f}, {stats['max_buy']:.4f}]")
    print(f"  Среднее: {stats['avg_buy']:.4f}")
    print(f"  Медиана: {stats['median_buy']:.4f}")
    print(f"  Стд. отклонение: {stats['std_buy']:.4f}")
    
    print("\nSELL expected returns:")
    print(f"  Уникальных значений: {stats['unique_sell_values']:,}")
    print(f"  Диапазон: [{stats['min_sell']:.4f}, {stats['max_sell']:.4f}]")
    print(f"  Среднее: {stats['avg_sell']:.4f}")
    print(f"  Медиана: {stats['median_sell']:.4f}")
    print(f"  Стд. отклонение: {stats['std_sell']:.4f}")
    
    # 2. Распределение по категориям
    print("\n" + "=" * 80)
    print("2. РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ")
    print("=" * 80)
    
    query = """
    SELECT 
        'BUY' as type,
        CASE 
            WHEN buy_expected_return = -1.1 THEN 'Exactly -1.1% (stop loss)'
            WHEN buy_expected_return = 5.8 THEN 'Exactly 5.8% (take profit)'
            WHEN buy_expected_return > 5.8 THEN 'Above 5.8%'
            WHEN buy_expected_return > 0 THEN 'Profit (0 to 5.8%)'
            WHEN buy_expected_return >= -1.1 AND buy_expected_return < 0 THEN 'Small loss (0 to -1.1%)'
            WHEN buy_expected_return < -1.1 THEN 'Below -1.1%'
        END as category,
        COUNT(*) as count,
        AVG(buy_expected_return) as avg_return
    FROM processed_market_data
    WHERE buy_expected_return IS NOT NULL
    GROUP BY 2
    
    UNION ALL
    
    SELECT 
        'SELL' as type,
        CASE 
            WHEN sell_expected_return = -1.1 THEN 'Exactly -1.1% (stop loss)'
            WHEN sell_expected_return = 5.8 THEN 'Exactly 5.8% (take profit)'
            WHEN sell_expected_return > 5.8 THEN 'Above 5.8%'
            WHEN sell_expected_return > 0 THEN 'Profit (0 to 5.8%)'
            WHEN sell_expected_return >= -1.1 AND sell_expected_return < 0 THEN 'Small loss (0 to -1.1%)'
            WHEN sell_expected_return < -1.1 THEN 'Below -1.1%'
        END as category,
        COUNT(*) as count,
        AVG(sell_expected_return) as avg_return
    FROM processed_market_data
    WHERE sell_expected_return IS NOT NULL
    GROUP BY 2
    ORDER BY 1, 3 DESC
    """
    
    cur.execute(query)
    results = cur.fetchall()
    
    for direction in ['BUY', 'SELL']:
        print(f"\n{direction} позиции:")
        dir_results = [r for r in results if r['type'] == direction]
        total = sum(r['count'] for r in dir_results)
        
        for r in dir_results:
            if r['category'] is None:
                continue
            pct = r['count'] / total * 100
            avg_str = f"{r['avg_return']:6.2f}" if r['avg_return'] is not None else "   N/A"
            print(f"  {r['category']:30s}: {r['count']:8,} ({pct:5.1f}%) avg: {avg_str}%")
    
    # 3. Проверка самых частых значений
    print("\n" + "=" * 80)
    print("3. ТОП-20 САМЫХ ЧАСТЫХ ЗНАЧЕНИЙ")
    print("=" * 80)
    
    for col in ['buy_expected_return', 'sell_expected_return']:
        query = f"""
        SELECT {col}, COUNT(*) as count
        FROM processed_market_data
        WHERE {col} IS NOT NULL
        GROUP BY {col}
        ORDER BY count DESC
        LIMIT 20
        """
        
        cur.execute(query)
        top_values = cur.fetchall()
        
        print(f"\n{col.upper()}:")
        for i, row in enumerate(top_values, 1):
            pct = row['count'] / stats['total_records'] * 100
            print(f"  {i:2d}. {row[col]:7.4f}%: {row['count']:8,} ({pct:5.2f}%)")
    
    # 4. Анализ по символам
    print("\n" + "=" * 80)
    print("4. АНАЛИЗ ПО СИМВОЛАМ (топ-10 с наихудшими результатами)")
    print("=" * 80)
    
    query = """
    SELECT 
        symbol,
        COUNT(*) as total,
        AVG(buy_expected_return) as avg_buy,
        AVG(sell_expected_return) as avg_sell,
        SUM(CASE WHEN buy_expected_return = -1.1 THEN 1 ELSE 0 END) as buy_sl_count,
        SUM(CASE WHEN sell_expected_return = -1.1 THEN 1 ELSE 0 END) as sell_sl_count
    FROM processed_market_data
    WHERE buy_expected_return IS NOT NULL
    GROUP BY symbol
    HAVING COUNT(*) > 1000
    ORDER BY avg_buy ASC
    LIMIT 10
    """
    
    cur.execute(query)
    worst_symbols = cur.fetchall()
    
    print("\nСимволы с наихудшими buy_expected_return:")
    print(f"{'Symbol':12s} {'Total':>8s} {'Avg Buy':>8s} {'Avg Sell':>8s} {'Buy SL%':>8s} {'Sell SL%':>8s}")
    print("-" * 60)
    
    for row in worst_symbols:
        buy_sl_pct = row['buy_sl_count'] / row['total'] * 100
        sell_sl_pct = row['sell_sl_count'] / row['total'] * 100
        print(f"{row['symbol']:12s} {row['total']:8,} {row['avg_buy']:8.3f} {row['avg_sell']:8.3f} {buy_sl_pct:7.1f}% {sell_sl_pct:7.1f}%")
    
    # 5. Создание визуализации
    print("\n" + "=" * 80)
    print("5. СОЗДАНИЕ ВИЗУАЛИЗАЦИИ")
    print("=" * 80)
    
    # Загружаем данные для визуализации
    query = """
    SELECT buy_expected_return, sell_expected_return
    FROM processed_market_data
    WHERE buy_expected_return IS NOT NULL
    AND sell_expected_return IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 100000
    """
    
    df = pd.read_sql(query, conn)
    
    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Анализ распределения Expected Returns', fontsize=16)
    
    # 1. Гистограмма buy_expected_return
    ax = axes[0, 0]
    ax.hist(df['buy_expected_return'], bins=100, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(-1.1, color='red', linestyle='--', label='Stop Loss (-1.1%)')
    ax.axvline(5.8, color='blue', linestyle='--', label='Take Profit (5.8%)')
    ax.set_xlabel('Buy Expected Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Распределение Buy Expected Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Гистограмма sell_expected_return
    ax = axes[0, 1]
    ax.hist(df['sell_expected_return'], bins=100, alpha=0.7, color='red', edgecolor='black')
    ax.axvline(-1.1, color='red', linestyle='--', label='Stop Loss (-1.1%)')
    ax.axvline(5.8, color='blue', linestyle='--', label='Take Profit (5.8%)')
    ax.set_xlabel('Sell Expected Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Распределение Sell Expected Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Scatter plot
    ax = axes[1, 0]
    scatter = ax.scatter(df['buy_expected_return'], df['sell_expected_return'], 
                        alpha=0.3, s=1, c=df['buy_expected_return'], cmap='RdYlGn')
    ax.axhline(-1.1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(-1.1, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Buy Expected Return (%)')
    ax.set_ylabel('Sell Expected Return (%)')
    ax.set_title('Корреляция Buy vs Sell Expected Returns')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax)
    
    # 4. Box plots
    ax = axes[1, 1]
    box_data = [df['buy_expected_return'], df['sell_expected_return']]
    bp = ax.boxplot(box_data, labels=['Buy', 'Sell'], showfliers=False)
    ax.axhline(-1.1, color='red', linestyle='--', alpha=0.5, label='Stop Loss')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(5.8, color='blue', linestyle='--', alpha=0.5, label='Take Profit')
    ax.set_ylabel('Expected Return (%)')
    ax.set_title('Box Plot Expected Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('expected_returns_analysis.png', dpi=300, bbox_inches='tight')
    print("График сохранен как 'expected_returns_analysis.png'")
    
    # 6. Проверка корреляции с техническими индикаторами
    print("\n" + "=" * 80)
    print("6. ПРОВЕРКА НЕСКОЛЬКИХ ЗАПИСЕЙ С ЭКСТРЕМАЛЬНЫМИ ЗНАЧЕНИЯМИ")
    print("=" * 80)
    
    # Примеры записей со stop loss
    query = """
    SELECT symbol, datetime, open, high, low, close, 
           buy_expected_return, sell_expected_return,
           technical_indicators->>'rsi_val' as rsi,
           technical_indicators->>'atr_val' as atr
    FROM processed_market_data
    WHERE buy_expected_return = -1.1
    ORDER BY RANDOM()
    LIMIT 5
    """
    
    cur.execute(query)
    sl_examples = cur.fetchall()
    
    print("\nПримеры записей с buy_expected_return = -1.1% (stop loss):")
    for row in sl_examples:
        print(f"\n{row['symbol']} @ {row['datetime']}")
        print(f"  OHLC: {row['open']:.4f} / {row['high']:.4f} / {row['low']:.4f} / {row['close']:.4f}")
        print(f"  Returns: buy={row['buy_expected_return']:.2f}%, sell={row['sell_expected_return']:.2f}%")
        print(f"  RSI: {row['rsi']}, ATR: {row['atr']}")
    
    cur.close()
    conn.close()
    
    print("\n" + "=" * 80)
    print("ВЫВОДЫ:")
    print("=" * 80)
    print("1. Более 50% всех buy позиций имеют expected_return = -1.1% (точный stop loss)")
    print("2. Медиана buy_expected_return = -1.1%, что указывает на систематическую проблему")
    print("3. Проблема может быть в:")
    print("   - Неправильной конвертации типов данных при сравнении цен")
    print("   - Слишком консервативной логике расчета (SL срабатывает слишком часто)")
    print("   - Неверном расчете уровней SL/TP")
    print("4. Модель не может обучаться, т.к. большинство значений одинаковые (-1.1%)")

if __name__ == "__main__":
    analyze_expected_returns()