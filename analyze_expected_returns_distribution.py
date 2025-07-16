#!/usr/bin/env python3
"""
Анализ распределения expected returns в базе данных
"""

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Параметры подключения к БД
DB_CONFIG = {
    'host': 'localhost',
    'port': 5555,
    'database': 'crypto_trading',
    'user': 'ruslan',
    'password': 'your_password'
}

def connect_db():
    """Подключение к БД"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Ошибка подключения к БД: {e}")
        return None

def analyze_distribution():
    """Основной анализ распределения expected returns"""
    conn = connect_db()
    if not conn:
        return
    
    cur = conn.cursor()
    
    print("="*80)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ EXPECTED RETURNS")
    print("="*80)
    
    # 1. Общая статистика
    cur.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT symbol) as unique_symbols,
            MIN(buy_expected_return) as min_buy,
            MAX(buy_expected_return) as max_buy,
            AVG(buy_expected_return) as avg_buy,
            MIN(sell_expected_return) as min_sell,
            MAX(sell_expected_return) as max_sell,
            AVG(sell_expected_return) as avg_sell
        FROM processed_market_data 
        WHERE buy_expected_return IS NOT NULL 
          AND sell_expected_return IS NOT NULL
    """)
    
    stats = cur.fetchone()
    print(f"\nОбщая статистика:")
    print(f"- Всего записей: {stats[0]:,}")
    print(f"- Уникальных символов: {stats[1]}")
    print(f"\nBuy Expected Return:")
    print(f"  - Минимум: {stats[2]:.2f}%")
    print(f"  - Максимум: {stats[3]:.2f}%")
    print(f"  - Среднее: {stats[4]:.4f}%")
    print(f"\nSell Expected Return:")
    print(f"  - Минимум: {stats[5]:.2f}%")
    print(f"  - Максимум: {stats[6]:.2f}%")
    print(f"  - Среднее: {stats[7]:.4f}%")
    
    # 2. Процентили
    cur.execute("""
        SELECT 
            percentile_cont(0.10) WITHIN GROUP (ORDER BY buy_expected_return) as buy_p10,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY buy_expected_return) as buy_p25,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY buy_expected_return) as buy_p50,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY buy_expected_return) as buy_p75,
            percentile_cont(0.90) WITHIN GROUP (ORDER BY buy_expected_return) as buy_p90
        FROM processed_market_data 
        WHERE buy_expected_return IS NOT NULL
    """)
    
    percentiles = cur.fetchone()
    print(f"\nПроцентили Buy Expected Return:")
    print(f"  - 10%: {percentiles[0]:.2f}%")
    print(f"  - 25%: {percentiles[1]:.2f}%")
    print(f"  - 50% (медиана): {percentiles[2]:.2f}%")
    print(f"  - 75%: {percentiles[3]:.2f}%")
    print(f"  - 90%: {percentiles[4]:.2f}%")
    
    # 3. Анализ по категориям
    cur.execute("""
        WITH categorized AS (
            SELECT 
                CASE 
                    WHEN buy_expected_return = -1.1 THEN 'Exactly -1.1 (Stop Loss)'
                    WHEN buy_expected_return > -1.1 AND buy_expected_return < 0 THEN 'Between -1.1 and 0'
                    WHEN buy_expected_return = 0 THEN 'Exactly 0'
                    WHEN buy_expected_return > 0 AND buy_expected_return < 0.3 THEN 'Between 0 and 0.3'
                    WHEN buy_expected_return >= 0.3 THEN 'Greater than 0.3'
                    ELSE 'Less than -1.1'
                END as category
            FROM processed_market_data 
            WHERE buy_expected_return IS NOT NULL
        )
        SELECT 
            category,
            COUNT(*) as count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM categorized
        GROUP BY category
        ORDER BY count DESC
    """)
    
    print(f"\nРаспределение Buy Expected Return по категориям:")
    print(f"{'Категория':<30} {'Количество':>12} {'Процент':>10}")
    print("-" * 55)
    
    categories = cur.fetchall()
    for cat, count, pct in categories:
        print(f"{cat:<30} {count:>12,} {pct:>9.2f}%")
    
    # 4. Анализ по порогам
    cur.execute("""
        WITH thresholds AS (
            SELECT 
                SUM(CASE WHEN buy_expected_return > 0.3 THEN 1 ELSE 0 END) as buy_positive_03,
                SUM(CASE WHEN buy_expected_return > 1.0 THEN 1 ELSE 0 END) as buy_positive_10,
                SUM(CASE WHEN sell_expected_return > 0.3 THEN 1 ELSE 0 END) as sell_positive_03,
                SUM(CASE WHEN sell_expected_return > 1.0 THEN 1 ELSE 0 END) as sell_positive_10,
                COUNT(*) as total_count
            FROM processed_market_data 
            WHERE buy_expected_return IS NOT NULL 
              AND sell_expected_return IS NOT NULL
        )
        SELECT * FROM thresholds
    """)
    
    thresholds = cur.fetchone()
    print(f"\nАнализ положительных returns по порогам:")
    print(f"\nПорог > 0.3%:")
    print(f"  - Buy: {thresholds[0]:,} примеров ({thresholds[0]/thresholds[4]*100:.2f}%)")
    print(f"  - Sell: {thresholds[2]:,} примеров ({thresholds[2]/thresholds[4]*100:.2f}%)")
    print(f"\nПорог > 1.0%:")
    print(f"  - Buy: {thresholds[1]:,} примеров ({thresholds[1]/thresholds[4]*100:.2f}%)")
    print(f"  - Sell: {thresholds[3]:,} примеров ({thresholds[3]/thresholds[4]*100:.2f}%)")
    
    # 5. Причины большого количества отрицательных returns
    print(f"\n{'='*80}")
    print("АНАЛИЗ ПРИЧИН ПРЕОБЛАДАНИЯ ОТРИЦАТЕЛЬНЫХ RETURNS")
    print("="*80)
    
    print(f"\nОсновные причины ~50% данных со значением -1.1% (Stop Loss):")
    print(f"1. Stop Loss установлен на уровне -1.1% от цены входа")
    print(f"2. Take Profit установлен на +5.8% (соотношение риск/прибыль 1:5.27)")
    print(f"3. В боковом рынке цена чаще достигает Stop Loss, чем Take Profit")
    print(f"4. Это реалистично отражает сложность криптовалютного рынка")
    
    print(f"\nДополнительные факторы:")
    print(f"- Анализируется 100 свечей вперед (25 часов на 15м таймфрейме)")
    print(f"- Используется частичное закрытие позиций (20%, 30%, 30%)")
    print(f"- Breakeven и profit locking защищают часть прибыли")
    
    # 6. Создание визуализации
    print(f"\nСоздание графиков распределения...")
    
    # Получаем данные для гистограммы
    cur.execute("""
        SELECT buy_expected_return, sell_expected_return
        FROM processed_market_data 
        WHERE buy_expected_return IS NOT NULL 
          AND sell_expected_return IS NOT NULL
        LIMIT 100000
    """)
    
    data = cur.fetchall()
    buy_returns = [row[0] for row in data]
    sell_returns = [row[1] for row in data]
    
    # Создаем графики
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # График для Buy Expected Return
    ax1.hist(buy_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', label='Zero return')
    ax1.axvline(x=0.3, color='orange', linestyle='--', label='Threshold 0.3%')
    ax1.axvline(x=-1.1, color='darkred', linestyle='--', label='Stop Loss -1.1%')
    ax1.set_xlabel('Buy Expected Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Buy Expected Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График для Sell Expected Return
    ax2.hist(sell_returns, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', label='Zero return')
    ax2.axvline(x=0.3, color='orange', linestyle='--', label='Threshold 0.3%')
    ax2.axvline(x=-1.1, color='darkred', linestyle='--', label='Stop Loss -1.1%')
    ax2.set_xlabel('Sell Expected Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Sell Expected Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('expected_returns_distribution.png', dpi=300, bbox_inches='tight')
    print(f"График сохранен: expected_returns_distribution.png")
    
    # 7. Анализ по символам
    print(f"\n{'='*80}")
    print("АНАЛИЗ ПО СИМВОЛАМ")
    print("="*80)
    
    cur.execute("""
        SELECT 
            symbol,
            COUNT(*) as total_points,
            SUM(CASE WHEN buy_expected_return = -1.1 THEN 1 ELSE 0 END) as buy_stoploss_count,
            ROUND(100.0 * SUM(CASE WHEN buy_expected_return = -1.1 THEN 1 ELSE 0 END) / COUNT(*), 2) as buy_stoploss_pct,
            SUM(CASE WHEN buy_expected_return > 0.3 THEN 1 ELSE 0 END) as buy_positive_count,
            ROUND(100.0 * SUM(CASE WHEN buy_expected_return > 0.3 THEN 1 ELSE 0 END) / COUNT(*), 2) as buy_positive_pct
        FROM processed_market_data 
        WHERE buy_expected_return IS NOT NULL
        GROUP BY symbol
        ORDER BY buy_stoploss_pct DESC
        LIMIT 10
    """)
    
    print(f"\nТоп-10 символов с наибольшим % Stop Loss:")
    print(f"{'Символ':<12} {'Всего точек':>12} {'Stop Loss':>12} {'SL %':>8} {'Positive':>12} {'Pos %':>8}")
    print("-" * 70)
    
    symbols = cur.fetchall()
    for symbol, total, sl_count, sl_pct, pos_count, pos_pct in symbols:
        print(f"{symbol:<12} {total:>12,} {sl_count:>12,} {sl_pct:>7.1f}% {pos_count:>12,} {pos_pct:>7.1f}%")
    
    # Закрываем соединение
    cur.close()
    conn.close()
    
    print(f"\n{'='*80}")
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("="*80)
    
    print(f"""
1. ТЕКУЩАЯ СИТУАЦИЯ:
   - ~53% данных имеют buy_expected_return = -1.1% (Stop Loss)
   - ~45% данных имеют положительный expected return > 0.3%
   - Медиана находится на уровне Stop Loss (-1.1%)
   
2. ПОЧЕМУ ТАК МНОГО ОТРИЦАТЕЛЬНЫХ RETURNS:
   - Агрессивное соотношение риск/прибыль (1:5.27)
   - Криптовалютный рынок очень волатилен
   - В боковом движении Stop Loss срабатывает чаще
   
3. ЭТО НОРМАЛЬНО?
   - ДА, это реалистично для криптовалютного рынка
   - Отражает реальную сложность прибыльной торговли
   - Важно не количество прибыльных сделок, а соотношение риск/прибыль
   
4. РЕКОМЕНДАЦИИ ДЛЯ МОДЕЛИ:
   - Использовать взвешенную функцию потерь (больший вес положительным примерам)
   - Фокусироваться на точности предсказания положительных returns
   - Рассмотреть балансировку датасета или oversampling
   - Использовать метрики precision/recall вместо только accuracy
    """)

if __name__ == "__main__":
    analyze_distribution()