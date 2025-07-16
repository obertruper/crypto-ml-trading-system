#!/usr/bin/env python3
"""
Скрипт для проверки проблем с загруженными данными.
Особое внимание уделяется 1000PEPEUSDT и аномальным значениям.
"""

import psycopg2
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate
import numpy as np

# Настройки подключения к БД
DB_CONFIG = {
    'dbname': 'crypto_trading',
    'user': 'ruslan',
    'password': 'ruslan',
    'host': 'localhost',
    'port': 5555
}

def check_symbol_data(conn, symbol):
    """Детальная проверка данных для конкретного символа"""
    print(f"\n{'='*80}")
    print(f"Детальная проверка символа: {symbol}")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    
    # 1. Общая статистика
    cursor.execute("""
        SELECT 
            COUNT(*) as total_records,
            to_timestamp(MIN(timestamp)/1000) as start_date,
            to_timestamp(MAX(timestamp)/1000) as end_date,
            MIN(open) as min_open,
            MAX(open) as max_open,
            MIN(high) as min_high,
            MAX(high) as max_high,
            MIN(low) as min_low,
            MAX(low) as max_low,
            MIN(close) as min_close,
            MAX(close) as max_close,
            MIN(volume) as min_volume,
            MAX(volume) as max_volume,
            AVG(volume) as avg_volume
        FROM raw_market_data
        WHERE symbol = %s
    """, (symbol,))
    
    stats = cursor.fetchone()
    
    if stats[0] == 0:
        print(f"[ОШИБКА] Нет данных для символа {symbol}")
        return
    
    print(f"\nОбщая статистика:")
    print(f"Количество записей: {stats[0]:,}")
    print(f"Период: с {stats[1]} по {stats[2]}")
    print(f"Длительность: {(stats[2] - stats[1]).days} дней")
    
    print(f"\nДиапазоны цен:")
    print(f"Open:  ${stats[3]:.8f} - ${stats[4]:.8f}")
    print(f"High:  ${stats[5]:.8f} - ${stats[6]:.8f}")
    print(f"Low:   ${stats[7]:.8f} - ${stats[8]:.8f}")
    print(f"Close: ${stats[9]:.8f} - ${stats[10]:.8f}")
    
    print(f"\nОбъемы:")
    print(f"Min: {stats[11]:,.2f}")
    print(f"Max: {stats[12]:,.2f}")
    print(f"Avg: {stats[13]:,.2f}")
    
    # 2. Проверка на нулевые и аномальные цены
    cursor.execute("""
        SELECT COUNT(*) 
        FROM raw_market_data 
        WHERE symbol = %s AND (open = 0 OR high = 0 OR low = 0 OR close = 0)
    """, (symbol,))
    zero_prices = cursor.fetchone()[0]
    
    if zero_prices > 0:
        print(f"\n[ВНИМАНИЕ] Найдено {zero_prices} записей с нулевыми ценами!")
        
        # Показать примеры
        cursor.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM raw_market_data 
            WHERE symbol = %s AND (open = 0 OR high = 0 OR low = 0 OR close = 0)
            ORDER BY timestamp
            LIMIT 10
        """, (symbol,))
        
        zero_examples = cursor.fetchall()
        print("\nПримеры записей с нулевыми ценами:")
        headers = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        print(tabulate(zero_examples, headers=headers, floatfmt=".8f"))
    
    # 3. Проверка на очень низкие цены (< $0.0000001)
    cursor.execute("""
        SELECT COUNT(*) 
        FROM raw_market_data 
        WHERE symbol = %s AND (open < 0.0000001 OR low < 0.0000001)
    """, (symbol,))
    very_low_prices = cursor.fetchone()[0]
    
    if very_low_prices > 0:
        print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Найдено {very_low_prices} записей с очень низкими ценами (< $0.0000001)")
    
    # 4. Проверка на пропуски во времени
    cursor.execute("""
        WITH time_gaps AS (
            SELECT 
                timestamp,
                LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) / 1000 / 60 as gap_minutes
            FROM raw_market_data
            WHERE symbol = %s
        )
        SELECT 
            COUNT(*) as gap_count,
            MAX(gap_minutes) as max_gap_minutes,
            AVG(gap_minutes) as avg_gap_minutes
        FROM time_gaps
        WHERE gap_minutes > 15
    """, (symbol,))
    
    gaps = cursor.fetchone()
    if gaps[0] and gaps[0] > 0:
        print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Пропуски в данных:")
        print(f"Количество пропусков (> 15 мин): {gaps[0]}")
        print(f"Максимальный пропуск: {gaps[1]:.1f} минут")
        print(f"Средний пропуск: {gaps[2]:.1f} минут")
        
        # Показать самые большие пропуски
        cursor.execute("""
            WITH time_gaps AS (
                SELECT 
                    timestamp,
                    LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                    (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) / 1000 / 60 as gap_minutes
                FROM raw_market_data
                WHERE symbol = %s
            )
            SELECT 
                to_timestamp(prev_timestamp/1000) as from_time,
                to_timestamp(timestamp/1000) as to_time,
                gap_minutes
            FROM time_gaps
            WHERE gap_minutes > 60
            ORDER BY gap_minutes DESC
            LIMIT 5
        """, (symbol,))
        
        big_gaps = cursor.fetchall()
        if big_gaps:
            print("\nСамые большие пропуски:")
            formatted_gaps = []
            for from_time, to_time, gap_minutes in big_gaps:
                formatted_gaps.append([
                    from_time.strftime('%Y-%m-%d %H:%M'),
                    to_time.strftime('%Y-%m-%d %H:%M'),
                    f"{gap_minutes:.1f} минут"
                ])
            headers = ['От', 'До', 'Пропуск']
            print(tabulate(formatted_gaps, headers=headers))
    
    # 5. Проверка консистентности данных
    cursor.execute("""
        SELECT COUNT(*)
        FROM raw_market_data
        WHERE symbol = %s AND (low > high OR open > high OR open < low OR close > high OR close < low)
    """, (symbol,))
    
    inconsistent = cursor.fetchone()[0]
    if inconsistent > 0:
        print(f"\n[ВНИМАНИЕ] Найдено {inconsistent} записей с некорректными OHLC данными!")
    
    # 6. Распределение цен по дням недели
    cursor.execute("""
        SELECT 
            EXTRACT(DOW FROM to_timestamp(timestamp/1000)) as day_of_week,
            COUNT(*) as count,
            AVG(close) as avg_close
        FROM raw_market_data
        WHERE symbol = %s
        GROUP BY day_of_week
        ORDER BY day_of_week
    """, (symbol,))
    
    dow_stats = cursor.fetchall()
    print(f"\nРаспределение по дням недели:")
    days = ['Вс', 'Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб']
    dow_data = []
    for dow, count, avg_close in dow_stats:
        dow_data.append([days[int(dow)], count, f"${avg_close:.8f}"])
    print(tabulate(dow_data, headers=['День', 'Записей', 'Средняя цена']))
    
    cursor.close()

def check_all_symbols(conn):
    """Проверка всех загруженных символов"""
    print(f"\n{'='*80}")
    print("Общая статистика по всем символам")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            symbol,
            COUNT(*) as records,
            to_timestamp(MIN(timestamp)/1000) as start_date,
            to_timestamp(MAX(timestamp)/1000) as end_date,
            MIN(low) as min_price,
            MAX(high) as max_price,
            AVG(volume) as avg_volume,
            SUM(CASE WHEN open = 0 OR low = 0 THEN 1 ELSE 0 END) as zero_prices
        FROM raw_market_data
        GROUP BY symbol
        ORDER BY symbol
    """)
    
    all_stats = cursor.fetchall()
    
    table_data = []
    problematic_symbols = []
    
    for symbol, records, start, end, min_price, max_price, avg_vol, zero_prices in all_stats:
        if start and end:
            days = (end - start).days
        else:
            days = 0
        
        # Определяем проблемные символы
        issues = []
        if zero_prices > 0:
            issues.append("[НУЛИ]")
        if min_price < 0.0000001:
            issues.append("[НИЗКИЕ]")
        if records < 1000:
            issues.append("[МАЛО]")
        
        if issues:
            problematic_symbols.append((symbol, issues))
        
        status = " ".join(issues) if issues else "OK"
        
        table_data.append([
            symbol,
            f"{records:,}",
            f"{days} дн",
            f"${min_price:.8f}",
            f"${max_price:.2f}",
            f"{avg_vol:,.0f}",
            zero_prices if zero_prices > 0 else "-",
            status
        ])
    
    headers = ['Символ', 'Записей', 'Период', 'Min цена', 'Max цена', 'Avg объем', 'Нули', 'Статус']
    print(tabulate(table_data, headers=headers))
    
    if problematic_symbols:
        print(f"\n[ПРОБЛЕМНЫЕ СИМВОЛЫ]:")
        for symbol, issues in problematic_symbols:
            print(f"  • {symbol}: {', '.join(issues)}")
    
    # Общая статистика
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT symbol) as total_symbols,
            COUNT(*) as total_records,
            to_timestamp(MIN(timestamp)/1000) as global_start,
            to_timestamp(MAX(timestamp)/1000) as global_end
        FROM raw_market_data
    """)
    
    global_stats = cursor.fetchone()
    print(f"\nИтого:")
    print(f"Символов: {global_stats[0]}")
    print(f"Всего записей: {global_stats[1]:,}")
    print(f"Период: с {global_stats[2]} по {global_stats[3]}")
    
    cursor.close()

def main():
    """Основная функция"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Сначала проверяем все символы
        check_all_symbols(conn)
        
        # Затем детальная проверка 1000PEPEUSDT
        check_symbol_data(conn, '1000PEPEUSDT')
        
        # Проверяем другие потенциально проблемные символы
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT symbol 
            FROM raw_market_data 
            WHERE symbol != '1000PEPEUSDT'
                AND EXISTS (
                    SELECT 1 FROM raw_market_data r2 
                    WHERE r2.symbol = raw_market_data.symbol 
                    AND (r2.open = 0 OR r2.low = 0 OR r2.low < 0.0000001)
                )
            LIMIT 5
        """)
        
        other_problematic = cursor.fetchall()
        if other_problematic:
            print(f"\n[ПРОВЕРКА] Другие проблемные символы:")
            for (symbol,) in other_problematic:
                check_symbol_data(conn, symbol)
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"[ОШИБКА]: {e}")
        raise

if __name__ == "__main__":
    main()