#!/usr/bin/env python3
"""
Скрипт для проверки загруженных данных в БД.
Выводит статистику, примеры записей и проверяет корректность данных.
"""

import os
import sys
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate
import random
from typing import Dict, List, Tuple
import numpy as np


def connect_to_db():
    """Подключение к базе данных."""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5555,
            database='crypto_trading',
            user='ruslan',
            password='ruslan'
        )
        return conn
    except Exception as e:
        print(f"❌ Ошибка подключения к БД: {e}")
        sys.exit(1)


def get_overall_stats(cursor) -> Dict:
    """Получить общую статистику по БД."""
    # Общее количество записей
    cursor.execute("SELECT COUNT(*) FROM raw_market_data")
    total_records = cursor.fetchone()[0]
    
    # Количество уникальных символов
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM raw_market_data")
    unique_symbols = cursor.fetchone()[0]
    
    # Временной диапазон
    cursor.execute("""
        SELECT MIN(timestamp), MAX(timestamp) 
        FROM raw_market_data
    """)
    min_time, max_time = cursor.fetchone()
    
    # Размер таблицы
    cursor.execute("""
        SELECT pg_size_pretty(pg_total_relation_size('raw_market_data'))
    """)
    table_size = cursor.fetchone()[0]
    
    return {
        'total_records': total_records,
        'unique_symbols': unique_symbols,
        'min_time': min_time,
        'max_time': max_time,
        'table_size': table_size
    }


def get_symbol_stats(cursor) -> pd.DataFrame:
    """Получить статистику по каждому символу."""
    query = """
    SELECT 
        symbol,
        COUNT(*) as record_count,
        MIN(timestamp) as first_timestamp,
        MAX(timestamp) as last_timestamp,
        MIN(close) as min_price,
        MAX(close) as max_price,
        AVG(close) as avg_price,
        STDDEV(close) as price_stddev,
        SUM(volume) as total_volume,
        AVG(volume) as avg_volume
    FROM raw_market_data
    GROUP BY symbol
    ORDER BY symbol
    """
    
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    
    return pd.DataFrame(data, columns=columns)


def get_sample_records(cursor, symbol: str) -> Tuple[Dict, Dict, Dict]:
    """Получить примеры записей для символа: первая, последняя, случайная."""
    # Первая запись
    cursor.execute("""
        SELECT * FROM raw_market_data 
        WHERE symbol = %s 
        ORDER BY timestamp ASC 
        LIMIT 1
    """, (symbol,))
    first = dict(cursor.fetchone())
    
    # Последняя запись
    cursor.execute("""
        SELECT * FROM raw_market_data 
        WHERE symbol = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, (symbol,))
    last = dict(cursor.fetchone())
    
    # Случайная запись
    cursor.execute("""
        SELECT * FROM raw_market_data 
        WHERE symbol = %s 
        ORDER BY RANDOM() 
        LIMIT 1
    """, (symbol,))
    random_rec = dict(cursor.fetchone())
    
    return first, last, random_rec


def check_time_intervals(cursor, symbol: str) -> Dict:
    """Проверить корректность временных интервалов для символа."""
    # Получаем все временные метки
    cursor.execute("""
        SELECT timestamp 
        FROM raw_market_data 
        WHERE symbol = %s 
        ORDER BY timestamp
    """, (symbol,))
    
    timestamps = [row[0] for row in cursor.fetchall()]
    
    if len(timestamps) < 2:
        return {
            'total_records': len(timestamps),
            'gaps_found': 0,
            'duplicates': 0,
            'wrong_intervals': 0
        }
    
    # Проверяем интервалы
    gaps = []
    wrong_intervals = []
    
    for i in range(1, len(timestamps)):
        diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
        
        if diff > 15:  # Пропуск
            gaps.append({
                'from': timestamps[i-1],
                'to': timestamps[i],
                'gap_minutes': diff
            })
        elif diff < 15 and diff > 0:  # Неправильный интервал
            wrong_intervals.append({
                'time': timestamps[i],
                'interval': diff
            })
    
    # Проверка на дубликаты
    unique_timestamps = len(set(timestamps))
    duplicates = len(timestamps) - unique_timestamps
    
    return {
        'total_records': len(timestamps),
        'gaps_found': len(gaps),
        'gaps': gaps[:5],  # Первые 5 пропусков
        'duplicates': duplicates,
        'wrong_intervals': len(wrong_intervals),
        'wrong_interval_samples': wrong_intervals[:5]
    }


def check_recent_data(cursor, hours: int = 24) -> pd.DataFrame:
    """Проверить наличие данных за последние N часов."""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    query = """
    SELECT 
        symbol,
        COUNT(*) as recent_records,
        MAX(timestamp) as last_timestamp,
        EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/3600 as hours_ago
    FROM raw_market_data
    WHERE timestamp > %s
    GROUP BY symbol
    ORDER BY symbol
    """
    
    cursor.execute(query, (cutoff_time,))
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    
    return pd.DataFrame(data, columns=columns)


def format_record(record: Dict) -> str:
    """Форматировать запись для вывода."""
    return (
        f"  Время: {record['timestamp']}\n"
        f"  Open: {record['open']:.4f}, High: {record['high']:.4f}, "
        f"Low: {record['low']:.4f}, Close: {record['close']:.4f}\n"
        f"  Volume: {record['volume']:.2f}"
    )


def main():
    """Основная функция."""
    print("🔍 Проверка загруженных данных в БД\n")
    
    # Подключаемся к БД
    conn = connect_to_db()
    cursor = conn.cursor(cursor_factory=DictCursor)
    
    try:
        # 1. Общая статистика
        print("📊 ОБЩАЯ СТАТИСТИКА")
        print("=" * 80)
        stats = get_overall_stats(cursor)
        print(f"Всего записей: {stats['total_records']:,}")
        print(f"Уникальных символов: {stats['unique_symbols']}")
        print(f"Период данных: {stats['min_time']} - {stats['max_time']}")
        print(f"Размер таблицы: {stats['table_size']}")
        print()
        
        # 2. Статистика по символам
        print("📈 СТАТИСТИКА ПО МОНЕТАМ")
        print("=" * 80)
        symbol_stats = get_symbol_stats(cursor)
        
        # Форматируем таблицу
        display_df = symbol_stats.copy()
        display_df['record_count'] = display_df['record_count'].apply(lambda x: f"{x:,}")
        display_df['min_price'] = display_df['min_price'].apply(lambda x: f"{x:.4f}")
        display_df['max_price'] = display_df['max_price'].apply(lambda x: f"{x:.4f}")
        display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"{x:.4f}")
        display_df['price_stddev'] = display_df['price_stddev'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        display_df['total_volume'] = display_df['total_volume'].apply(lambda x: f"{x:,.0f}")
        display_df['avg_volume'] = display_df['avg_volume'].apply(lambda x: f"{x:,.2f}")
        
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        print()
        
        # 3. Примеры записей для каждой монеты
        print("📝 ПРИМЕРЫ ЗАПИСЕЙ")
        print("=" * 80)
        
        for symbol in symbol_stats['symbol']:
            print(f"\n{symbol}:")
            first, last, random_rec = get_sample_records(cursor, symbol)
            
            print("\n  Первая запись:")
            print(format_record(first))
            
            print("\n  Последняя запись:")
            print(format_record(last))
            
            print("\n  Случайная запись:")
            print(format_record(random_rec))
        
        # 4. Проверка временных интервалов
        print("\n⏰ ПРОВЕРКА ВРЕМЕННЫХ ИНТЕРВАЛОВ")
        print("=" * 80)
        
        interval_issues = []
        for symbol in symbol_stats['symbol']:
            check_result = check_time_intervals(cursor, symbol)
            
            print(f"\n{symbol}:")
            print(f"  Всего записей: {check_result['total_records']:,}")
            print(f"  Найдено пропусков: {check_result['gaps_found']}")
            print(f"  Дубликатов: {check_result['duplicates']}")
            print(f"  Неправильных интервалов: {check_result['wrong_intervals']}")
            
            if check_result['gaps_found'] > 0:
                print("  Примеры пропусков:")
                for gap in check_result['gaps'][:3]:
                    print(f"    {gap['from']} -> {gap['to']} ({gap['gap_minutes']:.0f} минут)")
            
            if check_result['wrong_intervals'] > 0:
                interval_issues.append(symbol)
        
        # 5. Проверка свежести данных
        print("\n🕐 ПРОВЕРКА СВЕЖЕСТИ ДАННЫХ (последние 24 часа)")
        print("=" * 80)
        
        recent_data = check_recent_data(cursor, 24)
        
        if len(recent_data) == 0:
            print("⚠️  Нет данных за последние 24 часа!")
        else:
            recent_display = recent_data.copy()
            recent_display['recent_records'] = recent_display['recent_records'].apply(lambda x: f"{x:,}")
            recent_display['hours_ago'] = recent_display['hours_ago'].apply(lambda x: f"{x:.1f}ч назад")
            
            print(tabulate(recent_display[['symbol', 'recent_records', 'last_timestamp', 'hours_ago']], 
                         headers=['Символ', 'Записей за 24ч', 'Последняя запись', 'Обновлено'], 
                         tablefmt='grid', 
                         showindex=False))
        
        # 6. Дополнительная аналитика
        print("\n📊 ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА")
        print("=" * 80)
        
        # Проверка на отсутствующие символы за последние сутки
        all_symbols = set(symbol_stats['symbol'])
        recent_symbols = set(recent_data['symbol']) if len(recent_data) > 0 else set()
        missing_recent = all_symbols - recent_symbols
        
        if missing_recent:
            print(f"\n⚠️  Символы без данных за последние 24 часа: {', '.join(missing_recent)}")
        
        # Статистика по объемам
        print("\n💰 Топ-5 по среднему объему торгов:")
        top_volume = symbol_stats.nlargest(5, 'avg_volume')[['symbol', 'avg_volume']]
        for _, row in top_volume.iterrows():
            print(f"  {row['symbol']}: {row['avg_volume']:,.2f}")
        
        # Статистика по волатильности
        print("\n📊 Топ-5 по волатильности (стандартное отклонение цены):")
        symbol_stats['volatility_pct'] = (symbol_stats['price_stddev'] / symbol_stats['avg_price']) * 100
        top_volatility = symbol_stats.nlargest(5, 'volatility_pct')[['symbol', 'volatility_pct', 'avg_price']]
        for _, row in top_volatility.iterrows():
            print(f"  {row['symbol']}: {row['volatility_pct']:.2f}% (средняя цена: {row['avg_price']:.4f})")
        
        print("\n✅ Проверка завершена!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при проверке данных: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()