#!/usr/bin/env python3
"""
Полная проверка схемы базы данных processed_market_data
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

print("="*100)
print("ПОЛНАЯ СХЕМА ТАБЛИЦЫ processed_market_data")
print("="*100)

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Получаем полную информацию о колонках
    cursor.execute("""
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = 'processed_market_data' 
        ORDER BY ordinal_position
    """)
    
    columns = cursor.fetchall()
    
    print(f"\n📋 Всего колонок: {len(columns)}\n")
    
    # Группируем колонки по типам
    base_columns = []
    target_columns = []
    expected_return_columns = []
    entry_columns = []
    other_columns = []
    
    for col_name, data_type, max_length, nullable, default in columns:
        col_info = {
            'name': col_name,
            'type': data_type,
            'nullable': nullable,
            'default': default
        }
        
        if col_name in ['id', 'raw_data_id', 'symbol', 'timestamp', 'datetime', 
                       'open', 'high', 'low', 'close', 'volume']:
            base_columns.append(col_info)
        elif 'target' in col_name:
            target_columns.append(col_info)
        elif 'expected_return' in col_name or 'max_profit' in col_name or 'realized_profit' in col_name:
            expected_return_columns.append(col_info)
        elif 'entry' in col_name:
            entry_columns.append(col_info)
        else:
            other_columns.append(col_info)
    
    # Выводим информацию по группам
    print("🔹 БАЗОВЫЕ КОЛОНКИ:")
    for col in base_columns:
        print(f"   - {col['name']:<25} {col['type']:<20} (nullable: {col['nullable']})")
    
    print("\n🎯 ЦЕЛЕВЫЕ КОЛОНКИ (TARGETS):")
    for col in target_columns:
        print(f"   - {col['name']:<25} {col['type']:<20} (default: {col['default']})")
    
    print("\n💰 КОЛОНКИ ОЖИДАЕМЫХ РЕЗУЛЬТАТОВ:")
    for col in expected_return_columns:
        print(f"   - {col['name']:<25} {col['type']:<20} (default: {col['default']})")
    
    print("\n🚀 КОЛОНКИ ТОЧЕК ВХОДА:")
    for col in entry_columns:
        print(f"   - {col['name']:<25} {col['type']:<20} (default: {col['default']})")
    
    print("\n📊 ПРОЧИЕ КОЛОНКИ:")
    for col in other_columns:
        print(f"   - {col['name']:<25} {col['type']:<20}")
    
    # Проверяем индексы
    print("\n" + "="*100)
    print("ИНДЕКСЫ НА ТАБЛИЦЕ processed_market_data")
    print("="*100)
    
    cursor.execute("""
        SELECT 
            indexname,
            indexdef
        FROM pg_indexes 
        WHERE tablename = 'processed_market_data'
        ORDER BY indexname
    """)
    
    indexes = cursor.fetchall()
    print(f"\n📑 Всего индексов: {len(indexes)}\n")
    
    for idx_name, idx_def in indexes:
        print(f"🔸 {idx_name}")
        print(f"   {idx_def}\n")
    
    # Проверяем примеры данных
    print("="*100)
    print("ПРИМЕРЫ ДАННЫХ")
    print("="*100)
    
    # Проверяем есть ли данные с новыми колонками
    cursor.execute("""
        SELECT 
            symbol,
            timestamp,
            buy_expected_return,
            sell_expected_return,
            buy_max_profit,
            sell_max_profit,
            buy_realized_profit,
            sell_realized_profit,
            is_long_entry,
            is_short_entry
        FROM processed_market_data 
        WHERE buy_expected_return IS NOT NULL 
           OR sell_expected_return IS NOT NULL
        LIMIT 5
    """)
    
    sample_data = cursor.fetchall()
    
    if sample_data:
        print("\n✅ Найдены данные с заполненными новыми колонками:")
        df = pd.DataFrame(sample_data, columns=[
            'symbol', 'timestamp', 'buy_expected_return', 'sell_expected_return',
            'buy_max_profit', 'sell_max_profit', 'buy_realized_profit', 'sell_realized_profit',
            'is_long_entry', 'is_short_entry'
        ])
        print(df.to_string())
    else:
        print("\n⚠️ Данные с новыми колонками не найдены")
        
        # Проверяем есть ли вообще данные
        cursor.execute("SELECT COUNT(*) FROM processed_market_data")
        total_count = cursor.fetchone()[0]
        print(f"   Всего записей в таблице: {total_count:,}")
    
    # Проверяем данные в technical_indicators JSONB
    print("\n" + "="*100)
    print("ПРОВЕРКА ДАННЫХ В technical_indicators")
    print("="*100)
    
    cursor.execute("""
        SELECT 
            symbol,
            COUNT(*) as total,
            COUNT(CASE WHEN technical_indicators->>'buy_expected_return' IS NOT NULL THEN 1 END) as with_buy_er,
            COUNT(CASE WHEN technical_indicators->>'sell_expected_return' IS NOT NULL THEN 1 END) as with_sell_er
        FROM processed_market_data
        GROUP BY symbol
        HAVING COUNT(CASE WHEN technical_indicators->>'buy_expected_return' IS NOT NULL THEN 1 END) > 0
        ORDER BY symbol
        LIMIT 10
    """)
    
    json_data = cursor.fetchall()
    
    if json_data:
        print("\n📊 Данные в JSONB поле technical_indicators:")
        for symbol, total, with_buy, with_sell in json_data:
            print(f"   {symbol}: всего {total}, с buy_ER: {with_buy}, с sell_ER: {with_sell}")
    else:
        print("\n⚠️ В JSONB поле technical_indicators нет данных expected_return")
    
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
finally:
    if 'conn' in locals():
        conn.close()
        print("\n✅ Соединение закрыто")