#!/usr/bin/env python3
"""
Проверка статуса датасета с новыми метками
"""

import psycopg2
import yaml
import pandas as pd
import numpy as np

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

conn = psycopg2.connect(**db_config)

print("="*80)
print("ПРОВЕРКА СТАТУСА ДАТАСЕТА")
print("="*80)

# Проверяем общее количество записей
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM processed_market_data")
total_records = cursor.fetchone()[0]
print(f"\n📊 Всего записей в processed_market_data: {total_records:,}")

# Проверяем записи с новыми метками
cursor.execute("""
    SELECT COUNT(*) 
    FROM processed_market_data 
    WHERE technical_indicators->>'buy_expected_return' IS NOT NULL
""")
records_with_new_labels = cursor.fetchone()[0]
print(f"📈 Записей с новыми метками: {records_with_new_labels:,}")
print(f"📊 Прогресс: {records_with_new_labels/total_records*100:.1f}%")

# Проверяем по символам
cursor.execute("""
    SELECT 
        symbol,
        COUNT(*) as total,
        COUNT(CASE WHEN technical_indicators->>'buy_expected_return' IS NOT NULL THEN 1 END) as with_labels
    FROM processed_market_data
    GROUP BY symbol
    ORDER BY symbol
""")

print("\n📊 Статус по символам:")
print(f"{'Символ':<15} {'Всего':<10} {'С метками':<10} {'Прогресс':<10}")
print("-"*50)

for symbol, total, with_labels in cursor.fetchall():
    progress = with_labels/total*100 if total > 0 else 0
    print(f"{symbol:<15} {total:<10,} {with_labels:<10,} {progress:<10.1f}%")

# Если есть данные с новыми метками, анализируем их
if records_with_new_labels > 0:
    query = """
    SELECT 
        technical_indicators->>'buy_expected_return' as buy_return,
        technical_indicators->>'sell_expected_return' as sell_return
    FROM processed_market_data
    WHERE technical_indicators->>'buy_expected_return' IS NOT NULL
    LIMIT 10000
    """
    
    df = pd.read_sql(query, conn)
    df['buy_return'] = df['buy_return'].astype(float)
    df['sell_return'] = df['sell_return'].astype(float)
    
    print("\n📈 Статистика новых меток (выборка 10,000):")
    print("\nBUY ожидаемые результаты:")
    print(f"  Среднее: {df['buy_return'].mean():.2f}%")
    print(f"  Медиана: {df['buy_return'].median():.2f}%")
    print(f"  Std: {df['buy_return'].std():.2f}%")
    print(f"  Min: {df['buy_return'].min():.2f}%")
    print(f"  Max: {df['buy_return'].max():.2f}%")
    print(f"  Положительных: {(df['buy_return'] > 0).sum()} ({(df['buy_return'] > 0).mean()*100:.1f}%)")
    
    print("\nSELL ожидаемые результаты:")
    print(f"  Среднее: {df['sell_return'].mean():.2f}%")
    print(f"  Медиана: {df['sell_return'].median():.2f}%")
    print(f"  Std: {df['sell_return'].std():.2f}%")
    print(f"  Min: {df['sell_return'].min():.2f}%")
    print(f"  Max: {df['sell_return'].max():.2f}%")
    print(f"  Положительных: {(df['sell_return'] > 0).sum()} ({(df['sell_return'] > 0).mean()*100:.1f}%)")

conn.close()

print("\n" + "="*80)
if records_with_new_labels == total_records:
    print("✅ ДАТАСЕТ ПОЛНОСТЬЮ ГОТОВ К ОБУЧЕНИЮ!")
    print("   Запустите: python train_advanced_regression.py")
else:
    print("⏳ ДАТАСЕТ ЕЩЕ ОБРАБАТЫВАЕТСЯ...")
    print(f"   Обработано: {records_with_new_labels/total_records*100:.1f}%")
    print("   Дождитесь завершения prepare_dataset.py")
print("="*80)