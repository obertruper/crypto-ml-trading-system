#!/usr/bin/env python3
"""
Проверка валидности данных в таблице raw_market_data
"""

import psycopg2
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

print("="*80)
print("🔍 ПРОВЕРКА ВАЛИДНОСТИ RAW_MARKET_DATA")
print("="*80)

# Подключение к БД
conn = psycopg2.connect(**db_config)

# 1. Общая статистика
print("\n📊 ОБЩАЯ СТАТИСТИКА:")
query = """
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT symbol) as unique_symbols,
    COUNT(DISTINCT market_type) as market_types,
    COUNT(DISTINCT interval_minutes) as intervals,
    MIN(datetime) as start_date,
    MAX(datetime) as end_date
FROM raw_market_data
"""
df_stats = pd.read_sql(query, conn)
print(f"   Всего записей: {df_stats['total_records'][0]:,}")
print(f"   Уникальных символов: {df_stats['unique_symbols'][0]}")
print(f"   Типы рынка: {df_stats['market_types'][0]}")
print(f"   Интервалы: {df_stats['intervals'][0]}")
print(f"   Период данных: {df_stats['start_date'][0]} - {df_stats['end_date'][0]}")

# 2. Проверка типов рынка
print("\n📊 ТИПЫ РЫНКА:")
query = """
SELECT market_type, COUNT(*) as count, COUNT(DISTINCT symbol) as symbols
FROM raw_market_data
GROUP BY market_type
ORDER BY count DESC
"""
df_markets = pd.read_sql(query, conn)
for _, row in df_markets.iterrows():
    print(f"   {row['market_type']}: {row['count']:,} записей, {row['symbols']} символов")

# 3. Проверка интервалов
print("\n📊 ИНТЕРВАЛЫ:")
query = """
SELECT interval_minutes, COUNT(*) as count, COUNT(DISTINCT symbol) as symbols
FROM raw_market_data
GROUP BY interval_minutes
ORDER BY interval_minutes
"""
df_intervals = pd.read_sql(query, conn)
for _, row in df_intervals.iterrows():
    print(f"   {row['interval_minutes']} минут: {row['count']:,} записей, {row['symbols']} символов")

# 4. Проверка на NULL значения
print("\n❓ ПРОВЕРКА NULL ЗНАЧЕНИЙ:")
query = """
SELECT 
    SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) as null_open,
    SUM(CASE WHEN high IS NULL THEN 1 ELSE 0 END) as null_high,
    SUM(CASE WHEN low IS NULL THEN 1 ELSE 0 END) as null_low,
    SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
    SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume,
    SUM(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END) as null_timestamp,
    SUM(CASE WHEN datetime IS NULL THEN 1 ELSE 0 END) as null_datetime
FROM raw_market_data
"""
df_nulls = pd.read_sql(query, conn)
null_found = False
for col in df_nulls.columns:
    if df_nulls[col][0] > 0:
        print(f"   ⚠️ {col}: {df_nulls[col][0]:,} NULL значений")
        null_found = True
if not null_found:
    print("   ✅ NULL значений не найдено")

# 5. Проверка на отрицательные значения
print("\n📉 ПРОВЕРКА ОТРИЦАТЕЛЬНЫХ ЗНАЧЕНИЙ:")
query = """
SELECT 
    SUM(CASE WHEN open < 0 THEN 1 ELSE 0 END) as negative_open,
    SUM(CASE WHEN high < 0 THEN 1 ELSE 0 END) as negative_high,
    SUM(CASE WHEN low < 0 THEN 1 ELSE 0 END) as negative_low,
    SUM(CASE WHEN close < 0 THEN 1 ELSE 0 END) as negative_close,
    SUM(CASE WHEN volume < 0 THEN 1 ELSE 0 END) as negative_volume
FROM raw_market_data
"""
df_negative = pd.read_sql(query, conn)
negative_found = False
for col in df_negative.columns:
    if df_negative[col][0] > 0:
        print(f"   ⚠️ {col}: {df_negative[col][0]:,} отрицательных значений")
        negative_found = True
if not negative_found:
    print("   ✅ Отрицательных значений не найдено")

# 6. Проверка OHLC логики
print("\n📊 ПРОВЕРКА OHLC ЛОГИКИ:")
query = """
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN high < low THEN 1 ELSE 0 END) as high_less_than_low,
    SUM(CASE WHEN high < open THEN 1 ELSE 0 END) as high_less_than_open,
    SUM(CASE WHEN high < close THEN 1 ELSE 0 END) as high_less_than_close,
    SUM(CASE WHEN low > open THEN 1 ELSE 0 END) as low_greater_than_open,
    SUM(CASE WHEN low > close THEN 1 ELSE 0 END) as low_greater_than_close
FROM raw_market_data
WHERE market_type = 'futures' AND interval_minutes = 15
"""
df_ohlc = pd.read_sql(query, conn)
ohlc_errors = False
if df_ohlc['high_less_than_low'][0] > 0:
    print(f"   ⚠️ High < Low: {df_ohlc['high_less_than_low'][0]:,} записей")
    ohlc_errors = True
if df_ohlc['high_less_than_open'][0] > 0:
    print(f"   ⚠️ High < Open: {df_ohlc['high_less_than_open'][0]:,} записей")
    ohlc_errors = True
if df_ohlc['high_less_than_close'][0] > 0:
    print(f"   ⚠️ High < Close: {df_ohlc['high_less_than_close'][0]:,} записей")
    ohlc_errors = True
if df_ohlc['low_greater_than_open'][0] > 0:
    print(f"   ⚠️ Low > Open: {df_ohlc['low_greater_than_open'][0]:,} записей")
    ohlc_errors = True
if df_ohlc['low_greater_than_close'][0] > 0:
    print(f"   ⚠️ Low > Close: {df_ohlc['low_greater_than_close'][0]:,} записей")
    ohlc_errors = True
if not ohlc_errors:
    print("   ✅ OHLC логика корректна")

# 7. Проверка временных интервалов
print("\n⏰ ПРОВЕРКА ВРЕМЕННЫХ ИНТЕРВАЛОВ (15-минутные бары):")
query = """
WITH time_gaps AS (
    SELECT 
        symbol,
        timestamp,
        datetime,
        LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_timestamp,
        timestamp - LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as gap_seconds
    FROM raw_market_data
    WHERE market_type = 'futures' AND interval_minutes = 15
)
SELECT 
    COUNT(*) as total_gaps,
    SUM(CASE WHEN gap_seconds = 900 THEN 1 ELSE 0 END) as correct_gaps,
    SUM(CASE WHEN gap_seconds > 900 AND gap_seconds <= 3600 THEN 1 ELSE 0 END) as small_gaps,
    SUM(CASE WHEN gap_seconds > 3600 AND gap_seconds <= 86400 THEN 1 ELSE 0 END) as medium_gaps,
    SUM(CASE WHEN gap_seconds > 86400 THEN 1 ELSE 0 END) as large_gaps
FROM time_gaps
WHERE gap_seconds IS NOT NULL
"""
df_gaps = pd.read_sql(query, conn)
if df_gaps['total_gaps'][0] > 0:
    correct_pct = df_gaps['correct_gaps'][0] / df_gaps['total_gaps'][0] * 100
    print(f"   Корректные интервалы (15 мин): {df_gaps['correct_gaps'][0]:,} ({correct_pct:.1f}%)")
    if df_gaps['small_gaps'][0] > 0:
        print(f"   Малые пропуски (15м-1ч): {df_gaps['small_gaps'][0]:,}")
    if df_gaps['medium_gaps'][0] > 0:
        print(f"   Средние пропуски (1ч-24ч): {df_gaps['medium_gaps'][0]:,}")
    if df_gaps['large_gaps'][0] > 0:
        print(f"   Большие пропуски (>24ч): {df_gaps['large_gaps'][0]:,}")

# 8. Топ символов по количеству данных
print("\n📈 ТОП-10 СИМВОЛОВ ПО КОЛИЧЕСТВУ ДАННЫХ:")
query = """
SELECT 
    symbol, 
    COUNT(*) as records,
    MIN(datetime) as start_date,
    MAX(datetime) as end_date,
    ROUND(AVG(volume)::numeric, 2) as avg_volume
FROM raw_market_data
WHERE market_type = 'futures' AND interval_minutes = 15
GROUP BY symbol
ORDER BY records DESC
LIMIT 10
"""
df_symbols = pd.read_sql(query, conn)
for _, row in df_symbols.iterrows():
    days = (row['end_date'] - row['start_date']).days
    print(f"   {row['symbol']}: {row['records']:,} записей, {days} дней, avg vol: {row['avg_volume']:,.2f}")

# 9. Проверка на дубликаты
print("\n🔍 ПРОВЕРКА ДУБЛИКАТОВ:")
query = """
SELECT symbol, timestamp, COUNT(*) as count
FROM raw_market_data
WHERE market_type = 'futures' AND interval_minutes = 15
GROUP BY symbol, timestamp
HAVING COUNT(*) > 1
LIMIT 10
"""
df_duplicates = pd.read_sql(query, conn)
if len(df_duplicates) > 0:
    print(f"   ⚠️ Найдено {len(df_duplicates)} дубликатов!")
    for _, row in df_duplicates.iterrows():
        print(f"      {row['symbol']} at {pd.to_datetime(row['timestamp'], unit='s')}: {row['count']} записей")
else:
    print("   ✅ Дубликатов не найдено")

# 10. Проверка экстремальных значений
print("\n📊 ПРОВЕРКА ЭКСТРЕМАЛЬНЫХ ЗНАЧЕНИЙ:")
query = """
WITH price_changes AS (
    SELECT 
        symbol,
        datetime,
        close,
        LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close,
        CASE 
            WHEN LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) > 0 
            THEN ABS((close - LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp)) / 
                 LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp) * 100)
            ELSE 0 
        END as change_pct
    FROM raw_market_data
    WHERE market_type = 'futures' AND interval_minutes = 15
)
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN change_pct > 10 THEN 1 ELSE 0 END) as extreme_changes,
    MAX(change_pct) as max_change
FROM price_changes
WHERE change_pct IS NOT NULL
"""
df_extreme = pd.read_sql(query, conn)
if df_extreme['extreme_changes'][0] > 0:
    print(f"   ⚠️ Экстремальные изменения (>10% за 15 мин): {df_extreme['extreme_changes'][0]:,}")
    print(f"   ⚠️ Максимальное изменение: {df_extreme['max_change'][0]:.2f}%")
else:
    print("   ✅ Экстремальных изменений не найдено")

# 11. Итоговая оценка
print("\n" + "="*80)
print("📋 ИТОГОВАЯ ОЦЕНКА:")
print("="*80)

issues = []
if null_found:
    issues.append("NULL значения в данных")
if negative_found:
    issues.append("Отрицательные значения цен/объемов")
if ohlc_errors:
    issues.append("Нарушение OHLC логики")
if df_extreme['extreme_changes'][0] > 100:
    issues.append("Много экстремальных изменений цен")
if len(df_duplicates) > 0:
    issues.append("Дубликаты в данных")

if len(issues) == 0:
    print("✅ Данные валидны и готовы к использованию!")
    print("   - Нет NULL значений")
    print("   - Нет отрицательных цен")
    print("   - OHLC логика корректна")
    print("   - Нет дубликатов")
else:
    print("⚠️ Обнаружены проблемы:")
    for issue in issues:
        print(f"   - {issue}")
    print("\n💡 Рекомендуется очистить данные перед использованием")

# Проверка для фьючерсов
print("\n🔍 ПРОВЕРКА ФЬЮЧЕРСНЫХ ДАННЫХ:")
query = """
SELECT COUNT(*) as futures_count
FROM raw_market_data
WHERE market_type = 'futures' AND interval_minutes = 15
"""
futures_count = pd.read_sql(query, conn)['futures_count'][0]
print(f"   Фьючерсных 15-минутных баров: {futures_count:,}")

if futures_count > 1000000:
    print("   ✅ Достаточно данных для обучения модели")
else:
    print("   ⚠️ Мало данных, рекомендуется загрузить больше")

conn.close()
print("\n" + "="*80)