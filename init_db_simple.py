#!/usr/bin/env python3
import psycopg2
import yaml

print("Загрузка конфигурации...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database']
print(f"БД: {db_config['dbname']} на {db_config['host']}:{db_config['port']}")

# Подключаемся к БД
print("Подключение к БД...")
conn = psycopg2.connect(**db_config)
conn.autocommit = True
cursor = conn.cursor()

print("Создание таблиц...")

# Создаем таблицу raw_market_data
cursor.execute("""
CREATE TABLE IF NOT EXISTS raw_market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    datetime TIMESTAMP NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    turnover DECIMAL(20, 8) DEFAULT 0,
    interval_minutes INTEGER NOT NULL DEFAULT 15,
    market_type VARCHAR(20) DEFAULT 'spot',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, timestamp, interval_minutes)
);
""")
print("✅ Таблица raw_market_data создана")

# Создаем индексы
cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_raw_market_data_symbol_timestamp 
ON raw_market_data(symbol, timestamp);
""")
print("✅ Индексы созданы")

# Проверяем результат
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name;
""")

tables = cursor.fetchall()
print(f"\nСоздано таблиц: {len(tables)}")
for table in tables:
    print(f"  - {table[0]}")

cursor.close()
conn.close()
print("\n✅ Готово!")