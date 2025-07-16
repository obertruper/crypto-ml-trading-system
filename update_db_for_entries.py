#!/usr/bin/env python3
"""
Обновление схемы БД для поддержки реалистичных точек входа
"""

import psycopg2
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

logger.info("🔄 Обновление схемы БД для реалистичных точек входа...")

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Добавляем новые колонки в processed_market_data
    alter_queries = [
        """
        ALTER TABLE processed_market_data 
        ADD COLUMN IF NOT EXISTS is_long_entry BOOLEAN DEFAULT FALSE
        """,
        """
        ALTER TABLE processed_market_data 
        ADD COLUMN IF NOT EXISTS is_short_entry BOOLEAN DEFAULT FALSE
        """,
        """
        ALTER TABLE processed_market_data 
        ADD COLUMN IF NOT EXISTS long_entry_type VARCHAR(20)
        """,
        """
        ALTER TABLE processed_market_data 
        ADD COLUMN IF NOT EXISTS short_entry_type VARCHAR(20)
        """,
        """
        ALTER TABLE processed_market_data 
        ADD COLUMN IF NOT EXISTS long_entry_confidence FLOAT
        """,
        """
        ALTER TABLE processed_market_data 
        ADD COLUMN IF NOT EXISTS short_entry_confidence FLOAT
        """
    ]
    
    for query in alter_queries:
        cursor.execute(query)
        logger.info(f"✅ Выполнен запрос: {query.strip()[:50]}...")
    
    # Создаем индексы для оптимизации
    index_queries = [
        """
        CREATE INDEX IF NOT EXISTS idx_long_entries 
        ON processed_market_data(symbol, timestamp) 
        WHERE is_long_entry = TRUE
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_short_entries 
        ON processed_market_data(symbol, timestamp) 
        WHERE is_short_entry = TRUE
        """
    ]
    
    for query in index_queries:
        cursor.execute(query)
        logger.info(f"✅ Создан индекс: {query.strip()[:50]}...")
    
    conn.commit()
    logger.info("✅ Схема БД успешно обновлена!")
    
    # Проверяем структуру таблицы
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'processed_market_data' 
        AND column_name IN ('is_long_entry', 'is_short_entry', 
                           'long_entry_type', 'short_entry_type',
                           'long_entry_confidence', 'short_entry_confidence')
        ORDER BY ordinal_position
    """)
    
    columns = cursor.fetchall()
    logger.info("\n📊 Новые колонки в таблице:")
    for col_name, col_type in columns:
        logger.info(f"   - {col_name}: {col_type}")
    
except Exception as e:
    logger.error(f"❌ Ошибка обновления БД: {e}")
    raise
finally:
    if 'conn' in locals():
        conn.close()
        logger.info("📤 Соединение закрыто")