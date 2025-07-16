#!/usr/bin/env python3
"""
Обновление схемы базы данных для хранения ожидаемых результатов
"""

import psycopg2
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_database_schema():
    """Добавляет новые колонки в таблицу processed_market_data"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    try:
        logger.info("📊 Обновление схемы базы данных...")
        
        # Добавляем новые колонки для ожидаемых результатов
        alter_queries = [
            """
            ALTER TABLE processed_market_data 
            ADD COLUMN IF NOT EXISTS buy_expected_return DECIMAL(10, 4) DEFAULT 0.0
            """,
            """
            ALTER TABLE processed_market_data 
            ADD COLUMN IF NOT EXISTS sell_expected_return DECIMAL(10, 4) DEFAULT 0.0
            """,
            """
            ALTER TABLE processed_market_data 
            ADD COLUMN IF NOT EXISTS buy_max_profit DECIMAL(10, 4) DEFAULT 0.0
            """,
            """
            ALTER TABLE processed_market_data 
            ADD COLUMN IF NOT EXISTS sell_max_profit DECIMAL(10, 4) DEFAULT 0.0
            """,
            """
            ALTER TABLE processed_market_data 
            ADD COLUMN IF NOT EXISTS buy_realized_profit DECIMAL(10, 4) DEFAULT 0.0
            """,
            """
            ALTER TABLE processed_market_data 
            ADD COLUMN IF NOT EXISTS sell_realized_profit DECIMAL(10, 4) DEFAULT 0.0
            """
        ]
        
        for query in alter_queries:
            cursor.execute(query)
            logger.info(f"✅ Выполнен запрос: {query.strip()[:50]}...")
        
        # Создаем индексы для новых колонок
        index_queries = [
            """
            CREATE INDEX IF NOT EXISTS idx_buy_expected_return 
            ON processed_market_data(buy_expected_return)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sell_expected_return 
            ON processed_market_data(sell_expected_return)
            """
        ]
        
        for query in index_queries:
            cursor.execute(query)
            logger.info(f"✅ Создан индекс: {query.strip()[:50]}...")
        
        conn.commit()
        
        # Проверяем структуру таблицы
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'processed_market_data' 
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        
        logger.info("\n📋 Структура таблицы processed_market_data:")
        for col_name, col_type in columns:
            logger.info(f"   {col_name}: {col_type}")
        
        logger.info("\n✅ Схема базы данных успешно обновлена!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка обновления схемы: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    update_database_schema()