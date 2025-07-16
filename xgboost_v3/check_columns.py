#!/usr/bin/env python3
"""
Проверка колонок в данных
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Подключение к БД
conn = psycopg2.connect(
    host="localhost", port=5555, database="crypto_trading", user="ruslan"
)

with conn.cursor(cursor_factory=RealDictCursor) as cursor:
    cursor.execute("""
        SELECT * FROM processed_market_data
        WHERE symbol = 'BTCUSDT'
        LIMIT 1
    """)
    data = cursor.fetchall()

conn.close()

df = pd.DataFrame(data)
logger.info(f"Всего колонок: {len(df.columns)}")
logger.info("\nКолонки с 'target' или 'profit' или 'loss':")
for col in sorted(df.columns):
    if any(word in col.lower() for word in ['target', 'profit', 'loss']):
        logger.info(f"  - {col}")
        
logger.info("\nВсе колонки:")
for i, col in enumerate(sorted(df.columns)):
    logger.info(f"{i+1:3}. {col}")