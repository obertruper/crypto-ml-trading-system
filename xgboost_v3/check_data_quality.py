#!/usr/bin/env python3
"""
Проверка качества данных и expected returns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_expected_returns():
    """Проверка expected returns в базе данных"""
    # Подключение к БД
    conn = psycopg2.connect(
        host="localhost",
        port=5555,
        database="crypto_trading",
        user="ruslan"
    )
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # 1. Проверка наличия данных
            logger.info("📊 Проверка таблицы processed_market_data...")
            cursor.execute("""
                SELECT COUNT(*) as total, 
                       COUNT(DISTINCT symbol) as symbols,
                       MIN(timestamp) as min_date,
                       MAX(timestamp) as max_date
                FROM processed_market_data
            """)
            info = cursor.fetchone()
            logger.info(f"   Всего записей: {info['total']:,}")
            logger.info(f"   Символов: {info['symbols']}")
            logger.info(f"   Период: {info['min_date']} - {info['max_date']}")
            
            # 2. Проверка expected returns
            logger.info("\n📈 Проверка expected returns...")
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(buy_expected_return) as buy_not_null,
                    COUNT(sell_expected_return) as sell_not_null,
                    AVG(buy_expected_return) as buy_avg,
                    MIN(buy_expected_return) as buy_min,
                    MAX(buy_expected_return) as buy_max,
                    AVG(sell_expected_return) as sell_avg,
                    MIN(sell_expected_return) as sell_min,
                    MAX(sell_expected_return) as sell_max
                FROM processed_market_data
                WHERE buy_expected_return IS NOT NULL
            """)
            stats = cursor.fetchone()
            
            logger.info(f"   Записей с expected returns: {stats['buy_not_null']:,}")
            logger.info(f"   Buy Expected Return:")
            logger.info(f"      Среднее: {stats['buy_avg']:.2f}%")
            logger.info(f"      Мин: {stats['buy_min']:.2f}%")
            logger.info(f"      Макс: {stats['buy_max']:.2f}%")
            logger.info(f"   Sell Expected Return:")
            logger.info(f"      Среднее: {stats['sell_avg']:.2f}%")
            logger.info(f"      Мин: {stats['sell_min']:.2f}%")
            logger.info(f"      Макс: {stats['sell_max']:.2f}%")
            
            # 3. Распределение по порогам
            logger.info("\n📊 Распределение expected returns по порогам...")
            thresholds = [0, 0.5, 1.0, 1.5, 2.0]
            
            for threshold in thresholds:
                cursor.execute(f"""
                    SELECT 
                        COUNT(CASE WHEN buy_expected_return > {threshold} THEN 1 END) as buy_above,
                        COUNT(CASE WHEN sell_expected_return > {threshold} THEN 1 END) as sell_above,
                        COUNT(*) as total
                    FROM processed_market_data
                    WHERE buy_expected_return IS NOT NULL
                """)
                dist = cursor.fetchone()
                buy_pct = dist['buy_above'] / dist['total'] * 100 if dist['total'] > 0 else 0
                sell_pct = dist['sell_above'] / dist['total'] * 100 if dist['total'] > 0 else 0
                logger.info(f"   > {threshold}%: Buy={buy_pct:.1f}%, Sell={sell_pct:.1f}%")
            
            # 4. Проверка технических индикаторов
            logger.info("\n🔧 Проверка технических индикаторов...")
            cursor.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as records,
                    technical_indicators
                FROM processed_market_data
                WHERE technical_indicators IS NOT NULL
                GROUP BY symbol, technical_indicators
                LIMIT 1
            """)
            sample = cursor.fetchone()
            
            if sample and sample['technical_indicators']:
                indicators = list(sample['technical_indicators'].keys())
                logger.info(f"   Найдено {len(indicators)} индикаторов")
                logger.info(f"   Примеры: {', '.join(indicators[:10])}")
            else:
                logger.warning("   ⚠️ Технические индикаторы не найдены!")
            
            # 5. Проверка качества данных по символам
            logger.info("\n📊 Качество данных по топ символам...")
            cursor.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as records,
                    AVG(buy_expected_return) as buy_avg,
                    AVG(sell_expected_return) as sell_avg,
                    COUNT(CASE WHEN buy_expected_return > 0.5 THEN 1 END) * 100.0 / COUNT(*) as buy_profitable_pct,
                    COUNT(CASE WHEN sell_expected_return > 0.5 THEN 1 END) * 100.0 / COUNT(*) as sell_profitable_pct
                FROM processed_market_data
                WHERE buy_expected_return IS NOT NULL
                GROUP BY symbol
                ORDER BY records DESC
                LIMIT 10
            """)
            
            for row in cursor.fetchall():
                logger.info(f"   {row['symbol']:10} Records: {row['records']:6,}, "
                          f"Buy avg: {row['buy_avg']:5.2f}% ({row['buy_profitable_pct']:4.1f}% > 0.5%), "
                          f"Sell avg: {row['sell_avg']:5.2f}% ({row['sell_profitable_pct']:4.1f}% > 0.5%)")
            
            # 6. Проверка на аномалии
            logger.info("\n⚠️ Проверка на аномалии...")
            cursor.execute("""
                SELECT COUNT(*) as anomalies
                FROM processed_market_data
                WHERE buy_expected_return IS NOT NULL
                AND (buy_expected_return > 50 OR buy_expected_return < -50
                     OR sell_expected_return > 50 OR sell_expected_return < -50)
            """)
            anomalies = cursor.fetchone()
            if anomalies['anomalies'] > 0:
                logger.warning(f"   Найдено {anomalies['anomalies']} записей с экстремальными значениями (>50%)")
            else:
                logger.info("   ✅ Экстремальных значений не найдено")
                
    except Exception as e:
        logger.error(f"❌ Ошибка при проверке данных: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("ПРОВЕРКА КАЧЕСТВА ДАННЫХ ДЛЯ XGBOOST")
    logger.info("="*60)
    check_expected_returns()
    logger.info("\n✅ Проверка завершена")