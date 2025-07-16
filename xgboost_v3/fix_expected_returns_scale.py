#!/usr/bin/env python3
"""
Исправление масштаба expected returns - конвертация из долей в проценты
"""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_expected_returns_scale():
    """Конвертирует expected returns из долей в проценты"""
    
    # Подключение к БД
    conn = psycopg2.connect(
        host="localhost",
        port=5555,
        user="ruslan",
        password="",
        database="crypto_trading"
    )
    
    try:
        with conn.cursor() as cursor:
            # Проверяем текущий масштаб
            cursor.execute("""
                SELECT 
                    AVG(buy_expected_return) as avg_buy,
                    AVG(sell_expected_return) as avg_sell,
                    MAX(ABS(buy_expected_return)) as max_buy,
                    MAX(ABS(sell_expected_return)) as max_sell
                FROM processed_market_data
                WHERE symbol IN ('BTCUSDT', 'ETHUSDT')
            """)
            
            result = cursor.fetchone()
            avg_buy, avg_sell, max_buy, max_sell = result
            
            logger.info("📊 Текущий масштаб expected returns:")
            logger.info(f"   Среднее buy: {avg_buy:.6f}")
            logger.info(f"   Среднее sell: {avg_sell:.6f}")
            logger.info(f"   Макс buy: {max_buy:.6f}")
            logger.info(f"   Макс sell: {max_sell:.6f}")
            
            # Проверяем, нужна ли конвертация
            if max_buy < 1 and max_sell < 1:
                logger.info("\n✅ Обнаружены значения в долях (< 1), конвертируем в проценты...")
                
                # Обновляем значения - умножаем на 100
                cursor.execute("""
                    UPDATE processed_market_data
                    SET 
                        buy_expected_return = buy_expected_return * 100,
                        sell_expected_return = sell_expected_return * 100
                    WHERE buy_expected_return IS NOT NULL 
                    AND sell_expected_return IS NOT NULL
                """)
                
                updated_rows = cursor.rowcount
                logger.info(f"✅ Обновлено {updated_rows:,} записей")
                
                # Проверяем результат
                cursor.execute("""
                    SELECT 
                        AVG(buy_expected_return) as avg_buy,
                        AVG(sell_expected_return) as avg_sell,
                        MAX(ABS(buy_expected_return)) as max_buy,
                        MAX(ABS(sell_expected_return)) as max_sell
                    FROM processed_market_data
                    WHERE symbol IN ('BTCUSDT', 'ETHUSDT')
                """)
                
                result = cursor.fetchone()
                avg_buy, avg_sell, max_buy, max_sell = result
                
                logger.info("\n📊 Новый масштаб expected returns (в %):")
                logger.info(f"   Среднее buy: {avg_buy:.2f}%")
                logger.info(f"   Среднее sell: {avg_sell:.2f}%")
                logger.info(f"   Макс buy: {max_buy:.2f}%")
                logger.info(f"   Макс sell: {max_sell:.2f}%")
                
                # Анализ распределения для порога 1.5%
                cursor.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as total,
                        SUM(CASE WHEN buy_expected_return > 1.5 THEN 1 ELSE 0 END) as buy_positive,
                        SUM(CASE WHEN sell_expected_return > 1.5 THEN 1 ELSE 0 END) as sell_positive
                    FROM processed_market_data
                    WHERE symbol IN ('BTCUSDT', 'ETHUSDT')
                    GROUP BY symbol
                """)
                
                logger.info("\n📊 Распределение классов с порогом 1.5%:")
                for row in cursor.fetchall():
                    symbol, total, buy_pos, sell_pos = row
                    buy_ratio = buy_pos / total * 100
                    sell_ratio = sell_pos / total * 100
                    logger.info(f"   {symbol}: BUY={buy_pos:,}/{total:,} ({buy_ratio:.1f}%), SELL={sell_pos:,}/{total:,} ({sell_ratio:.1f}%)")
                
                # Коммитим изменения
                conn.commit()
                logger.info("\n✅ Изменения сохранены в БД")
                
            else:
                logger.info("\n✅ Expected returns уже в процентах, конвертация не нужна")
                
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    fix_expected_returns_scale()