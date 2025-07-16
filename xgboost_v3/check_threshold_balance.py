#!/usr/bin/env python3
"""
Проверка баланса классов при разных порогах
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_thresholds():
    """Проверка распределения классов при разных порогах"""
    # Подключение к БД
    conn = psycopg2.connect(
        host="localhost",
        port=5555,
        database="crypto_trading",
        user="ruslan"
    )
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Получаем expected returns для нескольких символов
            logger.info("📊 Загрузка expected returns...")
            cursor.execute("""
                SELECT buy_expected_return, sell_expected_return
                FROM processed_market_data
                WHERE symbol IN ('BTCUSDT', 'ETHUSDT')
                AND buy_expected_return IS NOT NULL
                LIMIT 50000
            """)
            
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            
            logger.info(f"   Загружено {len(df)} записей")
            
            # Проверяем разные пороги
            thresholds = [0.3, 0.5, 0.7, 1.0, 1.5]
            
            logger.info("\n📊 Распределение классов при разных порогах:")
            logger.info("Порог | Buy Class 1 % | Sell Class 1 %")
            logger.info("-" * 40)
            
            for threshold in thresholds:
                buy_class_1 = (df['buy_expected_return'] > threshold).mean() * 100
                sell_class_1 = (df['sell_expected_return'] > threshold).mean() * 100
                
                logger.info(f"{threshold:4.1f}% | {buy_class_1:13.1f}% | {sell_class_1:14.1f}%")
                
            # Детальная статистика для порога 0.7%
            logger.info("\n📊 Детальная статистика для порога 0.7%:")
            threshold = 0.7
            
            buy_positive = (df['buy_expected_return'] > threshold).sum()
            buy_negative = (df['buy_expected_return'] <= threshold).sum()
            sell_positive = (df['sell_expected_return'] > threshold).sum()
            sell_negative = (df['sell_expected_return'] <= threshold).sum()
            
            logger.info(f"   Buy:  Class 1: {buy_positive:,} ({buy_positive/len(df)*100:.1f}%), "
                       f"Class 0: {buy_negative:,} ({buy_negative/len(df)*100:.1f}%)")
            logger.info(f"   Sell: Class 1: {sell_positive:,} ({sell_positive/len(df)*100:.1f}%), "
                       f"Class 0: {sell_negative:,} ({sell_negative/len(df)*100:.1f}%)")
            
            # Соотношение классов
            buy_ratio = buy_negative / buy_positive if buy_positive > 0 else np.inf
            sell_ratio = sell_negative / sell_positive if sell_positive > 0 else np.inf
            
            logger.info(f"\n   Scale_pos_weight для buy: {buy_ratio:.2f}")
            logger.info(f"   Scale_pos_weight для sell: {sell_ratio:.2f}")
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("ПРОВЕРКА БАЛАНСА КЛАССОВ ПРИ РАЗНЫХ ПОРОГАХ")
    logger.info("="*60)
    check_thresholds()