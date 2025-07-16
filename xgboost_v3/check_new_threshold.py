#!/usr/bin/env python3
"""
Проверка баланса классов с новым порогом 1.5%
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

def check_new_threshold():
    """Проверка распределения классов с порогом 1.5%"""
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
                WHERE symbol IN ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
                AND buy_expected_return IS NOT NULL
                LIMIT 100000
            """)
            
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            
            # Конвертируем Decimal в float
            df['buy_expected_return'] = df['buy_expected_return'].astype(float)
            df['sell_expected_return'] = df['sell_expected_return'].astype(float)
            
            logger.info(f"   Загружено {len(df)} записей")
            
            # Статистика распределения
            logger.info("\n📈 Статистика expected returns:")
            logger.info(f"   Buy mean: {df['buy_expected_return'].mean():.3f}%")
            logger.info(f"   Buy std: {df['buy_expected_return'].std():.3f}%")
            logger.info(f"   Buy 95-percentile: {df['buy_expected_return'].quantile(0.95):.3f}%")
            logger.info(f"   Sell mean: {df['sell_expected_return'].mean():.3f}%")
            logger.info(f"   Sell std: {df['sell_expected_return'].std():.3f}%")
            logger.info(f"   Sell 95-percentile: {df['sell_expected_return'].quantile(0.95):.3f}%")
            
            # Проверяем разные пороги
            thresholds = [0.7, 1.0, 1.5, 2.0, 2.5]
            
            logger.info("\n📊 Распределение классов при разных порогах:")
            logger.info("Порог | Buy Class 1 % | Sell Class 1 % | Buy scale_pos_weight | Sell scale_pos_weight")
            logger.info("-" * 80)
            
            for threshold in thresholds:
                buy_class_1 = (df['buy_expected_return'] > threshold).mean() * 100
                sell_class_1 = (df['sell_expected_return'] > threshold).mean() * 100
                
                # Рассчитываем scale_pos_weight
                buy_pos = (df['buy_expected_return'] > threshold).sum()
                buy_neg = (df['buy_expected_return'] <= threshold).sum()
                sell_pos = (df['sell_expected_return'] > threshold).sum()
                sell_neg = (df['sell_expected_return'] <= threshold).sum()
                
                buy_spw = buy_neg / buy_pos if buy_pos > 0 else np.inf
                sell_spw = sell_neg / sell_pos if sell_pos > 0 else np.inf
                
                # Ограничиваем до 5
                buy_spw = min(buy_spw, 5.0)
                sell_spw = min(sell_spw, 5.0)
                
                logger.info(f"{threshold:4.1f}% | {buy_class_1:13.1f}% | {sell_class_1:14.1f}% | "
                           f"{buy_spw:20.2f} | {sell_spw:21.2f}")
            
            # Детальная статистика для порога 1.5%
            logger.info("\n📊 Детальная статистика для порога 1.5%:")
            threshold = 1.5
            
            buy_positive = (df['buy_expected_return'] > threshold).sum()
            buy_negative = (df['buy_expected_return'] <= threshold).sum()
            sell_positive = (df['sell_expected_return'] > threshold).sum()
            sell_negative = (df['sell_expected_return'] <= threshold).sum()
            
            logger.info(f"   Buy:  Class 1: {buy_positive:,} ({buy_positive/len(df)*100:.1f}%), "
                       f"Class 0: {buy_negative:,} ({buy_negative/len(df)*100:.1f}%)")
            logger.info(f"   Sell: Class 1: {sell_positive:,} ({sell_positive/len(df)*100:.1f}%), "
                       f"Class 0: {sell_negative:,} ({sell_negative/len(df)*100:.1f}%)")
            
            # Проверяем качество сигналов с порогом 1.5%
            logger.info("\n💡 Анализ качества сигналов с порогом 1.5%:")
            
            # Buy сигналы
            buy_signals = df[df['buy_expected_return'] > threshold]
            if len(buy_signals) > 0:
                logger.info(f"   Buy сигналы ({len(buy_signals)} шт):")
                logger.info(f"      Средний return: {buy_signals['buy_expected_return'].mean():.2f}%")
                logger.info(f"      Медиана return: {buy_signals['buy_expected_return'].median():.2f}%")
                logger.info(f"      90-percentile: {buy_signals['buy_expected_return'].quantile(0.9):.2f}%")
            
            # Sell сигналы
            sell_signals = df[df['sell_expected_return'] > threshold]
            if len(sell_signals) > 0:
                logger.info(f"   Sell сигналы ({len(sell_signals)} шт):")
                logger.info(f"      Средний return: {sell_signals['sell_expected_return'].mean():.2f}%")
                logger.info(f"      Медиана return: {sell_signals['sell_expected_return'].median():.2f}%")
                logger.info(f"      90-percentile: {sell_signals['sell_expected_return'].quantile(0.9):.2f}%")
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("ПРОВЕРКА НОВОГО ПОРОГА 1.5% ДЛЯ КЛАССИФИКАЦИИ")
    logger.info("="*80)
    check_new_threshold()