#!/usr/bin/env python3
"""
Анализ проблемы с обучением XGBoost v3
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

def analyze_data_distribution():
    """Анализ распределения данных и expected returns"""
    # Подключение к БД
    conn = psycopg2.connect(
        host="localhost",
        port=5555,
        database="crypto_trading",
        user="ruslan"
    )
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Получаем expected returns
            logger.info("📊 Загрузка expected returns...")
            cursor.execute("""
                SELECT buy_expected_return, sell_expected_return
                FROM processed_market_data
                WHERE symbol IN ('BTCUSDT', 'ETHUSDT')
                AND buy_expected_return IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 200000
            """)
            
            data = cursor.fetchall()
            df = pd.DataFrame(data)
            
            # Конвертируем в float
            df['buy_expected_return'] = df['buy_expected_return'].astype(float)
            df['sell_expected_return'] = df['sell_expected_return'].astype(float)
            
            logger.info(f"   Загружено {len(df)} записей")
            
            # Анализ распределения
            logger.info("\n📈 Статистика expected returns:")
            logger.info(f"   Buy - mean: {df['buy_expected_return'].mean():.3f}%, std: {df['buy_expected_return'].std():.3f}%")
            logger.info(f"   Buy - min: {df['buy_expected_return'].min():.3f}%, max: {df['buy_expected_return'].max():.3f}%")
            logger.info(f"   Buy - квантили: 25%={df['buy_expected_return'].quantile(0.25):.3f}%, "
                       f"50%={df['buy_expected_return'].quantile(0.5):.3f}%, "
                       f"75%={df['buy_expected_return'].quantile(0.75):.3f}%, "
                       f"95%={df['buy_expected_return'].quantile(0.95):.3f}%")
            
            # Проверка разных порогов
            thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            
            logger.info("\n📊 Баланс классов при разных порогах:")
            logger.info("Порог | Buy Class 1 % | scale_pos_weight | Рекомендация")
            logger.info("-" * 70)
            
            for threshold in thresholds:
                buy_class_1 = (df['buy_expected_return'] > threshold).mean() * 100
                
                if buy_class_1 > 0:
                    scale_pos_weight = (100 - buy_class_1) / buy_class_1
                    # С учетом ограничения в коде (70% от реального)
                    adjusted_spw = min(3.0, scale_pos_weight * 0.7)
                else:
                    scale_pos_weight = np.inf
                    adjusted_spw = 3.0
                
                # Рекомендация
                if buy_class_1 < 5:
                    rec = "❌ Слишком мало положительных"
                elif buy_class_1 > 40:
                    rec = "❌ Слишком много положительных"
                elif 15 <= buy_class_1 <= 25:
                    rec = "✅ Хороший баланс"
                else:
                    rec = "⚠️ Приемлемо"
                
                logger.info(f"{threshold:4.1f}% | {buy_class_1:13.1f}% | {adjusted_spw:16.2f} | {rec}")
            
            # Анализ информативности expected returns
            logger.info("\n🔍 Анализ информативности expected returns:")
            
            # Создаем метки для порога 1.5%
            y_buy = (df['buy_expected_return'] > 1.5).astype(int)
            
            # Смотрим на распределение returns в каждом классе
            logger.info(f"\n   Класс 0 (не входить): {(y_buy == 0).sum()} примеров")
            logger.info(f"   Expected return: mean={df.loc[y_buy == 0, 'buy_expected_return'].mean():.3f}%, "
                       f"std={df.loc[y_buy == 0, 'buy_expected_return'].std():.3f}%")
            
            logger.info(f"\n   Класс 1 (входить): {(y_buy == 1).sum()} примеров")
            logger.info(f"   Expected return: mean={df.loc[y_buy == 1, 'buy_expected_return'].mean():.3f}%, "
                       f"std={df.loc[y_buy == 1, 'buy_expected_return'].std():.3f}%")
            
            # Проверка перекрытия распределений
            class0_95p = df.loc[y_buy == 0, 'buy_expected_return'].quantile(0.95)
            class1_5p = df.loc[y_buy == 1, 'buy_expected_return'].quantile(0.05)
            
            logger.info(f"\n   Перекрытие классов:")
            logger.info(f"   Класс 0 (95-перцентиль): {class0_95p:.3f}%")
            logger.info(f"   Класс 1 (5-перцентиль): {class1_5p:.3f}%")
            
            if class0_95p < class1_5p:
                logger.info("   ✅ Классы хорошо разделены")
            else:
                logger.info("   ⚠️ Классы перекрываются - это усложняет обучение")
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("АНАЛИЗ ПРОБЛЕМЫ С ОБУЧЕНИЕМ XGBoost v3")
    logger.info("="*70)
    analyze_data_distribution()