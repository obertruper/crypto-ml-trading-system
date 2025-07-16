#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Проверка загрузки данных для обучения
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import Json
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """Менеджер для работы с PostgreSQL"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config.copy()
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
        self.connection = None
    
    def connect(self):
        """Создает подключение к БД"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = True
            logger.info("✅ Подключение к PostgreSQL установлено")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
            raise
    
    def disconnect(self):
        """Закрывает подключение к БД"""
        if self.connection:
            self.connection.close()
            logger.info("📤 Подключение к PostgreSQL закрыто")
    
    def fetch_dataframe(self, query: str, params=None) -> pd.DataFrame:
        """Выполняет запрос и возвращает результат как DataFrame"""
        try:
            return pd.read_sql_query(query, self.connection, params=params)
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса DataFrame: {e}")
            raise


def main():
    """Главная функция проверки"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    
    # Инициализируем менеджер БД
    db_manager = PostgreSQLManager(db_config)
    
    try:
        # Подключаемся к БД
        db_manager.connect()
        
        logger.info("\n" + "="*80)
        logger.info("ПРОВЕРКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        logger.info("="*80 + "\n")
        
        # 1. Проверка общего количества данных
        query_total = """
        SELECT COUNT(*) as total_count
        FROM processed_market_data
        WHERE technical_indicators IS NOT NULL
        """
        df_total = db_manager.fetch_dataframe(query_total)
        logger.info(f"📊 Всего записей с техническими индикаторами: {df_total['total_count'].iloc[0]:,}")
        
        # 2. Проверка данных для регрессии
        query_regression = """
        SELECT 
            p.symbol, p.timestamp, p.datetime,
            p.technical_indicators,
            p.buy_expected_return,
            p.sell_expected_return,
            p.open, p.high, p.low, p.close, p.volume
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
          AND (p.buy_expected_return <> 0 OR p.sell_expected_return <> 0)
        ORDER BY p.symbol, p.timestamp
        LIMIT 100
        """
        
        df_regression = db_manager.fetch_dataframe(query_regression)
        logger.info(f"\n📈 Данные для регрессии (первые 100 записей):")
        logger.info(f"   Количество: {len(df_regression)}")
        
        if len(df_regression) > 0:
            # Статистика по expected returns
            logger.info(f"\n📊 Статистика buy_expected_return:")
            logger.info(f"   Min: {df_regression['buy_expected_return'].min():.4f}%")
            logger.info(f"   Max: {df_regression['buy_expected_return'].max():.4f}%")
            logger.info(f"   Mean: {df_regression['buy_expected_return'].mean():.4f}%")
            logger.info(f"   Std: {df_regression['buy_expected_return'].std():.4f}%")
            logger.info(f"   Ненулевых: {(df_regression['buy_expected_return'] != 0).sum()}")
            
            logger.info(f"\n📊 Статистика sell_expected_return:")
            logger.info(f"   Min: {df_regression['sell_expected_return'].min():.4f}%")
            logger.info(f"   Max: {df_regression['sell_expected_return'].max():.4f}%")
            logger.info(f"   Mean: {df_regression['sell_expected_return'].mean():.4f}%")
            logger.info(f"   Std: {df_regression['sell_expected_return'].std():.4f}%")
            logger.info(f"   Ненулевых: {(df_regression['sell_expected_return'] != 0).sum()}")
            
            # Проверка technical_indicators
            first_row = df_regression.iloc[0]
            indicators = first_row['technical_indicators']
            
            logger.info(f"\n🔧 Технические индикаторы (первая запись):")
            logger.info(f"   Количество индикаторов: {len(indicators)}")
            logger.info(f"   Примеры индикаторов:")
            for key in list(indicators.keys())[:10]:
                logger.info(f"     {key}: {indicators[key]}")
            
            # Проверка наличия expected returns в JSON
            json_buy_return = indicators.get('buy_expected_return', 'NOT FOUND')
            json_sell_return = indicators.get('sell_expected_return', 'NOT FOUND')
            
            logger.info(f"\n⚠️ Expected returns в JSON:")
            logger.info(f"   buy_expected_return в JSON: {json_buy_return}")
            logger.info(f"   sell_expected_return в JSON: {json_sell_return}")
            logger.info(f"   buy_expected_return в колонке: {first_row['buy_expected_return']}")
            logger.info(f"   sell_expected_return в колонке: {first_row['sell_expected_return']}")
        
        # 3. Проверка полного запроса из train_universal_transformer.py
        query_full_count = """
        SELECT COUNT(*) as count
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
          AND (p.buy_expected_return <> 0 OR p.sell_expected_return <> 0)
        """
        
        df_count = db_manager.fetch_dataframe(query_full_count)
        logger.info(f"\n✅ Всего данных для обучения регрессии: {df_count['count'].iloc[0]:,}")
        
        # 4. Распределение по символам
        query_symbols = """
        SELECT p.symbol, COUNT(*) as count
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
          AND (p.buy_expected_return <> 0 OR p.sell_expected_return <> 0)
        GROUP BY p.symbol
        ORDER BY count DESC
        """
        
        df_symbols = db_manager.fetch_dataframe(query_symbols)
        logger.info(f"\n📊 Распределение по символам:")
        for _, row in df_symbols.iterrows():
            logger.info(f"   {row['symbol']}: {row['count']:,} записей")
        
        # 5. Проверка классификационных меток
        query_classification = """
        SELECT 
            SUM(buy_profit_target) as buy_profits,
            SUM(buy_loss_target) as buy_losses,
            SUM(sell_profit_target) as sell_profits,
            SUM(sell_loss_target) as sell_losses
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
        """
        
        df_class = db_manager.fetch_dataframe(query_classification)
        logger.info(f"\n🎯 Классификационные метки:")
        logger.info(f"   BUY прибыльных: {int(df_class['buy_profits'].iloc[0]):,}")
        logger.info(f"   BUY убыточных: {int(df_class['buy_losses'].iloc[0]):,}")
        logger.info(f"   SELL прибыльных: {int(df_class['sell_profits'].iloc[0]):,}")
        logger.info(f"   SELL убыточных: {int(df_class['sell_losses'].iloc[0]):,}")
        
        logger.info(f"\n🎉 Проверка завершена!")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()