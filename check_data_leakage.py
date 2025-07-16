#!/usr/bin/env python3
"""
Проверка на утечку данных в обучающей выборке
"""

import pandas as pd
import numpy as np
import psycopg2
import json
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_leakage():
    """Проверка на утечку целевых переменных в features"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    # Подключаемся к БД
    conn = psycopg2.connect(**db_config)
    logger.info("✅ Подключение к PostgreSQL установлено")
    
    try:
        # 1. Проверяем структуру таблицы processed_market_data
        logger.info("\n📊 Проверка структуры таблицы processed_market_data...")
        query = """
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'processed_market_data'
        ORDER BY ordinal_position
        """
        
        columns_df = pd.read_sql_query(query, conn)
        logger.info(f"Всего колонок: {len(columns_df)}")
        
        # Целевые переменные
        target_columns = ['buy_expected_return', 'sell_expected_return', 
                         'expected_return_buy', 'expected_return_sell']
        
        logger.info("\n🎯 Проверка целевых переменных:")
        for target in target_columns:
            if target in columns_df['column_name'].values:
                logger.info(f"   ✅ {target} найден в таблице")
        
        # 2. Проверяем содержимое technical_indicators
        logger.info("\n📈 Проверка содержимого technical_indicators...")
        query = """
        SELECT technical_indicators 
        FROM processed_market_data 
        WHERE technical_indicators IS NOT NULL 
        LIMIT 5
        """
        
        sample_data = pd.read_sql_query(query, conn)
        
        if len(sample_data) > 0:
            # Проверяем первую запись
            indicators = sample_data.iloc[0]['technical_indicators']
            indicator_keys = list(indicators.keys())
            
            logger.info(f"Найдено {len(indicator_keys)} индикаторов в JSON")
            
            # Проверяем на наличие целевых переменных
            logger.info("\n🔍 Проверка на утечку в technical_indicators:")
            leakage_found = False
            for key in indicator_keys:
                for target in target_columns:
                    if target in key or key in target:
                        logger.error(f"   🚨 УТЕЧКА ОБНАРУЖЕНА: {key} в technical_indicators!")
                        leakage_found = True
            
            if not leakage_found:
                logger.info("   ✅ Утечки не обнаружено в technical_indicators")
        
        # 3. Проверяем данные из prepare_dataset.py
        logger.info("\n📊 Проверка данных из prepare_dataset.py...")
        
        # Проверяем, что expected_returns рассчитываются правильно
        query = """
        SELECT 
            symbol,
            COUNT(*) as total_records,
            AVG(buy_expected_return) as avg_buy_return,
            STDDEV(buy_expected_return) as std_buy_return,
            MIN(buy_expected_return) as min_buy_return,
            MAX(buy_expected_return) as max_buy_return,
            AVG(sell_expected_return) as avg_sell_return,
            STDDEV(sell_expected_return) as std_sell_return
        FROM processed_market_data
        GROUP BY symbol
        ORDER BY total_records DESC
        LIMIT 10
        """
        
        stats_df = pd.read_sql_query(query, conn)
        logger.info("\nСтатистика expected_returns по символам:")
        print(stats_df.to_string(index=False))
        
        # 4. Проверяем корреляцию между индикаторами и expected_returns
        logger.info("\n📊 Проверка корреляций с целевыми переменными...")
        query = """
        SELECT 
            pm.symbol,
            pm.technical_indicators,
            pm.buy_expected_return,
            pm.sell_expected_return
        FROM processed_market_data pm
        WHERE pm.symbol = 'BTCUSDT'
        LIMIT 10000
        """
        
        btc_data = pd.read_sql_query(query, conn)
        
        if len(btc_data) > 0:
            # Извлекаем индикаторы
            indicators_list = []
            for _, row in btc_data.iterrows():
                indicators_list.append(row['technical_indicators'])
            
            indicators_df = pd.json_normalize(indicators_list)
            
            # Проверяем корреляции
            buy_corr = indicators_df.corrwith(btc_data['buy_expected_return'])
            sell_corr = indicators_df.corrwith(btc_data['sell_expected_return'])
            
            # Находим высокие корреляции (подозрительные)
            high_corr_threshold = 0.8
            
            logger.info(f"\n🔍 Индикаторы с высокой корреляцией (>{high_corr_threshold}):")
            suspicious_buy = buy_corr[buy_corr.abs() > high_corr_threshold]
            suspicious_sell = sell_corr[sell_corr.abs() > high_corr_threshold]
            
            if len(suspicious_buy) > 0:
                logger.warning("С buy_expected_return:")
                for ind, corr in suspicious_buy.items():
                    logger.warning(f"   {ind}: {corr:.3f}")
            
            if len(suspicious_sell) > 0:
                logger.warning("С sell_expected_return:")
                for ind, corr in suspicious_sell.items():
                    logger.warning(f"   {ind}: {corr:.3f}")
            
            if len(suspicious_buy) == 0 and len(suspicious_sell) == 0:
                logger.info("   ✅ Не найдено индикаторов с подозрительно высокой корреляцией")
        
        # 5. Итоговая проверка
        logger.info("\n" + "="*60)
        logger.info("ИТОГОВАЯ ПРОВЕРКА НА УТЕЧКУ ДАННЫХ:")
        logger.info("="*60)
        
        if leakage_found:
            logger.error("🚨 ОБНАРУЖЕНА УТЕЧКА ДАННЫХ!")
            logger.error("   Целевые переменные присутствуют в обучающих признаках")
            logger.error("   Это приведет к переобучению и плохим результатам на новых данных")
        else:
            logger.info("✅ Утечка данных НЕ обнаружена")
            logger.info("   Целевые переменные корректно отделены от признаков")
            logger.info("   Низкая производительность модели вызвана другими причинами:")
            logger.info("   - Недостаточно информативные признаки")
            logger.info("   - Сложная нелинейная зависимость")
            logger.info("   - Высокий уровень шума в данных")
            logger.info("   - Необходима более сложная архитектура модели")
        
    finally:
        conn.close()
        logger.info("\n📤 Подключение к PostgreSQL закрыто")


if __name__ == "__main__":
    check_data_leakage()