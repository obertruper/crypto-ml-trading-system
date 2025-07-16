#!/usr/bin/env python3
"""
Диагностика проблемы с бинарными признаками
"""

import pandas as pd
import numpy as np
import psycopg2
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_binary_features():
    # Загрузка конфигурации
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    
    # Подключение к БД
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['dbname'],
        user=db_config['user']
    )
    
    # Загружаем данные с техническими индикаторами
    query = """
    SELECT 
        pm.symbol, pm.timestamp,
        pm.technical_indicators,
        rm.open, rm.high, rm.low, rm.close, rm.volume
    FROM processed_market_data pm
    JOIN raw_market_data rm ON pm.raw_data_id = rm.id
    WHERE pm.symbol IN ('BTCUSDT', 'ETHUSDT')
    ORDER BY pm.timestamp DESC
    LIMIT 1000
    """
    
    df = pd.read_sql_query(query, conn)
    logger.info(f"📊 Загружено {len(df)} записей")
    
    # Извлекаем технические индикаторы
    indicators_df = pd.json_normalize(df['technical_indicators'])
    df = pd.concat([df, indicators_df], axis=1)
    
    # 1. Проверяем RSI
    logger.info("\n📈 АНАЛИЗ RSI:")
    if 'rsi_val' in df.columns:
        rsi_stats = df['rsi_val'].describe()
        logger.info(f"  Статистика: {rsi_stats}")
        
        # Создаем бинарные признаки
        rsi_oversold = (df['rsi_val'] < 30).astype(int)
        rsi_overbought = (df['rsi_val'] > 70).astype(int)
        
        logger.info(f"  Oversold (RSI < 30): {rsi_oversold.sum()} из {len(df)} ({rsi_oversold.mean()*100:.1f}%)")
        logger.info(f"  Overbought (RSI > 70): {rsi_overbought.sum()} из {len(df)} ({rsi_overbought.mean()*100:.1f}%)")
        
        # Показываем примеры
        oversold_examples = df[df['rsi_val'] < 30]['rsi_val'].head()
        if len(oversold_examples) > 0:
            logger.info(f"  Примеры oversold RSI: {oversold_examples.values}")
        
        overbought_examples = df[df['rsi_val'] > 70]['rsi_val'].head()
        if len(overbought_examples) > 0:
            logger.info(f"  Примеры overbought RSI: {overbought_examples.values}")
    else:
        logger.error("  ❌ RSI не найден!")
    
    # 2. Проверяем MACD
    logger.info("\n📈 АНАЛИЗ MACD:")
    if 'macd_hist' in df.columns:
        macd_stats = df['macd_hist'].describe()
        logger.info(f"  Статистика: {macd_stats}")
        
        # Создаем бинарный признак
        macd_bullish = (df['macd_hist'] > 0).astype(int)
        
        logger.info(f"  Bullish (MACD > 0): {macd_bullish.sum()} из {len(df)} ({macd_bullish.mean()*100:.1f}%)")
        logger.info(f"  Bearish (MACD <= 0): {(1-macd_bullish).sum()} из {len(df)} ({(1-macd_bullish).mean()*100:.1f}%)")
        
        # Показываем распределение
        positive_macd = df[df['macd_hist'] > 0]['macd_hist']
        negative_macd = df[df['macd_hist'] <= 0]['macd_hist']
        logger.info(f"  Позитивные значения: min={positive_macd.min():.4f}, max={positive_macd.max():.4f}")
        logger.info(f"  Негативные значения: min={negative_macd.min():.4f}, max={negative_macd.max():.4f}")
    else:
        logger.error("  ❌ MACD не найден!")
    
    # 3. Проверяем is_bullish
    logger.info("\n📈 АНАЛИЗ IS_BULLISH:")
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    logger.info(f"  Bullish свечей: {df['is_bullish'].sum()} из {len(df)} ({df['is_bullish'].mean()*100:.1f}%)")
    
    # 4. Проверяем проблему нормализации
    logger.info("\n🔍 ПРОВЕРКА НОРМАЛИЗАЦИИ:")
    
    # Симулируем нормализацию как в основном скрипте
    from sklearn.preprocessing import StandardScaler
    
    # Берем несколько индикаторов для теста
    test_features = ['rsi_val', 'macd_hist', 'adx_val'] 
    test_data = df[test_features].copy()
    
    # До нормализации
    logger.info("  До нормализации:")
    for col in test_features:
        if col in test_data.columns:
            logger.info(f"    {col}: mean={test_data[col].mean():.4f}, std={test_data[col].std():.4f}")
    
    # Нормализация
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(test_data)
    normalized_df = pd.DataFrame(normalized_data, columns=test_features)
    
    # После нормализации
    logger.info("  После нормализации:")
    for col in test_features:
        logger.info(f"    {col}: mean={normalized_df[col].mean():.4f}, std={normalized_df[col].std():.4f}")
    
    # 5. Проверяем, не теряются ли бинарные признаки после нормализации
    logger.info("\n⚠️ ПРОВЕРКА ПОТЕРИ БИНАРНЫХ ПРИЗНАКОВ:")
    
    # Создаем полный набор признаков
    df['rsi_oversold'] = (df['rsi_val'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_val'] > 70).astype(int)
    df['macd_bullish'] = (df['macd_hist'] > 0).astype(int)
    
    binary_features = ['rsi_oversold', 'rsi_overbought', 'macd_bullish', 'is_bullish']
    
    logger.info("  Значения бинарных признаков ДО обработки:")
    for feat in binary_features:
        if feat in df.columns:
            unique_vals = df[feat].unique()
            value_counts = df[feat].value_counts()
            logger.info(f"    {feat}: уникальные значения={unique_vals}, распределение={dict(value_counts)}")
    
    conn.close()
    logger.info("\n✅ Диагностика завершена")

if __name__ == "__main__":
    diagnose_binary_features()