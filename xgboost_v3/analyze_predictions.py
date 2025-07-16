#!/usr/bin/env python3
"""
Анализ предсказаний модели для проверки на переобучение
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_distribution():
    """Анализ распределения данных"""
    
    # 1. Загрузка данных
    logger.info("📥 Загрузка данных...")
    loader = DataLoader(settings)
    df = loader.load_data(no_cache=True)
    
    # 2. Инженерия признаков
    logger.info("🔬 Создание признаков...")
    engineer = FeatureEngineer()
    df = engineer.create_features(df)
    
    # 3. Предобработка
    logger.info("🔧 Предобработка...")
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df)
    
    # 4. Анализ expected returns
    logger.info("\n📊 АНАЛИЗ EXPECTED RETURNS:")
    
    for target in ['buy_expected_return', 'sell_expected_return']:
        logger.info(f"\n{target}:")
        data = df[target].astype(float)
        
        # Статистика
        logger.info(f"  Среднее: {data.mean():.3f}%")
        logger.info(f"  Медиана: {data.median():.3f}%")
        logger.info(f"  Std: {data.std():.3f}%")
        logger.info(f"  Min: {data.min():.3f}%")
        logger.info(f"  Max: {data.max():.3f}%")
        
        # Распределение по порогам
        thresholds = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
        logger.info("\n  Распределение по порогам:")
        for thresh in thresholds:
            pct = (data > thresh).mean() * 100
            logger.info(f"    > {thresh}%: {pct:.1f}%")
    
    # 5. Анализ по символам
    logger.info("\n📊 АНАЛИЗ ПО СИМВОЛАМ:")
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol]
        logger.info(f"\n{symbol} ({len(symbol_data)} записей):")
        
        for target in ['buy_expected_return', 'sell_expected_return']:
            data = symbol_data[target].astype(float)
            pos_rate = (data > 1.5).mean() * 100
            logger.info(f"  {target} > 1.5%: {pos_rate:.1f}%")
    
    # 6. Временной анализ
    logger.info("\n📊 ВРЕМЕННОЙ АНАЛИЗ:")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['date'] = df['timestamp'].dt.date
    
    # Группировка по дням
    daily_stats = df.groupby('date').agg({
        'buy_expected_return': lambda x: (x.astype(float) > 1.5).mean() * 100,
        'sell_expected_return': lambda x: (x.astype(float) > 1.5).mean() * 100
    })
    
    logger.info(f"  Дней в данных: {len(daily_stats)}")
    logger.info(f"  Buy > 1.5% (среднее по дням): {daily_stats['buy_expected_return'].mean():.1f}%")
    logger.info(f"  Buy > 1.5% (std по дням): {daily_stats['buy_expected_return'].std():.1f}%")
    
    # 7. Корреляция между buy и sell
    buy_binary = (df['buy_expected_return'].astype(float) > 1.5).astype(int)
    sell_binary = (df['sell_expected_return'].astype(float) > 1.5).astype(int)
    correlation = buy_binary.corr(sell_binary)
    logger.info(f"\n📊 Корреляция между buy и sell сигналами: {correlation:.3f}")
    
    # 8. Анализ последовательных сигналов
    logger.info("\n📊 АНАЛИЗ ПОСЛЕДОВАТЕЛЬНЫХ СИГНАЛОВ:")
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('timestamp')
        
        buy_signals = (symbol_data['buy_expected_return'].astype(float) > 1.5)
        
        # Проверка кластеризации сигналов
        signal_changes = buy_signals.diff().abs().sum()
        max_possible_changes = len(buy_signals) - 1
        change_rate = signal_changes / max_possible_changes
        
        logger.info(f"  {symbol} - частота изменения сигналов: {change_rate:.3f}")

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ДАННЫХ И ПРЕДСКАЗАНИЙ")
    logger.info("="*60)
    analyze_data_distribution()