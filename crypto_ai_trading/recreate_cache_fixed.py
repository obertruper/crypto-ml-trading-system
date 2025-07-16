#!/usr/bin/env python3
"""
Скрипт для пересоздания кэша данных с исправленным расчетом VWAP
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pickle
from datetime import datetime
import gc

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import CryptoDataLoader
from utils.logger import get_logger

def verify_vwap_fix():
    """Проверка что VWAP исправлен"""
    logger = get_logger("VWAPVerification")
    
    # Загрузка конфигурации
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Создаем загрузчик данных
    loader = CryptoDataLoader(config)
    
    # Загружаем небольшой сэмпл данных
    logger.info("🔍 Загрузка тестовых данных...")
    test_df = loader.load_raw_data(symbols=['BTCUSDT'], limit=1000)
    
    # Применяем feature engineering
    logger.info("🔧 Применение feature engineering...")
    from data.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(config)
    featured_df = engineer.create_features(test_df)
    
    # Проверяем close_vwap_ratio
    vwap_stats = featured_df['close_vwap_ratio'].describe()
    logger.info(f"📊 Статистика close_vwap_ratio:")
    logger.info(f"   Min: {vwap_stats['min']:.4f}")
    logger.info(f"   Max: {vwap_stats['max']:.4f}")
    logger.info(f"   Mean: {vwap_stats['mean']:.4f}")
    logger.info(f"   Std: {vwap_stats['std']:.4f}")
    
    if vwap_stats['max'] > 10:
        logger.error("❌ VWAP все еще имеет экстремальные значения!")
        return False
    
    logger.info("✅ VWAP исправлен успешно!")
    return True

def recreate_cache():
    """Пересоздание кэша с исправленными данными"""
    logger = get_logger("CacheRecreation")
    
    # Проверяем исправление
    if not verify_vwap_fix():
        logger.error("❌ Прерывание: VWAP не исправлен")
        return
    
    # Загрузка конфигурации
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Создаем загрузчик данных
    loader = CryptoDataLoader(config)
    
    # Путь к кэшу
    cache_dir = Path('data/processed')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Удаляем старый кэш
    logger.info("🗑️ Удаление старого кэша...")
    for file in ['train_data.parquet', 'val_data.parquet', 'test_data.parquet', 
                 'feature_columns.pkl', 'target_columns.pkl']:
        file_path = cache_dir / file
        if file_path.exists():
            file_path.unlink()
            logger.info(f"   Удален: {file}")
    
    # Загружаем и обрабатываем данные
    logger.info("📊 Загрузка полных данных из БД...")
    train_data, val_data, test_data = loader.prepare_trading_data(
        force_recreate=True,  # Принудительное пересоздание
        use_cache=False       # Не использовать кэш
    )
    
    # Проверяем результаты
    logger.info("\n📈 Статистика обновленных данных:")
    
    for name, df in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        if 'close_vwap_ratio' in df.columns:
            stats = df['close_vwap_ratio'].describe()
            logger.info(f"\n{name} - close_vwap_ratio:")
            logger.info(f"   Min: {stats['min']:.4f}")
            logger.info(f"   Max: {stats['max']:.4f}")
            logger.info(f"   Mean: {stats['mean']:.4f}")
            
            # Проверка на экстремальные значения
            extreme_count = (df['close_vwap_ratio'] > 10).sum()
            if extreme_count > 0:
                logger.warning(f"   ⚠️ Экстремальных значений (>10): {extreme_count}")
    
    # Проверяем размеры файлов
    logger.info("\n💾 Размеры кэш-файлов:")
    for file in cache_dir.glob('*.parquet'):
        size_mb = file.stat().st_size / (1024 * 1024)
        logger.info(f"   {file.name}: {size_mb:.2f} MB")
    
    logger.info("\n✅ Кэш успешно пересоздан с исправленными данными!")
    logger.info("🚀 Теперь можно запускать обучение: python main.py --mode train")

def check_current_cache():
    """Проверка текущего кэша на наличие проблем"""
    logger = get_logger("CacheCheck")
    
    cache_dir = Path('data/processed')
    
    logger.info("🔍 Проверка текущего кэша...")
    
    for file_name in ['train_data.parquet', 'val_data.parquet', 'test_data.parquet']:
        file_path = cache_dir / file_name
        
        if not file_path.exists():
            logger.warning(f"   ❌ {file_name} не найден")
            continue
            
        # Загружаем данные
        df = pd.read_parquet(file_path)
        logger.info(f"\n📊 {file_name}:")
        logger.info(f"   Размер: {len(df):,} записей")
        
        if 'close_vwap_ratio' in df.columns:
            stats = df['close_vwap_ratio'].describe()
            logger.info(f"   close_vwap_ratio - Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
            
            # Проверка на экстремальные значения
            extreme_count = (df['close_vwap_ratio'] > 10).sum()
            if extreme_count > 0:
                logger.error(f"   ❌ Найдено {extreme_count:,} экстремальных значений!")
                
                # Показываем примеры
                extreme_samples = df[df['close_vwap_ratio'] > 10][['symbol', 'datetime', 'close_vwap_ratio']].head(5)
                logger.error(f"   Примеры:\n{extreme_samples}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Управление кэшем данных')
    parser.add_argument('--check', action='store_true', help='Проверить текущий кэш')
    parser.add_argument('--recreate', action='store_true', help='Пересоздать кэш с исправлениями')
    parser.add_argument('--verify', action='store_true', help='Проверить исправление VWAP')
    
    args = parser.parse_args()
    
    if args.check:
        check_current_cache()
    elif args.verify:
        verify_vwap_fix()
    elif args.recreate:
        recreate_cache()
    else:
        # По умолчанию - проверка и предложение пересоздать
        check_current_cache()
        print("\n" + "="*80)
        print("Для пересоздания кэша с исправлениями запустите:")
        print("python recreate_cache_fixed.py --recreate")
        print("="*80)