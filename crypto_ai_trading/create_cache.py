#!/usr/bin/env python3
"""
Скрипт для создания кэша данных
"""

import yaml
from data.data_loader import CryptoDataLoader
from utils.logger import get_logger

def main():
    logger = get_logger("CacheCreation")
    
    # Загрузка конфигурации
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Создаем загрузчик данных
    loader = CryptoDataLoader(config)
    
    logger.info("🚀 Запуск создания кэша данных...")
    logger.info("📊 Это может занять 10-15 минут...")
    
    # Принудительное создание нового кэша
    train_data, val_data, test_data = loader.prepare_trading_data(
        force_recreate=True,  # Принудительное пересоздание
        use_cache=False       # Не использовать старый кэш
    )
    
    logger.info("\n✅ Кэш успешно создан!")
    logger.info(f"📈 Размеры датасетов:")
    logger.info(f"   - Train: {len(train_data):,} записей")
    logger.info(f"   - Val: {len(val_data):,} записей")
    logger.info(f"   - Test: {len(test_data):,} записей")
    
    # Проверка close_vwap_ratio
    if 'close_vwap_ratio' in train_data.columns:
        stats = train_data['close_vwap_ratio'].describe()
        logger.info(f"\n📊 Статистика close_vwap_ratio:")
        logger.info(f"   - Min: {stats['min']:.4f}")
        logger.info(f"   - Max: {stats['max']:.4f}")
        logger.info(f"   - Mean: {stats['mean']:.4f}")
        
        if stats['max'] > 10:
            logger.warning("⚠️ Обнаружены экстремальные значения! Проверьте feature_engineering.py")
        else:
            logger.info("✅ Все значения в норме!")

if __name__ == "__main__":
    main()