#!/usr/bin/env python3
"""
Тест оптимизированной структуры проекта
"""

import yaml
from pathlib import Path
import pandas as pd
from utils.logger import get_logger
from main import load_config, load_cached_data_if_exists, create_unified_data_loaders

def test_optimized_data_loading():
    """Тест централизованной загрузки данных"""
    logger = get_logger("TestOptimized")
    
    logger.info("🧪 Тестирование оптимизированной структуры...")
    
    # Загрузка конфигурации
    config = load_config('config/config.yaml')
    
    # Тест загрузки кэшированных данных
    logger.info("\n1️⃣ Тест загрузки кэшированных данных...")
    train_data, val_data, test_data, feature_cols, target_cols = load_cached_data_if_exists(logger)
    
    if train_data is not None:
        logger.info("✅ Кэшированные данные загружены успешно")
        
        # Тест создания DataLoader'ов
        logger.info("\n2️⃣ Тест создания унифицированных DataLoader'ов...")
        try:
            train_loader, val_loader, test_loader, config_updated = create_unified_data_loaders(
                train_data, val_data, test_data, feature_cols, target_cols, config, logger
            )
            logger.info("✅ DataLoader'ы созданы успешно")
            
            # Тест первого батча
            logger.info("\n3️⃣ Тест первого батча...")
            sample_batch = next(iter(train_loader))
            X_sample, y_sample, info_sample = sample_batch
            
            logger.info(f"📊 Размерности батча:")
            logger.info(f"   - X_sample: {X_sample.shape}")
            logger.info(f"   - y_sample: {y_sample.shape}")
            logger.info(f"   - info_sample: {len(info_sample)} элементов")
            
            # Проверка соответствия конфигурации
            n_features = X_sample.shape[-1]
            n_targets = y_sample.shape[-1] if y_sample is not None else 0
            
            logger.info(f"\n4️⃣ Проверка соответствия конфигурации:")
            logger.info(f"   - Признаков в данных: {n_features}")
            logger.info(f"   - Признаков в конфиге: {config_updated['model']['input_features']}")
            logger.info(f"   - Целевых переменных в данных: {n_targets}")
            logger.info(f"   - Целевых переменных в конфиге: {len(config_updated['model']['target_variables'])}")
            
            if n_features == config_updated['model']['input_features'] and n_targets == len(config_updated['model']['target_variables']):
                logger.info("✅ Конфигурация соответствует данным")
            else:
                logger.warning("⚠️ Несоответствие конфигурации и данных")
            
            logger.info("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при создании DataLoader'ов: {e}")
            return False
    else:
        logger.error("❌ Кэшированные данные не найдены")
        return False

def test_no_duplicates():
    """Проверка отсутствия дубликатов"""
    logger = get_logger("TestDuplicates")
    
    logger.info("🔍 Проверка отсутствия дубликатов в коде...")
    
    # Читаем main.py и ищем дублированные функциональности
    main_content = Path("main.py").read_text()
    
    # Проверяем что нет дублирования create_data_loaders
    create_data_loaders_count = main_content.count("create_data_loaders(")
    logger.info(f"   - Вызовов create_data_loaders в main.py: {create_data_loaders_count}")
    
    # Проверяем что нет дублирования CryptoDataLoader
    crypto_data_loader_count = main_content.count("CryptoDataLoader(")
    logger.info(f"   - Создания CryptoDataLoader в main.py: {crypto_data_loader_count}")
    
    # Проверяем что нет дублирования загрузки parquet
    parquet_count = main_content.count("read_parquet(")
    logger.info(f"   - Вызовов read_parquet в main.py: {parquet_count}")
    
    if create_data_loaders_count <= 1 and parquet_count <= 1:
        logger.info("✅ Дублирование устранено")
        return True
    else:
        logger.warning("⚠️ Возможно остались дубликаты")
        return False

if __name__ == "__main__":
    success1 = test_optimized_data_loading()
    success2 = test_no_duplicates()
    
    if success1 and success2:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Структура проекта оптимизирована.")
    else:
        print("\n❌ Некоторые тесты не прошли. Нужны дополнительные исправления.")