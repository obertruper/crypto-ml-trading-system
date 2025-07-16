#!/usr/bin/env python3
"""
Скрипт для валидации структуры данных перед обучением
"""

import pandas as pd
import yaml
from pathlib import Path
from data.constants import (
    TRADING_TARGET_VARIABLES, ADDITIONAL_TARGET_VARIABLES,
    SERVICE_COLUMNS, validate_data_structure
)
from utils.logger import get_logger

def main():
    logger = get_logger("DataValidator")
    
    logger.info("="*80)
    logger.info("🔍 Валидация структуры данных")
    logger.info("="*80)
    
    # Проверка наличия файлов
    data_dir = Path("data/processed")
    files = {
        'train': data_dir / "train_data.parquet",
        'val': data_dir / "val_data.parquet", 
        'test': data_dir / "test_data.parquet"
    }
    
    missing_files = [name for name, path in files.items() if not path.exists()]
    if missing_files:
        logger.error(f"❌ Отсутствуют файлы: {missing_files}")
        logger.error("Запустите: python prepare_trading_data.py")
        return False
    
    logger.info("✅ Все файлы найдены")
    
    # Загрузка конфигурации
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config_targets = config['model']['target_variables']
    logger.info(f"\n📋 Целевые переменные из config.yaml: {len(config_targets)}")
    
    # Проверка каждого файла
    all_valid = True
    for name, path in files.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Проверка {name}_data.parquet")
        logger.info(f"{'='*40}")
        
        df = pd.read_parquet(path)
        logger.info(f"Размер: {df.shape}")
        
        try:
            info = validate_data_structure(df)
            logger.info(f"✅ Признаков: {info['n_features']}")
            logger.info(f"✅ Целевых переменных: {info['n_targets']}")
            
            # Проверка соответствия с конфигом
            df_targets = set(info['target_cols'])
            config_targets_set = set(config_targets)
            
            if df_targets != config_targets_set:
                logger.warning("⚠️ Несоответствие целевых переменных с конфигом!")
                only_in_df = df_targets - config_targets_set
                only_in_config = config_targets_set - df_targets
                
                if only_in_df:
                    logger.warning(f"   Только в данных: {only_in_df}")
                if only_in_config:
                    logger.error(f"   ❌ Только в конфиге: {only_in_config}")
                    all_valid = False
                    
            # Проверка дополнительных переменных
            additional_found = [col for col in ADDITIONAL_TARGET_VARIABLES if col in df.columns]
            if additional_found:
                logger.info(f"ℹ️ Найдено {len(additional_found)} дополнительных целевых переменных")
                logger.info("   (они не используются для обучения)")
                
        except ValueError as e:
            logger.error(f"❌ Ошибка валидации: {e}")
            all_valid = False
    
    # Итоговый вердикт
    logger.info("\n" + "="*80)
    if all_valid:
        logger.info("✅ ВАЛИДАЦИЯ ПРОЙДЕНА УСПЕШНО!")
        logger.info("\nМожно запускать обучение:")
        logger.info("  python main.py --mode train")
    else:
        logger.error("❌ ВАЛИДАЦИЯ НЕ ПРОЙДЕНА!")
        logger.error("\nИсправьте ошибки и запустите валидацию снова")
    logger.info("="*80)
    
    return all_valid

if __name__ == "__main__":
    main()