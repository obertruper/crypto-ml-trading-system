#!/usr/bin/env python3
"""
Простой скрипт для пересоздания кэша данных
"""

import sys
import os
from pathlib import Path
import shutil

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger

def main():
    logger = get_logger("CacheRecreation")
    
    # Путь к кэшу
    cache_dir = Path('data/processed')
    
    logger.info("🗑️ Удаление старого кэша...")
    
    # Список файлов для удаления
    cache_files = [
        'train_data.parquet',
        'val_data.parquet', 
        'test_data.parquet',
        'feature_columns.pkl',
        'target_columns.pkl'
    ]
    
    for file_name in cache_files:
        file_path = cache_dir / file_name
        if file_path.exists():
            file_path.unlink()
            logger.info(f"   ✅ Удален: {file_name}")
    
    logger.info("\n📊 Теперь запустите main.py для создания нового кэша:")
    logger.info("   python main.py --mode train")
    logger.info("\n💡 Новый кэш будет создан автоматически с исправленными данными")

if __name__ == "__main__":
    main()