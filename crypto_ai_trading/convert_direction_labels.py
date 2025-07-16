"""
Конвертация категориальных direction переменных в числовые
"""

import pandas as pd
import numpy as np
from pathlib import Path

def convert_direction_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Конвертирует категориальные direction переменные в числовые
    
    Маппинг:
    - UP -> 0
    - DOWN -> 1  
    - FLAT -> 2
    """
    # Создаем копию
    df = df.copy()
    
    # Маппинг для конвертации
    direction_mapping = {
        'UP': 0,
        'DOWN': 1,
        'FLAT': 2
    }
    
    # Находим все direction колонки
    direction_cols = [col for col in df.columns if col.startswith('direction_')]
    
    print(f"🔄 Конвертация {len(direction_cols)} direction колонок...")
    
    for col in direction_cols:
        if pd.api.types.is_categorical_dtype(df[col]):
            # Конвертируем категориальные в строки, затем в числа
            df[col] = df[col].astype(str).map(direction_mapping)
        elif df[col].dtype == 'object':
            # Если уже строки
            df[col] = df[col].map(direction_mapping)
        else:
            # Уже числовые - проверяем диапазон
            unique_vals = df[col].unique()
            if not all(v in [0, 1, 2] for v in unique_vals if pd.notna(v)):
                print(f"⚠️ Неожиданные значения в {col}: {unique_vals}")
        
        # Проверка результата
        print(f"   {col}: {df[col].value_counts().sort_index().to_dict()}")
    
    return df


def convert_all_datasets():
    """Конвертирует все датасеты"""
    
    print("🚀 Начинаем конвертацию direction меток...")
    
    # Пути к файлам
    data_files = {
        'train': Path('data/processed/train_data.parquet'),
        'val': Path('data/processed/val_data.parquet'),
        'test': Path('data/processed/test_data.parquet')
    }
    
    # Создаем backup директорию
    backup_dir = Path('data/processed/backup')
    backup_dir.mkdir(exist_ok=True)
    
    for name, file_path in data_files.items():
        if file_path.exists():
            print(f"\n📊 Обработка {name} данных...")
            
            # Загружаем данные
            df = pd.read_parquet(file_path)
            print(f"   Загружено: {len(df)} строк")
            
            # Создаем backup
            backup_path = backup_dir / f"{name}_data_backup.parquet"
            df.to_parquet(backup_path)
            print(f"   Backup сохранен: {backup_path}")
            
            # Конвертируем
            df_converted = convert_direction_to_numeric(df)
            
            # Сохраняем обратно
            df_converted.to_parquet(file_path)
            print(f"   ✅ Сохранено: {file_path}")
    
    print("\n✅ Конвертация завершена!")
    print("📁 Backup файлы сохранены в data/processed/backup/")


if __name__ == "__main__":
    convert_all_datasets()