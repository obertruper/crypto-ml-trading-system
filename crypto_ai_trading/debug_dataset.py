"""
Отладка датасета для Direction модели
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Импорты из проекта
from train_direction_model import DirectionDatasetAdapter
from utils.config import load_config

def debug_dataset():
    """Проверка структуры данных в датасете"""
    
    print("🔍 Отладка датасета...")
    
    # Загружаем данные
    train_data = pd.read_parquet("data/processed/train_data.parquet")
    
    print(f"\n📊 Загружено данных: {len(train_data)} строк")
    print(f"   Колонки: {train_data.shape[1]}")
    
    # Проверяем direction колонки
    direction_cols = [col for col in train_data.columns if col.startswith('direction_')]
    print(f"\n🎯 Direction колонки: {direction_cols}")
    
    if direction_cols:
        for col in direction_cols[:2]:  # Первые 2 для примера
            print(f"\n   {col}:")
            print(f"   - Уникальные значения: {train_data[col].unique()}")
            print(f"   - Распределение: {train_data[col].value_counts().to_dict()}")
    
    # Проверяем future_return колонки
    return_cols = [col for col in train_data.columns if col.startswith('future_return_')]
    print(f"\n💰 Future return колонки: {return_cols}")
    
    # Создаем датасет
    config = load_config('configs/direction_only.yaml')
    
    feature_columns = [col for col in train_data.columns 
                      if col not in ['id', 'symbol', 'datetime', 'timestamp']
                      and not col.startswith(('target_', 'future_', 'direction_', 'optimal_'))]
    
    print(f"\n✨ Признаков: {len(feature_columns)}")
    
    # Создаем небольшой датасет для теста
    small_data = train_data.head(1000)
    
    dataset = DirectionDatasetAdapter(
        small_data,
        context_window=168,
        feature_cols=feature_columns,
        target_cols=direction_cols,
        stride=1,
        normalize=False
    )
    
    print(f"\n📦 Датасет создан: {len(dataset)} примеров")
    
    # Проверяем первый пример
    if len(dataset) > 0:
        features, targets, info = dataset[0]
        
        print(f"\n🔍 Первый пример:")
        print(f"   Features shape: {features.shape}")
        print(f"   Targets: {targets}")
        print(f"   Price changes: {info.get('price_changes', {})}")
        
        # Проверяем targets
        for key, value in targets.items():
            print(f"\n   {key}:")
            print(f"   - Shape: {value.shape}")
            print(f"   - Type: {value.dtype}")
            print(f"   - Value: {value.item() if value.numel() == 1 else value}")
    
    # Создаем DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    print("\n🚀 Проверка батча...")
    for i, (features, targets, info) in enumerate(loader):
        print(f"\n   Батч {i}:")
        print(f"   Features shape: {features.shape}")
        
        for key, value in targets.items():
            print(f"   {key} shape: {value.shape}, dtype: {value.dtype}")
            
        # Проверяем price_changes
        if 'price_changes' in info:
            for timeframe, changes in info['price_changes'].items():
                print(f"   price_changes[{timeframe}] shape: {changes.shape}")
        
        break  # Только первый батч
    
    print("\n✅ Отладка завершена!")


if __name__ == "__main__":
    debug_dataset()