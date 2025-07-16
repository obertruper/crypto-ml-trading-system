#!/usr/bin/env python3
"""
Удаление признаков с нулевой дисперсией из данных
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("🔧 Удаление константных признаков из данных")
print("="*60)

# Признаки для удаления
features_to_remove = ['vwap_extreme_deviation']

datasets = {
    'train': 'data/processed/train_data.parquet',
    'val': 'data/processed/val_data.parquet',
    'test': 'data/processed/test_data.parquet'
}

for name, path in datasets.items():
    print(f"\n📊 Обработка {name.upper()} набора:")
    
    # Загружаем данные
    df = pd.read_parquet(path)
    original_shape = df.shape
    
    # Удаляем проблемные признаки
    removed = []
    for feature in features_to_remove:
        if feature in df.columns:
            df = df.drop(columns=[feature])
            removed.append(feature)
    
    # Сохраняем обновленные данные
    df.to_parquet(path, index=False)
    
    print(f"   Исходный размер: {original_shape}")
    print(f"   Новый размер: {df.shape}")
    print(f"   Удалено признаков: {len(removed)}")
    if removed:
        print(f"   Удаленные: {', '.join(removed)}")

print("\n✅ Данные обновлены!")
print("\n📝 Теперь запустите проверку:")
print("   python verify_data_correctness.py")