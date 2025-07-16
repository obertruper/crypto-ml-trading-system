"""Проверка количества признаков из кэша"""
import h5py
import numpy as np
from pathlib import Path

print("🔍 ПРОВЕРКА РАЗМЕРНОСТИ ПРИЗНАКОВ В КЭШЕ")
print("="*60)

# Проверяем train кэш
cache_file = Path('cache/precomputed/train_w96_s1.h5')

if cache_file.exists():
    with h5py.File(cache_file, 'r') as f:
        X_shape = f['X'].shape
        y_shape = f['y'].shape
        
        print(f"📁 Файл: {cache_file}")
        print(f"  - X shape: {X_shape}")
        print(f"  - y shape: {y_shape}")
        print(f"  - Количество сэмплов: {X_shape[0]:,}")
        print(f"  - Длина окна (timesteps): {X_shape[1]}")
        print(f"  - КОЛИЧЕСТВО ПРИЗНАКОВ: {X_shape[2]}")
        print(f"  - Количество целевых переменных: {y_shape[2]}")
        
        # Проверяем первый батч данных
        X_sample = f['X'][0]
        print(f"\n📊 Первый сэмпл:")
        print(f"  - Shape: {X_sample.shape}")
        print(f"  - Min value: {X_sample.min():.4f}")
        print(f"  - Max value: {X_sample.max():.4f}")
        print(f"  - Mean value: {X_sample.mean():.4f}")
        print(f"  - Std value: {X_sample.std():.4f}")
        
        # Проверяем есть ли NaN или Inf
        has_nan = np.isnan(X_sample).any()
        has_inf = np.isinf(X_sample).any()
        print(f"\n🔍 Проверка данных:")
        print(f"  - Содержит NaN: {has_nan}")
        print(f"  - Содержит Inf: {has_inf}")
else:
    print(f"❌ Файл {cache_file} не найден!")

# Проверяем другие кэши
print(f"\n📂 Другие кэш-файлы:")
cache_dir = Path('cache/precomputed')
for file in cache_dir.glob('*.h5'):
    with h5py.File(file, 'r') as f:
        X_shape = f['X'].shape
        print(f"  - {file.name}: X={X_shape}, features={X_shape[2]}")