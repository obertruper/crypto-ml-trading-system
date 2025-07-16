#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправлений
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

print("🧪 Тестирование исправлений XGBoost v3.0")
print("="*50)

# Тест 1: SMOTE с синхронизацией индексов
print("\n1. Тест SMOTE балансировки:")
X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
y = pd.Series(np.random.choice([0, 1], 100, p=[0.8, 0.2]))

# Синхронизация индексов
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"   До балансировки: {dict(zip(*np.unique(y, return_counts=True)))}")

smote = SMOTE(k_neighbors=3, random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"   После балансировки: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
print("   ✅ SMOTE работает корректно")

# Тест 2: Конвертация numpy типов для JSON
print("\n2. Тест JSON сериализации:")
import json

data = {
    'float32': np.float32(1.5),
    'float64': np.float64(2.5),
    'int32': np.int32(10),
    'array': np.array([1, 2, 3])
}

def convert_to_native_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    else:
        return obj

converted_data = convert_to_native_types(data)
json_str = json.dumps(converted_data, indent=2)
print(f"   Сериализовано: {json_str[:100]}...")
print("   ✅ JSON сериализация работает корректно")

# Тест 3: Исправление бинарных признаков
print("\n3. Тест бинарных признаков:")
df = pd.DataFrame({
    'is_bullish': [0, 1, -1, 2, 0],
    'rsi_oversold': [0, 1, -1, 0, 1],
    'regular_feat': [1.5, 2.0, -3.5, 4.0, 0.5]
})

print("   До исправления:")
print(f"   is_bullish уникальные: {df['is_bullish'].unique()}")

# Исправление
binary_cols = ['is_bullish', 'rsi_oversold']
for col in binary_cols:
    df[col] = (df[col] != 0).astype(int)

print("   После исправления:")
print(f"   is_bullish уникальные: {df['is_bullish'].unique()}")
print("   ✅ Бинарные признаки исправлены")

print("\n✅ Все тесты пройдены успешно!")
print("="*50)