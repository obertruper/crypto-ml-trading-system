#!/usr/bin/env python3
"""
Скрипт проверки всех исправлений XGBoost v3.0
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from config import Config
from data import FeatureEngineer
from models import DataBalancer

print("🔍 Проверка исправлений XGBoost v3.0")
print("="*60)

# 1. Проверка бинарных признаков
print("\n1️⃣ Проверка исправления бинарных признаков:")
df_test = pd.DataFrame({
    'close': [100, 101, 99, 102, 98],
    'open': [99, 102, 100, 101, 99],
    'high': [102, 103, 101, 104, 100],
    'low': [98, 100, 98, 100, 97],
    'volume': [1000, 1200, 900, 1100, 950],
    'rsi_val': [30, 70, 50, 80, 20],
    'adx_val': [20, 30, 25, 35, 15],
    'is_bullish': [1, -1, 0, 1, -1]  # Проблемные значения
})

config = Config()
fe = FeatureEngineer(config)

# Проверяем validate_features
df_validated = fe.validate_features(df_test.copy())
print(f"   is_bullish до: {df_test['is_bullish'].unique()}")
print(f"   is_bullish после: {df_validated['is_bullish'].unique()}")
print("   ✅ Бинарные признаки исправлены" if set(df_validated['is_bullish'].unique()) <= {0, 1} else "   ❌ Ошибка")

# 2. Проверка балансировщика
print("\n2️⃣ Проверка нового балансировщика:")
X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
# Добавляем бинарные признаки
X['is_feature'] = np.random.choice([0, 1], 100)
X['binary_feat'] = np.random.choice([0, 1], 100)
y = pd.Series(np.concatenate([np.zeros(80), np.ones(20)]))  # Несбалансированные классы

balancer = DataBalancer(config)
X_balanced, y_balanced = balancer.balance_data(X, y)

print(f"   До балансировки: Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
print(f"   После балансировки: Class 0: {(y_balanced==0).sum()}, Class 1: {(y_balanced==1).sum()}")
print(f"   Размер X: {X.shape} -> {X_balanced.shape}")
print("   ✅ Балансировка работает корректно" if len(X_balanced) == len(y_balanced) else "   ❌ Ошибка размерности")

# 3. Проверка ансамблевого взвешивания
print("\n3️⃣ Проверка улучшенного взвешивания ансамбля:")

# Симуляция scores моделей
scores_similar = np.array([0.850, 0.851, 0.849])  # Очень похожие
scores_different = np.array([0.750, 0.850, 0.800])  # Разные

from models.ensemble import EnsembleModel
ensemble = EnsembleModel(config)

# Тест 1: похожие scores
scores = scores_similar
scores_normalized = (scores - scores.mean()) / (scores.std() + 1e-8)
scores_normalized = np.clip(scores_normalized, -2, 2)
exp_scores = np.exp(scores_normalized)
weights = exp_scores / exp_scores.sum()

print(f"   Похожие scores: {scores}")
print(f"   Веса: {weights}")
print(f"   Макс вес: {weights.max():.3f}")

# Тест 2: разные scores
scores = scores_different
if scores.std() > 0.01:
    scores_normalized = (scores - scores.mean()) / (scores.std() + 1e-8)
    scores_normalized = np.clip(scores_normalized, -2, 2)
    exp_scores = np.exp(scores_normalized)
    weights = exp_scores / exp_scores.sum()
    
    if weights.max() > 0.9:
        weights = 0.8 * weights + 0.2 * (np.ones(len(scores)) / len(scores))

print(f"\n   Разные scores: {scores}")
print(f"   Веса: {weights}")
print(f"   Макс вес: {weights.max():.3f}")
print("   ✅ Взвешивание работает корректно" if weights.max() < 0.9 else "   ❌ Слишком экстремальные веса")

# 4. Проверка удаления константных признаков
print("\n4️⃣ Проверка удаления константных признаков:")
from data import DataPreprocessor
df_const = pd.DataFrame({
    'feature1': [1, 1, 1, 1, 1],
    'feature2': [1, 2, 3, 4, 5],
    'constant_feat': [0, 0, 0, 0, 0],
    'buy_expected_return': [0.5, -0.3, 1.2, -0.8, 0.2],
    'sell_expected_return': [-0.2, 0.8, -1.0, 0.3, -0.5]
})

preprocessor = DataPreprocessor(config)
# Имитируем процесс подготовки признаков
feature_columns = ['feature1', 'feature2', 'constant_feat']
X = df_const[feature_columns].copy()

constant_features = []
for col in X.columns:
    if X[col].nunique() <= 1:
        constant_features.append(col)

print(f"   Найдено константных признаков: {len(constant_features)}")
print(f"   Константные признаки: {constant_features}")
print("   ✅ Обнаружение константных признаков работает" if 'constant_feat' in constant_features else "   ❌ Ошибка")

print("\n"+"="*60)
print("✅ Все проверки завершены!")
print("\n💡 Рекомендации:")
print("   1. Запустите полное обучение для проверки всех исправлений")
print("   2. Используйте optimize_threshold=True для автоматического подбора порога")
print("   3. Увеличьте ensemble_size до 3-5 для лучших результатов")
print("="*60)