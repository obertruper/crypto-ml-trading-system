#!/usr/bin/env python3
"""
Быстрый тест улучшений XGBoost v3.0
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from config import Config
from models import DataBalancer
from models.xgboost_trainer import XGBoostTrainer

print("🧪 Тестирование улучшений XGBoost v3.0")
print("="*60)

# Проверка 1: Новый балансировщик
print("\n1️⃣ Проверка нового DataBalancer:")
config = Config()
config.training.balance_method = "smote"

# Создаем тестовые данные
X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
y = pd.Series(np.concatenate([np.zeros(80), np.ones(20)]))

balancer = DataBalancer(config)
X_balanced, y_balanced = balancer.balance_data(X, y)

print(f"   До: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"   После: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
print("   ✅ DataBalancer работает корректно" if len(X_balanced) == len(y_balanced) else "   ❌ Ошибка")

# Проверка 2: XGBoostTrainer использует DataBalancer
print("\n2️⃣ Проверка интеграции в XGBoostTrainer:")
trainer = XGBoostTrainer(config, "test_model")
print(f"   Балансировщик: {type(trainer.data_balancer).__name__}")
print("   ✅ Используется DataBalancer" if hasattr(trainer, 'data_balancer') else "   ❌ Используется старый BalanceStrategy")

# Проверка 3: Конфигурация
print("\n3️⃣ Проверка обновленной конфигурации:")
print(f"   Размер ансамбля: {config.training.ensemble_size}")
print(f"   Метрика порога: {config.training.threshold_metric}")
print(f"   Оптимизация порога: {config.training.optimize_threshold}")

print("\n"+"="*60)
print("✅ Готово к запуску полного теста!")
print("\n🚀 Рекомендуемая команда:")
print("   python xgboost_v3/main.py --test-mode")
print("="*60)