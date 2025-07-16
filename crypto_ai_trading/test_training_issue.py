#!/usr/bin/env python3
"""
Тестовый скрипт для диагностики проблемы с обучением
"""
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Загружаем конфигурацию
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("="*80)
print("🔍 ДИАГНОСТИКА ПРОБЛЕМЫ С ОБУЧЕНИЕМ")
print("="*80)

# 1. Проверка загруженных данных
print("\n1️⃣ Проверка кэшированных данных:")
data_dir = Path("data/processed")
train_file = data_dir / "train_data.parquet"

if train_file.exists():
    train_data = pd.read_parquet(train_file)
    print(f"✅ Train data загружена: {len(train_data):,} записей")
    
    # Анализ колонок
    exclude_prefixes = ('target_', 'future_', 'long_tp', 'short_tp', 'long_sl', 'short_sl',
                       'long_optimal', 'short_optimal', 'long_expected', 'short_expected',
                       'best_direction', 'signal_strength', 'long_final', 'short_final')
    
    feature_cols = [col for col in train_data.columns 
                   if col not in ['id', 'symbol', 'datetime', 'timestamp', 'sector']
                   and not any(col.startswith(prefix) for prefix in exclude_prefixes)]
    
    target_cols = [col for col in train_data.columns 
                  if col.startswith(('target_', 'future_return_', 'long_tp', 'short_tp', 
                                   'long_sl', 'short_sl', 'long_optimal', 'short_optimal',
                                   'best_direction'))]
    
    print(f"\n📊 Структура данных:")
    print(f"   - Всего колонок: {len(train_data.columns)}")
    print(f"   - Признаков: {len(feature_cols)}")
    print(f"   - Целевых переменных: {len(target_cols)}")
    
    # Проверка на NaN
    nan_counts = train_data[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n⚠️  Найдены NaN в признаках:")
        for col, count in nan_counts[nan_counts > 0].head().items():
            print(f"   - {col}: {count} NaN")
    else:
        print(f"\n✅ NaN отсутствуют в признаках")
    
    # Проверка диапазонов значений
    print(f"\n📈 Статистика признаков:")
    numeric_features = train_data[feature_cols].select_dtypes(include=[np.number])
    print(f"   - Min значения: {numeric_features.min().min():.4f}")
    print(f"   - Max значения: {numeric_features.max().max():.4f}")
    print(f"   - Mean значения: {numeric_features.mean().mean():.4f}")
    print(f"   - Std значения: {numeric_features.std().mean():.4f}")
    
    # Проверка целевых переменных
    print(f"\n🎯 Статистика целевых переменных:")
    for col in target_cols[:5]:  # Первые 5 целевых
        if col in train_data.columns:
            unique_vals = train_data[col].unique()
            if len(unique_vals) < 10:
                print(f"   - {col}: уникальные значения = {unique_vals}")
            else:
                print(f"   - {col}: min={train_data[col].min():.4f}, max={train_data[col].max():.4f}, mean={train_data[col].mean():.4f}")

# 2. Проверка конфигурации модели
print(f"\n2️⃣ Проверка конфигурации модели:")
print(f"   - Model name: {config['model']['name']}")
print(f"   - Task type: {config['model'].get('task_type', 'regression')}")
print(f"   - Learning rate: {config['model']['learning_rate']}")
print(f"   - Batch size: {config['model']['batch_size']}")
print(f"   - Loss function: {config['loss']['name']}")

# 3. Создание простого батча для тестирования
print(f"\n3️⃣ Создание тестового батча:")
from data.dataset import TradingDataset

# Создаем маленький dataset для теста
test_dataset = TradingDataset(
    data=train_data.head(1000),  # Берем только 1000 записей
    config=config,
    context_window=config['model']['context_window'],
    prediction_window=config['model']['pred_len'],
    feature_cols=feature_cols,
    target_cols=target_cols
)

print(f"✅ Dataset создан: {len(test_dataset)} примеров")

# Получаем один батч
X, y, info = test_dataset[0]
print(f"\n📊 Размерности батча:")
print(f"   - X shape: {X.shape}")
print(f"   - y shape: {y.shape}")
print(f"   - X dtype: {X.dtype}")
print(f"   - y dtype: {y.dtype}")

# 4. Инициализация модели
print(f"\n4️⃣ Тест инициализации модели:")
try:
    # Обновляем конфигурацию
    config_copy = config.copy()
    config_copy['model']['input_features'] = len(feature_cols)
    config_copy['model']['n_features'] = len(feature_cols)
    config_copy['model']['target_variables'] = target_cols
    
    from models.patchtst import create_patchtst_model
    
    model = create_patchtst_model(config_copy)
    print(f"✅ Модель создана успешно")
    print(f"   - Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Тест forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_batch = X.unsqueeze(0).to(device)  # Добавляем batch dimension
    
    with torch.no_grad():
        output = model(X_batch)
    
    print(f"\n📊 Output shape: {output.shape}")
    print(f"   - Output min: {output.min().item():.4f}")
    print(f"   - Output max: {output.max().item():.4f}")
    print(f"   - Output mean: {output.mean().item():.4f}")
    
    # Тест loss
    from models.trading_losses import TradingMultiTaskLoss
    criterion = TradingMultiTaskLoss(config_copy)
    
    y_batch = y.unsqueeze(0).to(device)
    loss = criterion(output, y_batch)
    
    print(f"\n📊 Loss value: {loss.item():.4f}")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✅ Диагностика завершена")
print("="*80)