#!/usr/bin/env python3
"""
Диагностический скрипт для выявления проблем в обучении
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import traceback

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from models.patchtst_unified import UnifiedPatchTST, DirectionalMultiTaskLoss
from data.precomputed_dataset import custom_collate_fn
import yaml

# Загружаем конфигурацию
class Config:
    def __init__(self):
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.__dict__.update(config)

def diagnose_model_output():
    """Проверяем что возвращает модель"""
    print("\n" + "="*60)
    print("🔍 ДИАГНОСТИКА ВЫХОДА МОДЕЛИ")
    print("="*60)

    # Создаем простую модель
    config = Config()
    model = UnifiedPatchTST(config)
    model.eval()

    # Создаем случайный вход
    batch_size = 4
    seq_len = 96
    num_features = 261
    x = torch.randn(batch_size, seq_len, num_features)

    # Получаем выход
    with torch.no_grad():
        outputs = model(x)

    print(f"✅ Тип outputs: {type(outputs)}")
    print(f"✅ Тип outputs.__class__: {outputs.__class__.__name__}")

    # Проверяем атрибуты
    if hasattr(outputs, 'prediction'):
        print(f"✅ outputs.prediction shape: {outputs.prediction.shape}")
    if hasattr(outputs, '_direction_logits'):
        print(f"✅ outputs._direction_logits shape: {outputs._direction_logits.shape}")

    # Пытаемся индексировать
    try:
        result = outputs[:, 4]
        print(f"❌ ПРОБЛЕМА: outputs[:, 4] работает, но не должно!")
        print(f"   Результат: {result}")
    except Exception as e:
        print(f"✅ outputs[:, 4] вызывает ошибку (как и ожидалось): {e}")
        print(f"   Тип ошибки: {type(e).__name__}")

    # Проверяем как правильно получить direction outputs
    if isinstance(outputs, dict):
        print("✅ outputs - словарь")
        if 'direction_15m' in outputs:
            print(f"   direction_15m shape: {outputs['direction_15m'].shape}")
    else:
        print(f"⚠️ outputs не словарь, тип: {type(outputs)}")

    return outputs

def diagnose_loss_computation():
    """Проверяем вычисление loss"""
    print("\n" + "="*60)
    print("🔍 ДИАГНОСТИКА ВЫЧИСЛЕНИЯ LOSS")
    print("="*60)

    config = Config()
    model = UnifiedPatchTST(config)
    loss_fn = DirectionalMultiTaskLoss(config)

    # Создаем данные
    batch_size = 4
    seq_len = 96
    num_features = 261
    num_targets = 20

    x = torch.randn(batch_size, seq_len, num_features)
    targets = torch.randn(batch_size, num_targets)

    # Добавляем direction targets (классы 0, 1, 2)
    for i in range(4, 8):  # direction_15m, 1h, 4h, 12h
        targets[:, i] = torch.randint(0, 3, (batch_size,)).float()

    # Forward pass
    model.train()
    outputs = model(x)

    print(f"✅ Model outputs тип: {type(outputs)}")

    # Вычисляем loss
    try:
        loss = loss_fn(outputs, targets)
        print(f"✅ Loss вычислен: {type(loss)}")

        if isinstance(loss, dict):
            print(f"   total_loss: {loss['total_loss'].item():.4f}")
            for key, val in loss.items():
                if key != 'total_loss' and isinstance(val, torch.Tensor):
                    print(f"   {key}: {val.item():.4f}")
        else:
            print(f"   Loss значение: {loss.item():.4f}")

        # Проверяем градиенты
        if isinstance(loss, dict):
            total_loss = loss['total_loss']
        else:
            total_loss = loss

        total_loss.backward()

        # Проверяем есть ли градиенты
        has_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        if has_grads:
            print("✅ Градиенты распространяются")
        else:
            print("❌ ПРОБЛЕМА: Градиенты НЕ распространяются!")

    except Exception as e:
        print(f"❌ Ошибка при вычислении loss: {e}")
        traceback.print_exc()

    return loss

def diagnose_indexing_error():
    """Находим точное место ошибки индексации"""
    print("\n" + "="*60)
    print("🔍 ДИАГНОСТИКА ОШИБКИ ИНДЕКСАЦИИ")
    print("="*60)

    # Воспроизводим ошибку
    error_repr = "(slice(None, None, None), 4)"
    print(f"Ошибка: {error_repr}")

    # Это означает попытку индексации [:, 4]
    # Проверяем что может вызвать такую ошибку

    # Тест 1: кортеж вместо тензора
    test_tuple = (torch.randn(4, 10), torch.randn(4, 10))
    try:
        result = test_tuple[:, 4]
        print("❌ Кортеж индексируется без ошибки")
    except Exception as e:
        print(f"✅ Кортеж вызывает ошибку: {e}")
        if str(e) == error_repr or "slice" in str(e):
            print("   ⚠️ ЭТО ПОХОЖЕ НА НАШУ ОШИБКУ!")

    # Тест 2: UnifiedModelOutput
    from models.patchtst_unified import UnifiedModelOutput

    umo = UnifiedModelOutput(
        prediction=torch.randn(4, 20),
        direction_logits=None,
        confidence_scores=None,
        feature_importance=None
    )

    try:
        result = umo[:, 4]
        print(f"❌ UnifiedModelOutput индексируется: {result}")
    except Exception as e:
        print(f"✅ UnifiedModelOutput вызывает ошибку: {e}")
        if "(slice(None, None, None), 4)" in str(e):
            print("   🔴 ЭТО НАША ОШИБКА! UnifiedModelOutput не поддерживает индексацию!")

def check_actual_training():
    """Проверяем реальный процесс обучения"""
    print("\n" + "="*60)
    print("🔍 ПРОВЕРКА РЕАЛЬНОГО ОБУЧЕНИЯ")
    print("="*60)

    from training.optimized_trainer import OptimizedTrainer
    from torch.utils.data import DataLoader, TensorDataset

    config = Config()
    model = UnifiedPatchTST(config)
    trainer = OptimizedTrainer(model, config)

    # Создаем простой датасет
    n_samples = 100
    x = torch.randn(n_samples, 96, 261)
    y = torch.randn(n_samples, 20)

    # Direction targets
    for i in range(4, 8):
        y[:, i] = torch.randint(0, 3, (n_samples,)).float()

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # Пробуем одну эпоху
    try:
        print("Запускаем train_epoch...")
        loss, metrics = trainer.train_epoch(dataloader, 1)
        print(f"✅ train_epoch завершен")
        print(f"   Loss: {loss:.4f}")
        print(f"   Metrics: {metrics}")
    except Exception as e:
        print(f"❌ Ошибка в train_epoch: {e}")
        if "(slice(None, None, None)" in str(e):
            print("   🔴 ЭТО ОШИБКА ИНДЕКСАЦИИ!")

            # Пытаемся найти точное место
            import traceback
            tb = traceback.format_exc()
            print("\nТрассировка:")
            print(tb)

            # Ищем строку с ошибкой
            if "optimized_trainer.py" in tb:
                lines = tb.split('\n')
                for line in lines:
                    if "optimized_trainer.py" in line and "line" in line:
                        print(f"   📍 Ошибка в: {line.strip()}")

def main():
    """Запускаем все диагностики"""
    print("\n" + "="*80)
    print("🏥 ДИАГНОСТИКА ПРОБЛЕМ ОБУЧЕНИЯ")
    print("="*80)

    # 1. Проверяем выход модели
    outputs = diagnose_model_output()

    # 2. Проверяем вычисление loss
    loss = diagnose_loss_computation()

    # 3. Проверяем ошибку индексации
    diagnose_indexing_error()

    # 4. Проверяем реальное обучение
    check_actual_training()

    print("\n" + "="*80)
    print("📋 ВЫВОДЫ:")
    print("="*80)

    print("""
1. UnifiedModelOutput не поддерживает прямую индексацию [:, 4]
2. Нужно всегда использовать outputs.prediction или извлекать через атрибуты
3. Loss может быть 0 если модель возвращает None для direction_logits
4. Проверить все места где происходит индексация outputs
""")

if __name__ == "__main__":
    main()