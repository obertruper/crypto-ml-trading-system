#!/usr/bin/env python3
"""
Диагностика проблемы нулевого Loss
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

def check_loss_function():
    """Проверка работы Loss функции"""
    print("="*80)
    print("🔍 ДИАГНОСТИКА НУЛЕВОГО LOSS")
    print("="*80)

    # Импортируем необходимые модули
    from models.patchtst_unified import DirectionalMultiTaskLoss
    import yaml

    # Загружаем конфигурацию
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Преобразуем в объект для удобства
    class ConfigObj:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, ConfigObj(v))
                else:
                    setattr(self, k, v)
        def get(self, key, default=None):
            return getattr(self, key, default)

    config = ConfigObj(config)
    print("✅ Конфигурация загружена")

    # Создаем loss функцию
    print("\n📊 Создание Loss функции...")
    loss_fn = DirectionalMultiTaskLoss(config)

    print(f"   - Class weights: {loss_fn.class_weights}")
    print(f"   - Direction weights: {loss_fn.direction_task_weights}")
    print(f"   - Temperature: {loss_fn.temperature}")

    # Создаем тестовые данные
    batch_size = 32
    num_classes = 3

    # Имитируем outputs модели (dict с разными головами)
    outputs = {
        'direction_15m': torch.randn(batch_size, num_classes),
        'direction_1h': torch.randn(batch_size, num_classes),
        'direction_4h': torch.randn(batch_size, num_classes),
        'direction_12h': torch.randn(batch_size, num_classes),
        'future_return_15m': torch.randn(batch_size, 1),
        'future_return_1h': torch.randn(batch_size, 1),
        'future_return_4h': torch.randn(batch_size, 1),
        'future_return_12h': torch.randn(batch_size, 1),
    }

    # Создаем тестовые targets
    # Правильная структура targets для вашей системы
    targets = torch.zeros(batch_size, 20)  # 20 целевых переменных

    # Заполняем direction targets (индексы 4-7 для direction_15m, 1h, 4h, 12h)
    for i in range(4, 8):
        targets[:, i] = torch.randint(0, 3, (batch_size,)).float()  # 0=LONG, 1=SHORT, 2=FLAT

    # Заполняем return targets (индексы 0-3)
    for i in range(4):
        targets[:, i] = torch.randn(batch_size) * 0.01  # Небольшие возвраты

    print(f"\n📝 Тестовые данные:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Outputs keys: {list(outputs.keys())}")
    print(f"   - Targets shape: {targets.shape}")
    print(f"   - Direction targets (пример): {targets[0, 4:8].tolist()}")

    # Вычисляем loss
    print("\n🔧 Вычисление Loss...")
    try:
        loss_dict = loss_fn(outputs, targets)

        if isinstance(loss_dict, dict):
            print("✅ Loss вычислен (dict):")
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"   - {key}: {value.item():.6f}")
                else:
                    print(f"   - {key}: {value}")

            # Проверяем total loss
            if 'total' in loss_dict:
                total_loss = loss_dict['total']
                print(f"\n🎯 Total Loss: {total_loss.item():.6f}")

                if total_loss.item() == 0:
                    print("❌ ПРОБЛЕМА: Total Loss = 0!")

                    # Детальная проверка
                    print("\n🔍 Детальная проверка компонентов:")

                    # Проверяем направления
                    for key in ['direction_15m', 'direction_1h', 'direction_4h', 'direction_12h']:
                        if key in outputs:
                            logits = outputs[key]

                            # Индекс целевой переменной в targets
                            target_idx = 4 + ['direction_15m', 'direction_1h', 'direction_4h', 'direction_12h'].index(key)
                            target = targets[:, target_idx].long()

                            # Ручное вычисление CrossEntropy
                            ce = nn.CrossEntropyLoss()(logits, target)
                            print(f"   - {key} CE loss: {ce.item():.6f}")

                            # Проверка распределения предсказаний
                            probs = torch.softmax(logits, dim=-1)
                            preds = torch.argmax(probs, dim=-1)
                            unique, counts = torch.unique(preds, return_counts=True)
                            pred_dist = {int(u): int(c) for u, c in zip(unique, counts)}
                            print(f"     Распределение предсказаний: {pred_dist}")

                elif total_loss.item() < 1e-6:
                    print(f"⚠️ Очень малый Total Loss: {total_loss.item():.2e}")
                else:
                    print("✅ Loss в нормальном диапазоне")

        else:
            print(f"⚠️ Loss вернул не dict: {type(loss_dict)}")
            if isinstance(loss_dict, torch.Tensor):
                print(f"   Значение: {loss_dict.item():.6f}")

    except Exception as e:
        print(f"❌ Ошибка при вычислении Loss: {e}")
        import traceback
        traceback.print_exc()

    # Проверка конфигурации
    print("\n🔍 Проверка конфигурации Loss:")
    print(f"   - loss.type: {config.loss.type if hasattr(config.loss, 'type') else 'N/A'}")
    print(f"   - loss.class_weights: {config.loss.class_weights if hasattr(config.loss, 'class_weights') else 'N/A'}")
    print(f"   - loss.label_smoothing: {config.loss.label_smoothing if hasattr(config.loss, 'label_smoothing') else 'N/A'}")
    print(f"   - model.temperature: {config.model.temperature if hasattr(config.model, 'temperature') else 'N/A'}")

    # Проверка данных
    print("\n🔍 Проверка реальных данных:")
    try:
        from data.data_loader import create_dataloaders

        train_loader, _, _, _, _ = create_dataloaders(
            config=config,
            batch_size=32,
            num_workers=0
        )

        # Берем первый батч
        batch = next(iter(train_loader))
        inputs, targets = batch

        print(f"   - Input shape: {inputs.shape}")
        print(f"   - Target shape: {targets.shape}")
        print(f"   - Target dtype: {targets.dtype}")

        # Проверка распределения targets
        for i in range(4, 8):  # direction columns
            unique, counts = torch.unique(targets[:, i], return_counts=True)
            dist = {int(u): int(c) for u, c in zip(unique, counts)}
            col_name = ['direction_15m', 'direction_1h', 'direction_4h', 'direction_12h'][i-4]
            print(f"   - {col_name} распределение: {dist}")

            # Проверка на схлопывание
            max_class = max(dist.values())
            total = sum(dist.values())
            if max_class / total > 0.9:
                print(f"     ❌ СХЛОПЫВАНИЕ в {col_name}! {max_class/total:.1%} в одном классе")

    except Exception as e:
        print(f"❌ Ошибка при проверке данных: {e}")

    print("\n" + "="*80)
    print("💡 РЕКОМЕНДАЦИИ:")
    print("="*80)
    print("1. Проверьте что targets правильно передаются в loss функцию")
    print("2. Убедитесь что class_weights применяются корректно")
    print("3. Проверьте инициализацию весов модели")
    print("4. Возможно нужно увеличить learning rate")
    print("5. Проверьте что данные нормализованы правильно")


def check_model_initialization():
    """Проверка инициализации модели"""
    print("\n" + "="*80)
    print("🔍 ПРОВЕРКА ИНИЦИАЛИЗАЦИИ МОДЕЛИ")
    print("="*80)

    from models.patchtst_unified import UnifiedPatchTST
    from config.config import load_config

    config = load_config("config/config.yaml")

    # Создаем модель
    model = UnifiedPatchTST(
        num_features=261,  # Примерное количество признаков
        pred_len=config.model.pred_len,
        patch_len=config.model.patch_len,
        stride=config.model.stride,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
        activation=config.model.activation,
        norm_type=config.model.norm_type,
        num_targets=config.data.num_targets
    )

    # Проверяем веса direction heads
    print("\n📊 Проверка весов direction heads:")
    for name, param in model.named_parameters():
        if 'direction' in name and 'weight' in name:
            mean = param.mean().item()
            std = param.std().item()
            print(f"   {name}: mean={mean:.4f}, std={std:.4f}")

            if abs(mean) > 1.0:
                print(f"     ⚠️ Большое среднее значение!")
            if std < 0.001:
                print(f"     ⚠️ Очень малая дисперсия!")

    # Тестовый forward pass
    print("\n🔧 Тестовый forward pass:")
    batch_size = 32
    seq_len = 96
    num_features = 261

    x = torch.randn(batch_size, seq_len, num_features)

    with torch.no_grad():
        outputs = model(x)

    print(f"   - Output type: {type(outputs)}")
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: shape={value.shape}, mean={value.mean():.4f}, std={value.std():.4f}")

                # Для direction outputs проверяем распределение
                if 'direction' in key and value.dim() == 2:
                    probs = torch.softmax(value, dim=-1)
                    preds = torch.argmax(probs, dim=-1)
                    unique, counts = torch.unique(preds, return_counts=True)
                    dist = {int(u): int(c) for u, c in zip(unique, counts)}
                    print(f"     Распределение: {dist}")

                    # Проверка на схлопывание
                    max_count = max(counts)
                    if max_count / batch_size > 0.9:
                        print(f"     ❌ СХЛОПЫВАНИЕ! {max_count/batch_size:.1%} в одном классе")


if __name__ == "__main__":
    print("🚀 Запуск диагностики нулевого Loss...\n")

    # Проверка Loss функции
    check_loss_function()

    # Проверка модели
    check_model_initialization()

    print("\n✅ Диагностика завершена")