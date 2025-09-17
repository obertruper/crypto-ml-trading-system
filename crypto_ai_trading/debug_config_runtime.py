#!/usr/bin/env python3
"""
Отладка загрузки конфига при runtime
"""
import yaml
from omegaconf import DictConfig, OmegaConf
import sys
import os

# Добавляем путь к проекту
sys.path.append('/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading')

print("=" * 60)
print("🔍 ОТЛАДКА ЗАГРУЗКИ КОНФИГА")
print("=" * 60)

# 1. Загружаем конфиг как в main.py
config_path = 'config/config.yaml'
print(f"\n1️⃣ Загрузка конфига из: {config_path}")

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config = OmegaConf.create(config)

# 2. Проверяем наличие direction_bias в разных местах
print("\n2️⃣ Поиск direction_bias в конфиге:")

# В model секции
if 'model' in config and 'direction_bias' in config.model:
    print(f"   ✅ Найден в config.model: {config.model.direction_bias}")
else:
    print(f"   ❌ НЕ найден в config.model")

# В staged_training
if 'staged_training' in config:
    print(f"\n   Проверка staged_training:")
    if 'stages' in config.staged_training:
        stages = config.staged_training.stages
        print(f"   Количество этапов: {len(stages)}")
        for i, stage in enumerate(stages):
            if 'direction_bias' in stage:
                print(f"   ✅ Найден в stages[{i}]: {stage.direction_bias}")
            else:
                print(f"   ❌ НЕ найден в stages[{i}]")

# 3. Симулируем создание модели
print("\n3️⃣ Симуляция создания модели:")

# Получаем model config как в реальном коде
model_config = config.model
print(f"   Тип model_config: {type(model_config)}")
print(f"   Ключи в model_config: {list(model_config.keys())[:10]}...")  # первые 10 ключей

# Проверяем наличие direction_bias
if hasattr(model_config, 'direction_bias'):
    print(f"   ✅ model_config.direction_bias доступен: {model_config.direction_bias}")
else:
    print(f"   ❌ model_config.direction_bias НЕ доступен")

# Проверяем альтернативный доступ
if 'direction_bias' in model_config:
    print(f"   ✅ model_config['direction_bias'] доступен: {model_config['direction_bias']}")
else:
    print(f"   ❌ model_config['direction_bias'] НЕ доступен")

# 4. Проверяем, что будет передано в модель
print("\n4️⃣ Что получит модель:")

from models.patchtst_unified import UnifiedPatchTST

# Временно отключаем создание модели чтобы не загружать GPU
print("   Создаем модель с текущим конфигом...")

try:
    # Пробуем создать модель
    model = UnifiedPatchTST(model_config)
    print("   ✅ Модель создана успешно")

    # Проверяем что инициализировалось
    if hasattr(model, 'direction_head'):
        head = model.direction_head
        if hasattr(head, 'class_bias'):
            print(f"   class_bias shape: {head.class_bias.shape}")
            print(f"   class_bias values: {head.class_bias[0].tolist()}")
except Exception as e:
    print(f"   ⚠️ Ошибка при создании модели: {e}")

# 5. Рекомендации
print("\n5️⃣ АНАЛИЗ:")

if 'model' in config and 'direction_bias' in config.model:
    bias = config.model.direction_bias
    print(f"   ✅ direction_bias корректно настроен в config.model: {bias}")

    # Анализируем значения
    if isinstance(bias, list) and len(bias) == 3:
        long_bias = bias[0]
        short_bias = bias[1]
        flat_bias = bias[2]

        print(f"\n   Анализ текущих значений:")
        print(f"   LONG bias: {long_bias:.1f}")
        print(f"   SHORT bias: {short_bias:.1f}")
        print(f"   FLAT bias: {flat_bias:.1f}")

        # Предсказываем эффект
        import math

        # Softmax с bias даст примерные вероятности
        logits = [long_bias, short_bias, flat_bias]
        exp_logits = [math.exp(x) for x in logits]
        sum_exp = sum(exp_logits)
        probs = [x/sum_exp for x in exp_logits]

        print(f"\n   Ожидаемое распределение (примерное):")
        print(f"   LONG: {probs[0]*100:.1f}%")
        print(f"   SHORT: {probs[1]*100:.1f}%")
        print(f"   FLAT: {probs[2]*100:.1f}%")

        # Рекомендации
        if probs[2] < 0.6:
            print(f"\n   ⚠️ FLAT недостаточно! Увеличьте FLAT bias или уменьшите LONG/SHORT")
        elif probs[2] > 0.8:
            print(f"\n   ⚠️ FLAT слишком доминирует! Уменьшите FLAT bias или увеличьте LONG/SHORT")
        else:
            print(f"\n   ✅ Распределение близко к целевому!")
else:
    print(f"   ❌ direction_bias НЕ НАЙДЕН в config.model!")
    print(f"   Добавьте в config.yaml в секцию model:")
    print(f"   ```yaml")
    print(f"   model:")
    print(f"     direction_bias:")
    print(f"       - -0.8  # LONG")
    print(f"       - -0.8  # SHORT")
    print(f"       - 0.5   # FLAT")
    print(f"   ```")

print("\n" + "=" * 60)
print("✅ Отладка завершена!")
print("=" * 60)