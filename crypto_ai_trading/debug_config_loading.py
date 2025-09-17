#!/usr/bin/env python3
"""
Отладка загрузки конфига
"""
import yaml
from omegaconf import DictConfig, OmegaConf

# Загружаем конфиг
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config = OmegaConf.create(config)

print("🔍 СТРУКТУРА КОНФИГА:")
print(f"Ключи верхнего уровня: {list(config.keys())}")

# Проверяем model
if 'model' in config:
    model_config = config.model
    print(f"\n📦 model_config type: {type(model_config)}")
    print(f"   model_config keys: {list(model_config.keys())}")

    # Проверяем staged_training
    if hasattr(model_config, 'staged_training'):
        print(f"   ✅ model_config.staged_training существует")
    else:
        print(f"   ❌ model_config НЕ содержит staged_training")

# Проверяем staged_training на верхнем уровне
if 'staged_training' in config:
    print(f"\n📦 staged_training на верхнем уровне:")
    st = config.staged_training
    print(f"   staged_training type: {type(st)}")
    print(f"   staged_training keys: {list(st.keys())}")

    if 'stages' in st:
        stages = st.stages
        print(f"   stages type: {type(stages)}")
        print(f"   stages length: {len(stages)}")

        if stages and len(stages) > 0:
            stage0 = stages[0]
            print(f"\n   📝 Первый этап (stages[0]):")
            print(f"      type: {type(stage0)}")
            print(f"      keys: {list(stage0.keys())}")

            if 'direction_bias' in stage0:
                print(f"      ✅ direction_bias найден: {stage0.direction_bias}")
            else:
                print(f"      ❌ direction_bias НЕ найден в stages[0]")

print("\n🔍 ПУТЬ К direction_bias:")
print("   config.staged_training.stages[0].direction_bias")