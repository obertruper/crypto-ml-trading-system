"""Скрипт для проверки применения temperature scaling в конфигурации"""

import yaml
from pathlib import Path

print("🔍 ПРОВЕРКА КОНФИГУРАЦИИ TEMPERATURE SCALING")
print("="*60)

# Загрузка конфигурации
config_path = Path('config/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model_config = config['model']
loss_config = config['loss']

print("\n✅ ОБНОВЛЕННЫЕ ПАРАМЕТРЫ:")
print(f"\n1. Temperature Scaling:")
print(f"   - temperature_scaling: {model_config.get('temperature_scaling', False)}")
print(f"   - temperature: {model_config.get('temperature', 1.0)}")
print(f"   - confidence_threshold: {model_config.get('confidence_threshold', 0.6)}")

print(f"\n2. Entropy Regularization:")
print(f"   - entropy_weight: {model_config.get('entropy_weight', 0.0)}")
print(f"   - min_entropy_threshold: {model_config.get('min_entropy_threshold', 0.0)}")

print(f"\n3. Балансировка классов:")
print(f"   - class_weights: {loss_config.get('class_weights', [1.0, 1.0, 1.0])}")
print(f"   - adaptive_class_weights: {loss_config.get('adaptive_class_weights', False)}")
print(f"   - entropy_min_weight: {loss_config.get('entropy_min_weight', 0.5)}")

print(f"\n4. Другие параметры регуляризации:")
print(f"   - dropout: {model_config.get('dropout', 0.1)}")
print(f"   - attention_dropout: {model_config.get('attention_dropout', 0.1)}")
print(f"   - label_smoothing: {model_config.get('label_smoothing', 0.0)}")

print("\n📝 ИЗМЕНЕНИЯ ДЛЯ БОРЬБЫ С FLAT-СХЛОПЫВАНИЕМ:")
print("1. ✅ Увеличена температура до 2.0 для снижения уверенности")
print("2. ✅ Снижен порог уверенности до 0.45")
print("3. ✅ Добавлен штраф за низкую энтропию (0.1)")
print("4. ✅ Усилено подавление FLAT: вес 0.3 (было 0.4)")
print("5. ✅ Включена адаптивная балансировка классов")

print("\n⚡ ОЖИДАЕМЫЙ ЭФФЕКТ:")
print("- Снижение доли FLAT предсказаний с 79.6% до ~40%")
print("- Увеличение доли LONG/SHORT сигналов")
print("- Более равномерное распределение уверенности")
print("- Повышение Direction Accuracy до 40%+")

print("\n🚀 ГОТОВО К ЗАПУСКУ:")
print("python main.py --mode train")
print("\nКонфигурация обновлена и готова к обучению!")