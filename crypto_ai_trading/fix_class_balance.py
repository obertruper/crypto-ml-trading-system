#!/usr/bin/env python3
"""
Исправление баланса классов и настройка bias для предотвращения схлопывания
"""
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import sys
import numpy as np

# Добавляем путь к проекту
sys.path.append('/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading')

from models.patchtst_unified import UnifiedPatchTST
from utils.logger import get_logger

logger = get_logger("ClassBalanceFix")

print("=" * 60)
print("🔧 ИСПРАВЛЕНИЕ БАЛАНСА КЛАССОВ")
print("=" * 60)

# Загружаем конфиг
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config = OmegaConf.create(config)

print("\n1️⃣ АНАЛИЗ ТЕКУЩИХ НАСТРОЕК:")
print(f"   direction_bias в config.model: {config.model.get('direction_bias', 'НЕ НАЙДЕН')}")
print(f"   class_weights в config.loss: {config.loss.get('class_weights', 'НЕ НАЙДЕН')}")
print(f"   label_smoothing: {config.loss.get('label_smoothing', 0)}")

# Целевое распределение
target_distribution = {
    'LONG': 0.15,   # 15%
    'SHORT': 0.15,  # 15%
    'FLAT': 0.70    # 70%
}

print("\n2️⃣ ЦЕЛЕВОЕ РАСПРЕДЕЛЕНИЕ:")
for cls, ratio in target_distribution.items():
    print(f"   {cls}: {ratio:.1%}")

# Расчет оптимальных bias значений
# Формула: bias = -log(1/target_ratio - 1)
# Но для стабильности используем более консервативные значения

print("\n3️⃣ РАСЧЕТ ОПТИМАЛЬНЫХ BIAS ЗНАЧЕНИЙ:")

# Вариант 1: Агрессивный (сильное подавление LONG/SHORT)
aggressive_bias = [-2.0, -2.0, 1.5]  # LONG, SHORT, FLAT
print(f"   Агрессивный: {aggressive_bias}")

# Вариант 2: Сбалансированный
balanced_bias = [-1.0, -1.0, 0.8]
print(f"   Сбалансированный: {balanced_bias}")

# Вариант 3: Мягкий
soft_bias = [-0.3, -0.3, 0.5]
print(f"   Мягкий: {soft_bias}")

# Вариант 4: Основанный на логитах для целевого распределения
# log(p/(1-p)) для каждого класса
import math
logit_bias = [
    math.log(0.15/0.85),  # LONG: log(15%/85%) = -1.73
    math.log(0.15/0.85),  # SHORT: log(15%/85%) = -1.73
    math.log(0.70/0.30)   # FLAT: log(70%/30%) = 0.85
]
print(f"   На основе логитов: [{logit_bias[0]:.2f}, {logit_bias[1]:.2f}, {logit_bias[2]:.2f}]")

# Выбираем оптимальный вариант
optimal_bias = [-1.5, -1.5, 1.0]  # Компромисс между агрессивным и логитами
print(f"\n   ✅ ВЫБРАН ОПТИМАЛЬНЫЙ: {optimal_bias}")

# Тестируем с моделью
print("\n4️⃣ ТЕСТ С МОДЕЛЬЮ:")

# Обновляем конфиг
config.model.direction_bias = optimal_bias

# Создаем модель
model_config = config.model
model = UnifiedPatchTST(model_config)

# Проверяем инициализацию
if hasattr(model, 'direction_head'):
    head = model.direction_head
    print(f"   class_bias shape: {head.class_bias.shape}")
    print(f"   class_bias values:\n{head.class_bias}")

# Тест предсказаний
print("\n5️⃣ ТЕСТ РАСПРЕДЕЛЕНИЯ ПРЕДСКАЗАНИЙ:")
batch_size = 256  # Больше сэмплов для статистики
seq_len = 96
num_features = model_config.input_size

# Создаем тестовые данные
x = torch.randn(batch_size, seq_len, num_features)

with torch.no_grad():
    outputs = model(x, return_dict=True)

# Анализируем распределение
timeframes = ['15m', '1h', '4h', '12h']
all_predictions = []

for tf_name in timeframes:
    key = f'direction_{tf_name}'
    if key in outputs:
        logits = outputs[key]
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        long_pct = (preds == 0).float().mean() * 100
        short_pct = (preds == 1).float().mean() * 100
        flat_pct = (preds == 2).float().mean() * 100

        # Энтропия для оценки разнообразия
        avg_probs = probs.mean(dim=0)
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
        max_entropy = -math.log(1/3)  # Максимальная энтропия для 3 классов
        entropy_ratio = entropy / max_entropy

        print(f"\n   {tf_name}:")
        print(f"      LONG: {long_pct:.1f}% (цель: 15%)")
        print(f"      SHORT: {short_pct:.1f}% (цель: 15%)")
        print(f"      FLAT: {flat_pct:.1f}% (цель: 70%)")
        print(f"      Энтропия: {entropy_ratio:.2%} от максимума")

        # Проверка близости к целевому распределению
        long_error = abs(long_pct - 15)
        short_error = abs(short_pct - 15)
        flat_error = abs(flat_pct - 70)
        total_error = long_error + short_error + flat_error

        if total_error < 30:  # Суммарная ошибка < 30%
            print(f"      ✅ Распределение близко к целевому (ошибка: {total_error:.1f}%)")
        else:
            print(f"      ⚠️ Распределение далеко от целевого (ошибка: {total_error:.1f}%)")

        all_predictions.append(preds)

# Общая статистика
print("\n6️⃣ ОБЩАЯ СТАТИСТИКА:")
all_preds = torch.cat(all_predictions)
total_long = (all_preds == 0).float().mean() * 100
total_short = (all_preds == 1).float().mean() * 100
total_flat = (all_preds == 2).float().mean() * 100

print(f"   Среднее по всем таймфреймам:")
print(f"   LONG: {total_long:.1f}%")
print(f"   SHORT: {total_short:.1f}%")
print(f"   FLAT: {total_flat:.1f}%")

# Рекомендации по настройке
print("\n7️⃣ РЕКОМЕНДАЦИИ ПО НАСТРОЙКЕ:")

if total_flat < 60:
    print("   ⚠️ FLAT недостаточно! Рекомендации:")
    print("   1. Увеличьте bias для FLAT (например, на +0.5)")
    print("   2. Уменьшите bias для LONG/SHORT (например, на -0.5)")
    print(f"   3. Попробуйте bias: [{optimal_bias[0]-0.5:.1f}, {optimal_bias[1]-0.5:.1f}, {optimal_bias[2]+0.5:.1f}]")
elif total_flat > 80:
    print("   ⚠️ FLAT слишком доминирует! Рекомендации:")
    print("   1. Уменьшите bias для FLAT (например, на -0.3)")
    print("   2. Увеличьте bias для LONG/SHORT (например, на +0.3)")
    print(f"   3. Попробуйте bias: [{optimal_bias[0]+0.3:.1f}, {optimal_bias[1]+0.3:.1f}, {optimal_bias[2]-0.3:.1f}]")
else:
    print("   ✅ Распределение близко к оптимальному!")
    print("   Можно начинать обучение с текущими настройками")

# Проверка loss функции
print("\n8️⃣ ПРОВЕРКА LOSS ФУНКЦИИ:")
print(f"   class_weights: {config.loss.get('class_weights', [1.0, 1.0, 1.0])}")
print(f"   use_focal_loss: {config.loss.get('use_focal_loss', False)}")
print(f"   label_smoothing: {config.loss.get('label_smoothing', 0)}")

# Рекомендации для loss
print("\n   Рекомендации для loss:")
print("   1. class_weights: [2.0, 2.0, 0.5] - усилить редкие классы")
print("   2. label_smoothing: 0.1 - предотвратить переуверенность")
print("   3. use_focal_loss: True - фокус на сложных примерах")

# Сохранение оптимального конфига
print("\n9️⃣ СОХРАНЕНИЕ ОПТИМАЛЬНЫХ НАСТРОЕК:")

optimal_config = {
    'model': {
        'direction_bias': optimal_bias
    },
    'loss': {
        'class_weights': [2.0, 2.0, 0.5],  # Усиливаем LONG/SHORT
        'label_smoothing': 0.1,
        'use_focal_loss': True,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'use_dynamic_class_weights': False,  # Отключаем для стабильности
        'flat_preservation_weight': 0.3,
        'min_flat_ratio': 0.5,  # Минимум 50% FLAT
        'entropy_regularization': 0.1
    }
}

# Выводим YAML для копирования в конфиг
print("\n📋 YAML для добавления в config.yaml:")
print("```yaml")
print("model:")
print(f"  direction_bias: {optimal_bias}")
print("")
print("loss:")
print("  class_weights: [2.0, 2.0, 0.5]")
print("  label_smoothing: 0.1")
print("  use_focal_loss: true")
print("  focal_alpha: 0.25")
print("  focal_gamma: 2.0")
print("  use_dynamic_class_weights: false")
print("  flat_preservation_weight: 0.3")
print("  min_flat_ratio: 0.5")
print("  entropy_regularization: 0.1")
print("```")

print("\n✅ Анализ завершен!")
print("\n💡 СЛЕДУЮЩИЕ ШАГИ:")
print("1. Обновите config.yaml с рекомендованными настройками")
print("2. Запустите обучение и мониторьте распределение классов")
print("3. При необходимости корректируйте bias значения")