#!/usr/bin/env python3
"""
Быстрый тест модели для проверки готовности
"""

import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

print("🚀 Быстрый тест модели PatchTST...")
print("=" * 80)

# Загрузка checkpoint
checkpoint_path = 'models_saved/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"✅ Модель загружена")
print(f"📊 Эпоха: {checkpoint.get('epoch', 'N/A')}")
print(f"📊 Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

# Анализ конфигурации
config = checkpoint['config']['model']
print(f"\n🏗️ Параметры модели:")
print(f"   - Входные признаки: {config.get('input_size', 'N/A')}")
print(f"   - Выходные переменные: {config.get('output_size', 'N/A')}")
print(f"   - Размер батча при обучении: {config.get('batch_size', 'N/A')}")
print(f"   - Learning rate: {config.get('learning_rate', 'N/A')}")

# История обучения
if 'history' in checkpoint:
    history = checkpoint['history']
    if 'train_loss' in history and 'val_loss' in history:
        last_5_epochs = min(5, len(history['train_loss']))
        print(f"\n📈 Последние {last_5_epochs} эпох:")
        for i in range(-last_5_epochs, 0):
            epoch_num = len(history['train_loss']) + i + 1
            print(f"   Эпоха {epoch_num}: Train={history['train_loss'][i]:.6f}, Val={history['val_loss'][i]:.6f}")

# Оценка качества
val_loss = checkpoint.get('val_loss', float('inf'))
print(f"\n🎯 Оценка качества модели:")

if val_loss < 0.15:
    print(f"   ✅ ОТЛИЧНОЕ качество (Val Loss = {val_loss:.6f})")
    print("   Модель показывает высокую точность предсказаний")
elif val_loss < 0.20:
    print(f"   ✅ ХОРОШЕЕ качество (Val Loss = {val_loss:.6f})")
    print("   Модель готова к использованию")
elif val_loss < 0.25:
    print(f"   ⚠️ СРЕДНЕЕ качество (Val Loss = {val_loss:.6f})")
    print("   Рекомендуется дополнительная оптимизация")
else:
    print(f"   ❌ НИЗКОЕ качество (Val Loss = {val_loss:.6f})")
    print("   Требуется переобучение")

# Проверка совместимости данных
print(f"\n🔍 Проверка совместимости с данными:")
try:
    test_data = pd.read_parquet('data/processed/test_data.parquet')
    feature_cols = [col for col in test_data.columns 
                   if not col.startswith(('future_', 'direction_', 'volatility_', 'volume_change_', 'price_range_'))
                   and col not in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"   Признаков в данных: {len(feature_cols)}")
    print(f"   Признаков в модели: {config.get('input_size', 'N/A')}")
    
    if len(feature_cols) >= config.get('input_size', 240):
        print("   ✅ Данные совместимы с моделью")
    else:
        print("   ❌ Несоответствие размерностей данных")
        
except Exception as e:
    print(f"   ⚠️ Не удалось проверить данные: {str(e)}")

# Анализ целевых переменных
print(f"\n📊 Целевые переменные (20 выходов):")
targets = [
    "future_return_15m", "future_return_1h", "future_return_4h", "future_return_12h",
    "direction_15m", "direction_1h", "direction_4h", "direction_12h",
    "volatility_15m", "volatility_1h", "volatility_4h", "volatility_12h",
    "volume_change_15m", "volume_change_1h", "volume_change_4h", "volume_change_12h",
    "price_range_15m", "price_range_1h", "price_range_4h", "price_range_12h"
]

print("   Регрессионные задачи:")
for i in range(0, 16, 4):
    print(f"   - {targets[i:i+4]}")
    
# Рекомендации
print(f"\n💡 РЕКОМЕНДАЦИИ:")

if val_loss < 0.15:
    print("1. ✅ Модель готова к полномасштабному тестированию")
    print("2. ✅ Можно запускать бэктестинг на всех монетах")
    print("3. ✅ Рекомендуется начать с небольших позиций")
    print("\n🚀 ВЕРДИКТ: ЗАПУСКАЙТЕ ТЕСТИРОВАНИЕ!")
else:
    print("1. ⚠️ Рекомендуется провести дополнительную валидацию")
    print("2. ⚠️ Проверьте метрики на отдельных символах")
    print("3. ⚠️ Возможно требуется fine-tuning")
    print("\n⚠️ ВЕРДИКТ: ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА")

# Сохранение отчета
output_dir = Path('experiments/evaluation_results')
output_dir.mkdir(parents=True, exist_ok=True)

report_file = output_dir / f'quick_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

with open(report_file, 'w', encoding='utf-8') as f:
    f.write(f"Быстрый тест модели PatchTST\n")
    f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Val Loss: {val_loss:.6f}\n")
    f.write(f"Эпоха: {checkpoint.get('epoch', 'N/A')}\n")
    f.write(f"Вердикт: {'ГОТОВА' if val_loss < 0.15 else 'ТРЕБУЕТ ПРОВЕРКИ'}\n")

print(f"\n📄 Отчет сохранен: {report_file}")
print("=" * 80)