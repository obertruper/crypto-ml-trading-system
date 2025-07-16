"""Анализ уверенности модели в предсказаниях"""
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path

# Загрузка конфигурации
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('models_saved/best_model.pth', map_location=device)

print("🔍 АНАЛИЗ УВЕРЕННОСТИ МОДЕЛИ В ПРЕДСКАЗАНИЯХ")
print("="*60)

# Создание модели
from models.patchtst_unified import UnifiedPatchTSTForTrading
model = UnifiedPatchTSTForTrading(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Загрузка небольшой выборки данных для анализа
from data.precomputed_dataset import PrecomputedDataset
import pandas as pd

print("\n📊 Загрузка тестовых данных...")
# Используем кэшированные данные
cache_path = Path('cache/precomputed/val_w96_s4.h5')
if cache_path.exists():
    import h5py
    with h5py.File(cache_path, 'r') as f:
        # Берем первые 1000 примеров
        X_sample = torch.FloatTensor(f['X'][:1000])
        y_sample = torch.FloatTensor(f['y'][:1000])
        
    print(f"✅ Загружено {len(X_sample)} примеров")
    
    # Анализ предсказаний
    print("\n🎯 Анализ уверенности в предсказаниях направлений...")
    
    all_confidences = []
    all_entropies = []
    predictions_by_confidence = {
        'high': {'LONG': 0, 'SHORT': 0, 'FLAT': 0},
        'medium': {'LONG': 0, 'SHORT': 0, 'FLAT': 0},
        'low': {'LONG': 0, 'SHORT': 0, 'FLAT': 0}
    }
    
    with torch.no_grad():
        # Обрабатываем батчами
        batch_size = 100
        for i in range(0, len(X_sample), batch_size):
            batch_X = X_sample[i:i+batch_size].to(device)
            
            # Получаем предсказания
            outputs = model(batch_X)
            
            if hasattr(outputs, '_direction_logits'):
                # Извлекаем logits для direction_15m
                direction_logits = outputs._direction_logits[:, 0, :]  # [batch, 3]
                
                # Применяем softmax для получения вероятностей
                probs = F.softmax(direction_logits, dim=-1)
                
                # Максимальная вероятность = уверенность
                max_probs, predictions = torch.max(probs, dim=-1)
                
                # Энтропия распределения
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                
                all_confidences.extend(max_probs.cpu().numpy())
                all_entropies.extend(entropy.cpu().numpy())
                
                # Классификация по уровню уверенности
                for j, (conf, pred) in enumerate(zip(max_probs, predictions)):
                    conf_val = conf.item()
                    pred_val = pred.item()
                    
                    if conf_val > 0.7:
                        conf_level = 'high'
                    elif conf_val > 0.5:
                        conf_level = 'medium'
                    else:
                        conf_level = 'low'
                    
                    pred_class = ['LONG', 'SHORT', 'FLAT'][pred_val]
                    predictions_by_confidence[conf_level][pred_class] += 1
    
    # Статистика уверенности
    confidences = np.array(all_confidences)
    entropies = np.array(all_entropies)
    
    print("\n📈 СТАТИСТИКА УВЕРЕННОСТИ:")
    print(f"Средняя уверенность: {confidences.mean():.3f}")
    print(f"Медианная уверенность: {np.median(confidences):.3f}")
    print(f"Минимальная уверенность: {confidences.min():.3f}")
    print(f"Максимальная уверенность: {confidences.max():.3f}")
    print(f"Стандартное отклонение: {confidences.std():.3f}")
    
    print("\n🎲 ЭНТРОПИЯ ПРЕДСКАЗАНИЙ:")
    print(f"Средняя энтропия: {entropies.mean():.3f}")
    print(f"Максимальная энтропия (неуверенность): {entropies.max():.3f}")
    print(f"Минимальная энтропия (уверенность): {entropies.min():.3f}")
    
    print("\n📊 РАСПРЕДЕЛЕНИЕ УВЕРЕННОСТИ:")
    high_conf = (confidences > 0.7).sum()
    medium_conf = ((confidences > 0.5) & (confidences <= 0.7)).sum()
    low_conf = (confidences <= 0.5).sum()
    
    print(f"Высокая уверенность (>70%): {high_conf/len(confidences)*100:.1f}%")
    print(f"Средняя уверенность (50-70%): {medium_conf/len(confidences)*100:.1f}%")
    print(f"Низкая уверенность (<50%): {low_conf/len(confidences)*100:.1f}%")
    
    print("\n🎯 ПРЕДСКАЗАНИЯ ПО УРОВНЯМ УВЕРЕННОСТИ:")
    for level in ['high', 'medium', 'low']:
        total = sum(predictions_by_confidence[level].values())
        if total > 0:
            print(f"\n{level.upper()} уверенность:")
            for class_name, count in predictions_by_confidence[level].items():
                print(f"  {class_name}: {count/total*100:.1f}%")
    
    # Гистограмма уверенности
    print("\n📊 ГИСТОГРАММА УВЕРЕННОСТИ:")
    hist, bins = np.histogram(confidences, bins=10)
    for i in range(len(hist)):
        bar_width = int(hist[i] / len(confidences) * 50)
        print(f"{bins[i]:.2f}-{bins[i+1]:.2f}: {'█' * bar_width} {hist[i]}")
    
    # Проверка калибровки
    print("\n🎯 КАЛИБРОВКА МОДЕЛИ:")
    print("(Насколько уверенность соответствует реальной точности)")
    
    # Группируем по уровням уверенности и проверяем точность
    confidence_bins = [(0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    
    print("\nУверенность → Реальная точность:")
    for low, high in confidence_bins:
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            print(f"{low:.1f}-{high:.1f}: примеров {mask.sum()}")
            
else:
    print("❌ Кэш валидационных данных не найден")

print("\n💡 ВЫВОДЫ:")
if confidences.mean() > 0.8:
    print("⚠️ Модель СЛИШКОМ УВЕРЕНА - возможно переобучение")
elif confidences.mean() < 0.5:
    print("⚠️ Модель НЕУВЕРЕНА - требуется больше обучения")
else:
    print("✅ Уровень уверенности в норме")

if (confidences > 0.9).sum() / len(confidences) > 0.5:
    print("🔴 Более 50% предсказаний с уверенностью >90% - признак схлопывания!")