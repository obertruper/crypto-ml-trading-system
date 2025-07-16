"""
Тестирование обученной модели и анализ переобучения
"""
import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from collections import defaultdict

# Загрузка конфигурации
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('models_saved/best_model.pth', map_location=device)

# Создание модели
from models.patchtst_unified import UnifiedPatchTSTForTrading

# Используем конфигурацию из checkpoint
if 'config' in checkpoint:
    saved_config = checkpoint['config']
else:
    saved_config = config

model = UnifiedPatchTSTForTrading(saved_config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("✅ Модель загружена успешно")

# Загрузка тестовых данных
from data.precomputed_dataset import create_precomputed_data_loaders
from data.data_processor import DataProcessor

print("\n📊 Загрузка тестовых данных...")
processor = DataProcessor(config)
_, _, test_data = processor.load_cached_data()

# Создание только test loader
_, _, test_loader = create_precomputed_data_loaders(
    test_data, test_data, test_data,  # используем test_data для всех
    config,
    feature_cols=None,
    target_cols=None
)

print(f"✅ Загружено {len(test_loader)} батчей тестовых данных")

# Тестирование модели
print("\n🔍 Анализ предсказаний на тестовых данных...")

all_predictions = []
all_targets = []
direction_predictions = []
direction_targets = []

with torch.no_grad():
    for batch_idx, (inputs, targets, info) in enumerate(tqdm(test_loader, desc="Тестирование")):
        if batch_idx > 100:  # Ограничиваем для быстроты
            break
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Предсказание
        outputs = model(inputs)
        
        # Извлекаем direction предсказания (индекс 4 - direction_15m)
        if hasattr(outputs, '_direction_logits'):
            dir_logits = outputs._direction_logits[:, 0, :]  # 15m
            dir_preds = torch.argmax(torch.softmax(dir_logits, dim=-1), dim=-1)
            direction_predictions.extend(dir_preds.cpu().numpy())
            direction_targets.extend(targets[:, 0, 4].cpu().numpy())

# Анализ результатов
direction_predictions = np.array(direction_predictions)
direction_targets = np.array(direction_targets)

# Распределение предсказаний
pred_counts = np.bincount(direction_predictions.astype(int), minlength=3)
true_counts = np.bincount(direction_targets.astype(int), minlength=3)

print("\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
print("="*50)

print("\n🎯 Распределение классов:")
print(f"Истинное распределение:  LONG={true_counts[0]/len(direction_targets)*100:.1f}%, SHORT={true_counts[1]/len(direction_targets)*100:.1f}%, FLAT={true_counts[2]/len(direction_targets)*100:.1f}%")
print(f"Предсказания модели:     LONG={pred_counts[0]/len(direction_predictions)*100:.1f}%, SHORT={pred_counts[1]/len(direction_predictions)*100:.1f}%, FLAT={pred_counts[2]/len(direction_predictions)*100:.1f}%")

# Точность по классам
from sklearn.metrics import classification_report, confusion_matrix

print("\n📈 Метрики классификации:")
print(classification_report(direction_targets.astype(int), direction_predictions.astype(int), 
                          target_names=['LONG', 'SHORT', 'FLAT']))

# Confusion matrix
cm = confusion_matrix(direction_targets.astype(int), direction_predictions.astype(int))
print("\n🔢 Confusion Matrix:")
print("     Pred:  LONG  SHORT  FLAT")
print(f"True LONG:  {cm[0, 0]:4d}  {cm[0, 1]:4d}  {cm[0, 2]:4d}")
print(f"True SHORT: {cm[1, 0]:4d}  {cm[1, 1]:4d}  {cm[1, 2]:4d}")
print(f"True FLAT:  {cm[2, 0]:4d}  {cm[2, 1]:4d}  {cm[2, 2]:4d}")

# Direction accuracy
accuracy = (direction_predictions == direction_targets).mean()
print(f"\n✅ Direction Accuracy: {accuracy*100:.2f}%")

# Анализ переобучения
print("\n⚠️ АНАЛИЗ ПЕРЕОБУЧЕНИЯ:")
print("="*50)
print(f"Train Loss (из логов): 0.619")
print(f"Val Loss (из логов): 2.218")
print(f"Overfitting Ratio: {2.218/0.619:.2f}x")
print("\n🔴 КРИТИЧЕСКОЕ ПЕРЕОБУЧЕНИЕ! Val Loss в 3.6 раза больше Train Loss")

print("\n💡 РЕКОМЕНДАЦИИ:")
print("1. Увеличить dropout с 0.5 до 0.7")
print("2. Увеличить weight_decay с 0.01 до 0.1") 
print("3. Уменьшить размер модели (d_model, d_ff)")
print("4. Добавить больше данных или аугментации")
print("5. Использовать раннюю остановку с patience=10 (вместо 30)")