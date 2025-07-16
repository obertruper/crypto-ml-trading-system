#!/usr/bin/env python3
"""
Детальный анализ точности предсказаний модели PatchTST
Проверяет РЕАЛЬНОЕ качество предсказаний, а не только loss
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import yaml
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🔍 Детальный анализ точности предсказаний модели...")
print("=" * 80)

# Загрузка модели и данных
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Загрузка модели
from models.patchtst_unified import UnifiedPatchTSTForTrading as UnifiedPatchTST
from data.precomputed_dataset import PrecomputedDataLoaders

checkpoint = torch.load('models_saved/best_model.pth', map_location=device)
model = UnifiedPatchTST(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✅ Модель загружена (Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f})")

# 2. Загрузка тестовых данных
print("\n📊 Загрузка тестовых данных...")

# Используем precomputed dataloader для корректной загрузки
config = checkpoint['config']
data_loader_creator = PrecomputedDataLoaders(config)
train_loader, val_loader, test_loader = data_loader_creator.get_loaders()

print(f"✅ Загружено {len(test_loader)} батчей тестовых данных")

# 3. Получение предсказаний на тестовом наборе
print("\n🔮 Генерация предсказаний...")

all_predictions = []
all_targets = []
all_metadata = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 10:  # Ограничимся 10 батчами для быстрого анализа
            break
            
        # Извлечение данных из батча
        if isinstance(batch, dict):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            metadata = batch.get('metadata', {})
        else:
            features, targets, metadata = batch
            features = features.to(device)
            targets = targets.to(device)
        
        # Предсказание
        outputs = model(features)
        
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_metadata.append(metadata)
        
        if batch_idx % 5 == 0:
            print(f"   Обработано батчей: {batch_idx + 1}/10")

# Объединяем результаты
predictions = np.vstack(all_predictions)
targets = np.vstack(all_targets)

print(f"✅ Получено предсказаний: {predictions.shape}")
print(f"✅ Реальных значений: {targets.shape}")

# 4. Анализ точности по каждой целевой переменной
print("\n📊 АНАЛИЗ ТОЧНОСТИ ПРЕДСКАЗАНИЙ:")
print("=" * 70)

# Названия целевых переменных
target_names = [
    'future_return_15m', 'future_return_1h', 'future_return_4h', 'future_return_12h',
    'direction_15m', 'direction_1h', 'direction_4h', 'direction_12h',
    'volatility_15m', 'volatility_1h', 'volatility_4h', 'volatility_12h',
    'volume_change_15m', 'volume_change_1h', 'volume_change_4h', 'volume_change_12h',
    'price_range_15m', 'price_range_1h', 'price_range_4h', 'price_range_12h'
]

results = {}

# Анализ future returns (предсказание доходности)
print("\n🎯 ПРЕДСКАЗАНИЕ ДОХОДНОСТИ (future_return):")
print("-" * 50)

for i in range(4):
    pred = predictions[:, i]
    true = targets[:, i]
    
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    
    # Конвертируем в проценты для понятности
    mae_pct = mae * 100
    rmse_pct = rmse * 100
    
    # Точность предсказания направления (рост/падение)
    pred_direction = (pred > 0).astype(int)
    true_direction = (true > 0).astype(int)
    direction_accuracy = accuracy_score(true_direction, pred_direction)
    
    results[target_names[i]] = {
        'mae': mae_pct,
        'rmse': rmse_pct,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }
    
    print(f"{target_names[i]:20s} | MAE: {mae_pct:5.2f}% | Точность направления: {direction_accuracy:.1%}")

# Анализ direction (бинарная классификация)
print("\n🎯 ПРЕДСКАЗАНИЕ НАПРАВЛЕНИЯ (direction):")
print("-" * 50)

for i in range(4, 8):
    pred = predictions[:, i]
    true = targets[:, i]
    
    # Преобразуем в бинарные метки
    pred_binary = (pred > 0.5).astype(int)
    true_binary = (true > 0.5).astype(int)
    
    accuracy = accuracy_score(true_binary, pred_binary)
    
    # Подсчет True Positive, False Positive и т.д.
    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    results[target_names[i]] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    
    print(f"{target_names[i]:20s} | Точность: {accuracy:.1%} | Precision: {precision:.1%} | Recall: {recall:.1%}")

# Анализ volatility
print("\n🎯 ПРЕДСКАЗАНИЕ ВОЛАТИЛЬНОСТИ (volatility):")
print("-" * 50)

for i in range(8, 12):
    pred = predictions[:, i]
    true = targets[:, i]
    
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    
    # Корреляция между предсказанной и реальной волатильностью
    correlation = np.corrcoef(pred, true)[0, 1]
    
    results[target_names[i]] = {
        'mae': mae,
        'r2': r2,
        'correlation': correlation
    }
    
    print(f"{target_names[i]:20s} | MAE: {mae:.4f} | Корреляция: {correlation:.3f}")

# 5. Общая оценка качества
print("\n" + "=" * 70)
print("📊 СВОДНАЯ ОЦЕНКА КАЧЕСТВА:")
print("=" * 70)

# Средние метрики по типам
return_accuracies = [results[name]['direction_accuracy'] for name in target_names[:4]]
direction_accuracies = [results[name]['accuracy'] for name in target_names[4:8]]
volatility_correlations = [results[name]['correlation'] for name in target_names[8:12]]

avg_return_accuracy = np.mean(return_accuracies)
avg_direction_accuracy = np.mean(direction_accuracies)
avg_volatility_correlation = np.mean(volatility_correlations)

print(f"\n📈 Предсказание доходности:")
print(f"   Средняя точность направления: {avg_return_accuracy:.1%}")
print(f"   Средняя MAE: {np.mean([results[name]['mae'] for name in target_names[:4]]):.2f}%")

print(f"\n🎯 Предсказание направления:")
print(f"   Средняя точность: {avg_direction_accuracy:.1%}")

print(f"\n📊 Предсказание волатильности:")
print(f"   Средняя корреляция: {avg_volatility_correlation:.3f}")

# 6. Практическая оценка для торговли
print("\n" + "=" * 70)
print("💰 ПРАКТИЧЕСКАЯ ОЦЕНКА ДЛЯ ТОРГОВЛИ:")
print("=" * 70)

# Симуляция простой стратегии
# Открываем позицию если модель предсказывает движение > 0.5%
threshold = 0.005  # 0.5%

trades = 0
winning_trades = 0
total_return = 0

for i in range(len(predictions)):
    # Используем предсказание на 1 час
    pred_return = predictions[i, 1]  # future_return_1h
    true_return = targets[i, 1]
    
    if abs(pred_return) > threshold:
        trades += 1
        # Если предсказали правильное направление
        if (pred_return > 0 and true_return > 0) or (pred_return < 0 and true_return < 0):
            winning_trades += 1
            total_return += abs(true_return) - 0.001  # Минус комиссия 0.1%
        else:
            total_return -= 0.02  # Stop loss 2%

if trades > 0:
    win_rate = winning_trades / trades
    avg_return_per_trade = total_return / trades * 100
    
    print(f"\nРезультаты симуляции (порог {threshold*100}%):")
    print(f"   Всего сделок: {trades}")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Средний результат на сделку: {avg_return_per_trade:.2f}%")
    print(f"   Общий результат: {total_return*100:.2f}%")

# 7. Визуализация
print("\n📊 Создание визуализаций...")

output_dir = Path('experiments/accuracy_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# График 1: Scatter plot предсказаний vs реальность для future_return_1h
plt.figure(figsize=(10, 8))
sample_size = min(1000, len(predictions))
sample_idx = np.random.choice(len(predictions), sample_size, replace=False)

plt.scatter(targets[sample_idx, 1] * 100, predictions[sample_idx, 1] * 100, alpha=0.5, s=10)
plt.plot([-5, 5], [-5, 5], 'r--', lw=2, label='Идеальное предсказание')
plt.xlabel('Реальная доходность (%)')
plt.ylabel('Предсказанная доходность (%)')
plt.title('Предсказание доходности на 1 час')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.savefig(output_dir / 'return_prediction_scatter.png', dpi=300)
plt.close()

# График 2: Точность по временным горизонтам
plt.figure(figsize=(10, 6))
timeframes = ['15m', '1h', '4h', '12h']
accuracies = [results[f'direction_{tf}']['accuracy'] for tf in timeframes]

plt.bar(timeframes, accuracies)
plt.xlabel('Временной горизонт')
plt.ylabel('Точность предсказания направления')
plt.title('Точность предсказания направления по временным горизонтам')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.1%}', ha='center')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig(output_dir / 'accuracy_by_timeframe.png', dpi=300)
plt.close()

# 8. Финальный вердикт
print("\n" + "=" * 80)
print("🎯 ФИНАЛЬНЫЙ ВЕРДИКТ:")
print("=" * 80)

if avg_direction_accuracy > 0.55 and avg_return_accuracy > 0.55:
    print("✅ МОДЕЛЬ ПОКАЗЫВАЕТ ХОРОШУЮ ТОЧНОСТЬ!")
    print(f"   - Точность предсказания направления: {avg_direction_accuracy:.1%}")
    print(f"   - Практическая применимость: ВЫСОКАЯ")
    print("\n💡 Рекомендация: Модель готова к осторожному использованию в реальной торговле")
    verdict = "READY"
elif avg_direction_accuracy > 0.52:
    print("⚠️ МОДЕЛЬ ПОКАЗЫВАЕТ СРЕДНЮЮ ТОЧНОСТЬ")
    print(f"   - Точность предсказания направления: {avg_direction_accuracy:.1%}")
    print(f"   - Практическая применимость: СРЕДНЯЯ")
    print("\n💡 Рекомендация: Требуется дополнительная оптимизация")
    verdict = "NEEDS_OPTIMIZATION"
else:
    print("❌ МОДЕЛЬ ПОКАЗЫВАЕТ НИЗКУЮ ТОЧНОСТЬ")
    print(f"   - Точность предсказания направления: {avg_direction_accuracy:.1%}")
    print(f"   - Практическая применимость: НИЗКАЯ")
    print("\n💡 Рекомендация: Требуется переобучение или изменение архитектуры")
    verdict = "NOT_READY"

# Сохранение отчета
report = {
    'timestamp': datetime.now().isoformat(),
    'model_val_loss': checkpoint.get('val_loss', 'N/A'),
    'samples_analyzed': len(predictions),
    'avg_direction_accuracy': float(avg_direction_accuracy),
    'avg_return_accuracy': float(avg_return_accuracy),
    'avg_volatility_correlation': float(avg_volatility_correlation),
    'detailed_results': results,
    'verdict': verdict
}

report_file = output_dir / f'accuracy_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
with open(report_file, 'w') as f:
    yaml.dump(report, f, default_flow_style=False)

print(f"\n📄 Подробный отчет сохранен: {report_file}")
print("=" * 80)