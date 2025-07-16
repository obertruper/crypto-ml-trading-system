#!/usr/bin/env python3
"""
Быстрая проверка реальной точности предсказаний модели
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import yaml

print("🔍 Быстрая проверка точности предсказаний...")
print("=" * 80)

# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('models_saved/best_model.pth', map_location=device)

from models.patchtst_unified import UnifiedPatchTSTForTrading as UnifiedPatchTST
model = UnifiedPatchTST(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✅ Модель загружена (Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f})")

# Загрузка тестовых данных
print("\n📊 Загрузка тестовых данных...")
test_data = pd.read_parquet('data/processed/test_data.parquet')

# Берем последние 1000 записей для быстрого теста
test_sample = test_data.tail(1000).copy()
print(f"✅ Загружено {len(test_sample)} записей для анализа")

# Подготовка признаков
feature_cols = [col for col in test_sample.columns 
               if not col.startswith(('future_', 'direction_', 'volatility_', 'volume_change_', 'price_range_'))
               and col not in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

# Берем первые 240 признаков (как обучена модель)
feature_cols = feature_cols[:240]

# Целевые переменные
target_cols = [col for col in test_sample.columns if col.startswith(('future_', 'direction_', 'volatility_', 'volume_change_', 'price_range_'))][:20]

print(f"\n📊 Признаков: {len(feature_cols)}, Целевых переменных: {len(target_cols)}")

# Подготовка данных для модели
X = torch.tensor(test_sample[feature_cols].values, dtype=torch.float32).to(device)
y_true = test_sample[target_cols].values

# Добавляем размерности для модели [batch, seq_len, features]
X = X.unsqueeze(1)  # seq_len = 1

# Предсказание
print("\n🔮 Генерация предсказаний...")
with torch.no_grad():
    y_pred = model(X).cpu().numpy()

print(f"✅ Получено предсказаний: {y_pred.shape}")

# Анализ точности
print("\n" + "=" * 70)
print("📊 АНАЛИЗ ТОЧНОСТИ ПРЕДСКАЗАНИЙ:")
print("=" * 70)

# Анализ предсказания доходности (первые 4 переменные)
print("\n🎯 ТОЧНОСТЬ ПРЕДСКАЗАНИЯ ДОХОДНОСТИ:")
print("-" * 50)

timeframes = ['15m', '1h', '4h', '12h']
for i, tf in enumerate(timeframes):
    pred = y_pred[:, i]
    true = y_true[:, i]
    
    # MAE в процентах
    mae = np.mean(np.abs(pred - true)) * 100
    
    # Точность предсказания направления
    pred_direction = (pred > 0).astype(int)
    true_direction = (true > 0).astype(int)
    direction_accuracy = np.mean(pred_direction == true_direction)
    
    # Корреляция
    correlation = np.corrcoef(pred, true)[0, 1]
    
    print(f"future_return_{tf:3s} | MAE: {mae:5.2f}% | Точность направления: {direction_accuracy:.1%} | Корреляция: {correlation:.3f}")

# Анализ предсказания направления (следующие 4 переменные)
print("\n🎯 ТОЧНОСТЬ КЛАССИФИКАЦИИ НАПРАВЛЕНИЯ:")
print("-" * 50)

for i, tf in enumerate(timeframes):
    pred = y_pred[:, i+4]
    true = y_true[:, i+4]
    
    # Бинарная классификация
    pred_binary = (pred > 0.5).astype(int)
    true_binary = (true > 0.5).astype(int)
    
    accuracy = np.mean(pred_binary == true_binary)
    
    # Подсчет статистики
    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))
    total_positive = np.sum(true_binary == 1)
    total_negative = np.sum(true_binary == 0)
    
    print(f"direction_{tf:3s}     | Точность: {accuracy:.1%} | Угадано рост: {tp}/{total_positive} | Угадано падение: {tn}/{total_negative}")

# Практический тест торговой стратегии
print("\n" + "=" * 70)
print("💰 ПРАКТИЧЕСКИЙ ТЕСТ ТОРГОВОЙ СТРАТЕГИИ:")
print("=" * 70)

# Простая стратегия: торгуем когда модель уверена в движении > 0.5%
trades = []
for i in range(len(y_pred)):
    # Используем предсказание на 1 час
    pred_return = y_pred[i, 1]  # future_return_1h
    pred_direction = y_pred[i, 5]  # direction_1h
    true_return = y_true[i, 1]
    
    # Открываем позицию если предсказываем движение > 0.5%
    if abs(pred_return) > 0.005 and pred_direction > 0.6:  # Уверенность > 60%
        # Результат сделки
        if pred_return > 0:  # Long
            profit = true_return - 0.001  # Минус комиссия
        else:  # Short
            profit = -true_return - 0.001
            
        trades.append(profit)

if trades:
    trades = np.array(trades)
    winning_trades = np.sum(trades > 0)
    total_return = np.sum(trades)
    
    print(f"\nРезультаты стратегии (порог 0.5%, уверенность 60%):")
    print(f"   Всего сделок: {len(trades)}")
    print(f"   Прибыльных: {winning_trades} ({winning_trades/len(trades)*100:.1f}%)")
    print(f"   Средний результат: {np.mean(trades)*100:.3f}%")
    print(f"   Общий результат: {total_return*100:.2f}%")
    print(f"   Макс. прибыль: {np.max(trades)*100:.2f}%")
    print(f"   Макс. убыток: {np.min(trades)*100:.2f}%")
else:
    print("\n⚠️ Нет сделок по заданным критериям")

# Визуализация
print("\n📊 Создание визуализации...")

output_dir = Path('experiments/accuracy_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# График scatter plot для 1h предсказаний
plt.figure(figsize=(10, 8))
plt.scatter(y_true[:, 1] * 100, y_pred[:, 1] * 100, alpha=0.5, s=20)
plt.plot([-3, 3], [-3, 3], 'r--', lw=2, label='Идеальное предсказание')
plt.xlabel('Реальная доходность за 1ч (%)')
plt.ylabel('Предсказанная доходность за 1ч (%)')
plt.title('Точность предсказания доходности (1 час)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Добавляем статистику на график
mae_1h = np.mean(np.abs(y_pred[:, 1] - y_true[:, 1])) * 100
corr_1h = np.corrcoef(y_pred[:, 1], y_true[:, 1])[0, 1]
plt.text(0.05, 0.95, f'MAE: {mae_1h:.2f}%\nКорреляция: {corr_1h:.3f}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig(output_dir / 'quick_accuracy_check.png', dpi=300)
plt.close()

# Итоговый вердикт
print("\n" + "=" * 80)
print("🎯 ИТОГОВЫЙ ВЕРДИКТ:")
print("=" * 80)

# Считаем средние метрики
avg_direction_accuracy = np.mean([np.mean((y_pred[:, i] > 0) == (y_true[:, i] > 0)) for i in range(4)])
avg_mae = np.mean([np.mean(np.abs(y_pred[:, i] - y_true[:, i])) * 100 for i in range(4)])

if avg_direction_accuracy > 0.55:
    print(f"✅ МОДЕЛЬ РАБОТАЕТ! Точность направления: {avg_direction_accuracy:.1%}")
    print(f"   Средняя ошибка: {avg_mae:.2f}%")
    print("\n💡 Модель демонстрирует практическую ценность для торговли")
    verdict = "ГОТОВА К ИСПОЛЬЗОВАНИЮ"
else:
    print(f"⚠️ МОДЕЛЬ ТРЕБУЕТ УЛУЧШЕНИЯ. Точность направления: {avg_direction_accuracy:.1%}")
    print(f"   Средняя ошибка: {avg_mae:.2f}%")
    print("\n💡 Рекомендуется дополнительная оптимизация")
    verdict = "ТРЕБУЕТ ДОРАБОТКИ"

# Сохранение отчета
report = {
    'timestamp': datetime.now().isoformat(),
    'samples_analyzed': len(test_sample),
    'avg_direction_accuracy': float(avg_direction_accuracy),
    'avg_mae_percent': float(avg_mae),
    'trades_simulated': len(trades) if trades else 0,
    'verdict': verdict
}

report_file = output_dir / f'quick_accuracy_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
with open(report_file, 'w') as f:
    yaml.dump(report, f)

print(f"\n📄 Отчет сохранен: {report_file}")
print("✅ Анализ завершен!")
print("=" * 80)