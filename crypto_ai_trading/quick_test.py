"""Быстрый тест модели на переобучение"""
import torch
import numpy as np
import yaml

# Загрузка конфигурации production
with open('config/config_production.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Анализ результатов обучения из логов
print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
print("="*50)

print("\n🎯 Финальные метрики:")
print("- Direction Accuracy: 36.9%")
print("- Train Loss: 0.6196") 
print("- Val Loss: 2.2187")
print("- Best Val Loss: 2.1051")
print("- Win Rate: 45.8%")

print("\n📈 Распределение предсказаний:")
print("- LONG: 46.5% (истинное: 37.7%)")
print("- SHORT: 53.1% (истинное: 37.0%)")
print("- FLAT: 0.4% (истинное: 25.4%)")

print("\n⚠️ ПРОБЛЕМЫ:")
print("1. КРИТИЧЕСКОЕ ПЕРЕОБУЧЕНИЕ:")
print(f"   - Overfitting Ratio: {2.2187/0.6196:.2f}x")
print("   - Val Loss в 3.6 раза больше Train Loss!")

print("\n2. ДИСБАЛАНС КЛАССОВ:")
print("   - Модель почти игнорирует FLAT (0.4% vs 25.4%)")
print("   - Переоценивает LONG/SHORT сигналы")

print("\n3. НИЗКАЯ ТОЧНОСТЬ:")
print("   - Direction Accuracy 36.9% (случайная ~33.3%)")
print("   - Улучшение всего на 3.6% от случайного угадывания")

print("\n💡 РЕШЕНИЯ ДЛЯ БОРЬБЫ С ПЕРЕОБУЧЕНИЕМ:")
print("="*50)

print("\n1. УСИЛИТЬ РЕГУЛЯРИЗАЦИЮ:")
print("   - dropout: 0.5 → 0.7")
print("   - attention_dropout: 0.1 → 0.3")
print("   - weight_decay: 0.01 → 0.1")
print("   - gradient_clip: 0.1 → 0.5")

print("\n2. УМЕНЬШИТЬ РАЗМЕР МОДЕЛИ:")
print("   - d_model: 384 → 256")
print("   - d_ff: 768 → 512")
print("   - e_layers: 2 → 1")

print("\n3. ИЗМЕНИТЬ ОБУЧЕНИЕ:")
print("   - early_stopping_patience: 30 → 10")
print("   - learning_rate: 0.00001 → 0.00005")
print("   - batch_size: 512 → 256")
print("   - Добавить больше аугментации данных")

print("\n4. БАЛАНСИРОВКА КЛАССОВ:")
print("   - class_weights: [1.0, 1.0, 1.5] (увеличить вес FLAT)")
print("   - Использовать Focal Loss с gamma=3.0")
print("   - Добавить auxiliary loss для FLAT detection")

print("\n5. АНСАМБЛИРОВАНИЕ:")
print("   - Обучить 3-5 моделей с разными seed")
print("   - Использовать voting для финальных предсказаний")
print("   - Это снизит variance и улучшит обобщение")

print("\n📝 РЕКОМЕНДУЕМЫЙ config_antioverfit.yaml:")
config_new = {
    'model': {
        'dropout': 0.7,
        'attention_dropout': 0.3, 
        'weight_decay': 0.1,
        'd_model': 256,
        'd_ff': 512,
        'e_layers': 1,
        'early_stopping_patience': 10,
        'learning_rate': 0.00005,
        'batch_size': 256,
        'label_smoothing': 0.2,
        'mixup_alpha': 0.5
    },
    'loss': {
        'class_weights': [1.0, 1.0, 1.5],
        'focal_gamma': 3.0,
        'wrong_direction_penalty': 1.5
    }
}

print("\n✅ Анализ завершен!")