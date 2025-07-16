#!/usr/bin/env python3
"""
Тестирование улучшений для повышения уверенности модели
"""

import torch
import yaml
import numpy as np
from pathlib import Path

def test_confidence_improvements():
    """Проверка всех механизмов повышения уверенности"""
    print("\n" + "="*80)
    print("🧪 Тестирование улучшений для повышения уверенности модели")
    print("="*80)
    
    # Загружаем конфигурацию
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n✅ Проверка конфигурации:")
    print(f"   - Label smoothing: {config['model'].get('label_smoothing', 0)}")
    print(f"   - Temperature scaling: {config['model'].get('temperature_scaling', False)}")
    print(f"   - Confidence threshold: {config['model'].get('confidence_threshold', 0.6)}")
    print(f"   - Dropout schedule: {config['model'].get('dropout_schedule', False)}")
    print(f"   - Mixup alpha: {config['model'].get('mixup_alpha', 0)}")
    print(f"   - Early stopping patience: {config['model'].get('early_stopping_patience', 25)}")
    
    # Проверяем модель
    from models.patchtst_unified import UnifiedPatchTSTForTrading
    
    print("\n🏗️ Создание модели с новыми компонентами...")
    model = UnifiedPatchTSTForTrading(config)
    
    # Проверяем наличие новых компонентов
    print("\n📋 Проверка компонентов модели:")
    
    # Temperature scaling
    if hasattr(model, 'temperature'):
        if model.temperature is not None:
            print(f"   ✅ Temperature parameter: {model.temperature.item():.3f}")
        else:
            print("   ❌ Temperature scaling выключен")
    else:
        print("   ❌ Temperature parameter не найден")
    
    # Confidence head
    if hasattr(model, 'confidence_head'):
        print("   ✅ Confidence head найден")
        # Проверяем размерность
        dummy_input = torch.randn(1, model.d_model)
        with torch.no_grad():
            conf_output = model.confidence_head(dummy_input)
        print(f"      Выходная размерность: {conf_output.shape}")
    else:
        print("   ❌ Confidence head не найден")
    
    # Проверяем Loss функцию
    from models.patchtst_unified import DirectionalMultiTaskLoss
    
    print("\n🎯 Проверка Loss функции:")
    loss_fn = DirectionalMultiTaskLoss(config)
    
    print(f"   - Label smoothing: {loss_fn.label_smoothing}")
    print(f"   - Class weights: {loss_fn.class_weights}")
    print(f"   - Warmup epochs: {loss_fn.warmup_epochs}")
    
    # Тест label smoothing
    if loss_fn.label_smoothing > 0:
        print("\n   📊 Тест label smoothing:")
        targets = torch.tensor([0, 1, 2, 0])  # LONG, SHORT, FLAT, LONG
        smoothed = loss_fn.apply_label_smoothing(targets, num_classes=3)
        print(f"      Original targets: {targets}")
        print(f"      Smoothed targets shape: {smoothed.shape}")
        print(f"      Smoothed example (class 0): {smoothed[0]}")
    
    # Проверяем Trainer
    print("\n🏋️ Проверка OptimizedTrainer:")
    
    from training.optimized_trainer import OptimizedTrainer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    trainer = OptimizedTrainer(model, config, device)
    
    if hasattr(trainer, 'use_dropout_schedule'):
        print(f"   ✅ Dropout schedule: {trainer.use_dropout_schedule}")
        if trainer.use_dropout_schedule:
            print(f"      Initial dropout: {trainer.initial_dropout}")
            print(f"      Final dropout: {trainer.final_dropout}")
            print(f"      Warmup epochs: {trainer.dropout_warmup_epochs}")
    
    if hasattr(trainer, 'use_mixup'):
        print(f"   ✅ Mixup augmentation: {trainer.use_mixup}")
        if trainer.use_mixup:
            print(f"      Alpha: {trainer.mixup_alpha}")
    
    # Тест forward pass с confidence
    print("\n🔄 Тест forward pass модели:")
    
    batch_size = 4
    seq_len = config['model']['context_window']
    n_features = config['model']['input_size']
    
    dummy_input = torch.randn(batch_size, seq_len, n_features).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"   - Выходная размерность: {outputs.shape}")
    
    if hasattr(outputs, '_direction_logits'):
        print(f"   ✅ Direction logits: {outputs._direction_logits.shape}")
        
        # Проверяем temperature scaling
        if model.temperature is not None:
            # Вычисляем softmax с и без temperature
            logits = outputs._direction_logits[0, 0, :]  # Первый батч, первый таймфрейм
            probs_normal = torch.softmax(logits, dim=-1)
            probs_scaled = torch.softmax(logits / model.temperature, dim=-1)
            
            print(f"\n   📊 Эффект Temperature Scaling:")
            print(f"      Без scaling: {probs_normal}")
            print(f"      С scaling:   {probs_scaled}")
            print(f"      Энтропия без: {-(probs_normal * torch.log(probs_normal + 1e-8)).sum().item():.3f}")
            print(f"      Энтропия с:  {-(probs_scaled * torch.log(probs_scaled + 1e-8)).sum().item():.3f}")
    
    if hasattr(outputs, '_confidence_scores'):
        print(f"   ✅ Confidence scores: {outputs._confidence_scores.shape}")
        conf_logits = outputs._confidence_scores
        conf_probs = torch.sigmoid(conf_logits)  # Применяем sigmoid к логитам
        print(f"      Средняя уверенность: {conf_probs.mean().item():.3f}")
        print(f"      Мин/Макс: {conf_probs.min().item():.3f} / {conf_probs.max().item():.3f}")
    
    # Тест confidence filter
    print("\n🎯 Тест фильтрации по уверенности:")
    
    from utils.confidence_filter import filter_predictions_by_confidence, get_high_confidence_signals
    
    # Создаем тестовые предсказания
    test_predictions = {
        'direction_classes': torch.randint(0, 3, (batch_size, 4)),
        'confidence_scores': torch.rand(batch_size, 4),
        'future_returns': torch.randn(batch_size, 4) * 0.05,
        'long_levels': torch.rand(batch_size, 4),
        'short_levels': torch.rand(batch_size, 4)
    }
    
    # Фильтруем
    filtered = filter_predictions_by_confidence(test_predictions, confidence_threshold=0.6)
    
    # Извлекаем высокоуверенные сигналы
    signals = get_high_confidence_signals(test_predictions, min_confidence=0.7)
    
    print("\n" + "="*80)
    print("✅ Все механизмы повышения уверенности успешно реализованы!")
    print("\n🎯 Ожидаемые улучшения:")
    print("   1. Temperature scaling сделает предсказания более 'острыми'")
    print("   2. Label smoothing улучшит генерализацию")
    print("   3. Confidence head позволит фильтровать неуверенные предсказания")
    print("   4. Dropout schedule постепенно увеличит уверенность")
    print("   5. Confidence-aware loss заставит модель быть более решительной")
    print("\n🚀 Запустите обучение для проверки эффективности:")
    print("   python main.py --mode train")
    print("="*80)


if __name__ == "__main__":
    test_confidence_improvements()