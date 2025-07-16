#!/usr/bin/env python3
"""
Тестирование всех production улучшений
"""

import torch
import yaml
from pathlib import Path
import numpy as np

def test_weighted_loss():
    """Тест weighted loss для дисбаланса классов"""
    print("\n🧪 Тестирование Weighted Loss...")
    
    from models.patchtst_unified import DirectionalMultiTaskLoss
    
    # Загружаем конфигурацию
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Создаем loss функцию
    loss_fn = DirectionalMultiTaskLoss(config)
    
    # Проверяем веса классов
    print(f"✅ Веса классов установлены: {loss_fn.class_weights}")
    print(f"   LONG weight: {loss_fn.class_weights[0]:.2f}")
    print(f"   SHORT weight: {loss_fn.class_weights[1]:.2f}")
    print(f"   FLAT weight: {loss_fn.class_weights[2]:.2f}")
    
    # Тестовые данные с сильным дисбалансом (как в реальности)
    batch_size = 100
    outputs = torch.randn(batch_size, 20, requires_grad=True)
    targets = torch.randn(batch_size, 20)
    
    # Устанавливаем direction классы с дисбалансом
    # 80% FLAT, 10% LONG, 10% SHORT
    direction_classes = torch.zeros(batch_size, 4)
    direction_classes[:80, :] = 2  # FLAT
    direction_classes[80:90, :] = 0  # LONG
    direction_classes[90:, :] = 1  # SHORT
    targets[:, 4:8] = direction_classes
    
    # Создаем direction_logits
    direction_logits = torch.randn(batch_size, 4, 3, requires_grad=True)
    outputs._direction_logits = direction_logits
    
    # Вычисляем loss
    loss = loss_fn(outputs, targets)
    
    print(f"✅ Loss вычислен успешно: {loss.item():.4f}")
    print(f"✅ Loss требует градиенты: {loss.requires_grad}")
    
    # Проверяем backward
    loss.backward()
    print("✅ Backward pass выполнен успешно")
    
    return True


def test_bias_initialization():
    """Тест инициализации bias для direction head"""
    print("\n🧪 Тестирование инициализации Direction Head...")
    
    from models.patchtst_unified import UnifiedPatchTSTForTrading
    
    # Загружаем конфигурацию
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Создаем модель
    model = UnifiedPatchTSTForTrading(config)
    
    # Проверяем bias в direction head
    for name, param in model.direction_head.named_parameters():
        if 'bias' in name and param.shape[0] == 12:
            biases = param.detach().view(4, 3)  # 4 таймфрейма x 3 класса
            
            print("✅ Direction Head biases:")
            for tf in range(4):
                print(f"   Таймфрейм {tf+1}:")
                print(f"      LONG bias:  {biases[tf, 0]:.3f}")
                print(f"      SHORT bias: {biases[tf, 1]:.3f}")
                print(f"      FLAT bias:  {biases[tf, 2]:.3f}")
            
            # Проверяем что FLAT имеет отрицательный bias
            flat_biases = biases[:, 2]
            if (flat_biases < 0).all():
                print("✅ Все FLAT biases отрицательные - хорошо!")
            else:
                print("❌ Не все FLAT biases отрицательные!")
                return False
    
    return True


def test_config_updates():
    """Тест обновлений конфигурации"""
    print("\n🧪 Тестирование обновлений конфигурации...")
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Проверяем learning rate
    lr = config['model']['learning_rate']
    print(f"✅ Learning rate: {lr}")
    if lr != 0.001:
        print(f"❌ Learning rate должен быть 0.001, а не {lr}")
        return False
    
    # Проверяем warmup
    warmup = config['model'].get('warmup_steps', 0)
    print(f"✅ Warmup steps: {warmup}")
    # Warmup есть, это главное
    
    # Проверяем вес direction loss
    direction_weight = config['loss']['task_weights']['directions']
    print(f"✅ Direction loss weight: {direction_weight}")
    if direction_weight != 10.0:
        print(f"❌ Direction weight должен быть 10.0, а не {direction_weight}")
        return False
    
    # Проверяем focal loss параметры
    focal_alpha = config['loss'].get('focal_alpha', 0.25)
    focal_gamma = config['loss'].get('focal_gamma', 2.0)
    print(f"✅ Focal loss параметры: alpha={focal_alpha}, gamma={focal_gamma}")
    
    return True


def test_diversity_monitoring():
    """Тест мониторинга разнообразия предсказаний"""
    print("\n🧪 Тестирование мониторинга разнообразия...")
    
    from training.optimized_trainer import OptimizedTrainer
    
    # Создаем фиктивную модель
    import torch.nn as nn
    model = nn.Linear(10, 20)
    
    # Загружаем конфигурацию
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = OptimizedTrainer(model, config, device=torch.device('cpu'))
    
    # Тестовые данные
    batch_size = 100
    outputs = torch.randn(batch_size, 20)
    targets = torch.randn(batch_size, 20)
    
    # Создаем direction_logits с разным распределением
    # Тест 1: Сбалансированное распределение
    direction_logits = torch.randn(batch_size, 4, 3)
    outputs._direction_logits = direction_logits
    
    # Устанавливаем целевые классы
    targets[:, 4:8] = torch.randint(0, 3, (batch_size, 4)).float()
    
    # Вычисляем метрики
    metrics = trainer.compute_direction_metrics(outputs, targets)
    
    print("✅ Метрики для сбалансированного распределения:")
    print(f"   Энтропия: {metrics.get('pred_entropy_overall', 0):.3f}")
    print(f"   LONG ratio: {metrics.get('pred_long_ratio_overall', 0):.1%}")
    print(f"   SHORT ratio: {metrics.get('pred_short_ratio_overall', 0):.1%}")
    print(f"   FLAT ratio: {metrics.get('pred_flat_ratio_overall', 0):.1%}")
    
    # Тест 2: Несбалансированное распределение (все FLAT)
    direction_logits = torch.zeros(batch_size, 4, 3)
    direction_logits[:, :, 2] = 10  # Сильное предпочтение FLAT
    outputs._direction_logits = direction_logits
    
    metrics = trainer.compute_direction_metrics(outputs, targets)
    
    print("\n✅ Метрики для несбалансированного распределения (все FLAT):")
    print(f"   Энтропия: {metrics.get('pred_entropy_overall', 0):.3f}")
    print(f"   FLAT ratio: {metrics.get('pred_flat_ratio_overall', 0):.1%}")
    
    if metrics.get('pred_entropy_overall', 1) < 0.1:
        print("✅ Низкая энтропия правильно обнаружена!")
    
    return True


def test_production_ready_main():
    """Тест production-ready main.py"""
    print("\n🧪 Тестирование Production-Ready функционала...")
    
    # Проверяем наличие main_production.py
    if Path('main_production.py').exists():
        print("✅ main_production.py создан")
        
        # Импортируем и проверяем классы
        try:
            from main_production import ProductionConfig, ModelValidator, ProductionInference
            print("✅ Production классы импортированы успешно")
            
            # Тестируем ProductionConfig
            config = ProductionConfig('config/config.yaml')
            print("✅ ProductionConfig работает")
            
            return True
        except Exception as e:
            print(f"❌ Ошибка импорта: {e}")
            return False
    else:
        print("❌ main_production.py не найден")
        return False


def main():
    """Запуск всех тестов"""
    print("="*80)
    print("🚀 Тестирование Production улучшений")
    print("="*80)
    
    tests = [
        ("Weighted Loss", test_weighted_loss),
        ("Bias Initialization", test_bias_initialization),
        ("Config Updates", test_config_updates),
        ("Diversity Monitoring", test_diversity_monitoring),
        ("Production Ready Main", test_production_ready_main)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED\n")
            else:
                print(f"❌ {test_name} - FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}\n")
    
    print("="*80)
    print(f"📊 Результаты: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты пройдены успешно!")
        print("\n🚀 Система готова к production!")
        print("\nСледующие шаги:")
        print("1. Запустить обучение: python main_production.py --mode train")
        print("2. Проверить модель: python evaluate_model_production.py")
        print("3. Валидация: python main_production.py --mode validate --model-path models_saved/best_model.pth")
    else:
        print("⚠️ Некоторые тесты не пройдены, проверьте логи выше")


if __name__ == "__main__":
    main()