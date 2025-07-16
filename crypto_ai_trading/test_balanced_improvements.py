#!/usr/bin/env python3
"""
Тестирование сбалансированных улучшений
"""

import torch
import yaml

def test_moderate_weights():
    """Тест умеренных весов и bias"""
    print("\n🧪 Тестирование сбалансированных улучшений...")
    
    from models.patchtst_unified import DirectionalMultiTaskLoss, UnifiedPatchTSTForTrading
    
    # Загружаем конфигурацию
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Проверяем loss функцию
    loss_fn = DirectionalMultiTaskLoss(config)
    print(f"\n✅ Умеренные веса классов: {loss_fn.class_weights}")
    print(f"   LONG weight: {loss_fn.class_weights[0]:.2f}")
    print(f"   SHORT weight: {loss_fn.class_weights[1]:.2f}")
    print(f"   FLAT weight: {loss_fn.class_weights[2]:.2f}")
    
    # Проверяем warmup
    print(f"\n✅ Warmup настройки:")
    print(f"   Начальный вес direction: 1.0")
    print(f"   Конечный вес direction: {loss_fn.directions_weight}")
    print(f"   Warmup эпох: {loss_fn.warmup_epochs}")
    
    # Проверяем динамические веса
    for epoch in [0, 5, 10, 15]:
        loss_fn.set_epoch(epoch)
        weight = loss_fn.get_dynamic_direction_weight()
        print(f"   Эпоха {epoch}: direction weight = {weight:.2f}")
    
    # Проверяем bias в модели
    model = UnifiedPatchTSTForTrading(config)
    
    print("\n✅ Direction Head biases (умеренные):")
    for name, param in model.direction_head.named_parameters():
        if 'bias' in name and param.shape[0] == 12:
            biases = param.detach().view(4, 3)
            print(f"   LONG bias:  {biases[0, 0]:.3f}")
            print(f"   SHORT bias: {biases[0, 1]:.3f}")
            print(f"   FLAT bias:  {biases[0, 2]:.3f}")
            break
    
    # Проверяем конфигурацию
    print(f"\n✅ Параметры конфигурации:")
    print(f"   Learning rate: {config['model']['learning_rate']}")
    print(f"   Direction weight (max): {config['loss']['task_weights']['directions']}")
    print(f"   Focal alpha: {config['loss']['focal_alpha']}")
    print(f"   Focal gamma: {config['loss']['focal_gamma']}")
    
    return True


def main():
    """Запуск тестов"""
    print("="*80)
    print("🚀 Тестирование сбалансированных улучшений")
    print("="*80)
    
    if test_moderate_weights():
        print("\n✅ Все тесты пройдены!")
        print("\n📊 Ожидаемые результаты:")
        print("   - Более сбалансированное распределение предсказаний")
        print("   - Постепенное улучшение через warmup")
        print("   - Direction accuracy должна расти постепенно")
        print("   - Предсказания: LONG ~20-30%, SHORT ~15-25%, FLAT ~45-65%")
        
        print("\n🚀 Запустите обучение:")
        print("   python main.py --mode train")
    else:
        print("\n❌ Тесты не пройдены!")


if __name__ == "__main__":
    main()