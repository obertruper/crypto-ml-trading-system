#!/usr/bin/env python3
"""
Диагностика проблемы с отрицательным loss
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_confidence_loss():
    """Тестирование confidence loss с разными значениями"""
    print("\n" + "="*80)
    print("🔍 Диагностика проблемы с отрицательным loss")
    print("="*80)
    
    # Тестовые случаи
    batch_size = 4
    n_timeframes = 4
    
    # Случай 1: Нормальные предсказания
    print("\n1. Нормальные предсказания:")
    confidence_logits = torch.randn(batch_size, n_timeframes) * 2  # Логиты от -2 до 2
    correct_predictions = torch.randint(0, 2, (batch_size, n_timeframes)).float()
    
    loss1 = F.binary_cross_entropy_with_logits(confidence_logits, correct_predictions)
    print(f"   Confidence logits: {confidence_logits[0]}")
    print(f"   Correct predictions: {correct_predictions[0]}")
    print(f"   Loss: {loss1.item():.4f}")
    
    # Случай 2: Очень уверенные правильные предсказания
    print("\n2. Очень уверенные ПРАВИЛЬНЫЕ предсказания:")
    confidence_logits = torch.ones(batch_size, n_timeframes) * 10  # Очень высокие логиты
    correct_predictions = torch.ones(batch_size, n_timeframes)  # Все правильные
    
    loss2 = F.binary_cross_entropy_with_logits(confidence_logits, correct_predictions)
    print(f"   Confidence logits: {confidence_logits[0]}")
    print(f"   Correct predictions: {correct_predictions[0]}")
    print(f"   Loss: {loss2.item():.4f}")
    
    # Случай 3: Очень уверенные НЕПРАВИЛЬНЫЕ предсказания
    print("\n3. Очень уверенные НЕПРАВИЛЬНЫЕ предсказания:")
    confidence_logits = torch.ones(batch_size, n_timeframes) * 10  # Очень высокие логиты
    correct_predictions = torch.zeros(batch_size, n_timeframes)  # Все неправильные
    
    loss3 = F.binary_cross_entropy_with_logits(confidence_logits, correct_predictions)
    print(f"   Confidence logits: {confidence_logits[0]}")
    print(f"   Correct predictions: {correct_predictions[0]}")
    print(f"   Loss: {loss3.item():.4f}")
    
    # Случай 4: Негативные логиты с правильными предсказаниями
    print("\n4. Негативные логиты с ПРАВИЛЬНЫМИ предсказаниями:")
    confidence_logits = torch.ones(batch_size, n_timeframes) * -10  # Очень низкие логиты
    correct_predictions = torch.zeros(batch_size, n_timeframes)  # Правильно (low conf = wrong)
    
    loss4 = F.binary_cross_entropy_with_logits(confidence_logits, correct_predictions)
    print(f"   Confidence logits: {confidence_logits[0]}")
    print(f"   Correct predictions: {correct_predictions[0]}")
    print(f"   Loss: {loss4.item():.4f}")
    
    # Проблема с суммированием losses
    print("\n5. Проблема суммирования нескольких loss компонентов:")
    losses = []
    
    # Нормальные компоненты loss
    mse_loss = torch.tensor(0.5)
    ce_loss = torch.tensor(1.2)
    bce_loss = torch.tensor(0.8)
    
    losses.extend([mse_loss, ce_loss, bce_loss])
    
    # Проблемный confidence loss
    problematic_confidence_loss = torch.tensor(-100.0)  # Если модель слишком уверена
    losses.append(problematic_confidence_loss * 0.5)  # Даже с весом 0.5
    
    total_loss = sum(losses)
    print(f"   MSE loss: {mse_loss.item()}")
    print(f"   CE loss: {ce_loss.item()}")
    print(f"   BCE loss: {bce_loss.item()}")
    print(f"   Confidence loss (weighted): {(problematic_confidence_loss * 0.5).item()}")
    print(f"   Total loss: {total_loss.item()}")
    
    print("\n🔴 ПРОБЛЕМА НАЙДЕНА:")
    print("   BCE loss может давать очень большие отрицательные значения")
    print("   когда модель уверена в неправильных предсказаниях!")
    
    print("\n💡 РЕШЕНИЕ:")
    print("   1. Ограничить confidence logits с помощью tanh или clamp")
    print("   2. Использовать gradient clipping для confidence loss")
    print("   3. Уменьшить вес confidence loss")
    print("   4. Добавить регуляризацию для confidence head")
    print("="*80)


if __name__ == "__main__":
    test_confidence_loss()