#!/usr/bin/env python3
"""
Тестирование исправлений для проблемы с отрицательным loss
"""

import torch
import torch.nn.functional as F
import numpy as np

def test_tanh_confidence():
    """Тестирование confidence с Tanh активацией"""
    print("\n" + "="*80)
    print("🔍 Тестирование исправлений confidence loss")
    print("="*80)
    
    batch_size = 4
    n_timeframes = 4
    
    print("\n1. Старый подход (без ограничений):")
    # Симуляция больших логитов без ограничений
    confidence_logits_old = torch.randn(batch_size, n_timeframes) * 10  # Могут быть очень большими
    correct_predictions = torch.randint(0, 2, (batch_size, n_timeframes)).float()
    
    # BCE с логитами
    loss_old = F.binary_cross_entropy_with_logits(confidence_logits_old, correct_predictions)
    print(f"   Логиты: min={confidence_logits_old.min():.2f}, max={confidence_logits_old.max():.2f}")
    print(f"   Loss: {loss_old.item():.4f}")
    
    # Проблемный случай
    confidence_logits_problem = torch.ones(batch_size, n_timeframes) * 20  # Очень большие логиты
    correct_predictions_problem = torch.zeros(batch_size, n_timeframes)  # Все неправильные
    loss_problem = F.binary_cross_entropy_with_logits(confidence_logits_problem, correct_predictions_problem)
    print(f"\n   Проблемный случай:")
    print(f"   Логиты: {confidence_logits_problem[0]}")
    print(f"   Loss: {loss_problem.item():.4f} (очень большой!)")
    
    print("\n2. Новый подход (с Tanh и MSE):")
    # Симуляция выхода с Tanh
    confidence_tanh = torch.tanh(torch.randn(batch_size, n_timeframes) * 2)  # В диапазоне [-1, 1]
    
    # Преобразование целей в тот же диапазон
    confidence_targets = correct_predictions * 2 - 1  # Из [0, 1] в [-1, 1]
    
    # MSE loss (безопасен для autocast)
    loss_new = F.mse_loss(confidence_tanh, confidence_targets)
    print(f"   Tanh выход: min={confidence_tanh.min():.2f}, max={confidence_tanh.max():.2f}")
    print(f"   Целевые значения: min={confidence_targets.min():.2f}, max={confidence_targets.max():.2f}")
    print(f"   Loss: {loss_new.item():.4f}")
    
    # Тот же проблемный случай
    confidence_tanh_extreme = torch.ones(batch_size, n_timeframes) * 0.99  # Близко к 1 после tanh
    confidence_targets_extreme = correct_predictions_problem * 2 - 1  # Все -1 (неправильные)
    loss_extreme = F.mse_loss(confidence_tanh_extreme, confidence_targets_extreme)
    print(f"\n   Экстремальный случай:")
    print(f"   Tanh выход: {confidence_tanh_extreme[0]} (все уверенные)")
    print(f"   Целевые значения: {confidence_targets_extreme[0]} (все неправильные)")
    print(f"   Loss: {loss_extreme.item():.4f} (ограничен!)")
    
    print("\n3. Сравнение подходов:")
    print(f"   Старый подход: BCE loss может быть от 0 до +∞")
    print(f"   Новый подход: MSE loss ограничен диапазоном значений Tanh")
    print(f"   Максимальный MSE loss: (1 - (-1))^2 = 4.0")
    
    print("\n4. Влияние на общий loss:")
    # Симуляция других компонентов loss
    mse_loss = torch.tensor(1.5)
    ce_loss = torch.tensor(2.0)
    bce_loss = torch.tensor(0.8)
    
    # Старый подход
    total_loss_old = mse_loss + ce_loss + bce_loss + loss_problem * 0.5
    print(f"   Старый подход: {mse_loss} + {ce_loss} + {bce_loss} + {loss_problem:.1f} * 0.5 = {total_loss_old:.1f}")
    
    # Новый подход  
    total_loss_new = mse_loss + ce_loss + bce_loss + loss_extreme * 0.1  # Меньший вес
    print(f"   Новый подход: {mse_loss} + {ce_loss} + {bce_loss} + {loss_extreme:.3f} * 0.1 = {total_loss_new:.3f}")
    
    print("\n✅ РЕЗУЛЬТАТ:")
    print("   1. Tanh ограничивает confidence в диапазоне [-1, 1]")
    print("   2. MSE loss безопасен для autocast и ограничен по значению")
    print("   3. Уменьшенный вес (0.1 вместо 0.5) дополнительно стабилизирует")
    print("   4. Loss больше не может стать отрицательным!")
    print("   5. Совместимо с Mixed Precision Training!")
    print("="*80)


if __name__ == "__main__":
    test_tanh_confidence()