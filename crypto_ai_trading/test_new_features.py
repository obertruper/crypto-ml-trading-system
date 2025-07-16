#!/usr/bin/env python3
"""
Тестовый скрипт для проверки новых функций улучшения качества предсказаний
"""

import torch
import numpy as np
from pathlib import Path

def test_direction_metrics():
    """Тест метрик направления"""
    print("🧪 Тестирование direction_metrics...")
    
    from utils.direction_metrics import DirectionMetricsCalculator, quick_direction_evaluation
    
    # Создаем тестовые данные
    batch_size = 1000
    n_timeframes = 4
    
    # Генерируем случайные данные
    predictions = torch.randint(0, 3, (batch_size, n_timeframes)).float()
    targets = torch.randint(0, 3, (batch_size, n_timeframes)).float()
    returns = torch.randn(batch_size, n_timeframes) * 2  # ±2% доходности
    
    # Быстрая оценка
    quick_results = quick_direction_evaluation(predictions, targets, returns)
    
    print(f"📊 Direction Accuracy: {quick_results['direction_accuracy']:.3f}")
    print(f"💰 Win Rate: {quick_results['win_rate']:.3f}")
    print(f"📈 Profit Factor: {quick_results['profit_factor']:.2f}")
    print(f"🔢 Total Trades: {quick_results['total_trades']}")
    
    print("✅ Direction metrics тест пройден!")
    return True

def test_directional_trading_loss():
    """Тест DirectionalTradingLoss"""
    print("\n🧪 Тестирование DirectionalTradingLoss...")
    
    from models.patchtst_unified import DirectionalTradingLoss
    
    # Создаем тестовые данные
    batch_size = 32
    
    # Словари с предсказаниями и целями для разных таймфреймов
    predictions = {
        'direction_15m': torch.randn(batch_size, 3),  # Логиты для 3 классов
        'direction_1h': torch.randn(batch_size, 3),
        'direction_4h': torch.randn(batch_size, 3),
        'direction_12h': torch.randn(batch_size, 3)
    }
    
    targets = {
        'direction_15m': torch.randint(0, 3, (batch_size,)),
        'direction_1h': torch.randint(0, 3, (batch_size,)),
        'direction_4h': torch.randint(0, 3, (batch_size,)),
        'direction_12h': torch.randint(0, 3, (batch_size,))
    }
    
    price_changes = {
        '15m': torch.randn(batch_size) * 0.02,  # ±2% изменения
        '1h': torch.randn(batch_size) * 0.03,
        '4h': torch.randn(batch_size) * 0.05,
        '12h': torch.randn(batch_size) * 0.08
    }
    
    # Создаем loss функцию
    loss_fn = DirectionalTradingLoss(commission=0.001, profit_focus_weight=5.0)
    
    # Вычисляем loss
    loss = loss_fn(predictions, targets, price_changes)
    
    print(f"📉 DirectionalTradingLoss: {loss.item():.4f}")
    print(f"🔧 Loss имеет градиенты: {loss.requires_grad}")
    
    print("✅ DirectionalTradingLoss тест пройден!")
    return True

def test_optimized_trainer_metrics():
    """Тест новых метрик в OptimizedTrainer"""
    print("\n🧪 Тестирование метрик OptimizedTrainer...")
    
    from training.optimized_trainer import OptimizedTrainer
    
    # Создаем тестовые данные (имитируем выходы модели)
    batch_size = 100
    n_outputs = 20  # 20 выходов модели
    
    outputs = torch.randn(batch_size, n_outputs)
    targets = torch.randn(batch_size, n_outputs)
    
    # Устанавливаем direction targets как целые числа [0, 1, 2]
    targets[:, 4:8] = torch.randint(0, 3, (batch_size, 4)).float()
    
    # Создаем временный трейнер для доступа к методам
    import yaml
    config = {
        'model': {'epochs': 1, 'learning_rate': 1e-4},
        'performance': {'mixed_precision': False, 'gradient_accumulation_steps': 1},
        'loss': {}
    }
    
    # Создаем простую модель для тестирования
    import torch.nn as nn
    test_model = nn.Linear(10, n_outputs)
    
    trainer = OptimizedTrainer(test_model, config, device=torch.device('cpu'))
    
    # Тестируем метрики
    direction_metrics = trainer.compute_direction_metrics(outputs, targets)
    trading_metrics = trainer.compute_trading_metrics(outputs, targets)
    
    print("📊 Direction Metrics:")
    for key, value in direction_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
    
    print("\n💰 Trading Metrics:")
    for key, value in trading_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
    
    print("✅ OptimizedTrainer метрики тест пройден!")
    return True

def test_converted_data():
    """Тест конвертированных direction меток"""
    print("\n🧪 Проверка конвертированных данных...")
    
    import pandas as pd
    
    # Проверяем один из файлов
    data_file = Path("data/processed/train_data.parquet")
    
    if not data_file.exists():
        print("⚠️ Файл train_data.parquet не найден, пропускаем тест")
        return True
    
    df = pd.read_parquet(data_file)
    
    direction_cols = ['direction_15m', 'direction_1h', 'direction_4h', 'direction_12h']
    
    print(f"📊 Размер данных: {len(df):,} записей")
    
    for col in direction_cols:
        if col in df.columns:
            unique_values = sorted(df[col].dropna().unique())
            value_counts = df[col].value_counts()
            
            print(f"   {col}: {unique_values}")
            print(f"     Распределение: {dict(value_counts)}")
            
            # Проверяем что все значения в диапазоне [0, 1, 2]
            if all(v in [0, 1, 2] for v in unique_values):
                print(f"     ✅ Корректные значения")
            else:
                print(f"     ❌ Некорректные значения!")
                return False
    
    print("✅ Конвертированные данные валидны!")
    return True

def main():
    """Главная функция тестирования"""
    print("=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ НОВЫХ ФУНКЦИЙ УЛУЧШЕНИЯ КАЧЕСТВА")
    print("=" * 60)
    
    tests = [
        test_direction_metrics,
        test_directional_trading_loss, 
        test_optimized_trainer_metrics,
        test_converted_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Тест {test_func.__name__} провален: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 РЕЗУЛЬТАТЫ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("\n🚀 Система готова к обучению с новыми улучшениями:")
        print("   python main.py --mode train --direction-focus --loss-type directional")
        print("   python main.py --mode train --target-focus directions --ensemble-count 3")
        print("   python main.py --mode train --large-movement-weight 5.0 --min-movement-threshold 0.01")
    else:
        print("⚠️ Некоторые тесты не пройдены, проверьте логи выше")
    
    print("=" * 60)

if __name__ == "__main__":
    main()