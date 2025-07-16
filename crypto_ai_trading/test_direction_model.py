"""
Быстрый тест Direction модели
"""

import torch
import numpy as np
from models.direction_predictor import DirectionPredictor
from utils.config import load_config

def test_direction_model():
    """Тестирование создания и forward pass модели"""
    
    print("🧪 Тестирование Direction модели...")
    
    # Загружаем конфигурацию
    config = load_config('configs/direction_only.yaml')
    config['model']['n_features'] = 254  # Из данных
    
    # Создаем модель
    print("📦 Создание модели...")
    model = DirectionPredictor(config['model'])
    print(f"✅ Модель создана успешно!")
    print(f"   Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Тестовый батч
    batch_size = 16
    seq_len = 168
    n_features = 254
    
    # Создаем случайные данные
    print("\n🎲 Создание тестовых данных...")
    test_input = torch.randn(batch_size, seq_len, n_features)
    
    # Forward pass
    print("🚀 Forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(test_input)
    
    # Проверяем выходы
    print("\n📊 Результаты:")
    for key, value in outputs.items():
        print(f"   {key}: shape={value.shape}, min={value.min():.4f}, max={value.max():.4f}")
        
        # Проверяем softmax
        probs = torch.softmax(value, dim=-1)
        print(f"      Probabilities sum: {probs.sum(dim=-1).mean():.4f}")
        
        # Предсказания
        predictions = torch.argmax(value, dim=-1)
        unique_preds = torch.unique(predictions)
        print(f"      Unique predictions: {unique_preds.tolist()}")
    
    # Тест confidence через softmax
    print("\n🎯 Тест уверенности предсказаний...")
    with torch.no_grad():
        single_output = model(test_input[:1])
        
    for key, logits in single_output.items():
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_class = torch.max(probs, dim=-1)
        print(f"   {key}: класс={pred_class.item()}, уверенность={confidence.item():.2%}")
    
    print("\n✅ Все тесты пройдены успешно!")
    
    # Проверим совместимость с loss
    from models.direction_predictor import DirectionalTradingLoss
    print("\n🎯 Тест loss функции...")
    
    criterion = DirectionalTradingLoss()
    
    # Создаем фиктивные targets
    targets = {}
    price_changes = {}
    
    for timeframe in ['15m', '1h', '4h', '12h']:
        key = f'direction_{timeframe}'
        # Случайные классы 0, 1, 2
        targets[key] = torch.randint(0, 3, (batch_size,))
        # Случайные изменения цен
        price_changes[timeframe] = torch.randn(batch_size) * 0.05  # ±5%
    
    # Вычисляем loss
    loss = criterion(outputs, targets, price_changes)
    print(f"✅ Loss вычислен: {loss.item():.4f}")
    
    return model


if __name__ == "__main__":
    try:
        model = test_direction_model()
        print("\n🎉 Модель готова к обучению!")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()