#!/usr/bin/env python3
"""
Тестовый скрипт для проверки архитектуры Transformer v3.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from config import Config
from models.tft_model import create_tft_model, TemporalFusionTransformer
from data.sequence_creator import SequenceCreator

def test_config():
    """Тест конфигурации"""
    print("🧪 Тестирование конфигурации...")
    
    # Создаем конфигурацию
    config = Config()
    
    # Проверяем значения по умолчанию
    assert config.model.hidden_size == 160
    assert config.model.sequence_length == 100
    assert config.model.num_heads == 4
    assert config.training.task_type == "regression"
    
    # Валидация
    config.validate()
    
    print("✅ Конфигурация работает корректно")

def test_tft_model():
    """Тест TFT модели"""
    print("🧪 Тестирование TFT модели...")
    
    config = Config()
    config.model.hidden_size = 64  # Уменьшаем для теста
    config.model.sequence_length = 20
    config.model.batch_size = 4
    
    # Создаем модель
    input_shape = (20, 50)  # 20 timesteps, 50 features
    model = create_tft_model(config, input_shape)
    
    # Проверяем архитектуру
    assert model is not None
    print(f"   Параметров в модели: {model.count_params():,}")
    
    # Тестовые данные
    batch_size = 4
    test_input = tf.random.normal((batch_size, 20, 50))
    
    # Forward pass
    output = model(test_input)
    
    # Проверяем размерность выхода
    assert output.shape == (batch_size, 1)
    
    print("✅ TFT модель работает корректно")

def test_sequence_creator():
    """Тест создания последовательностей"""
    print("🧪 Тестирование создания последовательностей...")
    
    config = Config()
    config.model.sequence_length = 10  # Маленькая последовательность для теста
    
    sequence_creator = SequenceCreator(config)
    
    # Создаем тестовые данные
    n_samples = 50
    n_features = 20
    
    # DataFrame с признаками
    import pandas as pd
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    X['symbol'] = ['TEST'] * n_samples
    X['timestamp'] = range(n_samples)
    
    # Целевые переменные
    y_buy = pd.Series(np.random.randn(n_samples))
    y_sell = pd.Series(np.random.randn(n_samples))
    
    # Создаем последовательности
    sequences_data = sequence_creator.create_sequences(X, y_buy, y_sell)
    
    # Проверяем результат
    assert 'X' in sequences_data
    assert 'y_buy' in sequences_data
    assert 'y_sell' in sequences_data
    
    X_seq = sequences_data['X']
    assert len(X_seq.shape) == 3  # [samples, timesteps, features]
    assert X_seq.shape[1] == config.model.sequence_length
    
    print(f"   Создано последовательностей: {X_seq.shape[0]}")
    print(f"   Размерность: {X_seq.shape}")
    print("✅ Создание последовательностей работает корректно")

def test_integration():
    """Интеграционный тест"""
    print("🧪 Интеграционный тест...")
    
    config = Config()
    config.model.hidden_size = 32
    config.model.sequence_length = 10
    config.model.batch_size = 2
    config.model.epochs = 1
    config.model.use_mixed_precision = False  # Отключаем mixed precision для теста
    config.training.task_type = "regression"
    
    # Создаем модель
    input_shape = (10, 20)
    model = create_tft_model(config, input_shape)
    
    # Тестовые данные для обучения
    n_train = 100
    X_train = np.random.randn(n_train, 10, 20)
    y_train = np.random.randn(n_train)
    
    X_val = np.random.randn(20, 10, 20)
    y_val = np.random.randn(20)
    
    # Компилируем и тестируем обучение
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Краткое обучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=2,
        verbose=0
    )
    
    # Предсказания
    predictions = model.predict(X_val[:5])
    assert predictions.shape == (5, 1)
    
    print("✅ Интеграционный тест прошел успешно")

def main():
    """Запуск всех тестов"""
    print("""
    ╔══════════════════════════════════════════╗
    ║     Transformer v3.0 - Architecture      ║
    ║            Testing Suite                 ║
    ╚══════════════════════════════════════════╝
    """)
    
    try:
        # Проверяем доступность TensorFlow
        print(f"🔍 TensorFlow версия: {tf.__version__}")
        
        if tf.config.list_physical_devices('GPU'):
            print("🖥️ GPU доступен")
        else:
            print("💻 Используется CPU")
        
        # Запускаем тесты
        test_config()
        test_tft_model()
        test_sequence_creator()
        test_integration()
        
        print("\n🎉 Все тесты прошли успешно!")
        print("✅ Архитектура Transformer v3.0 готова к использованию")
        
    except Exception as e:
        print(f"\n❌ Ошибка в тестах: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())