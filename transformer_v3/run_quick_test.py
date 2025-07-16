#!/usr/bin/env python3
"""
Быстрый тест Transformer v3.0 с синтетическими данными
Для проверки работоспособности без подключения к БД
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from datetime import datetime

from config import Config
from data.sequence_creator import SequenceCreator
from models.tft_trainer import TFTTrainer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 50):
    """Генерация синтетических данных для тестирования"""
    logger.info(f"🔄 Генерация синтетических данных: {n_samples} samples, {n_features} features")
    
    # Создаем временной ряд с трендом и шумом
    time_steps = np.arange(n_samples)
    
    # Создаем базовые признаки
    features = {}
    
    # Ценовые данные
    base_price = 100 + 0.01 * time_steps + 5 * np.sin(time_steps * 0.1) + np.random.normal(0, 2, n_samples)
    features['price'] = base_price
    features['volume'] = np.exp(5 + np.random.normal(0, 0.5, n_samples))
    
    # Технические индикаторы (имитация)
    features['rsi_val'] = 50 + 20 * np.sin(time_steps * 0.05) + np.random.normal(0, 5, n_samples)
    features['rsi_val'] = np.clip(features['rsi_val'], 0, 100)
    
    features['macd_val'] = 0.1 * np.sin(time_steps * 0.03) + np.random.normal(0, 0.1, n_samples)
    features['adx_val'] = 25 + 15 * np.abs(np.sin(time_steps * 0.02)) + np.random.normal(0, 3, n_samples)
    
    # Волатильность
    features['atr'] = 0.5 + 0.3 * np.abs(np.sin(time_steps * 0.04)) + np.random.normal(0, 0.1, n_samples)
    
    # BTC корреляционные признаки
    features['btc_correlation_20'] = 0.7 + 0.2 * np.sin(time_steps * 0.01) + np.random.normal(0, 0.1, n_samples)
    features['btc_return_1h'] = np.random.normal(0, 0.02, n_samples)
    
    # Временные признаки
    features['hour_sin'] = np.sin(2 * np.pi * (time_steps % 24) / 24)
    features['hour_cos'] = np.cos(2 * np.pi * (time_steps % 24) / 24)
    features['dow_sin'] = np.sin(2 * np.pi * (time_steps % 7) / 7)
    
    # Дополняем случайными признаками до нужного количества
    for i in range(len(features), n_features):
        features[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Создаем DataFrame
    df = pd.DataFrame(features)
    
    # Добавляем метаданные
    df['symbol'] = 'SYNTHETIC'
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='15min')
    
    # Создаем целевые переменные (ожидаемые доходности)
    # Простая модель: следующая доходность зависит от RSI и MACD
    returns = 0.001 * (50 - df['rsi_val']) / 50 + 0.5 * df['macd_val'] + np.random.normal(0, 0.01, n_samples)
    
    df['buy_expected_return'] = returns
    df['sell_expected_return'] = -returns  # Противоположная стратегия
    
    logger.info("✅ Синтетические данные созданы")
    return df

def main():
    """Основная функция быстрого теста"""
    logger.info("""
    ╔══════════════════════════════════════════╗
    ║       Transformer v3.0 Quick Test        ║
    ║        Синтетические данные              ║
    ╚══════════════════════════════════════════╝
    """)
    
    # Конфигурация для быстрого теста
    config = Config()
    config.model.hidden_size = 64
    config.model.sequence_length = 20  # Короткая последовательность
    config.model.batch_size = 16
    config.model.epochs = 5  # Мало эпох для быстроты
    config.model.use_mixed_precision = False
    config.training.task_type = "regression"
    config.training.use_data_augmentation = False
    config.training.save_plots = True
    config.training.save_models = False  # Не сохраняем модели в тесте
    
    try:
        # 1. Генерация синтетических данных
        df = generate_synthetic_data(n_samples=500, n_features=30)
        
        # 2. Создание последовательностей
        logger.info("🔄 Создание временных последовательностей...")
        sequence_creator = SequenceCreator(config)
        
        # Подготавливаем данные
        feature_cols = [col for col in df.columns if col not in ['symbol', 'timestamp', 'buy_expected_return', 'sell_expected_return']]
        X = df[feature_cols]
        y_buy = df['buy_expected_return']
        y_sell = df['sell_expected_return']
        
        # Создаем последовательности
        sequences_data = sequence_creator.create_sequences(X, y_buy, y_sell)
        sequences_splits = sequence_creator.split_sequences(sequences_data)
        
        # 3. Тестирование обучения для buy модели
        logger.info("🚀 Тестирование обучения модели...")
        
        # Создаем trainer
        trainer = TFTTrainer(config, model_name="test_buy_predictor")
        
        # Получаем данные
        X_train = sequences_splits['buy']['X_train']
        y_train = sequences_splits['buy']['y_train']
        X_val = sequences_splits['buy']['X_val']
        y_val = sequences_splits['buy']['y_val']
        X_test = sequences_splits['buy']['X_test']
        y_test = sequences_splits['buy']['y_test']
        
        logger.info(f"   Размеры данных:")
        logger.info(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Обучение модели
        model = trainer.train(X_train, y_train, X_val, y_val)
        
        # Оценка модели
        test_metrics = trainer.evaluate(X_test, y_test, "Test")
        
        # 4. Тестирование предсказаний
        logger.info("🔮 Тестирование предсказаний...")
        predictions = trainer.predict(X_test[:10])  # Первые 10 семплов
        
        logger.info(f"   Примеры предсказаний:")
        for i in range(5):
            true_val = y_test[i]
            pred_val = predictions[i][0] if len(predictions[i]) > 0 else predictions[i]
            logger.info(f"   {i+1}. Истинное: {true_val:.4f}, Предсказанное: {pred_val:.4f}")
        
        # 5. Финальная статистика
        logger.info("\n📊 Результаты быстрого теста:")
        logger.info(f"   Последовательностей обработано: {len(sequences_data['X'])}")
        logger.info(f"   Размерность входа: {X_train.shape}")
        logger.info(f"   Параметров модели: {model.count_params():,}")
        
        for metric_name, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {metric_name}: {value:.4f}")
        
        logger.info("\n🎉 Быстрый тест завершен успешно!")
        logger.info("✅ Transformer v3.0 работает корректно")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Ошибка в быстром тесте: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())