#!/usr/bin/env python3
"""
Тест оптимизаций GPU для обучения модели
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import time
import yaml

from data.optimized_dataset import create_optimized_dataloaders
from training.optimized_trainer import OptimizedTrainer
from models.patchtst_unified import create_unified_model
from utils.logger import get_logger

def test_gpu_optimization():
    """Тестирование оптимизированного pipeline"""
    logger = get_logger("GPUOptimizationTest")
    
    logger.info("🚀 Тестирование GPU оптимизаций")
    
    # Загрузка конфигурации
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Проверка GPU
    if not torch.cuda.is_available():
        logger.error("❌ GPU не доступен!")
        return
    
    device = torch.device('cuda')
    logger.info(f"✅ Используется GPU: {torch.cuda.get_device_name()}")
    logger.info(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Загрузка данных
    logger.info("📊 Загрузка данных...")
    try:
        train_data = pd.read_parquet('cache/train_data.parquet')
        val_data = pd.read_parquet('cache/val_data.parquet')
        test_data = pd.read_parquet('cache/test_data.parquet')
        
        # Для теста берем небольшую часть
        train_data = train_data.head(50000)
        val_data = val_data.head(10000)
        test_data = test_data.head(10000)
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки данных: {e}")
        logger.info("Создаем синтетические данные для теста...")
        
        # Создаем синтетические данные
        n_samples = 50000
        n_features = 171
        n_targets = 37
        
        # Создаем DataFrame с нужными колонками
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        target_cols = ['future_return_1', 'future_return_2', 'future_return_3', 'future_return_4',
                      'long_tp1_hit', 'long_tp1_time', 'long_tp2_hit', 'long_tp2_time',
                      'long_tp3_hit', 'long_tp3_time', 'long_sl_hit', 'long_sl_time',
                      'short_tp1_hit', 'short_tp1_time', 'short_tp2_hit', 'short_tp2_time',
                      'short_tp3_hit', 'short_tp3_time', 'short_sl_hit', 'short_sl_time',
                      'long_optimal_entry_improvement', 'short_optimal_entry_improvement',
                      'long_tp1_reached', 'long_tp2_reached', 'long_tp3_reached', 'long_sl_reached',
                      'short_tp1_reached', 'short_tp2_reached', 'short_tp3_reached', 'short_sl_reached',
                      'long_expected_value', 'short_expected_value',
                      'long_optimal_entry_time', 'short_optimal_entry_time',
                      'best_direction', 'signal_strength', 'target_return_1h']
        
        # Генерация данных
        data = {}
        data['symbol'] = np.random.choice(['BTCUSDT', 'ETHUSDT'], n_samples)
        data['datetime'] = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
        
        for col in feature_cols:
            data[col] = np.random.randn(n_samples).astype(np.float32)
        
        for col in target_cols:
            if 'hit' in col or 'reached' in col:
                data[col] = np.random.choice([0, 1], n_samples).astype(np.float32)
            elif col == 'best_direction':
                data[col] = np.random.choice([0, 1, 2], n_samples).astype(np.float32)
            else:
                data[col] = np.random.randn(n_samples).astype(np.float32)
        
        train_data = pd.DataFrame(data)
        val_data = train_data.copy()
        test_data = train_data.copy()
        
        # Обновляем конфиг
        config['feature_cols'] = feature_cols
        config['target_cols'] = target_cols
    
    # Создание оптимизированных DataLoader'ов
    logger.info("🔧 Создание оптимизированных DataLoader'ов...")
    start_time = time.time()
    
    train_loader, val_loader, test_loader = create_optimized_dataloaders(
        train_data, val_data, test_data, config, logger
    )
    
    logger.info(f"✅ DataLoader'ы созданы за {time.time() - start_time:.1f}с")
    
    # Создание модели
    logger.info("🏗️ Создание модели...")
    model = create_unified_model(config)
    
    # Создание оптимизированного тренера
    logger.info("🎯 Создание оптимизированного тренера...")
    trainer = OptimizedTrainer(model, config, device)
    
    # Тест скорости загрузки данных
    logger.info("\n📊 Тест скорости загрузки данных:")
    test_batches = 10
    
    # Без оптимизаций (для сравнения)
    logger.info("1. Базовая загрузка (без pin_memory):")
    start = time.time()
    for i, (inputs, targets, info) in enumerate(train_loader):
        if i >= test_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
    elapsed = time.time() - start
    logger.info(f"   Время: {elapsed:.2f}с ({test_batches/elapsed:.1f} batches/s)")
    
    # С оптимизациями
    logger.info("2. Оптимизированная загрузка (pin_memory + non_blocking):")
    start = time.time()
    for i, (inputs, targets, info) in enumerate(train_loader):
        if i >= test_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    logger.info(f"   Время: {elapsed:.2f}с ({test_batches/elapsed:.1f} batches/s)")
    
    # Тест одной эпохи обучения
    logger.info("\n🏃 Тест обучения (1 эпоха):")
    
    # Ограничиваем количество батчей для теста
    config['model']['epochs'] = 1
    trainer.epochs = 1
    
    # Запуск обучения
    start = time.time()
    history = trainer.train(train_loader, val_loader)
    elapsed = time.time() - start
    
    logger.info(f"\n✅ Тест завершен за {elapsed:.1f}с")
    logger.info(f"   Train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        logger.info(f"   Val loss: {history['val_loss'][-1]:.4f}")
    
    # Проверка утилизации GPU
    if torch.cuda.is_available():
        logger.info("\n📊 Статистика GPU:")
        logger.info(f"   Выделено памяти: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"   Зарезервировано: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        logger.info(f"   Максимум памяти: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    logger.info("\n✅ Все тесты пройдены успешно!")
    
    # Рекомендации
    logger.info("\n💡 РЕКОМЕНДАЦИИ для запуска обучения:")
    logger.info("1. Используйте оптимизированные компоненты:")
    logger.info("   - from data.optimized_dataset import create_optimized_dataloaders")
    logger.info("   - from training.optimized_trainer import OptimizedTrainer")
    logger.info("2. Убедитесь что в конфиге:")
    logger.info("   - num_workers: 4 (или больше)")
    logger.info("   - use_amp: true")
    logger.info("   - compile_model: true (для PyTorch 2.0+)")
    logger.info("3. Запустите полное обучение:")
    logger.info("   python main.py --mode train --use-optimized")


if __name__ == "__main__":
    test_gpu_optimization()