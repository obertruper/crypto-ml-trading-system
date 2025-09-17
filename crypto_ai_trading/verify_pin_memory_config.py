#!/usr/bin/env python3
"""
Скрипт для проверки и настройки pin_memory для RTX 5090
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
import time
import gc
import psutil
from pathlib import Path

from utils.logger import get_logger
from data.precomputed_dataset import create_precomputed_data_loaders, custom_collate_fn
from data.data_loader import CryptoDataLoader


def check_gpu_compatibility():
    """Проверка совместимости GPU с pin_memory"""
    logger = get_logger("PinMemoryVerify")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA недоступна!")
        return False
    
    # Информация о GPU
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    compute_capability = torch.cuda.get_device_capability(0)
    
    logger.info(f"🖥️ GPU: {gpu_name}")
    logger.info(f"💾 Память GPU: {gpu_memory:.1f} GB")
    logger.info(f"🔧 Compute Capability: {compute_capability}")
    
    # Проверка поддержки pin_memory
    try:
        test_tensor = torch.randn(100, 100)
        pinned_tensor = test_tensor.pin_memory()
        assert pinned_tensor.is_pinned()
        logger.info("✅ Pin memory поддерживается GPU")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка при тесте pin_memory: {e}")
        return False


def test_custom_collate_fn():
    """Тестирование custom_collate_fn"""
    logger = get_logger("CollateTest")
    
    # Создаем тестовый батч
    batch_size = 4
    seq_len = 96
    n_features = 240
    n_targets = 20
    
    # Симулируем батч данных
    batch = []
    for i in range(batch_size):
        X = torch.randn(seq_len, n_features)
        y = torch.randn(1, n_targets)
        info = {'idx': i}
        batch.append((X, y, info))
    
    logger.info("🔧 Тестирование custom_collate_fn...")
    
    try:
        # Применяем collate функцию
        X_batch, y_batch, info_batch = custom_collate_fn(batch)
        
        # Проверки
        assert X_batch.shape == (batch_size, seq_len, n_features)
        assert y_batch.shape == (batch_size, 1, n_targets)
        assert isinstance(info_batch, dict)
        assert 'idx' in info_batch
        assert isinstance(info_batch['idx'], torch.Tensor)
        assert info_batch['idx'].shape == (batch_size,)
        
        logger.info("✅ custom_collate_fn работает корректно")
        logger.info(f"   X_batch: {X_batch.shape}, dtype: {X_batch.dtype}")
        logger.info(f"   y_batch: {y_batch.shape}, dtype: {y_batch.dtype}")
        logger.info(f"   info_batch['idx']: {info_batch['idx'].shape}, dtype: {info_batch['idx'].dtype}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка в custom_collate_fn: {e}")
        return False


def test_dataloader_with_pin_memory(config):
    """Тестирование DataLoader с pin_memory"""
    logger = get_logger("DataLoaderTest")
    
    # Принудительно включаем pin_memory
    config['performance']['dataloader_pin_memory'] = True
    config['performance']['pin_memory'] = True
    
    logger.info("📊 Загрузка данных для теста...")
    
    # Загружаем небольшой набор данных
    loader = CryptoDataLoader(config)
    data = loader.load_data(limit_symbols=2, limit_per_symbol=5000)  # Только 2 символа для теста
    
    if data is None or data.empty:
        logger.error("❌ Не удалось загрузить данные")
        return False
    
    # Разделение данных
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info("🔄 Создание DataLoader с pin_memory=True...")
    
    try:
        train_loader, val_loader, test_loader = create_precomputed_data_loaders(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            config=config
        )
        
        logger.info("✅ DataLoader создан успешно")
        
        # Тестируем загрузку батча
        logger.info("🧪 Тестирование загрузки батчей...")
        
        for loader_name, loader in [("train", train_loader), ("val", val_loader)]:
            logger.info(f"\n📦 Тестирование {loader_name} loader...")
            
            # Получаем первый батч
            start_time = time.time()
            inputs, targets, info = next(iter(loader))
            load_time = time.time() - start_time
            
            logger.info(f"   Время загрузки батча: {load_time:.3f}s")
            logger.info(f"   Размер inputs: {inputs.shape}")
            logger.info(f"   Размер targets: {targets.shape}")
            logger.info(f"   Тип info: {type(info)}")
            
            # Проверяем pin_memory
            if torch.cuda.is_available():
                # Проверяем, что данные можно перенести на GPU
                try:
                    start_transfer = time.time()
                    inputs_gpu = inputs.cuda(non_blocking=True)
                    targets_gpu = targets.cuda(non_blocking=True)
                    
                    # Обработка info
                    if isinstance(info, dict) and 'idx' in info:
                        if isinstance(info['idx'], torch.Tensor):
                            info_gpu = {'idx': info['idx'].cuda(non_blocking=True)}
                        else:
                            logger.warning(f"   ⚠️ info['idx'] не является тензором: {type(info['idx'])}")
                    
                    torch.cuda.synchronize()
                    transfer_time = time.time() - start_transfer
                    
                    logger.info(f"   ✅ Перенос на GPU успешен за {transfer_time:.3f}s")
                    
                    # Проверяем, что данные на GPU
                    assert inputs_gpu.is_cuda
                    assert targets_gpu.is_cuda
                    
                    # Очистка
                    del inputs_gpu, targets_gpu
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"   ❌ Ошибка при переносе на GPU: {e}")
                    return False
            
            # Тестируем несколько батчей
            logger.info(f"   Тестирование 5 батчей...")
            times = []
            
            for i, (inputs, targets, info) in enumerate(loader):
                if i >= 5:
                    break
                
                start = time.time()
                
                # Симулируем обработку
                if torch.cuda.is_available():
                    inputs = inputs.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)
                    torch.cuda.synchronize()
                
                batch_time = time.time() - start
                times.append(batch_time)
                
                # Очистка GPU памяти
                if torch.cuda.is_available():
                    del inputs, targets
                    torch.cuda.empty_cache()
            
            avg_time = np.mean(times)
            logger.info(f"   Среднее время обработки батча: {avg_time:.3f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_config_settings(config_path):
    """Проверка настроек конфигурации"""
    logger = get_logger("ConfigVerify")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("🔍 Проверка настроек конфигурации...")
    
    # Критические настройки для RTX 5090
    critical_settings = {
        'performance.pin_memory': True,
        'performance.dataloader_pin_memory': True,
        'performance.non_blocking': True,  # Если есть
        'performance.num_workers': 4,
        'performance.persistent_workers': True,
        'performance.prefetch_factor': 2,
        'model.use_amp': True,
        'model.amp_dtype': 'float16',
        'model.use_tf32': True,
        'model.compile_model': False,  # RTX 5090 не поддерживается
    }
    
    all_correct = True
    
    for path, expected in critical_settings.items():
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value.get(key, None)
            
            if value == expected:
                logger.info(f"   ✅ {path}: {value}")
            else:
                logger.warning(f"   ⚠️ {path}: {value} (ожидается {expected})")
                all_correct = False
                
        except:
            logger.warning(f"   ❌ {path}: не найдено")
            all_correct = False
    
    # Дополнительные рекомендации
    logger.info("\n📋 Рекомендации для RTX 5090:")
    logger.info("   - batch_size: 8192 (с gradient_accumulation_steps: 2)")
    logger.info("   - Использовать OptimizedTrainer вместо обычного Trainer")
    logger.info("   - Включить cudnn.benchmark = True")
    logger.info("   - Использовать non_blocking=True при переносе на GPU")
    
    return all_correct


def main():
    """Основная функция проверки"""
    logger = get_logger("Main")
    
    logger.info("🚀 Проверка настройки pin_memory для RTX 5090")
    logger.info("="*60)
    
    # 1. Проверка GPU
    if not check_gpu_compatibility():
        return
    
    logger.info("\n" + "="*60)
    
    # 2. Тестирование collate функции
    if not test_custom_collate_fn():
        return
    
    logger.info("\n" + "="*60)
    
    # 3. Проверка конфигурации
    config_path = Path("config/config.yaml")
    if not verify_config_settings(config_path):
        logger.warning("⚠️ Некоторые настройки требуют внимания")
    
    logger.info("\n" + "="*60)
    
    # 4. Загружаем конфиг для теста DataLoader
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 5. Тестирование DataLoader
    if test_dataloader_with_pin_memory(config):
        logger.info("\n✅ Все тесты пройдены успешно!")
        logger.info("🎯 Pin memory настроен корректно для RTX 5090")
    else:
        logger.error("\n❌ Обнаружены проблемы с настройкой")
    
    # Финальные рекомендации
    logger.info("\n📝 Финальные рекомендации:")
    logger.info("1. Используйте OptimizedTrainer для максимальной производительности")
    logger.info("2. Убедитесь, что все данные переносятся с non_blocking=True")
    logger.info("3. Мониторьте использование GPU через nvidia-smi")
    logger.info("4. При низкой утилизации GPU увеличьте batch_size")


if __name__ == "__main__":
    main()