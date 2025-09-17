#!/usr/bin/env python
"""
Запуск обучения с улучшенной архитектурой и защитой от коллапса
"""

import torch
import yaml
import sys
from pathlib import Path
import logging
from datetime import datetime

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).parent))

from models.patchtst_unified import UnifiedPatchTSTForTrading
from data.precomputed_dataset import PrecomputedDataset
from training.optimized_trainer import OptimizedTrainer
from utils.logger import setup_logging, get_logger

def main():
    """Главная функция запуска обучения"""

    # Настройка логирования
    setup_logging()
    logger = get_logger("EnhancedTraining")

    logger.info("="*60)
    logger.info("🚀 ЗАПУСК ОБУЧЕНИЯ С УЛУЧШЕННОЙ АРХИТЕКТУРОЙ")
    logger.info("="*60)

    # Загружаем конфигурацию
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Устанавливаем device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"📱 Device: {device}")

    if device.type == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Создаем датасеты
    logger.info("\n📊 Загрузка данных...")

    # Исправление путей для совместимости с конфигом
    cache_dir = Path(config.get('cache_dir', 'cache')) / 'precomputed'
    models_dir = Path(config.get('models_dir', 'models_saved'))
    logs_dir = Path(config.get('logs_dir', 'logs'))

    train_dataset = PrecomputedDataset(
        cache_dir=cache_dir,
        mode='train',
        window_size=config['model'].get('context_window', 96),
        step_size=config['data'].get('train_stride', 1)
    )

    val_dataset = PrecomputedDataset(
        cache_dir=cache_dir,
        mode='val',
        window_size=config['model'].get('context_window', 96),
        step_size=config['data'].get('val_stride', 1)
    )

    logger.info(f"   Train samples: {len(train_dataset):,}")
    logger.info(f"   Val samples: {len(val_dataset):,}")

    # Создаем DataLoader'ы
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=False,  # Отключаем из-за custom collate
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
        persistent_workers=True
    )

    # Создаем модель с улучшенной архитектурой
    logger.info("\n🏗️ Создание модели с улучшенной архитектурой...")

    # Проверяем что в конфиге есть поддержка новой архитектуры
    if 'use_separate_direction_heads' not in config['model']:
        config['model']['use_separate_direction_heads'] = True
        logger.info("   ✅ Включены отдельные direction heads для каждого таймфрейма")

    model = UnifiedPatchTSTForTrading(
        num_features=261,  # 171 индикаторов + 90 дополнительных фич
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        e_layers=config['model']['e_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        patch_len=config['model']['patch_len'],
        stride=config['model']['stride']
    ).to(device)

    # Проверяем наличие новых direction heads
    if hasattr(model, 'direction_heads'):
        logger.info(f"   ✅ Создано {len(model.direction_heads)} отдельных direction heads")
        logger.info("   🛡️ Архитектура защищена от коллапса")
    else:
        logger.warning("   ⚠️ Используется старая архитектура")

    # Инициализация весов для предотвращения коллапса
    logger.info("\n🎲 Инициализация весов...")

    # Специальная инициализация для direction heads
    if hasattr(model, 'direction_heads'):
        for idx, head in enumerate(model.direction_heads):
            # Инициализация последнего слоя с большей дисперсией
            for layer in head:
                if isinstance(layer, torch.nn.Linear):
                    # Xavier с увеличенным gain для активации
                    torch.nn.init.xavier_uniform_(layer.weight, gain=2.0)
                    if layer.bias is not None:
                        # Смещение против FLAT класса
                        if layer.out_features == 3:
                            layer.bias.data = torch.tensor([0.2, 0.2, -0.4])
                        else:
                            layer.bias.data.zero_()

    logger.info("   ✅ Веса инициализированы против коллапса")

    # Создаем тренера
    logger.info("\n🎯 Создание тренера...")

    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Проверяем параметры защиты от коллапса
    logger.info("\n🛡️ Параметры защиты от коллапса:")
    logger.info(f"   Class weights: {config['loss'].get('class_weights', 'default')}")
    logger.info(f"   Entropy regularization: {config['loss'].get('entropy_regularization', 0.0)}")
    logger.info(f"   Temperature: {config['loss'].get('temperature', 1.0)}")
    logger.info(f"   Label smoothing: {config['loss'].get('label_smoothing', 0.0)}")
    logger.info(f"   Focal loss gamma: {config['loss'].get('focal_gamma', 0.0)}")

    # Запуск обучения
    logger.info("\n" + "="*60)
    logger.info("🚀 НАЧАЛО ОБУЧЕНИЯ")
    logger.info("="*60)

    try:
        # Обучение
        trainer.train(num_epochs=config['training']['num_epochs'])

        logger.info("\n" + "="*60)
        logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.info("\n⚠️ Обучение прервано пользователем")

    except Exception as e:
        logger.error(f"\n❌ Ошибка при обучении: {e}")
        raise

    finally:
        # Сохраняем финальную модель
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_path = models_dir / f"enhanced_final_{timestamp}.pth"

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'architecture': 'enhanced_with_separate_heads'
        }, final_path)

        logger.info(f"\n💾 Финальная модель сохранена: {final_path}")

if __name__ == "__main__":
    main()