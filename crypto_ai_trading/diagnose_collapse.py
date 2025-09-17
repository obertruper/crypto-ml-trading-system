#!/usr/bin/env python3
"""
Диагностика проблемы схлопывания модели в LONG класс
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from omegaconf import OmegaConf
import sys

from utils.logger import get_logger
from data.precomputed_dataset import PrecomputedDataset
from data.dataset import create_unified_data_loaders
from models.patchtst_unified import UnifiedPatchTST

logger = get_logger("DiagnoseCollapse")

def check_data_distribution(data_loader, num_batches=10):
    """Проверка распределения классов в данных"""
    logger.info("📊 Анализ распределения классов в данных...")

    all_targets = []
    for i, (X, y, info) in enumerate(data_loader):
        if i >= num_batches:
            break
        # Извлекаем direction targets (индексы 4-7 для direction_15m/1h/4h/12h)
        direction_targets = y[:, 4:8]  # direction_15m, direction_1h, direction_4h, direction_12h
        all_targets.append(direction_targets)

    all_targets = torch.cat(all_targets, dim=0)

    # Анализ для каждого таймфрейма
    timeframes = ['15m', '1h', '4h', '12h']
    for i, tf in enumerate(timeframes):
        targets = all_targets[:, i]
        unique, counts = torch.unique(targets, return_counts=True)
        total = counts.sum().item()

        logger.info(f"\n📈 Direction {tf}:")
        class_names = ['LONG', 'SHORT', 'FLAT']
        for cls_idx in range(3):
            if cls_idx in unique:
                idx = (unique == cls_idx).nonzero(as_tuple=True)[0]
                count = counts[idx].item()
                pct = count / total * 100
                logger.info(f"   {class_names[cls_idx]}: {count} ({pct:.1f}%)")
            else:
                logger.info(f"   {class_names[cls_idx]}: 0 (0.0%)")

    return all_targets

def check_model_bias(config):
    """Проверка настроек bias в модели"""
    logger.info("\n🔍 Проверка настроек direction_bias в модели...")

    # Создаем модель
    model_config = config['model']
    model = UnifiedPatchTST(model_config)

    # Проверяем direction_bias
    if hasattr(model, 'direction_bias'):
        bias = model.direction_bias
        logger.info(f"✅ direction_bias в модели: {bias.data.tolist()}")
    else:
        logger.warning("⚠️ direction_bias не найден в модели!")

    # Проверяем веса direction head
    if hasattr(model, 'direction_head'):
        for name, param in model.direction_head.named_parameters():
            if 'bias' in name and param.shape[0] == 12:  # 4 таймфрейма * 3 класса
                bias_values = param.data.view(4, 3)  # Reshape к [4 timeframes, 3 classes]
                logger.info(f"\n📊 Bias в direction head ({name}):")
                timeframes = ['15m', '1h', '4h', '12h']
                for i, tf in enumerate(timeframes):
                    logger.info(f"   {tf}: LONG={bias_values[i, 0]:.3f}, SHORT={bias_values[i, 1]:.3f}, FLAT={bias_values[i, 2]:.3f}")

    return model

def simulate_predictions(model, data_loader, device='cuda'):
    """Симуляция предсказаний модели"""
    logger.info("\n🔮 Симуляция предсказаний модели...")

    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, (X, y, info) in enumerate(data_loader):
            if i >= 5:  # Только несколько батчей для теста
                break

            X = X.to(device)
            outputs = model(X)

            # Извлекаем direction logits
            if hasattr(outputs, '_direction_logits'):
                direction_logits = outputs._direction_logits  # [batch, 4, 3]

                # Применяем softmax и получаем предсказания
                probs = torch.softmax(direction_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)  # [batch, 4]

                predictions.append(preds.cpu())

    if predictions:
        predictions = torch.cat(predictions, dim=0)

        # Анализ предсказаний
        timeframes = ['15m', '1h', '4h', '12h']
        for i, tf in enumerate(timeframes):
            preds_tf = predictions[:, i]
            unique, counts = torch.unique(preds_tf, return_counts=True)
            total = counts.sum().item()

            logger.info(f"\n🎯 Предсказания direction {tf}:")
            class_names = ['LONG', 'SHORT', 'FLAT']
            for cls_idx in range(3):
                if cls_idx in unique:
                    idx = (unique == cls_idx).nonzero(as_tuple=True)[0]
                    count = counts[idx].item()
                    pct = count / total * 100
                    logger.info(f"   {class_names[cls_idx]}: {count} ({pct:.1f}%)")
                else:
                    logger.info(f"   {class_names[cls_idx]}: 0 (0.0%)")

def analyze_loss_components(config):
    """Анализ компонентов loss функции"""
    logger.info("\n📉 Анализ настроек loss функции...")

    loss_config = config.get('loss', {})

    # Проверяем class weights
    class_weights = loss_config.get('class_weights', [1.0, 1.0, 1.0])
    logger.info(f"📊 Class weights: LONG={class_weights[0]}, SHORT={class_weights[1]}, FLAT={class_weights[2]}")

    # Проверяем другие параметры
    logger.info(f"🔄 Label smoothing: {config['model'].get('label_smoothing', 0.0)}")
    logger.info(f"🌡️ Temperature: {config['model'].get('temperature', 1.0)}")
    logger.info(f"📈 Gradient clip: {config['model'].get('gradient_clip', 1.0)}")

    # Проверяем staged training параметры
    if 'staged_training' in config and config['staged_training'].get('enabled'):
        logger.info("\n🎯 Staged training параметры:")
        stages = config['staged_training'].get('stages', [])
        for i, stage in enumerate(stages):
            logger.info(f"\n   Этап {i+1} ({stage['name']}):")
            if 'direction_bias' in stage:
                bias = stage['direction_bias']
                logger.info(f"      ✅ direction_bias: LONG={bias[0]}, SHORT={bias[1]}, FLAT={bias[2]}")
            else:
                logger.warning(f"      ⚠️ direction_bias НЕ УКАЗАН!")
            logger.info(f"      gradient_clip: {stage.get('gradient_clip', 'не указан')}")
            if 'class_weights' in stage:
                weights = stage['class_weights']
                logger.info(f"      class_weights: LONG={weights[0]}, SHORT={weights[1]}, FLAT={weights[2]}")

def main():
    """Основная функция диагностики"""
    logger.info("=" * 80)
    logger.info("🔍 ДИАГНОСТИКА СХЛОПЫВАНИЯ МОДЕЛИ В LONG КЛАСС")
    logger.info("=" * 80)

    # Загружаем конфигурацию
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Анализ конфигурации
    analyze_loss_components(config)

    # 2. Проверка модели
    model = check_model_bias(config)

    # 3. Загрузка данных (небольшая выборка)
    logger.info("\n📦 Загрузка данных для анализа...")
    from data.data_loader import CryptoDataLoader
    from data.feature_engineering import FeatureEngineer

    # Загружаем только несколько символов для теста
    config['data']['symbols'] = ['BTCUSDT', 'ETHUSDT']
    config['data']['max_symbols'] = 2

    data_loader = CryptoDataLoader(config)
    symbols_to_load = config['data']['symbols']

    raw_data = data_loader.load_data(
        symbols=symbols_to_load,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )

    feature_engineer = FeatureEngineer(config)
    train_data, val_data, test_data = feature_engineer.create_features_with_train_split(
        raw_data,
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Создаем data loaders
    from data.constants import get_feature_columns, get_target_columns, validate_data_structure

    data_info = validate_data_structure(train_data)
    feature_cols = data_info['feature_cols']
    target_cols = data_info['target_cols']

    # Уменьшаем batch size для теста
    config['model']['batch_size'] = 512

    train_loader, val_loader, _, _ = create_unified_data_loaders(
        train_data, val_data, test_data, feature_cols, target_cols, config, logger
    )

    # 4. Анализ распределения данных
    check_data_distribution(train_loader)

    # 5. Симуляция предсказаний
    if torch.cuda.is_available():
        simulate_predictions(model, val_loader)

    logger.info("\n" + "=" * 80)
    logger.info("📊 ВЫВОДЫ:")
    logger.info("=" * 80)
    logger.info("""
1. Если модель схлопывается в LONG (99.5%), возможные причины:
   - direction_bias инвертирован (положительный для LONG вместо отрицательного)
   - Слишком высокий gradient (12.9) разрушает веса
   - Данные имеют сильный дисбаланс в сторону LONG

2. Решения:
   - Проверить знаки direction_bias (должны быть отрицательные для LONG/SHORT)
   - Снизить gradient_clip до 0.1
   - Использовать более агрессивные class_weights для балансировки
   - Проверить качество данных на наличие утечек
""")

if __name__ == "__main__":
    main()