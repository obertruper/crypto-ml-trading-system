#!/usr/bin/env python3
"""
Анализ распределения классов в данных для понимания дисбаланса
"""

import torch
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from data.precomputed_dataset import create_precomputed_data_loaders
from utils.logger import get_logger

logger = get_logger("ClassDistributionAnalysis")

def analyze_direction_distribution(dataloader, dataset_name="Dataset"):
    """
    Анализирует распределение классов direction в датасете
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Анализ распределения классов для {dataset_name}")
    logger.info(f"{'='*60}")
    
    all_directions = {
        'direction_15m': [],
        'direction_1h': [],
        'direction_4h': [],
        'direction_12h': []
    }
    
    all_returns = {
        'future_return_15m': [],
        'future_return_1h': [],
        'future_return_4h': [],
        'future_return_12h': []
    }
    
    total_samples = 0
    
    # Собираем данные
    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        if batch_idx % 10 == 0:
            logger.info(f"Обработка батча {batch_idx}...")
        
        # Приводим targets к правильной размерности
        if targets.dim() == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        batch_size = targets.shape[0]
        total_samples += batch_size
        
        # Извлекаем direction классы (индексы 4-7)
        for i, timeframe in enumerate(['15m', '1h', '4h', '12h']):
            direction_key = f'direction_{timeframe}'
            directions = targets[:, 4 + i].cpu().numpy()
            all_directions[direction_key].extend(directions)
            
            # Также собираем returns для анализа
            return_key = f'future_return_{timeframe}'
            returns = targets[:, i].cpu().numpy()
            all_returns[return_key].extend(returns)
    
    logger.info(f"\n✅ Всего проанализировано: {total_samples} примеров")
    
    # Анализ распределения классов
    results = {}
    
    for timeframe in ['15m', '1h', '4h', '12h']:
        direction_key = f'direction_{timeframe}'
        directions = np.array(all_directions[direction_key])
        
        # Подсчет классов
        class_counts = Counter(directions.astype(int))
        total = len(directions)
        
        logger.info(f"\n📈 Распределение классов для {timeframe}:")
        logger.info(f"   LONG (0):  {class_counts.get(0, 0):6d} ({class_counts.get(0, 0)/total*100:5.1f}%)")
        logger.info(f"   SHORT (1): {class_counts.get(1, 0):6d} ({class_counts.get(1, 0)/total*100:5.1f}%)")
        logger.info(f"   FLAT (2):  {class_counts.get(2, 0):6d} ({class_counts.get(2, 0)/total*100:5.1f}%)")
        
        # Расчет весов классов для балансировки
        class_weights = {}
        for cls in [0, 1, 2]:
            if class_counts.get(cls, 0) > 0:
                # Используем sqrt для более умеренной коррекции
                weight = np.sqrt(total / (3.0 * class_counts.get(cls, 1)))
                class_weights[cls] = weight
            else:
                class_weights[cls] = 1.0
        
        # Нормализация весов
        mean_weight = np.mean(list(class_weights.values()))
        normalized_weights = {k: v/mean_weight for k, v in class_weights.items()}
        
        logger.info(f"   Рекомендуемые веса классов:")
        logger.info(f"   LONG:  {normalized_weights[0]:.3f}")
        logger.info(f"   SHORT: {normalized_weights[1]:.3f}")
        logger.info(f"   FLAT:  {normalized_weights[2]:.3f}")
        
        # Анализ returns
        return_key = f'future_return_{timeframe}'
        returns = np.array(all_returns[return_key])
        
        logger.info(f"\n   Статистика returns:")
        logger.info(f"   Mean: {np.mean(returns):.4f}%")
        logger.info(f"   Std:  {np.std(returns):.4f}%")
        logger.info(f"   Min:  {np.min(returns):.4f}%")
        logger.info(f"   Max:  {np.max(returns):.4f}%")
        
        # Сохраняем результаты
        results[timeframe] = {
            'class_distribution': dict(class_counts),
            'class_weights': normalized_weights,
            'total_samples': total,
            'return_stats': {
                'mean': float(np.mean(returns)),
                'std': float(np.std(returns)),
                'min': float(np.min(returns)),
                'max': float(np.max(returns))
            }
        }
    
    return results

def create_distribution_plots(results, save_path="experiments/class_distribution"):
    """
    Создает визуализации распределения классов
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # График распределения классов
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, timeframe in enumerate(['15m', '1h', '4h', '12h']):
        ax = axes[idx]
        
        dist = results[timeframe]['class_distribution']
        classes = ['LONG', 'SHORT', 'FLAT']
        counts = [dist.get(0, 0), dist.get(1, 0), dist.get(2, 0)]
        
        # Барплот
        bars = ax.bar(classes, counts, color=['green', 'red', 'gray'])
        
        # Добавляем проценты
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count/total*100:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_title(f'Распределение классов - {timeframe}')
        ax.set_ylabel('Количество примеров')
        ax.set_ylim(0, max(counts) * 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/class_distribution.png", dpi=300)
    plt.close()
    
    # График рекомендуемых весов
    fig, ax = plt.subplots(figsize=(10, 6))
    
    timeframes = ['15m', '1h', '4h', '12h']
    x = np.arange(len(timeframes))
    width = 0.25
    
    long_weights = [results[tf]['class_weights'][0] for tf in timeframes]
    short_weights = [results[tf]['class_weights'][1] for tf in timeframes]
    flat_weights = [results[tf]['class_weights'][2] for tf in timeframes]
    
    ax.bar(x - width, long_weights, width, label='LONG', color='green', alpha=0.8)
    ax.bar(x, short_weights, width, label='SHORT', color='red', alpha=0.8)
    ax.bar(x + width, flat_weights, width, label='FLAT', color='gray', alpha=0.8)
    
    ax.set_xlabel('Таймфрейм')
    ax.set_ylabel('Рекомендуемый вес класса')
    ax.set_title('Рекомендуемые веса классов для балансировки')
    ax.set_xticks(x)
    ax.set_xticklabels(timeframes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/recommended_weights.png", dpi=300)
    plt.close()
    
    logger.info(f"\n✅ Графики сохранены в {save_path}/")

def generate_recommendations(results):
    """
    Генерирует рекомендации на основе анализа
    """
    logger.info(f"\n{'='*60}")
    logger.info("🎯 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ ОБУЧЕНИЯ")
    logger.info(f"{'='*60}")
    
    # Анализируем общий дисбаланс
    all_distributions = []
    for tf in results.values():
        dist = tf['class_distribution']
        total = sum(dist.values())
        all_distributions.append({
            'LONG': dist.get(0, 0) / total,
            'SHORT': dist.get(1, 0) / total,
            'FLAT': dist.get(2, 0) / total
        })
    
    # Средние значения
    avg_long = np.mean([d['LONG'] for d in all_distributions])
    avg_short = np.mean([d['SHORT'] for d in all_distributions])
    avg_flat = np.mean([d['FLAT'] for d in all_distributions])
    
    logger.info(f"\n📊 Средний дисбаланс классов:")
    logger.info(f"   LONG:  {avg_long*100:.1f}%")
    logger.info(f"   SHORT: {avg_short*100:.1f}%")
    logger.info(f"   FLAT:  {avg_flat*100:.1f}%")
    
    # Рекомендации
    logger.info(f"\n💡 Рекомендации:")
    
    if avg_flat > 0.7:
        logger.info("   1. ⚠️ КРИТИЧЕСКИЙ дисбаланс - FLAT доминирует!")
        logger.info("      - Использовать Focal Loss с gamma=3-5")
        logger.info("      - Применить SMOTE или ADASYN для балансировки")
        logger.info("      - Увеличить порог для определения FLAT (сейчас слишком чувствительный)")
    
    if avg_long < 0.1 or avg_short < 0.1:
        logger.info("   2. ⚠️ Недостаточно примеров LONG или SHORT")
        logger.info("      - Использовать weighted sampling в DataLoader")
        logger.info("      - Увеличить веса редких классов в loss")
        logger.info("      - Рассмотреть data augmentation для minority классов")
    
    # Средние веса для конфига
    avg_weights = {
        'LONG': np.mean([results[tf]['class_weights'][0] for tf in results]),
        'SHORT': np.mean([results[tf]['class_weights'][1] for tf in results]),
        'FLAT': np.mean([results[tf]['class_weights'][2] for tf in results])
    }
    
    logger.info(f"\n📝 Рекомендуемые веса для config.yaml:")
    logger.info(f"   class_weights: [{avg_weights['LONG']:.3f}, {avg_weights['SHORT']:.3f}, {avg_weights['FLAT']:.3f}]")
    
    # Сохраняем рекомендации в файл
    with open("experiments/class_distribution/recommendations.txt", "w") as f:
        f.write(f"Анализ от {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Средний дисбаланс классов:\n")
        f.write(f"LONG:  {avg_long*100:.1f}%\n")
        f.write(f"SHORT: {avg_short*100:.1f}%\n")
        f.write(f"FLAT:  {avg_flat*100:.1f}%\n\n")
        f.write(f"Рекомендуемые веса классов:\n")
        f.write(f"[{avg_weights['LONG']:.3f}, {avg_weights['SHORT']:.3f}, {avg_weights['FLAT']:.3f}]\n")

def main():
    """Основная функция анализа"""
    logger.info("🚀 Запуск анализа распределения классов...")
    
    # Загружаем конфигурацию
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Создаем dataloaders
    logger.info("📥 Загрузка данных...")
    train_loader, val_loader, test_loader = create_precomputed_data_loaders(config)
    
    # Анализируем каждый датасет
    results = {}
    
    logger.info("\n🔍 Анализ обучающего датасета...")
    results['train'] = analyze_direction_distribution(train_loader, "Training Dataset")
    
    logger.info("\n🔍 Анализ валидационного датасета...")
    results['val'] = analyze_direction_distribution(val_loader, "Validation Dataset")
    
    # Создаем визуализации
    create_distribution_plots(results['train'], "experiments/class_distribution/train")
    create_distribution_plots(results['val'], "experiments/class_distribution/val")
    
    # Генерируем рекомендации
    generate_recommendations(results['train'])
    
    logger.info("\n✅ Анализ завершен! Результаты сохранены в experiments/class_distribution/")

if __name__ == "__main__":
    main()