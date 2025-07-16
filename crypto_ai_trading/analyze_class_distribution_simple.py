#!/usr/bin/env python3
"""
Упрощенный анализ распределения классов в данных
"""

import torch
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import h5py

from utils.logger import get_logger

logger = get_logger("ClassDistributionAnalysis")

def analyze_h5_file(file_path, dataset_name="Dataset"):
    """
    Анализирует распределение классов в HDF5 файле
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Анализ {dataset_name}: {file_path}")
    logger.info(f"{'='*60}")
    
    if not Path(file_path).exists():
        logger.error(f"Файл не найден: {file_path}")
        return None
    
    with h5py.File(file_path, 'r') as f:
        # Проверяем структуру
        logger.info(f"Ключи в файле: {list(f.keys())}")
        
        if 'y' not in f:
            logger.error("Ключ 'y' не найден в файле")
            return None
        
        targets = f['y'][:]
        logger.info(f"Размерность targets: {targets.shape}")
        
        # Если targets имеют размерность (N, 1, 20), убираем среднюю размерность
        if targets.ndim == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        total_samples = targets.shape[0]
        logger.info(f"Всего примеров: {total_samples}")
        
        # Анализируем direction классы (индексы 4-7)
        results = {}
        
        for i, timeframe in enumerate(['15m', '1h', '4h', '12h']):
            directions = targets[:, 4 + i].astype(int)
            
            # Подсчет классов
            class_counts = Counter(directions)
            
            logger.info(f"\n📈 Распределение классов для {timeframe}:")
            logger.info(f"   LONG (0):  {class_counts.get(0, 0):6d} ({class_counts.get(0, 0)/total_samples*100:5.1f}%)")
            logger.info(f"   SHORT (1): {class_counts.get(1, 0):6d} ({class_counts.get(1, 0)/total_samples*100:5.1f}%)")
            logger.info(f"   FLAT (2):  {class_counts.get(2, 0):6d} ({class_counts.get(2, 0)/total_samples*100:5.1f}%)")
            
            # Расчет весов классов
            class_weights = {}
            for cls in [0, 1, 2]:
                if class_counts.get(cls, 0) > 0:
                    weight = np.sqrt(total_samples / (3.0 * class_counts.get(cls, 1)))
                    class_weights[cls] = weight
                else:
                    class_weights[cls] = 1.0
            
            # Нормализация весов
            mean_weight = np.mean(list(class_weights.values()))
            normalized_weights = {k: v/mean_weight for k, v in class_weights.items()}
            
            logger.info(f"   Рекомендуемые веса:")
            logger.info(f"   LONG:  {normalized_weights[0]:.3f}")
            logger.info(f"   SHORT: {normalized_weights[1]:.3f}")
            logger.info(f"   FLAT:  {normalized_weights[2]:.3f}")
            
            # Анализ returns
            returns = targets[:, i]
            logger.info(f"\n   Статистика returns:")
            logger.info(f"   Mean: {np.mean(returns):.4f}%")
            logger.info(f"   Std:  {np.std(returns):.4f}%")
            
            results[timeframe] = {
                'class_distribution': dict(class_counts),
                'class_weights': normalized_weights,
                'total_samples': total_samples
            }
    
    return results

def create_summary_plot(all_results, save_path="experiments/class_distribution"):
    """
    Создает сводный график распределения классов
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Подготовка данных для графика
    datasets = list(all_results.keys())
    timeframes = ['15m', '1h', '4h', '12h']
    
    # Создаем график
    fig, axes = plt.subplots(len(datasets), len(timeframes), figsize=(16, 4*len(datasets)))
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    
    for ds_idx, dataset in enumerate(datasets):
        for tf_idx, timeframe in enumerate(timeframes):
            ax = axes[ds_idx, tf_idx]
            
            if all_results[dataset] and timeframe in all_results[dataset]:
                dist = all_results[dataset][timeframe]['class_distribution']
                classes = ['LONG', 'SHORT', 'FLAT']
                counts = [dist.get(0, 0), dist.get(1, 0), dist.get(2, 0)]
                total = sum(counts)
                
                # Барплот
                bars = ax.bar(classes, counts, color=['green', 'red', 'gray'])
                
                # Добавляем проценты
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    if total > 0:
                        ax.annotate(f'{count/total*100:.1f}%',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom')
                
                ax.set_title(f'{dataset} - {timeframe}')
                ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/class_distribution_summary.png", dpi=300)
    plt.close()
    
    logger.info(f"✅ График сохранен: {save_path}/class_distribution_summary.png")

def generate_final_recommendations(all_results):
    """
    Генерирует финальные рекомендации
    """
    logger.info(f"\n{'='*60}")
    logger.info("🎯 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ")
    logger.info(f"{'='*60}")
    
    # Собираем общую статистику
    all_weights = []
    all_distributions = []
    
    for dataset, results in all_results.items():
        if results:
            for tf, data in results.items():
                dist = data['class_distribution']
                total = sum(dist.values())
                if total > 0:
                    all_distributions.append({
                        'LONG': dist.get(0, 0) / total,
                        'SHORT': dist.get(1, 0) / total,
                        'FLAT': dist.get(2, 0) / total
                    })
                    all_weights.append(data['class_weights'])
    
    # Средние значения
    if all_distributions:
        avg_long = np.mean([d['LONG'] for d in all_distributions])
        avg_short = np.mean([d['SHORT'] for d in all_distributions])
        avg_flat = np.mean([d['FLAT'] for d in all_distributions])
        
        logger.info(f"\n📊 Общий дисбаланс классов:")
        logger.info(f"   LONG:  {avg_long*100:.1f}%")
        logger.info(f"   SHORT: {avg_short*100:.1f}%")
        logger.info(f"   FLAT:  {avg_flat*100:.1f}%")
        
        # Средние веса
        if all_weights:
            avg_weights = {
                0: np.mean([w[0] for w in all_weights]),
                1: np.mean([w[1] for w in all_weights]),
                2: np.mean([w[2] for w in all_weights])
            }
            
            logger.info(f"\n📝 Рекомендуемые веса классов для loss функции:")
            logger.info(f"   class_weights = [{avg_weights[0]:.3f}, {avg_weights[1]:.3f}, {avg_weights[2]:.3f}]")
            
            # Сохраняем в файл
            save_path = Path("experiments/class_distribution")
            save_path.mkdir(parents=True, exist_ok=True)
            
            with open(save_path / "recommended_weights.txt", "w") as f:
                f.write(f"Анализ от {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Дисбаланс классов:\n")
                f.write(f"LONG:  {avg_long*100:.1f}%\n")
                f.write(f"SHORT: {avg_short*100:.1f}%\n")
                f.write(f"FLAT:  {avg_flat*100:.1f}%\n\n")
                f.write(f"Рекомендуемые веса:\n")
                f.write(f"class_weights = [{avg_weights[0]:.3f}, {avg_weights[1]:.3f}, {avg_weights[2]:.3f}]\n")
            
            logger.info(f"\n✅ Рекомендации сохранены в {save_path}/recommended_weights.txt")

def main():
    """Основная функция анализа"""
    logger.info("🚀 Запуск анализа распределения классов...")
    
    # Пути к предвычисленным данным
    cache_dir = Path("cache/precomputed")
    
    # Анализируем доступные файлы
    h5_files = {
        'train': cache_dir / "train_w168_s1.h5",
        'val': cache_dir / "val_w168_s4.h5",
        'test': cache_dir / "test_w168_s4.h5"
    }
    
    all_results = {}
    
    for dataset_name, file_path in h5_files.items():
        if file_path.exists():
            results = analyze_h5_file(file_path, dataset_name.upper())
            all_results[dataset_name.upper()] = results
        else:
            logger.warning(f"⚠️ Файл не найден: {file_path}")
    
    if all_results:
        # Создаем визуализации
        create_summary_plot(all_results)
        
        # Генерируем рекомендации
        generate_final_recommendations(all_results)
    else:
        logger.error("❌ Не найдено ни одного файла для анализа!")
        logger.info("💡 Сначала запустите подготовку данных: python main.py --mode data")

if __name__ == "__main__":
    main()