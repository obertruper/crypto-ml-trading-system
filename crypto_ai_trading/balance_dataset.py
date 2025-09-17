"""
Скрипт для балансировки датасета через undersampling FLAT класса
"""

import pandas as pd
import numpy as np
from pathlib import Path
import h5py
import logging
from sklearn.utils import resample
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('DataBalancer')

def balance_dataset(df, target_column='direction_15m', strategy='undersample'):
    """
    Балансирует датасет по целевой колонке

    Args:
        df: DataFrame с данными
        target_column: колонка для балансировки
        strategy: 'undersample' или 'oversample'
    """
    logger.info(f"📊 Начальное распределение {target_column}:")

    # Статистика до балансировки
    value_counts = df[target_column].value_counts()
    for val, count in value_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"   Класс {val}: {count:,} ({pct:.1f}%)")

    if strategy == 'undersample':
        # Находим минимальный класс
        min_class_size = value_counts.min()
        logger.info(f"🎯 Undersampling до размера: {min_class_size:,}")

        balanced_dfs = []
        for class_val in value_counts.index:
            class_df = df[df[target_column] == class_val]

            if len(class_df) > min_class_size:
                # Undersample
                class_df_balanced = resample(
                    class_df,
                    n_samples=min_class_size,
                    random_state=42,
                    replace=False
                )
                logger.info(f"   Класс {class_val}: {len(class_df)} → {min_class_size}")
            else:
                class_df_balanced = class_df

            balanced_dfs.append(class_df_balanced)

        # Объединяем и перемешиваем
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    elif strategy == 'mixed':
        # Смешанная стратегия: частичный undersample FLAT, oversample LONG/SHORT
        flat_df = df[df[target_column] == 2]
        long_df = df[df[target_column] == 0]
        short_df = df[df[target_column] == 1]

        target_size = int(len(df) / 4)  # Целевой размер для каждого класса

        # Undersample FLAT до целевого размера
        flat_balanced = resample(flat_df, n_samples=target_size, random_state=42, replace=False)

        # Oversample LONG и SHORT
        long_balanced = resample(long_df, n_samples=target_size, random_state=42, replace=True)
        short_balanced = resample(short_df, n_samples=target_size, random_state=42, replace=True)

        balanced_df = pd.concat([flat_balanced, long_balanced, short_balanced], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"🎯 Mixed strategy: каждый класс → {target_size:,}")

    # Статистика после балансировки
    logger.info(f"✅ Распределение после балансировки:")
    value_counts_after = balanced_df[target_column].value_counts()
    for val, count in value_counts_after.items():
        pct = (count / len(balanced_df)) * 100
        logger.info(f"   Класс {val}: {count:,} ({pct:.1f}%)")

    logger.info(f"📉 Размер датасета: {len(df):,} → {len(balanced_df):,}")

    return balanced_df

def process_cached_data():
    """Обрабатывает кэшированные данные"""

    cache_dir = Path("cache")

    # Обработка train данных
    train_cache = cache_dir / "train_data.pkl"
    if train_cache.exists():
        logger.info("📥 Загрузка train данных...")
        train_df = pd.read_pickle(train_cache)

        # Балансировка для всех direction колонок
        direction_cols = ['direction_15m', 'direction_1h', 'direction_4h', 'direction_12h']

        for col in direction_cols:
            if col in train_df.columns:
                logger.info(f"\n🔄 Балансировка {col}:")
                train_df = balance_dataset(train_df, col, strategy='mixed')

        # Сохранение
        balanced_cache = cache_dir / "train_data_balanced.pkl"
        train_df.to_pickle(balanced_cache)
        logger.info(f"💾 Сохранено: {balanced_cache}")

        return train_df
    else:
        logger.error("❌ Train cache не найден!")
        return None

def update_precomputed_windows():
    """Обновляет предвычисленные окна с балансировкой"""

    precomputed_dir = Path("cache/precomputed")

    for h5_file in precomputed_dir.glob("train_*.h5"):
        logger.info(f"📦 Обработка {h5_file.name}")

        with h5py.File(h5_file, 'r') as f:
            features = f['X'][:]  # Правильное имя ключа
            targets = f['y'][:]   # Правильное имя ключа

        # Извлекаем direction колонки (индексы 4-7)
        direction_15m = targets[:, 0, 4]  # shape: (N, 1, 20) -> направление на индексе 4

        # Статистика
        unique, counts = np.unique(direction_15m, return_counts=True)
        logger.info(f"   Распределение: {dict(zip(unique, counts))}")

        # Балансировка через индексы
        min_count = counts.min()
        balanced_indices = []

        for class_val in unique:
            class_indices = np.where(direction_15m == class_val)[0]
            if len(class_indices) > min_count:
                # Random undersample
                selected = np.random.choice(class_indices, min_count, replace=False)
            else:
                selected = class_indices
            balanced_indices.extend(selected)

        # Перемешиваем
        np.random.shuffle(balanced_indices)

        # Применяем балансировку
        features_balanced = features[balanced_indices]
        targets_balanced = targets[balanced_indices]

        # Сохраняем с правильными именами ключей
        balanced_file = h5_file.parent / f"balanced_{h5_file.name}"
        with h5py.File(balanced_file, 'w') as f:
            f.create_dataset('X', data=features_balanced, compression='gzip')
            f.create_dataset('y', data=targets_balanced, compression='gzip')

        logger.info(f"   ✅ Сохранено: {balanced_file.name}")
        logger.info(f"   📉 Размер: {len(features)} → {len(features_balanced)}")

def create_curriculum_config():
    """Создает конфигурацию для curriculum learning"""

    curriculum_config = {
        'curriculum_learning': {
            'enabled': True,
            'stages': [
                {
                    'name': 'Easy Examples',
                    'description': 'Обучение на явных сигналах',
                    'epochs': 10,
                    'data_filter': {
                        'min_return_threshold': 0.02,  # Только сильные движения > 2%
                        'exclude_flat': False,
                        'confidence_threshold': 0.7
                    },
                    'loss_weights': {
                        'directions': 1.5,
                        'returns': 0.5
                    }
                },
                {
                    'name': 'Medium Difficulty',
                    'description': 'Добавление средних сигналов',
                    'epochs': 15,
                    'data_filter': {
                        'min_return_threshold': 0.01,  # Движения > 1%
                        'exclude_flat': False,
                        'confidence_threshold': 0.5
                    },
                    'loss_weights': {
                        'directions': 1.2,
                        'returns': 0.8
                    }
                },
                {
                    'name': 'Full Dataset',
                    'description': 'Обучение на полном датасете',
                    'epochs': 20,
                    'data_filter': {
                        'min_return_threshold': 0.0,
                        'exclude_flat': False,
                        'confidence_threshold': 0.3
                    },
                    'loss_weights': {
                        'directions': 1.0,
                        'returns': 1.0
                    }
                }
            ]
        }
    }

    # Сохраняем конфигурацию
    with open('config/curriculum.yaml', 'w') as f:
        yaml.dump(curriculum_config, f, default_flow_style=False)

    logger.info("📚 Создана конфигурация curriculum learning")

    return curriculum_config

def main():
    """Главная функция"""

    logger.info("="*60)
    logger.info("🎯 Балансировка датасета для решения проблемы коллапса")
    logger.info("="*60)

    # 1. Балансировка кэшированных данных
    balanced_df = process_cached_data()

    # 2. Обновление предвычисленных окон
    update_precomputed_windows()

    # 3. Создание конфигурации curriculum learning
    create_curriculum_config()

    logger.info("="*60)
    logger.info("✅ Балансировка завершена!")
    logger.info("🎯 Следующие шаги:")
    logger.info("   1. Модификация архитектуры модели")
    logger.info("   2. Запуск обучения с balanced данными")
    logger.info("="*60)

if __name__ == "__main__":
    main()