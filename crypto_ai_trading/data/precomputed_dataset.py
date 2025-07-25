"""
PrecomputedDataset для быстрой загрузки предвычисленных временных окон
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import gc
import h5py
from typing import List, Dict, Optional, Tuple
import pandas as pd
import time
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

from utils.logger import get_logger
from data.dataset import TimeSeriesDataset
from torch.utils.data import WeightedRandomSampler


def custom_collate_fn(batch):
    """Кастомная функция для правильной обработки батчей с pin_memory
    
    Решает проблему CUDA error при использовании pin_memory с PyTorch 2.9.0.dev
    """
    # Разделяем батч на компоненты
    X_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    
    # Собираем info словарь безопасным способом
    info_batch = {
        'idx': torch.tensor([item[2]['idx'] for item in batch], dtype=torch.long)
    }
    
    return X_batch, y_batch, info_batch


def calculate_sample_weights(dataset: 'PrecomputedDataset', 
                           direction_indices: List[int] = [4, 5, 6, 7],
                           class_weights: List[float] = [2.5, 2.5, 0.3]) -> torch.Tensor:
    """
    Рассчитывает веса для каждого сэмпла на основе распределения классов direction
    
    Args:
        dataset: PrecomputedDataset
        direction_indices: индексы direction переменных в targets (по умолчанию 4-7)
        class_weights: веса для классов [LONG, SHORT, FLAT]
        
    Returns:
        torch.Tensor с весом для каждого сэмпла
    """
    logger = get_logger("SampleWeights")
    logger.info("📊 Расчет весов сэмплов для балансировки классов...")
    
    # Загружаем все таргеты для анализа
    all_targets = []
    cache_file = dataset._get_cache_path()
    
    if dataset.use_hdf5 and cache_file.exists():
        with h5py.File(cache_file, 'r') as f:
            targets = f['y'][:]  # (n_samples, 1, n_targets)
            if targets.ndim == 3:
                targets = targets.squeeze(1)  # (n_samples, n_targets)
    else:
        # Fallback на обычную загрузку
        for i in range(len(dataset)):
            _, y, _ = dataset[i]
            all_targets.append(y)
        targets = torch.stack(all_targets).numpy()
    
    # Рассчитываем веса для каждого сэмпла
    sample_weights = np.ones(len(targets))
    
    # Анализируем каждый direction индекс
    for idx in direction_indices:
        if idx < targets.shape[1]:
            directions = targets[:, idx].astype(int)
            
            # Подсчет классов
            unique, counts = np.unique(directions, return_counts=True)
            class_dist = {int(cls): cnt for cls, cnt in zip(unique, counts)}
            
            # Применяем веса классов к каждому сэмплу
            for i, direction in enumerate(directions):
                if 0 <= direction <= 2:  # LONG=0, SHORT=1, FLAT=2
                    sample_weights[i] *= class_weights[direction]
            
            # Логирование распределения
            total = len(directions)
            logger.info(f"   Direction idx {idx}: LONG={class_dist.get(0,0)/total:.1%}, "
                       f"SHORT={class_dist.get(1,0)/total:.1%}, FLAT={class_dist.get(2,0)/total:.1%}")
    
    # Нормализуем веса
    sample_weights = sample_weights / sample_weights.mean()
    
    logger.info(f"✅ Веса рассчитаны: min={sample_weights.min():.2f}, "
                f"max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")
    
    return torch.from_numpy(sample_weights).float()


class PrecomputedDataset(Dataset):
    """Dataset с предвычисленными окнами для максимальной скорости загрузки"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 context_window: int = 168,
                 prediction_window: int = 4,
                 feature_cols: List[str] = None,
                 target_cols: List[str] = None,
                 stride: int = 1,
                 cache_dir: str = "cache/precomputed",
                 dataset_name: str = "train",
                 use_hdf5: bool = True,
                 normalize: bool = True,
                 scaler_path: Optional[str] = None,
                 fit_scaler: bool = False):
        """
        Args:
            data: DataFrame с данными
            context_window: размер входного окна
            prediction_window: размер окна предсказания
            feature_cols: список признаков
            target_cols: список целевых переменных
            stride: шаг между окнами
            cache_dir: директория для кэша
            dataset_name: имя датасета (train/val/test)
            use_hdf5: использовать HDF5 для хранения (экономия памяти)
        """
        self.logger = get_logger("PrecomputedDataset")
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.stride = stride
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.use_hdf5 = use_hdf5
        
        # Определение признаков и целевых переменных
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns 
                               if col not in ['id', 'symbol', 'datetime', 'timestamp', 'sector']
                               and not col.startswith(('target_', 'future_', 'optimal_'))]
        else:
            self.feature_cols = feature_cols
            
        if target_cols is None:
            self.target_cols = [col for col in data.columns 
                              if col.startswith(('target_', 'future_return_', 'long_tp', 'short_tp', 
                                               'long_sl', 'short_sl', 'long_optimal', 'short_optimal',
                                               'best_direction'))]
        else:
            self.target_cols = target_cols
        
        # Создаем временный датасет для подготовки данных
        # ВАЖНО: не держим все данные в памяти одновременно
        self.temp_dataset = TimeSeriesDataset(
            data=data,
            context_window=context_window,
            prediction_window=prediction_window,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            stride=stride,
            normalize=normalize,
            scaler_path=scaler_path,
            fit_scaler=fit_scaler
        )
        
        # Освобождаем оригинальные данные из памяти
        del data
        gc.collect()
        
        # Проверяем доступную память перед созданием кэша
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.logger.info(f"🖥️ Доступная память: {available_memory_gb:.2f} GB")
        
        # Проверяем наличие кэша
        cache_file = self._get_cache_path()
        
        if cache_file.exists():
            self.logger.info(f"✅ Загрузка предвычисленных данных из {cache_file}")
            self._load_cache(cache_file)
        else:
            self.logger.info(f"📊 Предвычисление всех окон для {dataset_name}...")
            self._precompute_all_windows()
            self.logger.info(f"💾 Сохранение в кэш: {cache_file}")
            self._save_cache(cache_file)
        
        self.logger.info(f"✅ PrecomputedDataset готов: {len(self)} примеров")
    
    def _get_cache_path(self) -> Path:
        """Получение пути к файлу кэша"""
        cache_name = f"{self.dataset_name}_w{self.context_window}_s{self.stride}"
        if self.use_hdf5:
            return self.cache_dir / f"{cache_name}.h5"
        else:
            return self.cache_dir / f"{cache_name}.pkl"
    
    def _precompute_all_windows(self):
        """Предвычисление всех окон"""
        n_samples = len(self.temp_dataset)
        
        if self.use_hdf5:
            # Используем HDF5 для экономии памяти
            cache_file = self._get_cache_path()
            
            # Получаем размерности из первого примера
            X_sample, y_sample, _ = self.temp_dataset[0]
            X_shape = (n_samples,) + X_sample.shape
            y_shape = (n_samples,) + y_sample.shape
            
            # Оценка размера данных
            memory_estimate_gb = (np.prod(X_shape) + np.prod(y_shape)) * 4 / (1024**3)
            self.logger.info(f"💾 Оценочный размер кэша: {memory_estimate_gb:.2f} GB")
            
            # Улучшенный расчет размера батча
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Размер одного окна в байтах
            window_size_bytes = (np.prod(X_sample.shape) + np.prod(y_sample.shape)) * 4
            
            # Используем только 30% доступной памяти для безопасности
            safe_memory_bytes = available_memory_gb * 1024**3 * 0.3
            
            # Размер батча с учетом параллельной обработки
            n_workers = min(mp.cpu_count() - 1, 8)  # Ограничиваем 8 воркерами
            batch_size = int(safe_memory_bytes / (window_size_bytes * n_workers))
            batch_size = min(10000, max(500, batch_size))  # От 500 до 10000
            
            self.logger.info(f"🔄 Батчевая обработка: {batch_size} окон за раз")
            self.logger.info(f"⚡ Параллелизация: {n_workers} CPU ядер")
            self.logger.info(f"💾 Используем {safe_memory_bytes/(1024**3):.1f} GB памяти")
            
            with h5py.File(cache_file, 'w') as f:
                # Создаем датасеты с оптимальными chunk-ами
                chunk_size = min(2048, batch_size)  # Увеличенные chunks для batch_size=2048
                X_dataset = f.create_dataset('X', shape=X_shape, dtype='float32', 
                                           chunks=(chunk_size,) + X_sample.shape,
                                           compression=None)  # Без сжатия для максимальной скорости
                y_dataset = f.create_dataset('y', shape=y_shape, dtype='float32',
                                           chunks=(chunk_size,) + y_sample.shape,
                                           compression=None)  # Без сжатия для максимальной скорости
                
                # Заполняем данными батчами
                n_batches = (n_samples + batch_size - 1) // batch_size
                
                # Главный прогресс-бар для батчей
                batch_pbar = tqdm(range(n_batches), desc="Батчи", position=0)
                total_processed = 0
                
                for batch_idx in batch_pbar:
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_samples)
                    current_batch_size = end_idx - start_idx
                    
                    # Обновляем информацию о прогрессе
                    batch_pbar.set_postfix({
                        'Обработано': f'{total_processed}/{n_samples}',
                        'Память': f'{psutil.virtual_memory().percent:.1f}%'
                    })
                    
                    # Предварительное выделение памяти для батча
                    X_batch = np.zeros((current_batch_size,) + X_sample.shape, dtype=np.float32)
                    y_batch = np.zeros((current_batch_size,) + y_sample.shape, dtype=np.float32)
                    
                    # Простая последовательная обработка с прогресс-баром
                    # ProcessPoolExecutor может вызывать проблемы с памятью при большом количестве данных
                    window_pbar = tqdm(
                        range(current_batch_size),
                        desc=f"Окна батча {batch_idx+1}/{n_batches}",
                        position=1,
                        leave=False
                    )
                    
                    for i in window_pbar:
                        try:
                            idx = start_idx + i
                            X, y, info = self.temp_dataset[idx]
                            X_batch[i] = X.numpy().astype(np.float32)
                            y_batch[i] = y.numpy().astype(np.float32)
                            
                            # Периодическая очистка для предотвращения накопления мусора
                            if i % 100 == 0 and i > 0:
                                gc.collect(0)  # Быстрая сборка мусора
                                
                        except Exception as e:
                            self.logger.error(f"Ошибка обработки окна {idx}: {e}")
                            # Заполняем нулями в случае ошибки
                            X_batch[i] = np.zeros(X_sample.shape, dtype=np.float32)
                            y_batch[i] = np.zeros(y_sample.shape, dtype=np.float32)
                    
                    # Записываем батч в HDF5
                    X_dataset[start_idx:end_idx] = X_batch
                    y_dataset[start_idx:end_idx] = y_batch
                    
                    # Обновляем счетчик
                    total_processed += current_batch_size
                    
                    # Агрессивная очистка памяти
                    del X_batch, y_batch
                    gc.collect()
                    
                    # Проверка памяти и адаптация размера батча
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 80:
                        self.logger.warning(f"⚠️ Высокое использование памяти: {memory_percent:.1f}%")
                        # Уменьшаем размер батча
                        batch_size = max(500, int(batch_size * 0.7))
                        self.logger.info(f"📉 Уменьшен размер батча до {batch_size}")
                        time.sleep(2)  # Даем системе время на очистку
                    elif memory_percent < 50 and batch_size < 10000:
                        # Можем увеличить размер батча
                        batch_size = min(10000, int(batch_size * 1.2))
                        self.logger.info(f"📈 Увеличен размер батча до {batch_size}")
            
            # Открываем файл для чтения
            self.h5_file = h5py.File(cache_file, 'r')
            self.X_data = self.h5_file['X']
            self.y_data = self.h5_file['y']
            
        else:
            # Загружаем все в память (быстрее, но требует больше RAM)
            self.logger.info("⚠️ Загрузка всех данных в память...")
            
            X_list = []
            y_list = []
            
            # Предвычисляем все окна
            for i in tqdm(range(n_samples), desc="Предвычисление окон"):
                X, y, info = self.temp_dataset[i]
                X_list.append(X.numpy())
                y_list.append(y.numpy())
                
                # Периодическая очистка памяти
                if i % 10000 == 0:
                    gc.collect()
            
            # Конвертируем в numpy массивы
            self.X_data = np.stack(X_list)
            self.y_data = np.stack(y_list)
            
            # Очистка памяти
            del X_list, y_list
            gc.collect()
    
    def _save_cache(self, cache_file: Path):
        """Сохранение кэша"""
        if not self.use_hdf5:
            # Сохраняем pickle
            cache_data = {
                'X': self.X_data,
                'y': self.y_data,
                'feature_cols': self.feature_cols,
                'target_cols': self.target_cols,
                'context_window': self.context_window,
                'prediction_window': self.prediction_window,
                'stride': self.stride
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_cache(self, cache_file: Path):
        """Загрузка кэша"""
        if self.use_hdf5:
            # Открываем HDF5 файл
            self.h5_file = h5py.File(cache_file, 'r')
            self.X_data = self.h5_file['X']
            self.y_data = self.h5_file['y']
        else:
            # Загружаем pickle
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.X_data = cache_data['X']
            self.y_data = cache_data['y']
    
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, idx):
        """Быстрое получение предвычисленного примера"""
        # Преобразуем в тензоры
        X = torch.FloatTensor(self.X_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        
        # Минимальная информация для совместимости
        info = {
            'idx': idx
        }
        
        return X, y, info
    
    def __del__(self):
        """Закрытие HDF5 файла при удалении объекта"""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()


def create_precomputed_data_loaders(train_data: pd.DataFrame,
                                   val_data: pd.DataFrame,
                                   test_data: pd.DataFrame,
                                   config: Dict,
                                   feature_cols: List[str] = None,
                                   target_cols: List[str] = None) -> Tuple[torch.utils.data.DataLoader, 
                                                                           torch.utils.data.DataLoader, 
                                                                           torch.utils.data.DataLoader]:
    """Создание DataLoader'ов с предвычисленными данными для максимальной скорости"""
    
    logger = get_logger("PrecomputedDataLoaders")
    
    batch_size = config['model']['batch_size']
    context_window = config['model']['context_window']
    pred_window = config['model']['pred_len']
    num_workers = config['performance']['num_workers']
    persistent_workers = config['performance'].get('persistent_workers', True) if num_workers > 0 else False
    prefetch_factor = config['performance'].get('prefetch_factor', 2)
    
    # Получаем параметры из конфига
    normalize = config.get('data', {}).get('normalize', True)
    scaler_path = config.get('data', {}).get('scaler_path', 'models_saved/data_scaler.pkl')
    pin_memory = config['performance'].get('dataloader_pin_memory', True)
    drop_last = config['performance'].get('dataloader_drop_last', True)
    
    # Параметры stride
    train_stride = config.get('data', {}).get('train_stride', 1)
    val_stride = config.get('data', {}).get('val_stride', 4)
    
    # Проверка наличия scaler
    from pathlib import Path
    scaler_exists = Path(scaler_path).exists()
    
    logger.info("🚀 Создание PrecomputedDataset для быстрой загрузки...")
    
    # Создание датасетов
    train_dataset = PrecomputedDataset(
        data=train_data,
        context_window=context_window,
        prediction_window=pred_window,
        feature_cols=feature_cols,
        target_cols=target_cols,
        stride=train_stride,
        dataset_name="train",
        use_hdf5=True,  # Используем HDF5 для больших данных
        normalize=normalize,
        scaler_path=scaler_path,
        fit_scaler=not scaler_exists
    )
    
    val_dataset = PrecomputedDataset(
        data=val_data,
        context_window=context_window,
        prediction_window=pred_window,
        feature_cols=feature_cols,
        target_cols=target_cols,
        stride=val_stride,
        dataset_name="val",
        use_hdf5=True,
        normalize=normalize,
        scaler_path=scaler_path,
        fit_scaler=False
    )
    
    test_dataset = PrecomputedDataset(
        data=test_data,
        context_window=context_window,
        prediction_window=pred_window,
        feature_cols=feature_cols,
        target_cols=target_cols,
        stride=4,  # Фиксированный stride для теста
        dataset_name="test",
        use_hdf5=True,
        normalize=normalize,
        scaler_path=scaler_path,
        fit_scaler=False
    )
    
    logger.info(f"📊 Размеры предвычисленных датасетов:")
    logger.info(f"   - Train: {len(train_dataset):,} окон")
    logger.info(f"   - Val: {len(val_dataset):,} окон")
    logger.info(f"   - Test: {len(test_dataset):,} окон")
    
    # Проверяем нужно ли использовать WeightedRandomSampler
    use_weighted_sampling = config.get('training', {}).get('use_weighted_sampling', False)
    
    # Создание DataLoader'ов
    if use_weighted_sampling:
        logger.info("⚖️ Используем WeightedRandomSampler для балансировки классов...")
        
        # Получаем веса классов из конфига
        class_weights = config.get('loss', {}).get('class_weights', [2.5, 2.5, 0.3])
        
        # Рассчитываем веса для каждого сэмпла
        sample_weights = calculate_sample_weights(train_dataset, class_weights=class_weights)
        
        # Создаем sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=False  # Без дублирования для более стабильного обучения
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Используем sampler вместо shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=custom_collate_fn
        )
    else:
        # Проверяем настройку shuffle
        import os
        shuffle_enabled = config.get('performance', {}).get('shuffle_train', True)
        if os.environ.get('DISABLE_SHUFFLE', '0') == '1':
            shuffle_enabled = False
        
        if not shuffle_enabled:
            logger.warning("⚠️ Shuffle отключен для ускорения работы с HDF5 кэшем")
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_enabled,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            collate_fn=custom_collate_fn
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=custom_collate_fn  # Используем кастомную функцию для pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=custom_collate_fn  # Используем кастомную функцию для pin_memory
    )
    
    logger.info("✅ PrecomputedDataLoader'ы созданы успешно!")
    
    return train_loader, val_loader, test_loader