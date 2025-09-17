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
    
    Решает проблему CUDA error при использовании pin_memory с RTX 5090
    """
    # Разделяем батч на компоненты
    X_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    
    # Создаем словарь info с тензором индексов
    # ВАЖНО: Оставляем структуру словаря для совместимости с тренером
    # Проверяем, есть ли ключ 'idx' в info, если нет - создаем его
    if 'idx' in batch[0][2]:
        idx_list = [item[2]['idx'] for item in batch]
    else:
        # Создаем индексы на основе позиции в батче
        idx_list = list(range(len(batch)))
    
    idx_tensor = torch.tensor(idx_list, dtype=torch.long)
    
    # Базовый info словарь
    info_batch = {'idx': idx_tensor}
    
    # Безопасно проверяем наличие метаданных в первом элементе батча
    try:
        first_info = batch[0][2]
        
        if 'symbol' in first_info:
            info_batch['symbol'] = [item[2]['symbol'] for item in batch]
        
        if 'timestamp' in first_info:
            info_batch['timestamp'] = [item[2]['timestamp'] for item in batch]
        
        if 'close_price' in first_info:
            info_batch['close_price'] = torch.tensor([item[2]['close_price'] for item in batch], dtype=torch.float32)
    except (KeyError, IndexError, TypeError):
        # Если возникают проблемы с метаданными, игнорируем их
        pass
    
    return X_batch, y_batch, info_batch


def calculate_sample_weights(dataset: 'PrecomputedDataset',
                           direction_indices: List[int] = None,  # Автоматически находим direction колонки
                           class_weights: List[float] = [2.5, 2.5, 0.3]) -> torch.Tensor:
    """
    Рассчитывает веса для каждого сэмпла на основе распределения классов direction
    
    Args:
        dataset: PrecomputedDataset
        direction_indices: индексы direction переменных в targets (по умолчанию только direction_15m)
        class_weights: веса для классов [LONG, SHORT, FLAT]
        
    Returns:
        torch.Tensor с весом для каждого сэмпла
    """
    logger = get_logger("SampleWeights")
    logger.info("📊 Расчет весов сэмплов для балансировки классов...")
    logger.info(f"📌 Входные class_weights: {class_weights}")
    
    # Автоматически находим индексы direction колонок если не указаны
    if direction_indices is None:
        direction_indices = []
        if hasattr(dataset, 'target_cols'):
            for i, col in enumerate(dataset.target_cols):
                if 'direction' in col.lower():
                    direction_indices.append(i)
        if not direction_indices:
            # Fallback на стандартные индексы (обновлено: direction колонки на позициях 5-8)
            direction_indices = [5, 6, 7, 8]
        logger.info(f"🔍 Автоматически найдены индексы direction: {direction_indices}")

    # Загружаем все таргеты для анализа
    all_targets = []

    # Добавим отладочную информацию
    logger.info(f"📊 Размер dataset: {len(dataset)}")
    logger.info(f"📊 use_cache: {dataset.use_cache}, use_hdf5: {dataset.use_hdf5}")

    # Если кэш отключен или не используем HDF5
    if not dataset.use_cache or not dataset.use_hdf5:
        # Загружаем напрямую из датасета
        for i in range(len(dataset)):
            _, y, _ = dataset[i]
            # Если y имеет размерность [prediction_window, n_targets], берем только первый timestep
            if y.dim() > 1 and y.shape[0] > 1:
                y = y[0]  # Берем только первый временной шаг
            all_targets.append(y)
        targets = torch.stack(all_targets).numpy()
    else:
        # Используем кэш если доступен
        cache_file = dataset._get_cache_path()
        if cache_file.exists():
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
    # Инициализируем единицами как безопасное значение по умолчанию
    sample_weights = np.ones(len(targets))
    
    # Отладка размерности targets
    logger.info(f"📊 Исходная размерность targets: {targets.shape}")
    logger.info(f"📊 Тип данных targets: {targets.dtype}")

    # Сжимаем размерность если нужно
    if targets.ndim == 3 and targets.shape[1] == 1:
        targets = targets.squeeze(1)  # (n_samples, n_targets)
        logger.info(f"✅ Targets сжаты до размерности: {targets.shape}")
    
    # Собираем направления по указанным индексам (фильтруем выход за диапазон)
    valid_indices = [i for i in direction_indices if i < targets.shape[1]]
    if not valid_indices:
        logger.error(f"❌ Ни один из индексов {direction_indices} не попадает в диапазон targets.shape={targets.shape}")
        return torch.from_numpy(sample_weights).float()
    logger.info(f"📍 Используем индексы для direction: {valid_indices}")
    logger.info(f"📊 Размерность targets после сжатия: {targets.shape}")

    # Отладка: проверяем первые значения direction колонок
    for idx in valid_indices[:2]:  # Проверяем первые 2 индекса
        sample_values = targets[:5, idx]  # Первые 5 значений
        unique_vals, counts = np.unique(targets[:, idx], return_counts=True)
        logger.info(f"  📍 Колонка {idx}: первые 5 = {sample_values}")
        logger.info(f"     Уникальные значения: {unique_vals}, counts: {counts}")

    directions_multi = [targets[:, i].astype(int) for i in valid_indices]
    directions = directions_multi[0]  # для логов и длины
    logger.info(f"✅ Загружены direction данные: {len(directions)} samples x {len(valid_indices)} tf")
    
    # Подсчет классов по первому таймфрейму для отображения
    first_tf = directions_multi[0]  # direction_15m
    unique_first, counts_first = np.unique(first_tf, return_counts=True)
    class_dist_first = {int(cls): cnt for cls, cnt in zip(unique_first, counts_first)}
    n_samples = len(first_tf)

    # Логирование распределения классов для первого таймфрейма
    logger.info(f"📊 Распределение классов direction_15m:")
    logger.info(f"   LONG: {class_dist_first.get(0,0):,} ({class_dist_first.get(0,0)/n_samples:.1%})")
    logger.info(f"   SHORT: {class_dist_first.get(1,0):,} ({class_dist_first.get(1,0)/n_samples:.1%})")
    logger.info(f"   FLAT: {class_dist_first.get(2,0):,} ({class_dist_first.get(2,0)/n_samples:.1%})")

    # Подсчет общих классов по всем таймфреймам для расчета весов
    stacked = np.concatenate(directions_multi).astype(int)
    unique, counts = np.unique(stacked, return_counts=True)
    class_dist = {int(cls): cnt for cls, cnt in zip(unique, counts)}
    total_samples = len(stacked)
    
    # Рассчитываем веса балансировки с inverse frequency weighting
    balanced_weights = np.zeros(3)
    for cls in range(3):
        class_count = class_dist.get(cls, 0)
        if class_count > 0:
            # Инверсная частота с умножением на class_weights
            balanced_weights[cls] = (total_samples / (3.0 * class_count)) * class_weights[cls]
        else:
            # Если класса нет в данных, используем большой вес
            balanced_weights[cls] = class_weights[cls] * 10.0
    
    logger.info(f"⚖️ Рассчитанные веса классов:")
    logger.info(f"   LONG weight: {balanced_weights[0]:.3f}")
    logger.info(f"   SHORT weight: {balanced_weights[1]:.3f}")
    logger.info(f"   FLAT weight: {balanced_weights[2]:.3f}")
    
    # Применяем веса к каждому сэмплу — как среднее веса по всем tf
    for i in range(len(sample_weights)):
        per_tf_weights = []
        for arr in directions_multi:
            d = int(arr[i])
            if 0 <= d <= 2:
                per_tf_weights.append(balanced_weights[d])
        sample_weights[i] = np.mean(per_tf_weights) if per_tf_weights else 1.0
    
    # Проверяем уникальность весов перед нормализацией
    unique_weights = np.unique(sample_weights)
    logger.info(f"📊 Уникальные веса до нормализации: {unique_weights}")
    
    # Проверка на нулевые веса
    if sample_weights.min() <= 0:
        logger.error("❌ Обнаружены нулевые или отрицательные веса! Сброс на единицы.")
        sample_weights = np.ones_like(sample_weights)
    
    # Нормализуем веса ТОЛЬКО если они не все одинаковые
    if len(unique_weights) > 1 and sample_weights.min() > 0:
        sample_weights = sample_weights / sample_weights.mean()
        logger.info(f"✅ Веса нормализованы: min={sample_weights.min():.2f}, "
                    f"max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")
    else:
        if len(unique_weights) == 1:
            logger.warning(f"⚠️ Все веса одинаковые ({unique_weights[0]:.2f}), нормализация пропущена!")
    
    logger.info(f"✅ Финальные веса: min={sample_weights.min():.2f}, "
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
                 fit_scaler: bool = False,
                 shard_max_samples: Optional[int] = None,
                 use_cache: bool = False):  # НОВЫЙ ПАРАМЕТР: отключение кэша
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
            use_cache: использовать кэширование (False = прямая загрузка без кэша)
        """
        self.logger = get_logger("PrecomputedDataset")
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.stride = stride
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.use_hdf5 = use_hdf5
        self.shard_max_samples = shard_max_samples
        self.use_cache = use_cache  # Сохраняем флаг использования кэша
        
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
        
        # Если кэш отключен - используем TimeSeriesDataset напрямую
        if not self.use_cache:
            self.logger.info("⚡ Кэш отключен - используем прямую загрузку из TimeSeriesDataset")
            self.logger.info(f"✅ PrecomputedDataset готов (без кэша): {len(self)} примеров")
            # Не освобождаем данные и не создаем кэш
            return
        
        # Освобождаем оригинальные данные из памяти (только если используем кэш)
        del data
        gc.collect()
        
        # Проверяем доступную память перед созданием кэша
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.logger.info(f"🖥️ Доступная память: {available_memory_gb:.2f} GB")
        
        # Проверяем наличие кэша
        cache_file = self._get_cache_path()

        # Проверяем наличие шардов (train может быть очень большим)
        shard_files = list(self.cache_dir.glob(self._get_shard_glob())) if self.use_hdf5 else []
        if shard_files:
            shard_files = sorted(shard_files)
            self.logger.info(f"✅ Найдены {len(shard_files)} шард(ов) предвычисленных данных")
            self._load_shards(shard_files)
        elif cache_file.exists():
            self.logger.info(f"✅ Загрузка предвычисленных данных из {cache_file}")
            self._load_cache(cache_file)
        else:
            self.logger.info(f"📊 Предвычисление всех окон для {dataset_name}...")
            if self.use_hdf5 and self.shard_max_samples and self.dataset_name == 'train':
                self._precompute_all_windows_sharded(self.shard_max_samples)
            else:
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

    def _get_shard_name(self, shard_idx: int) -> str:
        return f"{self.dataset_name}_w{self.context_window}_s{self.stride}_part{shard_idx:03d}.h5"

    def _get_shard_glob(self) -> str:
        return f"{self.dataset_name}_w{self.context_window}_s{self.stride}_part*.h5"
    
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
            
            # ВАЖНО: Создаем случайный порядок индексов для перемешивания
            # Это позволит потом читать данные последовательно без shuffle
            self.logger.info("🎲 Создание случайного порядка индексов для перемешивания данных...")
            shuffled_indices = np.random.permutation(n_samples)
            self.logger.info(f"✅ Индексы перемешаны - данные будут сохранены в случайном порядке")
            self.logger.info(f"   Первые 10 оригинальных индексов: {list(range(10))}")
            self.logger.info(f"   Первые 10 перемешанных индексов: {shuffled_indices[:10].tolist()}")
            
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
                
                # Создаем датасеты для метаданных
                # Используем строковый тип для символа (UTF-8)
                symbol_dtype = h5py.string_dtype(encoding='utf-8')
                symbols_dataset = f.create_dataset('symbols', shape=(n_samples,), 
                                                 dtype=symbol_dtype, chunks=(chunk_size,))
                # Timestamps как float64 (Unix timestamps)
                timestamps_dataset = f.create_dataset('timestamps', shape=(n_samples,), 
                                                    dtype='float64', chunks=(chunk_size,))
                # Цены как float32
                prices_dataset = f.create_dataset('prices', shape=(n_samples,), 
                                                dtype='float32', chunks=(chunk_size,))
                
                # Сохраняем информацию о перемешивании для отладки
                f.create_dataset('shuffled_indices', data=shuffled_indices, dtype='int64')
                f.attrs['is_shuffled'] = True
                # Используем строку ASCII вместо Unicode для совместимости с HDF5
                import datetime
                f.attrs['creation_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode('ascii')
                self.logger.info("💾 Сохранена информация о перемешивании в HDF5")
                
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
                    # Массивы для метаданных
                    symbols_batch = []
                    timestamps_batch = np.zeros(current_batch_size, dtype=np.float64)
                    prices_batch = np.zeros(current_batch_size, dtype=np.float32)
                    
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
                            # Используем перемешанный индекс вместо последовательного
                            idx = shuffled_indices[start_idx + i]
                            X, y, info = self.temp_dataset[idx]
                            X_batch[i] = X.numpy().astype(np.float32)
                            y_batch[i] = y.numpy().astype(np.float32)
                            
                            # Сохраняем метаданные
                            symbols_batch.append(info['symbol'])
                            # Преобразуем timestamp в Unix time
                            from datetime import datetime
                            timestamp_str = info['context_end_time']
                            try:
                                # Парсим строку timestamp
                                dt = pd.to_datetime(timestamp_str)
                                timestamps_batch[i] = dt.timestamp()
                            except:
                                timestamps_batch[i] = 0.0
                            
                            # Извлекаем последнюю цену закрытия из контекста
                            # Предполагаем, что цена close находится в определенном индексе
                            try:
                                # Ищем индекс признака 'close' среди feature_cols
                                close_idx = None
                                for feat_idx, col in enumerate(self.temp_dataset.feature_cols):
                                    if 'close' in col.lower() and 'volume' not in col.lower():
                                        close_idx = feat_idx
                                        break
                                
                                if close_idx is not None:
                                    # Берем последнюю цену из последней временной точки контекста
                                    prices_batch[i] = X[-1, close_idx].item()
                                else:
                                    # Fallback: используем среднее значение первых нескольких признаков
                                    prices_batch[i] = X[-1, :5].mean().item()
                            except:
                                prices_batch[i] = 0.0
                            
                            # Периодическая очистка для предотвращения накопления мусора
                            if i % 100 == 0 and i > 0:
                                gc.collect(0)  # Быстрая сборка мусора
                                
                        except Exception as e:
                            self.logger.error(f"Ошибка обработки окна {idx}: {e}")
                            # Заполняем значениями по умолчанию в случае ошибки
                            X_batch[i] = np.zeros(X_sample.shape, dtype=np.float32)
                            y_batch[i] = np.zeros(y_sample.shape, dtype=np.float32)
                            symbols_batch.append('UNKNOWN')
                            timestamps_batch[i] = 0.0
                            prices_batch[i] = 0.0
                    
                    # Записываем батч в HDF5
                    X_dataset[start_idx:end_idx] = X_batch
                    y_dataset[start_idx:end_idx] = y_batch
                    # Записываем метаданные
                    symbols_dataset[start_idx:end_idx] = symbols_batch
                    timestamps_dataset[start_idx:end_idx] = timestamps_batch
                    prices_dataset[start_idx:end_idx] = prices_batch
                    
                    # Обновляем счетчик
                    total_processed += current_batch_size
                    
                    # Агрессивная очистка памяти
                    del X_batch, y_batch
                    gc.collect()
                
            # После завершения записи в контекстном менеджере, файл закрылся
            # Теперь открываем его для чтения
            self.h5_file = h5py.File(cache_file, 'r')
            self.X_data = self.h5_file['X']
            self.y_data = self.h5_file['y']
            self.total_len = n_samples
            self.has_metadata = True
            if 'symbols' in self.h5_file:
                self.symbols_data = self.h5_file['symbols']
                self.timestamps_data = self.h5_file['timestamps']
                self.prices_data = self.h5_file['prices']
                    
    def _precompute_all_windows_sharded(self, shard_max_samples: int):
        """Предвычисление всех окон в несколько HDF5 шардов для снижения нагрузки"""
        n_samples = len(self.temp_dataset)
        X_sample, y_sample, _ = self.temp_dataset[0]
        X_shape = X_sample.shape
        y_shape = y_sample.shape

        # Параметры батчей, как в обычной версии
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        window_size_bytes = (np.prod(X_sample.shape) + np.prod(y_sample.shape)) * 4
        safe_memory_bytes = available_memory_gb * 1024**3 * 0.1  # используем 10% памяти
        n_workers = min(mp.cpu_count() - 1, 4)
        batch_size = int(safe_memory_bytes / (window_size_bytes * max(1, n_workers)))
        batch_size = min(2000, max(500, batch_size))  # 500..2000

        self.logger.info(f"🔄 Шардинг: max {shard_max_samples} окон на шарду, batch={batch_size}, workers={n_workers}")

        total_written = 0
        shard_idx = 0
        while total_written < n_samples:
            remaining = n_samples - total_written
            cur_count = int(min(shard_max_samples, remaining))
            shard_path = self.cache_dir / self._get_shard_name(shard_idx)
            self.logger.info(f"💾 Создание шарда {shard_idx} на {cur_count:,} окон → {shard_path}")

            with h5py.File(shard_path, 'w') as f:
                chunk_size = min(1024, batch_size)
                X_dataset = f.create_dataset('X', shape=(cur_count,) + X_shape, dtype='float32',
                                             chunks=(chunk_size,) + X_shape, compression=None)
                y_dataset = f.create_dataset('y', shape=(cur_count,) + y_shape, dtype='float32',
                                             chunks=(chunk_size,) + y_shape, compression=None)

                n_batches = (cur_count + batch_size - 1) // batch_size
                batch_pbar = tqdm(range(n_batches), desc=f"Шард {shard_idx}")
                for b in batch_pbar:
                    start = total_written + b * batch_size
                    end = min(total_written + (b + 1) * batch_size, total_written + cur_count)
                    current_batch_size = end - start

                    X_batch = np.zeros((current_batch_size,) + X_shape, dtype=np.float32)
                    y_batch = np.zeros((current_batch_size,) + y_shape, dtype=np.float32)

                    for i in range(current_batch_size):
                        try:
                            X, y, _ = self.temp_dataset[start + i]
                            X_batch[i] = X.numpy().astype(np.float32)
                            y_batch[i] = y.numpy().astype(np.float32)
                        except Exception as e:
                            self.logger.error(f"Ошибка окна {start+i}: {e}")
                            X_batch[i] = np.zeros(X_shape, dtype=np.float32)
                            y_batch[i] = np.zeros(y_shape, dtype=np.float32)

                    # Записываем батч
                    X_dataset[start - total_written:end - total_written] = X_batch
                    y_dataset[start - total_written:end - total_written] = y_batch

                    del X_batch, y_batch
                    gc.collect()

            total_written += cur_count
            shard_idx += 1

        self.logger.info(f"✅ Шардинг завершен: {shard_idx} шард(ов)")
        
        # Загружаем созданные шарды
        shard_files = sorted(list(self.cache_dir.glob(self._get_shard_glob())))
        if shard_files:
            self._load_shards(shard_files)

    def _load_shards(self, shard_files: List[Path]):
        """Загрузка набора шардов с авто‑fallback при HDF5 ошибках."""
        import os, time
        self.shard_files = [str(p) for p in shard_files]
        self.shard_handlers = []
        self.shard_sizes = []
        corrupted = False
        for p in shard_files:
            try:
                f = h5py.File(p, 'r')
                self.shard_handlers.append(f)
                sz = f['X'].shape[0]
                self.shard_sizes.append(sz)
            except (BlockingIOError, OSError) as e:
                msg = str(e)
                self.logger.warning(f"⚠️ Ошибка открытия шард-файла {p}: {msg}")
                # Попытка обойти блокировки и несовместимости
                os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
                try:
                    time.sleep(0.5)
                    f = h5py.File(p, 'r')
                    self.shard_handlers.append(f)
                    sz = f['X'].shape[0]
                    self.shard_sizes.append(sz)
                except Exception as e2:
                    self.logger.error(f"❌ Не удалось открыть {p} повторно: {e2}")
                    corrupted = True
                    break
        if corrupted:
            # Закрываем уже открытые файлы и пересоздаём шардированный кэш
            for f in self.shard_handlers:
                try:
                    f.close()
                except Exception:
                    pass
            self.shard_handlers = []
            self.logger.warning("🧹 Удаляем повреждённые шард-файлы и пересоздаём кэш...")
            for p in shard_files:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
            # Пересоздание
            if self.use_hdf5 and self.shard_max_samples and self.dataset_name == 'train':
                self._precompute_all_windows_sharded(self.shard_max_samples)
                return
        # Успешная загрузка
        import numpy as _np
        self.cum_sizes = _np.cumsum(self.shard_sizes)
        self.total_len = int(self.cum_sizes[-1]) if len(self.cum_sizes) > 0 else 0
        self.has_metadata = False
    
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
        """Загрузка кэша с авто‑fallback при HDF5 ошибках."""
        if self.use_hdf5:
            import os, time
            # Попытка открыть с обходом блокировок
            def try_open(path):
                last = None
                for attempt in range(3):
                    try:
                        return h5py.File(path, 'r')
                    except (BlockingIOError, OSError) as e:
                        last = e
                        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
                        time.sleep(0.5 * (attempt + 1))
                if last:
                    raise last
            try:
                self.h5_file = try_open(cache_file)
                self.X_data = self.h5_file['X']
                self.y_data = self.h5_file['y']
                self.total_len = len(self.X_data)
                if 'symbols' in self.h5_file:
                    self.symbols_data = self.h5_file['symbols']
                    self.timestamps_data = self.h5_file['timestamps']
                    self.prices_data = self.h5_file['prices']
                    self.has_metadata = True
                else:
                    self.has_metadata = False
            except (BlockingIOError, OSError) as e:
                msg = str(e)
                self.logger.error(f"❌ Ошибка открытия HDF5 кэша {cache_file}: {msg}")
                # Если файл поврежден (bad header) или не удаётся снять блокировку — пересоздаем
                try:
                    cache_file.unlink(missing_ok=True)
                except Exception:
                    pass
                self.logger.warning("🔄 Пересоздание HDF5 кэша из исходных данных...")
                if self.use_hdf5 and self.shard_max_samples and self.dataset_name == 'train':
                    self._precompute_all_windows_sharded(self.shard_max_samples)
                    return
                else:
                    self._precompute_all_windows()
                    self._save_cache(cache_file)
                    # Повторная загрузка
                    self.h5_file = h5py.File(cache_file, 'r')
                    self.X_data = self.h5_file['X']
                    self.y_data = self.h5_file['y']
                    self.total_len = len(self.X_data)
                    self.has_metadata = 'symbols' in self.h5_file
        else:
            # Загружаем pickle
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.X_data = cache_data['X']
            self.y_data = cache_data['y']
            # Устанавливаем total_len для pickle тоже
            self.total_len = len(self.X_data)
            self.has_metadata = False
    
    def __len__(self):
        # Если кэш отключен - используем длину temp_dataset
        if not self.use_cache:
            return len(self.temp_dataset)
            
        if hasattr(self, 'total_len'):
            return self.total_len
        elif hasattr(self, 'X_data'):
            return len(self.X_data)
        else:
            return 0
    
    def __getitem__(self, idx):
        """Быстрое получение предвычисленного примера"""
        # Если кэш отключен - получаем данные из TimeSeriesDataset напрямую
        if not self.use_cache:
            return self.temp_dataset[idx]
            
        # Преобразуем в тензоры
        if hasattr(self, 'shard_handlers'):
            # Находим шард и локальный индекс
            shard_idx = int(np.searchsorted(self.cum_sizes, idx, side='right'))
            prev = 0 if shard_idx == 0 else int(self.cum_sizes[shard_idx - 1])
            local_idx = int(idx - prev)
            f = self.shard_handlers[shard_idx]
            # Убеждаемся что local_idx это int, а не tuple
            if isinstance(local_idx, tuple):
                local_idx = local_idx[0]
            X = torch.FloatTensor(f['X'][local_idx])
            y = torch.FloatTensor(f['y'][local_idx])
        else:
            X = torch.FloatTensor(self.X_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
        
        # Создаем info словарь с метаданными
        info = {
            'idx': idx
        }
        
        # Добавляем метаданные если они доступны
        if hasattr(self, 'has_metadata') and self.has_metadata:
            # Декодируем символ из bytes в строку
            symbol_val = self.symbols_data[idx]
            if isinstance(symbol_val, bytes):
                info['symbol'] = symbol_val.decode('utf-8')
            else:
                info['symbol'] = symbol_val
            # Преобразуем Unix timestamp обратно в datetime
            info['timestamp'] = pd.to_datetime(self.timestamps_data[idx], unit='s')
            info['close_price'] = float(self.prices_data[idx])
        
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
    # Важно: при наличии 'symbol_id' не отключаем нормализацию целиком.
    # Исключение самой колонки из нормализации реализовано внутри TimeSeriesDataset.
    scaler_path = config.get('data', {}).get('scaler_path', 'models_saved/data_scaler.pkl')
    pin_memory = config['performance'].get('dataloader_pin_memory', True)
    drop_last = config['performance'].get('dataloader_drop_last', True)
    
    # Pin memory теперь работает корректно с обновленным custom_collate_fn
    if pin_memory:
        logger.info("✅ Pin memory включен для ускорения передачи данных на GPU")
    
    # Параметры stride
    train_stride = config.get('data', {}).get('train_stride', 1)
    val_stride = config.get('data', {}).get('val_stride', 4)
    
    # Проверка наличия scaler
    from pathlib import Path
    scaler_exists = Path(scaler_path).exists()
    
    logger.info("🚀 Создание PrecomputedDataset для быстрой загрузки...")
    
    # Проверяем флаг использования кэша из конфига
    use_cache = config.get('performance', {}).get('use_precomputed_cache', False)
    
    if not use_cache:
        logger.info("⚡ Кэширование отключено - используем прямую загрузку данных")
    
    # Создание датасетов
    # На train всегда обучаем scaler на текущих признаках и сохраняем
    if scaler_exists:
        logger.info(f"✅ Найден существующий scaler: {scaler_path} (будет переобучен на train, если отличается)")
    else:
        logger.info(f"⚠️ Scaler не найден, будет создан новый: {scaler_path}")

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
        fit_scaler=True,  # ВАЖНО: всегда фитим на train для согласования признаков
        shard_max_samples=config.get('performance', {}).get('precomputed_shard_max_samples', 200000),
        use_cache=use_cache  # Передаем флаг использования кэша
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
        fit_scaler=False,
        shard_max_samples=config.get('performance', {}).get('precomputed_shard_max_samples', 200000),
        use_cache=use_cache  # Передаем флаг использования кэша
    )
    
    # Используем stride из конфига или 4 по умолчанию для совместимости с существующим кэшом
    test_stride = config.get('data', {}).get('test_stride', 4)  # По умолчанию 4
    
    test_dataset = PrecomputedDataset(
        data=test_data,
        context_window=context_window,
        prediction_window=pred_window,
        feature_cols=feature_cols,
        target_cols=target_cols,
        stride=test_stride,  # Используем меньший stride для большего количества данных
        dataset_name="test",
        use_hdf5=True,
        normalize=normalize,
        scaler_path=scaler_path,
        fit_scaler=False,
        shard_max_samples=config.get('performance', {}).get('precomputed_shard_max_samples', 200000),
        use_cache=use_cache  # Передаем флаг использования кэша
    )
    
    logger.info(f"📊 Размеры предвычисленных датасетов:")
    logger.info(f"   - Train: {len(train_dataset):,} окон")
    logger.info(f"   - Val: {len(val_dataset):,} окон")
    logger.info(f"   - Test: {len(test_dataset):,} окон")
    
    # Проверяем нужно ли использовать WeightedRandomSampler
    # Поддерживаем оба места в конфиге: training.use_weighted_sampling и loss.use_weighted_sampling
    use_weighted_sampling = (
        config.get('training', {}).get('use_weighted_sampling', False)
        or config.get('loss', {}).get('use_weighted_sampling', False)
    )
    
    # Создание DataLoader'ов
    if use_weighted_sampling:
        logger.info("⚖️ Используем WeightedRandomSampler для балансировки классов...")
        
        # Получаем веса классов из конфига
        class_weights = config.get('loss', {}).get('class_weights', [2.5, 2.5, 0.3])
        
        # Рассчитываем веса для каждого сэмпла
        sample_weights = calculate_sample_weights(train_dataset, class_weights=class_weights)
        
        # Создаем sampler с replacement=True для реального oversampling
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # ВАЖНО: включаем oversampling для балансировки классов
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
