"""
Утилиты для быстрой работы с файлами и данными
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any, Dict, Optional, Union
import logging
from pathlib import Path
import gc
import concurrent.futures
from functools import partial

logger = logging.getLogger(__name__)


class FastIO:
    """Класс для оптимизированной работы с файлами"""
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str, compress: int = 3) -> None:
        """
        Быстрое сохранение объекта в pickle с сжатием
        
        Args:
            obj: Объект для сохранения
            filepath: Путь к файлу
            compress: Уровень сжатия (0-9, где 3 - оптимальный баланс)
        """
        try:
            # Используем joblib для более быстрого сохранения больших массивов
            if isinstance(obj, (np.ndarray, pd.DataFrame)):
                joblib.dump(obj, filepath, compress=compress)
                logger.info(f"✅ Сохранено с joblib: {filepath}")
            else:
                # Для обычных объектов используем pickle с оптимизацией
                with open(filepath, 'wb') as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"✅ Сохранено с pickle: {filepath}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения {filepath}: {e}")
            raise
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """
        Быстрая загрузка объекта из pickle
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            Загруженный объект
        """
        try:
            # Проверяем, был ли файл сохранен через joblib
            try:
                obj = joblib.load(filepath)
                logger.info(f"✅ Загружено с joblib: {filepath}")
                return obj
            except:
                # Если не joblib, пробуем обычный pickle
                with open(filepath, 'rb') as f:
                    obj = pickle.load(f)
                logger.info(f"✅ Загружено с pickle: {filepath}")
                return obj
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {filepath}: {e}")
            raise
    
    @staticmethod
    def save_parquet_fast(df: pd.DataFrame, filepath: str, 
                         compression: str = 'snappy',
                         use_pyarrow: bool = True) -> None:
        """
        Быстрое сохранение DataFrame в parquet
        
        Args:
            df: DataFrame для сохранения
            filepath: Путь к файлу
            compression: Тип сжатия ('snappy', 'gzip', 'brotli')
            use_pyarrow: Использовать pyarrow для ускорения
        """
        try:
            if use_pyarrow:
                # Используем pyarrow напрямую для максимальной скорости
                table = pa.Table.from_pandas(df, preserve_index=False)
                pq.write_table(
                    table, 
                    filepath,
                    compression=compression,
                    use_dictionary=True,  # Для категориальных данных
                    write_statistics=True,  # Для быстрого чтения
                    row_group_size=50000  # Оптимальный размер группы
                )
            else:
                # Fallback на pandas
                df.to_parquet(
                    filepath, 
                    compression=compression,
                    engine='pyarrow',
                    index=False
                )
            
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            logger.info(f"✅ Сохранено в parquet: {filepath} ({size_mb:.1f} MB)")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения parquet {filepath}: {e}")
            raise
    
    @staticmethod
    def load_parquet_fast(filepath: str, 
                         columns: Optional[list] = None,
                         use_pyarrow: bool = True) -> pd.DataFrame:
        """
        Быстрая загрузка DataFrame из parquet
        
        Args:
            filepath: Путь к файлу
            columns: Список колонок для загрузки (None = все)
            use_pyarrow: Использовать pyarrow для ускорения
            
        Returns:
            Загруженный DataFrame
        """
        try:
            if use_pyarrow:
                # Используем pyarrow с оптимизациями
                parquet_file = pq.ParquetFile(filepath)
                
                # Если нужны только определенные колонки
                if columns:
                    df = parquet_file.read(columns=columns).to_pandas()
                else:
                    df = parquet_file.read().to_pandas()
            else:
                # Fallback на pandas
                df = pd.read_parquet(filepath, columns=columns, engine='pyarrow')
            
            logger.info(f"✅ Загружено из parquet: {filepath} ({len(df):,} строк)")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки parquet {filepath}: {e}")
            raise
    
    @staticmethod
    def parallel_save_models(models: Dict[str, Any], 
                           base_path: str,
                           n_workers: int = 4) -> None:
        """
        Параллельное сохранение нескольких моделей
        
        Args:
            models: Словарь {имя: модель}
            base_path: Базовый путь для сохранения
            n_workers: Количество воркеров
        """
        os.makedirs(base_path, exist_ok=True)
        
        def save_model(item):
            name, model = item
            filepath = os.path.join(base_path, f"{name}.pkl")
            FastIO.save_pickle(model, filepath)
            return name
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(save_model, item): item[0] 
                      for item in models.items()}
            
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    logger.info(f"✅ Сохранена модель: {result}")
                except Exception as e:
                    logger.error(f"❌ Ошибка сохранения модели {name}: {e}")
    
    @staticmethod
    def optimize_memory(df: pd.DataFrame, 
                       verbose: bool = True) -> pd.DataFrame:
        """
        Оптимизация использования памяти DataFrame
        
        Args:
            df: DataFrame для оптимизации
            verbose: Выводить информацию
            
        Returns:
            Оптимизированный DataFrame
        """
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                # Оптимизация int
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                # Оптимизация float
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        # Оптимизация категориальных данных
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage().sum() / 1024**2
        
        if verbose:
            logger.info(f'🎯 Оптимизация памяти: {start_mem:.1f} MB → {end_mem:.1f} MB '
                       f'({100 * (start_mem - end_mem) / start_mem:.1f}% экономии)')
        
        return df
    
    @staticmethod
    def batch_process_files(file_paths: list,
                          process_func: callable,
                          n_workers: int = 4,
                          **kwargs) -> list:
        """
        Параллельная обработка нескольких файлов
        
        Args:
            file_paths: Список путей к файлам
            process_func: Функция обработки
            n_workers: Количество воркеров
            **kwargs: Дополнительные аргументы для process_func
            
        Returns:
            Список результатов
        """
        process_func_with_args = partial(process_func, **kwargs)
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_file = {executor.submit(process_func_with_args, fp): fp 
                            for fp in file_paths}
            
            for future in concurrent.futures.as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"✅ Обработан: {filepath}")
                except Exception as e:
                    logger.error(f"❌ Ошибка обработки {filepath}: {e}")
                    results.append(None)
        
        return results


# Глобальный экземпляр для удобства
fast_io = FastIO()