"""
Кеширование данных для Transformer v3
"""

import pickle
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import json

from config import Config

logger = logging.getLogger(__name__)


class CacheManager:
    """Менеджер для кеширования подготовленных данных"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path("cache/transformer_v3")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """Генерация уникального ключа для кеша"""
        # Создаем строку из параметров
        param_str = json.dumps(params, sort_keys=True)
        # Генерируем хеш
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def save_data(self, data: Any, cache_name: str):
        """
        Сохранение данных в кеш
        
        Args:
            data: Данные для сохранения
            cache_name: Имя файла кеша
        """
        cache_path = self.cache_dir / f"{cache_name}.pkl"
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"💾 Данные сохранены в кеш: {cache_path}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения в кеш: {e}")
            
    def load_data(self, cache_name: str) -> Optional[Any]:
        """
        Загрузка данных из кеша
        
        Args:
            cache_name: Имя файла кеша
            
        Returns:
            Загруженные данные или None
        """
        cache_path = self.cache_dir / f"{cache_name}.pkl"
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"📂 Данные загружены из кеша: {cache_path}")
            return data
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки из кеша: {e}")
            return None
            
    def save_sequences(self, sequences: Dict[str, Dict[str, np.ndarray]], 
                      model_type: str, task_type: str):
        """
        Сохранение последовательностей
        
        Args:
            sequences: Словарь с последовательностями
            model_type: 'buy' или 'sell'
            task_type: 'regression' или 'classification_binary'
        """
        cache_name = f"sequences_{model_type}_{task_type}"
        self.save_data(sequences, cache_name)
        
    def load_sequences(self, model_type: str, task_type: str) -> Optional[Dict]:
        """
        Загрузка последовательностей
        
        Args:
            model_type: 'buy' или 'sell'
            task_type: 'regression' или 'classification_binary'
            
        Returns:
            Словарь с последовательностями или None
        """
        cache_name = f"sequences_{model_type}_{task_type}"
        return self.load_data(cache_name)
        
    def save_processed_data(self, df: pd.DataFrame, stage: str):
        """
        Сохранение обработанных данных
        
        Args:
            df: DataFrame с данными
            stage: Этап обработки ('raw', 'features', 'normalized')
        """
        cache_name = f"data_{stage}"
        
        # Для больших DataFrame используем parquet
        if len(df) > 100000:
            cache_path = self.cache_dir / f"{cache_name}.parquet"
            df.to_parquet(cache_path, compression='snappy')
            logger.info(f"💾 Данные сохранены в parquet: {cache_path}")
        else:
            self.save_data(df, cache_name)
            
    def load_processed_data(self, stage: str) -> Optional[pd.DataFrame]:
        """
        Загрузка обработанных данных
        
        Args:
            stage: Этап обработки
            
        Returns:
            DataFrame или None
        """
        # Сначала пробуем parquet
        parquet_path = self.cache_dir / f"data_{stage}.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                logger.info(f"📂 Данные загружены из parquet: {parquet_path}")
                return df
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки parquet: {e}")
                
        # Затем пробуем pickle
        return self.load_data(f"data_{stage}")
        
    def clear_cache(self):
        """Очистка всего кеша"""
        for file in self.cache_dir.glob("*"):
            file.unlink()
        logger.info("🗑️ Кеш очищен")
        
    def get_cache_info(self) -> Dict[str, Any]:
        """Информация о кеше"""
        cache_files = list(self.cache_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        info = {
            'cache_dir': str(self.cache_dir),
            'n_files': len(cache_files),
            'total_size_mb': total_size / 1024 / 1024,
            'files': [f.name for f in cache_files]
        }
        
        return info