"""
Менеджер кэширования для ускорения загрузки данных
"""

import pickle
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CacheManager:
    """Класс для управления кэшированием данных"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Загрузка метаданных кэша"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_metadata(self):
        """Сохранение метаданных кэша"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def _generate_key(self, identifier: str, params: Optional[Dict] = None) -> str:
        """Генерация уникального ключа для кэша"""
        key_data = {'identifier': identifier}
        if params:
            key_data.update(params)
            
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def get(self, identifier: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Получение данных из кэша"""
        cache_key = self._generate_key(identifier, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    
                logger.info(f"✅ Данные загружены из кэша: {identifier}")
                
                # Обновляем метаданные
                if cache_key in self.metadata:
                    self.metadata[cache_key]['access_count'] += 1
                    self.metadata[cache_key]['last_access'] = pd.Timestamp.now().isoformat()
                    self._save_metadata()
                    
                return data
                
            except Exception as e:
                logger.error(f"Ошибка при загрузке из кэша: {e}")
                # Удаляем поврежденный файл
                cache_file.unlink()
                
        return None
        
    def set(self, identifier: str, data: Any, params: Optional[Dict] = None):
        """Сохранение данных в кэш"""
        cache_key = self._generate_key(identifier, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            # Обновляем метаданные
            self.metadata[cache_key] = {
                'identifier': identifier,
                'params': params,
                'created': pd.Timestamp.now().isoformat(),
                'last_access': pd.Timestamp.now().isoformat(),
                'access_count': 0,
                'size_mb': cache_file.stat().st_size / 1024 / 1024
            }
            
            # Добавляем информацию о данных
            if isinstance(data, pd.DataFrame):
                self.metadata[cache_key]['data_info'] = {
                    'type': 'DataFrame',
                    'shape': data.shape,
                    'columns': len(data.columns)
                }
            elif isinstance(data, dict):
                self.metadata[cache_key]['data_info'] = {
                    'type': 'dict',
                    'keys': list(data.keys())
                }
                
            self._save_metadata()
            
            logger.info(f"💾 Данные сохранены в кэш: {identifier} ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении в кэш: {e}")
            if cache_file.exists():
                cache_file.unlink()
                
    def clear(self, identifier: Optional[str] = None):
        """Очистка кэша"""
        if identifier:
            # Очищаем конкретный идентификатор
            keys_to_remove = []
            for cache_key, info in self.metadata.items():
                if info['identifier'] == identifier:
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    if cache_file.exists():
                        cache_file.unlink()
                    keys_to_remove.append(cache_key)
                    
            for key in keys_to_remove:
                del self.metadata[key]
                
            self._save_metadata()
            logger.info(f"🗑️ Очищен кэш для: {identifier}")
            
        else:
            # Очищаем весь кэш
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                
            self.metadata = {}
            self._save_metadata()
            logger.info("🗑️ Весь кэш очищен")
            
    def get_cache_info(self) -> Dict:
        """Получение информации о кэше"""
        total_size = sum(
            (self.cache_dir / f"{key}.pkl").stat().st_size 
            for key in self.metadata.keys()
            if (self.cache_dir / f"{key}.pkl").exists()
        ) / 1024 / 1024  # MB
        
        info = {
            'total_items': len(self.metadata),
            'total_size_mb': total_size,
            'cache_dir': str(self.cache_dir),
            'items': []
        }
        
        for cache_key, metadata in self.metadata.items():
            info['items'].append({
                'identifier': metadata['identifier'],
                'created': metadata['created'],
                'last_access': metadata['last_access'],
                'access_count': metadata['access_count'],
                'size_mb': metadata['size_mb']
            })
            
        # Сортируем по последнему доступу
        info['items'].sort(key=lambda x: x['last_access'], reverse=True)
        
        return info
        
    def cleanup_old_cache(self, days: int = 7):
        """Удаление старого кэша"""
        current_time = pd.Timestamp.now()
        keys_to_remove = []
        
        for cache_key, metadata in self.metadata.items():
            last_access = pd.Timestamp(metadata['last_access'])
            
            if (current_time - last_access).days > days:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                keys_to_remove.append(cache_key)
                
        for key in keys_to_remove:
            del self.metadata[key]
            
        if keys_to_remove:
            self._save_metadata()
            logger.info(f"🗑️ Удалено {len(keys_to_remove)} старых файлов кэша")
            
    def cache_dataframe(self, df: pd.DataFrame, identifier: str, 
                       compression: bool = True) -> bool:
        """Специальный метод для кэширования DataFrame с компрессией"""
        if compression:
            # Используем parquet для эффективного хранения
            cache_key = self._generate_key(identifier)
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            try:
                df.to_parquet(cache_file, compression='snappy', index=False)
                
                # Обновляем метаданные
                self.metadata[cache_key] = {
                    'identifier': identifier,
                    'type': 'parquet',
                    'created': pd.Timestamp.now().isoformat(),
                    'last_access': pd.Timestamp.now().isoformat(),
                    'access_count': 0,
                    'size_mb': cache_file.stat().st_size / 1024 / 1024,
                    'shape': df.shape,
                    'columns': list(df.columns)
                }
                self._save_metadata()
                
                logger.info(f"💾 DataFrame сохранен в parquet: {identifier} ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка при сохранении parquet: {e}")
                return False
        else:
            self.set(identifier, df)
            return True
            
    def load_dataframe(self, identifier: str) -> Optional[pd.DataFrame]:
        """Загрузка DataFrame из кэша"""
        cache_key = self._generate_key(identifier)
        
        # Проверяем parquet файл
        parquet_file = self.cache_dir / f"{cache_key}.parquet"
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                logger.info(f"✅ DataFrame загружен из parquet: {identifier}")
                
                # Обновляем метаданные
                if cache_key in self.metadata:
                    self.metadata[cache_key]['access_count'] += 1
                    self.metadata[cache_key]['last_access'] = pd.Timestamp.now().isoformat()
                    self._save_metadata()
                    
                return df
            except Exception as e:
                logger.error(f"Ошибка при загрузке parquet: {e}")
                
        # Пробуем обычный pickle
        return self.get(identifier)