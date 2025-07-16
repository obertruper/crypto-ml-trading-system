"""
Менеджер кеша для Universal LSP Server
"""

import json
import gzip
import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib

from .config import CacheConfig

logger = logging.getLogger(__name__)

class CacheManager:
    """Менеджер кеша для LSP сервера"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.directory)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # В памяти кеш для быстрого доступа
        self.memory_cache: Dict[str, Any] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0
        }
        
        # Запускаем периодическую очистку
        self.cleanup_task = None
        if config.enabled:
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    def _get_cache_key(self, key: str) -> str:
        """Получить хеш ключа для имени файла"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Получить путь к файлу кеша"""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        """Получить значение из кеша"""
        if not self.config.enabled:
            return None
        
        # Проверяем память
        if key in self.memory_cache:
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]['data']
        
        # Проверяем файл
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self.cache_stats['misses'] += 1
            return None
        
        try:
            # Проверяем время жизни
            if cache_path.stat().st_mtime + self.config.ttl < datetime.now().timestamp():
                cache_path.unlink()
                self.cache_stats['misses'] += 1
                return None
            
            # Читаем из файла
            if self.config.compression:
                async with aiofiles.open(cache_path, 'rb') as f:
                    compressed = await f.read()
                    data = gzip.decompress(compressed)
                    value = json.loads(data.decode())
            else:
                async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    value = json.loads(content)
            
            # Сохраняем в память
            self.memory_cache[key] = {
                'data': value,
                'timestamp': datetime.now()
            }
            
            self.cache_stats['hits'] += 1
            return value
            
        except Exception as e:
            logger.error(f"Ошибка чтения кеша {key}: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any):
        """Сохранить значение в кеш"""
        if not self.config.enabled:
            return
        
        try:
            # Сохраняем в память
            self.memory_cache[key] = {
                'data': value,
                'timestamp': datetime.now()
            }
            
            # Сохраняем в файл
            cache_path = self._get_cache_path(key)
            
            if self.config.compression:
                data = json.dumps(value, ensure_ascii=False).encode()
                compressed = gzip.compress(data)
                async with aiofiles.open(cache_path, 'wb') as f:
                    await f.write(compressed)
            else:
                async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(value, ensure_ascii=False))
            
            self.cache_stats['writes'] += 1
            
        except Exception as e:
            logger.error(f"Ошибка записи кеша {key}: {e}")
    
    async def delete(self, key: str):
        """Удалить значение из кеша"""
        # Удаляем из памяти
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Удаляем файл
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
    
    async def clear(self):
        """Очистить весь кеш"""
        # Очищаем память
        self.memory_cache.clear()
        
        # Удаляем все файлы
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
        
        logger.info("Кеш очищен")
    
    async def get_index(self, directory: Path) -> Optional[Dict[str, Any]]:
        """Получить кешированный индекс директории"""
        key = f"index:{directory}"
        return await self.get(key)
    
    async def save_index(self, directory: Path, index_data: Dict[str, Any]):
        """Сохранить индекс директории в кеш"""
        key = f"index:{directory}"
        await self.set(key, index_data)
    
    async def flush(self):
        """Сбросить кеш на диск"""
        # В нашей реализации данные сразу сохраняются на диск
        logger.info(f"Статистика кеша: {self.cache_stats}")
    
    async def _periodic_cleanup(self):
        """Периодическая очистка устаревших записей"""
        while True:
            try:
                await asyncio.sleep(3600)  # Каждый час
                await self._cleanup_expired()
                await self._check_size_limit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка очистки кеша: {e}")
    
    async def _cleanup_expired(self):
        """Удалить устаревшие записи"""
        now = datetime.now().timestamp()
        expired_count = 0
        
        # Очищаем из памяти
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if entry['timestamp'].timestamp() + self.config.ttl < now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            expired_count += 1
        
        # Очищаем файлы
        for cache_file in self.cache_dir.glob("*.cache"):
            if cache_file.stat().st_mtime + self.config.ttl < now:
                cache_file.unlink()
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Удалено {expired_count} устаревших записей кеша")
    
    async def _check_size_limit(self):
        """Проверить и соблюсти ограничение размера"""
        total_size = 0
        cache_files = []
        
        # Собираем информацию о файлах
        for cache_file in self.cache_dir.glob("*.cache"):
            size = cache_file.stat().st_size
            mtime = cache_file.stat().st_mtime
            cache_files.append((cache_file, size, mtime))
            total_size += size
        
        # Конвертируем лимит в байты
        max_size_bytes = self.config.max_size * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Сортируем по времени изменения (старые первые)
            cache_files.sort(key=lambda x: x[2])
            
            # Удаляем старые файлы пока не уложимся в лимит
            for cache_file, size, _ in cache_files:
                if total_size <= max_size_bytes:
                    break
                
                cache_file.unlink()
                total_size -= size
                logger.debug(f"Удален файл кеша для освобождения места: {cache_file.name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику кеша"""
        # Подсчитываем размер
        total_size = 0
        file_count = 0
        
        for cache_file in self.cache_dir.glob("*.cache"):
            total_size += cache_file.stat().st_size
            file_count += 1
        
        return {
            'enabled': self.config.enabled,
            'memory_entries': len(self.memory_cache),
            'file_entries': file_count,
            'total_size_mb': total_size / 1024 / 1024,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'writes': self.cache_stats['writes'],
            'hit_rate': self.cache_stats['hits'] / max(1, self.cache_stats['hits'] + self.cache_stats['misses'])
        }