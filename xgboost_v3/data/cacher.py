"""
Кэширование данных для ускорения загрузки
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


class CacheManager:
    """Класс для управления кэшированием данных"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_path(self) -> Path:
        """Получить путь к файлу кэша"""
        if self.config.training.test_mode:
            return self.cache_dir / "test_data.parquet"
        else:
            return self.cache_dir / "full_data.parquet"
            
    def load_from_cache(self) -> Optional[pd.DataFrame]:
        """Загрузить данные из кэша"""
        cache_path = self.get_cache_path()
        
        if not cache_path.exists():
            logger.info("📭 Кэш не найден")
            return None
            
        try:
            logger.info(f"📥 Загрузка данных из кэша: {cache_path}")
            df = pd.read_parquet(cache_path)
            logger.info(f"✅ Загружено {len(df):,} записей из кэша")
            return df
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки кэша: {e}")
            return None
            
    def save_to_cache(self, df: pd.DataFrame):
        """Сохранить данные в кэш"""
        cache_path = self.get_cache_path()
        
        try:
            logger.info(f"💾 Сохранение данных в кэш: {cache_path}")
            df.to_parquet(cache_path, index=False)
            logger.info(f"✅ Сохранено {len(df):,} записей в кэш")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения кэша: {e}")
            
    def clear_cache(self):
        """Очистить кэш"""
        cache_files = list(self.cache_dir.glob("*.parquet"))
        
        for file in cache_files:
            file.unlink()
            
        logger.info(f"🗑️ Удалено {len(cache_files)} файлов кэша")