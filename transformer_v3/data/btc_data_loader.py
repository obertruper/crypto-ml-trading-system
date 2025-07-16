"""
Загрузчик данных BTC для корреляционных признаков
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

from config import Config, BTC_DATA_PARAMS, MARKET_FEATURES

logger = logging.getLogger(__name__)


class BTCDataLoader:
    """Класс для загрузки данных BTC"""
    
    def __init__(self, config: Config):
        self.config = config
        self.btc_symbol = BTC_DATA_PARAMS['symbol']
        self.cache = {}
        
    def load_btc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Загружает данные BTC и добавляет их в DataFrame
        
        Args:
            df: DataFrame с основными данными (должен содержать timestamp)
            
        Returns:
            DataFrame с добавленной колонкой btc_close
        """
        logger.info(f"📊 Загрузка данных {self.btc_symbol}...")
        
        if 'timestamp' not in df.columns:
            logger.error("❌ Отсутствует колонка timestamp в DataFrame")
            return df
            
        # Определяем временной диапазон
        min_timestamp = df['timestamp'].min()
        max_timestamp = df['timestamp'].max()
        
        # Пытаемся загрузить из базы данных
        btc_data = self._load_from_database(min_timestamp, max_timestamp)
        
        if btc_data is None or btc_data.empty:
            logger.warning(f"⚠️ Не удалось загрузить данные {self.btc_symbol} из БД")
            
            # Проверяем настройку fallback
            if not BTC_DATA_PARAMS.get('fallback_to_synthetic', False):
                logger.error("❌ Синтетические данные отключены, возвращаем исходный DataFrame")
                return df
            
            # Если включен fallback на синтетические данные
            if BTC_DATA_PARAMS['fallback_to_synthetic']:
                logger.warning("⚠️ Используются синтетические данные BTC! Необходимо загрузить реальные данные.")
                df['btc_close'] = df['close'] * np.random.uniform(
                    MARKET_FEATURES['btc_synthetic_min'], 
                    MARKET_FEATURES['btc_synthetic_max'], 
                    len(df)
                )
            else:
                logger.error(f"❌ Данные {self.btc_symbol} недоступны. Пропускаем создание BTC признаков.")
                return df
        else:
            # Объединяем данные
            logger.info(f"✅ Загружено {len(btc_data)} записей {self.btc_symbol}")
            
            # Merge по timestamp
            df = df.merge(
                btc_data[['timestamp', 'close']].rename(columns={'close': 'btc_close'}),
                on='timestamp',
                how='left'
            )
            
            # Заполняем пропуски методом forward fill
            if df['btc_close'].isnull().any():
                null_count = df['btc_close'].isnull().sum()
                logger.warning(f"⚠️ Обнаружено {null_count} пропусков в данных BTC, заполняем...")
                df['btc_close'] = df['btc_close'].ffill().bfill()
            
            # Валидация данных
            stats = self.validate_btc_coverage(df)
            logger.info(f"📊 Покрытие данных BTC: {stats['coverage']:.1f}%")
            
            # Проверка на синтетические данные
            if not stats['is_synthetic']:
                logger.info("✅ Используются реальные данные BTC")
            else:
                logger.error("❌ Обнаружены признаки синтетических данных!")
                
        return df
        
    def _load_from_database(self, min_timestamp: int, max_timestamp: int) -> Optional[pd.DataFrame]:
        """
        Загружает данные BTC из базы данных
        
        Args:
            min_timestamp: Минимальная временная метка
            max_timestamp: Максимальная временная метка
            
        Returns:
            DataFrame с данными BTC или None
        """
        try:
            # Подключаемся к БД с таймаутом
            conn_params = {
                'host': self.config.database.host,
                'port': self.config.database.port,
                'database': self.config.database.database,
                'user': self.config.database.user,
                'password': self.config.database.password,
                'connect_timeout': 10,  # 10 секунд таймаут на подключение
                'options': '-c statement_timeout=30000'  # 30 секунд таймаут на запросы
            }
            
            logger.info(f"🔌 Подключаемся к БД {self.config.database.host}:{self.config.database.port}...")
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Запрос данных BTC
                    query = """
                        SELECT timestamp, close
                        FROM raw_market_data
                        WHERE symbol = %s
                        AND timestamp >= %s
                        AND timestamp <= %s
                        ORDER BY timestamp
                    """
                    
                    logger.info(f"🔍 Выполняем запрос для {self.btc_symbol} с {min_timestamp} по {max_timestamp}...")
                    cursor.execute(query, (self.btc_symbol, int(min_timestamp), int(max_timestamp)))
                    
                    logger.info("📥 Загружаем результаты из БД...")
                    results = cursor.fetchall()
                    
                    if results:
                        logger.info(f"📊 Преобразуем {len(results)} записей в DataFrame...")
                        df = pd.DataFrame(results)
                        # Преобразуем decimal в float
                        df['close'] = df['close'].astype(float)
                        logger.info(f"✅ Загружено {len(df)} записей {self.btc_symbol} из БД")
                        return df
                    else:
                        logger.warning(f"⚠️ Данные {self.btc_symbol} не найдены в БД")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке данных из БД: {e}")
            return None
            
    def validate_btc_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Проверяет покрытие данных BTC
        
        Args:
            df: DataFrame с данными включая btc_close
            
        Returns:
            Словарь со статистикой покрытия
        """
        if 'btc_close' not in df.columns:
            return {'coverage': 0.0, 'is_synthetic': True}
            
        # Проверяем на синтетические данные
        # Синтетические данные будут иметь высокую корреляцию с исходными ценами
        # и коэффициент будет в диапазоне 0.8-1.2
        # Убеждаемся что оба столбца float
        btc_close = df['btc_close'].astype(float)
        close = df['close'].astype(float)
        ratio = btc_close / close
        is_synthetic = (ratio.min() >= 0.79) and (ratio.max() <= 1.21) and (ratio.std() < 0.1)
        
        # Считаем покрытие
        non_null_count = df['btc_close'].notna().sum()
        coverage = non_null_count / len(df) * 100
        
        stats = {
            'coverage': coverage,
            'is_synthetic': is_synthetic,
            'non_null_count': non_null_count,
            'total_count': len(df)
        }
        
        if is_synthetic:
            logger.warning("⚠️ Обнаружены синтетические данные BTC!")
        else:
            logger.info(f"✅ Используются реальные данные BTC (покрытие: {coverage:.1f}%)")
            
        return stats


def update_feature_engineer_btc_loading():
    """
    Обновляет метод _create_market_features в feature_engineer.py
    для использования BTCDataLoader
    """
    update_code = '''
    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание рыночных признаков"""
        logger.info("🌍 Создание рыночных признаков...")
        
        # Загружаем данные BTC для корреляций
        if 'btc_close' not in df.columns:
            from data.btc_data_loader import BTCDataLoader
            btc_loader = BTCDataLoader(self.config)
            df = btc_loader.load_btc_data(df)
            
            # Проверяем успешность загрузки
            if 'btc_close' not in df.columns:
                logger.warning("⚠️ Не удалось загрузить данные BTC, пропускаем рыночные признаки")
                return df
        
        # Далее идет существующий код для создания признаков...
    '''
    
    logger.info("ℹ️ Для использования BTCDataLoader обновите метод _create_market_features в feature_engineer.py")
    logger.info("ℹ️ Замените синтетическую генерацию данных на вызов BTCDataLoader")