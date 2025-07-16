"""
Загрузчик данных из PostgreSQL для Transformer v3
Адаптировано из train_universal_transformer.py
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, Json
import logging
from typing import Dict, Optional, List
import time

from config import Config, EXCLUDE_SYMBOLS

logger = logging.getLogger(__name__)


class DataLoader:
    """Загрузчик данных из PostgreSQL"""
    
    def __init__(self, config: Config):
        self.config = config
        self.connection = None
        self._connect()
        
    def _connect(self):
        """Создает подключение к БД"""
        try:
            self.connection = psycopg2.connect(**self.config.database.connection_params)
            self.connection.autocommit = True
            logger.info("✅ Подключение к PostgreSQL установлено")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
            raise
            
    def disconnect(self):
        """Закрывает подключение к БД"""
        if self.connection:
            self.connection.close()
            logger.info("📤 Подключение к PostgreSQL закрыто")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        
    def fetch_dataframe(self, query: str, params=None) -> pd.DataFrame:
        """Выполняет запрос и возвращает результат как DataFrame"""
        try:
            return pd.read_sql_query(query, self.connection, params=params)
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса: {e}")
            raise
            
    def load_data(self, 
                  symbols: Optional[List[str]] = None,
                  limit: Optional[int] = None) -> pd.DataFrame:
        """
        Загрузка данных из БД
        
        Args:
            symbols: Список символов для загрузки (если None - все символы)
            limit: Ограничение количества записей (для тестирования)
            
        Returns:
            DataFrame с данными
        """
        logger.info("📊 Загрузка данных из PostgreSQL...")
        start_time = time.time()
        
        # Базовый запрос
        query = """
        SELECT 
            p.symbol, 
            p.timestamp, 
            p.datetime,
            p.technical_indicators,
            p.buy_expected_return,
            p.sell_expected_return,
            p.is_long_entry,
            p.is_short_entry,
            p.open, 
            p.high, 
            p.low, 
            p.close, 
            p.volume,
            r.market_type
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
        """
        
        # Добавляем фильтр по символам
        conditions = []
        
        # Исключаем тестовые символы
        if EXCLUDE_SYMBOLS:
            exclude_list = "', '".join(EXCLUDE_SYMBOLS)
            conditions.append(f"p.symbol NOT IN ('{exclude_list}')")
            
        # Фильтр по конкретным символам
        if symbols:
            symbols_list = "', '".join(symbols)
            conditions.append(f"p.symbol IN ('{symbols_list}')")
            
        if conditions:
            query += " AND " + " AND ".join(conditions)
            
        query += " ORDER BY p.symbol, p.timestamp"
        
        # Добавляем лимит для тестирования
        if limit:
            query += f" LIMIT {limit}"
            
        # Выполняем запрос
        df = self.fetch_dataframe(query)
        
        load_time = time.time() - start_time
        logger.info(f"✅ Загружено {len(df):,} записей за {load_time:.2f} сек")
        
        if len(df) == 0:
            raise ValueError("Нет данных для обучения\!")
            
        # Статистика по символам
        self._log_data_statistics(df)
        
        return df
        
    def load_symbols_list(self) -> List[str]:
        """Загрузка списка доступных символов"""
        if EXCLUDE_SYMBOLS:
            # Создаем плейсхолдеры для IN оператора
            placeholders = ','.join(['%s'] * len(EXCLUDE_SYMBOLS))
            query = f"""
            SELECT DISTINCT p.symbol
            FROM processed_market_data p
            JOIN raw_market_data r ON p.raw_data_id = r.id
            WHERE p.technical_indicators IS NOT NULL
              AND r.market_type = 'futures'
              AND p.symbol NOT IN ({placeholders})
            ORDER BY p.symbol
            """
            df = self.fetch_dataframe(query, EXCLUDE_SYMBOLS)
        else:
            query = """
            SELECT DISTINCT p.symbol
            FROM processed_market_data p
            JOIN raw_market_data r ON p.raw_data_id = r.id
            WHERE p.technical_indicators IS NOT NULL
              AND r.market_type = 'futures'
            ORDER BY p.symbol
            """
            df = self.fetch_dataframe(query)
        
        return df['symbol'].tolist()
        
    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Загрузка данных для конкретного символа"""
        return self.load_data(symbols=[symbol])
        
    def load_symbol_updates(self, symbol: str, after_timestamp: int) -> pd.DataFrame:
        """Загрузка только новых данных для символа после указанного timestamp"""
        logger.info(f"📊 Проверка обновлений для {symbol} после {pd.to_datetime(after_timestamp, unit='ms')}")
        
        query = """
        SELECT 
            p.symbol, 
            p.timestamp, 
            p.datetime,
            p.technical_indicators,
            p.buy_expected_return,
            p.sell_expected_return,
            p.is_long_entry,
            p.is_short_entry,
            p.open, 
            p.high, 
            p.low, 
            p.close, 
            p.volume,
            r.market_type
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
          AND p.symbol = %s
          AND p.timestamp > %s
        ORDER BY p.timestamp
        """
        
        df = self.fetch_dataframe(query, (symbol, after_timestamp))
        
        if len(df) > 0:
            logger.info(f"🆕 Найдено {len(df)} новых записей для {symbol}")
        
        return df
        
    def _log_data_statistics(self, df: pd.DataFrame):
        """Логирование статистики по данным"""
        symbol_counts = df['symbol'].value_counts()
        
        logger.info("📊 Распределение по символам:")
        logger.info(f"   Всего символов: {len(symbol_counts)}")
        logger.info(f"   Среднее записей на символ: {symbol_counts.mean():.0f}")
        
        # Топ-10 символов
        logger.info("   Топ-10 символов:")
        for symbol, count in symbol_counts.head(10).items():
            logger.info(f"     {symbol}: {count:,} записей")
            
        # Проверка expected returns
        if 'buy_expected_return' in df.columns:
            buy_stats = df['buy_expected_return'].describe()
            sell_stats = df['sell_expected_return'].describe()
            
            logger.info("\n📊 Статистика expected returns:")
            logger.info(f"   Buy - mean: {buy_stats['mean']:.3f}%, std: {buy_stats['std']:.3f}%")
            logger.info(f"   Buy - min/max: {buy_stats['min']:.3f}% / {buy_stats['max']:.3f}%")
            logger.info(f"   Sell - mean: {sell_stats['mean']:.3f}%, std: {sell_stats['std']:.3f}%")
            logger.info(f"   Sell - min/max: {sell_stats['min']:.3f}% / {sell_stats['max']:.3f}%")
            
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Проверка качества данных"""
        quality_report = {
            'total_records': len(df),
            'unique_symbols': df['symbol'].nunique(),
            'date_range': {
                'start': df['datetime'].min(),
                'end': df['datetime'].max()
            },
            'missing_values': {},
            'outliers': {}
        }
        
        # Проверка пропущенных значений
        for col in ['buy_expected_return', 'sell_expected_return']:
            if col in df.columns:
                missing = df[col].isna().sum()
                quality_report['missing_values'][col] = missing
                
        # Проверка выбросов
        for col in ['buy_expected_return', 'sell_expected_return']:
            if col in df.columns:
                outliers = ((df[col] < -1.1) | (df[col] > 5.8)).sum()
                quality_report['outliers'][col] = outliers
                
        return quality_report