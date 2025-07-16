"""
Загрузчик данных из PostgreSQL
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки данных из PostgreSQL"""
    
    def __init__(self, config: Config):
        self.config = config
        self.connection = None
        self.technical_indicators = [
            'ema_15', 'adx_val', 'adx_plus_di', 'adx_minus_di', 'adx_diff',
            'macd_val', 'macd_signal', 'macd_hist', 'macd_signal_ratio',
            'sar', 'sar_trend', 'sar_distance',
            'ich_tenkan', 'ich_kijun', 'ich_senkou_a', 'ich_senkou_b',
            'ich_chikou', 'ich_tenkan_kijun_signal', 'ich_price_kumo',
            'aroon_up', 'aroon_down', 'aroon_oscillator',
            'rsi_val', 'rsi_ma', 'stoch_k', 'stoch_d', 'stoch_signal',
            'cci', 'williams_r', 'obv', 'obv_slope', 'cmf', 'mfi',
            'volume_ratio', 'atr', 'bb_upper', 'bb_middle', 'bb_lower',
            'bb_width', 'bb_position', 'kc_upper', 'kc_lower',
            'dc_upper', 'dc_lower'
        ]
        
    def connect(self):
        """Установка соединения с БД"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database,
                user=self.config.database.user,
                password=self.config.database.password,
                connect_timeout=10,  # 10 секунд таймаут на подключение
                options='-c statement_timeout=60000'  # 60 секунд таймаут на запросы
            )
            logger.info("✅ Подключение к PostgreSQL установлено")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к БД: {e}")
            raise
            
    def disconnect(self):
        """Закрытие соединения с БД"""
        if self.connection:
            self.connection.close()
            logger.info("📤 Подключение к PostgreSQL закрыто")
            
    def get_available_symbols(self) -> List[str]:
        """Получить список доступных символов"""
        query = """
        SELECT DISTINCT symbol 
        FROM processed_market_data 
        ORDER BY symbol
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            symbols = [row[0] for row in cursor.fetchall()]
            
        # Фильтруем исключенные символы
        symbols = [s for s in symbols if s not in self.config.training.exclude_symbols]
        
        logger.info(f"📋 Найдено {len(symbols)} символов для загрузки")
        return symbols
        
    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Загрузка данных для одного символа"""
        query = """
        SELECT 
            timestamp, symbol, open, high, low, close, volume,
            technical_indicators, buy_expected_return, sell_expected_return
        FROM processed_market_data
        WHERE symbol = %s
        ORDER BY timestamp
        """
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (symbol,))
            data = cursor.fetchall()
            
        if not data:
            logger.warning(f"⚠️ Нет данных для {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Преобразуем decimal.Decimal в float
        for col in ['open', 'high', 'low', 'close', 'volume', 'buy_expected_return', 'sell_expected_return']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Извлекаем технические индикаторы
        df = self._extract_technical_indicators(df)
        
        # Expected returns уже есть в данных как отдельные колонки
        # df = self._extract_expected_returns(df)
        
        logger.info(f"   ✅ Загружено {len(df):,} записей для {symbol}")
        return df
        
    def load_data(self, symbols: Optional[List[str]] = None, 
                  max_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Параллельная загрузка данных для списка символов
        
        Args:
            symbols: Список символов для загрузки (None = все доступные)
            max_workers: Количество потоков для параллельной загрузки
            
        Returns:
            DataFrame с объединенными данными
        """
        if symbols is None:
            symbols = self.get_available_symbols()
            
        if self.config.training.test_mode:
            symbols = self.config.training.test_symbols
            logger.info(f"⚡ Тестовый режим: загружаем только {symbols}")
            
        # Автоматическое определение оптимальных параметров
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if max_workers is None:
            # Для мощных серверов используем больше воркеров
            if cpu_count > 64:
                max_workers = min(50, cpu_count // 2)
                logger.info(f"🚀 Обнаружен мощный сервер: {cpu_count} CPU, используем {max_workers} воркеров")
            else:
                max_workers = min(10, cpu_count)
            
        logger.info(f"⚡ Параллельная загрузка данных для {len(symbols)} символов...")
        
        all_data = []
        
        # Батчевая загрузка - увеличиваем размер для серверов с большой памятью
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb > 100:  # Больше 100 GB свободной памяти
            batch_size = 50
            logger.info(f"💾 Доступно {available_memory_gb:.1f} GB RAM, используем батчи по {batch_size}")
        else:
            batch_size = 10
            
        batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
        
        with tqdm(total=len(batches), desc="Загрузка батчей") as pbar:
            for batch in batches:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    logger.info(f"📦 Загружаем батч из {len(batch)} символов...")
                    futures = {
                        executor.submit(self.load_symbol_data, symbol): symbol
                        for symbol in batch
                    }
                    
                    for future in as_completed(futures, timeout=300):  # 5 минут таймаут на батч
                        try:
                            df = future.result(timeout=60)  # 1 минута таймаут на символ
                            if not df.empty:
                                all_data.append(df)
                                logger.debug(f"✅ Загружен {futures[future]}: {len(df)} записей")
                        except Exception as e:
                            symbol = futures[future]
                            logger.error(f"❌ Ошибка загрузки {symbol}: {e}")
                            
                pbar.update(1)
                
        if not all_data:
            raise ValueError("Не удалось загрузить данные")
            
        # Объединяем все данные
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Загружено всего {len(df):,} записей")
        
        return df
        
    def load_market_data(self, reference_symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Загрузка данных для расчета рыночных корреляций"""
        query = """
        SELECT timestamp, close, volume
        FROM processed_market_data
        WHERE symbol = %s
        ORDER BY timestamp
        """
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (reference_symbol,))
            data = cursor.fetchall()
            
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ Загружены данные {reference_symbol}: {len(df)} записей")
        return df
        
    def _extract_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение технических индикаторов из JSON"""
        logger.info("   📊 Извлечение технических индикаторов из JSON...")
        
        # Извлекаем индикаторы
        indicators_dict = {}
        for idx, row in df.iterrows():
            if row['technical_indicators']:
                # Проверяем, является ли это уже словарем или JSON строкой
                if isinstance(row['technical_indicators'], dict):
                    indicators = row['technical_indicators']
                else:
                    indicators = json.loads(row['technical_indicators'])
                    
                for indicator in self.technical_indicators:
                    if indicator not in indicators_dict:
                        indicators_dict[indicator] = []
                    indicators_dict[indicator].append(indicators.get(indicator))
                    
        # Добавляем колонки в DataFrame
        for indicator, values in indicators_dict.items():
            df[indicator] = values
            
        # Удаляем исходную JSON колонку
        df.drop('technical_indicators', axis=1, inplace=True)
        
        # Проверка
        found_indicators = [ind for ind in self.technical_indicators if ind in df.columns]
        missing_indicators = [ind for ind in self.technical_indicators if ind not in df.columns]
        
        logger.info(f"   ✅ Найдено {len(found_indicators)} из {len(self.technical_indicators)} индикаторов")
        
        if missing_indicators:
            logger.warning(f"   ⚠️ Отсутствуют индикаторы: {missing_indicators[:10]}...")
        
        # Детальная статистика по группам индикаторов
        logger.info(f"   📊 Статистика индикаторов по группам:")
        
        # Трендовые индикаторы
        trend_indicators = ['ema_15', 'adx_val', 'macd_val', 'sar', 'ich_tenkan', 'aroon_up']
        trend_found = sum(1 for ind in trend_indicators if ind in df.columns)
        logger.info(f"      Трендовые: {trend_found}/{len(trend_indicators)}")
        
        # Осцилляторы
        oscillators = ['rsi_val', 'stoch_k', 'cci', 'williams_r', 'mfi']
        osc_found = sum(1 for ind in oscillators if ind in df.columns)
        logger.info(f"      Осцилляторы: {osc_found}/{len(oscillators)}")
        
        # Объемные индикаторы
        volume_indicators = ['obv', 'cmf', 'mfi', 'volume_ratio']
        vol_found = sum(1 for ind in volume_indicators if ind in df.columns)
        logger.info(f"      Объемные: {vol_found}/{len(volume_indicators)}")
        
        # Волатильность
        volatility_indicators = ['atr', 'bb_upper', 'bb_width', 'kc_upper', 'dc_upper']
        volat_found = sum(1 for ind in volatility_indicators if ind in df.columns)
        logger.info(f"      Волатильность: {volat_found}/{len(volatility_indicators)}")
        
        # Примеры значений ключевых индикаторов
        if 'rsi_val' in df.columns:
            logger.info(f"   📊 Проверка диапазонов ключевых индикаторов:")
            logger.info(f"      RSI: min={df['rsi_val'].min():.2f}, max={df['rsi_val'].max():.2f}, mean={df['rsi_val'].mean():.2f}")
            
        if 'macd_hist' in df.columns:
            logger.info(f"      MACD hist: min={df['macd_hist'].min():.4f}, max={df['macd_hist'].max():.4f}, mean={df['macd_hist'].mean():.4f}")
            
        if 'adx_val' in df.columns:
            logger.info(f"      ADX: min={df['adx_val'].min():.2f}, max={df['adx_val'].max():.2f}, mean={df['adx_val'].mean():.2f}")
            
        if 'bb_position' in df.columns:
            logger.info(f"      BB position: min={df['bb_position'].min():.2f}, max={df['bb_position'].max():.2f}")
            
        if 'volume_ratio' in df.columns:
            logger.info(f"      Volume ratio: min={df['volume_ratio'].min():.2f}, max={df['volume_ratio'].max():.2f}")
            
        return df
        
    def _extract_expected_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение expected returns из JSON"""
        if 'expected_returns' not in df.columns:
            logger.warning("   ⚠️ Колонка expected_returns не найдена")
            return df
            
        # Извлекаем buy и sell returns
        buy_returns = []
        sell_returns = []
        
        for idx, row in df.iterrows():
            if row['expected_returns']:
                returns = json.loads(row['expected_returns'])
                buy_returns.append(returns.get('buy_expected_return'))
                sell_returns.append(returns.get('sell_expected_return'))
            else:
                buy_returns.append(None)
                sell_returns.append(None)
                
        df['buy_expected_return'] = buy_returns
        df['sell_expected_return'] = sell_returns
        
        # Удаляем исходную JSON колонку
        df.drop('expected_returns', axis=1, inplace=True)
        
        logger.info("   ✅ Целевые колонки найдены и будут использованы ТОЛЬКО как targets")
        
        return df
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Валидация загруженных данных"""
        logger.info("🔍 Валидация данных...")
        
        # Проверка размера
        if len(df) < 1000:
            logger.warning(f"⚠️ Слишком мало данных: {len(df)} записей")
            return False
            
        # Проверка на пропуски
        missing_counts = df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(df) * 0.1]
        if not high_missing.empty:
            logger.warning(f"⚠️ Колонки с большим количеством пропусков: {high_missing.index.tolist()}")
            
        # Проверка целевых переменных
        if 'buy_expected_return' not in df.columns or 'sell_expected_return' not in df.columns:
            logger.error("❌ Отсутствуют целевые переменные")
            return False
            
        # Проверка на бесконечности
        numeric_df = df.select_dtypes(include=[np.number])
        inf_mask = np.isinf(numeric_df).any()
        inf_columns = numeric_df.columns[inf_mask].tolist()
        if inf_columns:
            logger.warning(f"⚠️ Колонки с бесконечными значениями: {inf_columns}")
            
        logger.info("✅ Валидация пройдена")
        return True