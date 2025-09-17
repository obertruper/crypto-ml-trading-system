"""
Загрузчик данных из PostgreSQL
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import pickle
from pathlib import Path
import hashlib
from tqdm import tqdm
from contextlib import contextmanager
import os

from utils.logger import get_logger
from utils.config_validator import validate_dataframe

class CryptoDataLoader:
    """Загрузчик исторических данных криптовалютных фьючерсов"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("DataLoader")
        self.cache_dir = Path(config.get('performance', {}).get('cache_dir', 'cache'))
        self.cache_dir.mkdir(exist_ok=True)
        self.engine = self._create_engine()
    
    def get_data_stats(self) -> Dict:
        """Получение статистики по данным в БД"""
        with self.engine.connect() as conn:
            # Общее количество записей
            total_records = conn.execute(
                text("SELECT COUNT(*) FROM raw_market_data")
            ).scalar()
            
            # Количество уникальных символов
            unique_symbols = conn.execute(
                text("SELECT COUNT(DISTINCT symbol) FROM raw_market_data")
            ).scalar()
            
            # Диапазон дат (timestamp хранится как Unix timestamp в миллисекундах)
            date_range = conn.execute(
                text("""
                    SELECT to_timestamp(MIN(timestamp)/1000)::date as min_date, 
                           to_timestamp(MAX(timestamp)/1000)::date as max_date 
                    FROM raw_market_data
                """)
            ).fetchone()
            
            return {
                'total_records': total_records or 0,
                'unique_symbols': unique_symbols or 0,
                'date_range': {
                    'min': str(date_range.min_date) if date_range else 'N/A',
                    'max': str(date_range.max_date) if date_range else 'N/A'
                }
            }
        
    def _create_engine(self):
        """Создание SQLAlchemy engine с поддержкой переменных окружения"""
        db_config = self.config['database'].copy()
        
        # ИСПРАВЛЕННАЯ обработка переменных окружения с безопасным парсингом
        for key, value in db_config.items():
            if isinstance(value, str) and value.startswith('${'):
                try:
                    # Проверяем наличие разделителя ':'
                    if ':' in value:
                        # Извлечение имени переменной и значения по умолчанию
                        var_name = value.split(':')[0][2:]
                        default_value = value.split(':')[1].rstrip('}')
                    else:
                        # Если разделителя нет, используем всю строку как имя переменной
                        var_name = value[2:-1]  # Убираем ${ и }
                        default_value = None
                    
                    # Получаем значение из окружения
                    env_value = os.getenv(var_name, default_value)
                    
                    # Если ни в окружении, ни в дефолте нет значения
                    if env_value is None:
                        self.logger.warning(f"Переменная окружения {var_name} не найдена и нет значения по умолчанию")
                        env_value = ""  # Используем пустую строку вместо None
                    
                    db_config[key] = env_value
                    
                except Exception as e:
                    self.logger.error(f"Ошибка парсинга переменной окружения {value}: {e}")
                    # Оставляем исходное значение
                    db_config[key] = value
        
        # Преобразование порта в int
        db_config['port'] = int(db_config['port'])
        
        # Создание connection string
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        self.logger.info(f"Подключение к БД {db_config['database']} на {db_config['host']}:{db_config['port']}...")
        
        # Создание engine с дополнительными параметрами для стабильности
        engine = create_engine(
            connection_string,
            pool_size=db_config.get('pool_size', 10),
            max_overflow=db_config.get('max_overflow', 20),
            pool_pre_ping=True,  # Проверка соединения перед использованием
            echo=False  # Отключаем SQL логирование
        )
        
        # Тест подключения
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            self.logger.info("✅ Подключение к БД успешно установлено")
        except Exception as e:
            self.logger.error(f"❌ Ошибка подключения к БД: {e}")
            raise
            
        return engine
    
    def _get_cache_key(self, symbols: List[str], start_date: str, end_date: str) -> str:
        """Генерация ключа кэша"""
        key_string = f"{','.join(sorted(symbols))}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Загрузка данных из кэша"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists() and self.config.get('performance', {}).get('cache_features', True):
            self.logger.info(f"Загрузка данных из кэша: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки кэша: {e}")
                return None
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Безопасное сохранение данных в кэш с ограничением размера"""
        if not self.config.get('performance', {}).get('cache_features', True):
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Проверка размера данных
        data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        max_cache_size_mb = self.config.get('performance', {}).get('max_cache_size_mb', 500)
        
        if data_size_mb > max_cache_size_mb:
            self.logger.warning(f"Данные слишком большие для кэша: {data_size_mb:.1f}MB > {max_cache_size_mb}MB")
            return
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"Кэш сохранен: {cache_file} ({data_size_mb:.1f}MB)")
            
            # Очистка старых файлов кэша
            self._cleanup_old_cache_files()
            
        except Exception as e:
            self.logger.warning(f"Ошибка сохранения кэша: {e}")
    
    def _cleanup_old_cache_files(self):
        """Очистка старых файлов кэша"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        max_cache_files = self.config.get('performance', {}).get('max_cache_files', 10)
        
        if len(cache_files) > max_cache_files:
            # Сортируем по времени модификации и удаляем старые
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            files_to_remove = cache_files[:-max_cache_files]
            
            for file in files_to_remove:
                file.unlink()
                self.logger.info(f"Удален старый кэш файл: {file}")
    
    def load_data(self, 
                  symbols: Optional[List[str]] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  interval_minutes: Optional[float] = None) -> pd.DataFrame:
        """ИСПРАВЛЕННАЯ загрузка данных с поддержкой symbols: all и различных интервалов"""
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильная обработка symbols
        if symbols is None:
            config_symbols = self.config['data']['symbols']
            if config_symbols == 'all':
                self.logger.info("Загрузка всех доступных символов...")
                symbols = self.get_available_symbols()
            elif isinstance(config_symbols, list):
                symbols = config_symbols
            elif isinstance(config_symbols, str):
                symbols = [config_symbols]  # Один символ как строка
            else:
                raise ValueError(f"Неподдерживаемый тип symbols: {type(config_symbols)}")
        
        start_date = start_date or self.config['data']['start_date']
        end_date = end_date or self.config['data']['end_date']
        
        # Определяем интервал (по умолчанию из конфига)
        if interval_minutes is None:
            interval_minutes = self.config['data'].get('interval_minutes', 15)
        
        self.logger.info(f"📊 Загрузка данных с интервалом {interval_minutes} минут")

        # Поддержка динамических границ дат: 'earliest'/'min' и 'latest'/'max'
        try:
            if isinstance(end_date, str) and end_date.strip().lower() in {"latest", "max", "now", "today"}:
                with self.engine.connect() as conn:
                    end_row = conn.execute(text(f"SELECT MAX(datetime) FROM raw_market_data WHERE market_type IN ('futures', 'spot') AND interval_minutes={interval_minutes}")).fetchone()
                    if end_row and end_row[0] is not None:
                        end_date = str(end_row[0])
                        self.logger.info(f"⏱️ end_date=latest → используем MAX(datetime) из БД: {end_date}")
            if isinstance(start_date, str) and start_date.strip().lower() in {"earliest", "min"}:
                with self.engine.connect() as conn:
                    start_row = conn.execute(text(f"SELECT MIN(datetime) FROM raw_market_data WHERE market_type IN ('futures', 'spot') AND interval_minutes={interval_minutes}")).fetchone()
                    if start_row and start_row[0] is not None:
                        start_date = str(start_row[0])
                        self.logger.info(f"⏱️ start_date=earliest → используем MIN(datetime) из БД: {start_date}")
        except Exception as e_bounds:
            self.logger.warning(f"⚠️ Не удалось определить динамические границы дат: {e_bounds}")
        
        # Валидация параметров
        self._validate_symbols(symbols)
        self._validate_dates(start_date, end_date)
        
        cache_key = self._get_cache_key(symbols, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        self.logger.start_stage("data_loading", symbols_count=len(symbols))

        try:
            # Диагностика: поэтапные подсчёты записей под фильтрами
            try:
                with self.engine.connect() as conn:
                    total_all = conn.execute(text("SELECT COUNT(*) FROM raw_market_data")).scalar() or 0
                    total_interval = conn.execute(text(
                        """
                        SELECT COUNT(*) FROM raw_market_data
                        WHERE market_type IN ('futures', 'spot') AND interval_minutes = :interval
                        """
                    ), {"interval": interval_minutes}).scalar() or 0
                    total_date = conn.execute(text(
                        """
                        SELECT COUNT(*) FROM raw_market_data
                        WHERE market_type IN ('futures', 'spot') AND interval_minutes = :interval
                          AND datetime >= :start_date AND datetime <= :end_date
                        """
                    ), {"start_date": start_date, "end_date": end_date, "interval": interval_minutes}).scalar() or 0
                    total_symbols = conn.execute(text(
                        """
                        SELECT COUNT(*) FROM raw_market_data
                        WHERE market_type IN ('futures', 'spot') AND interval_minutes = :interval
                          AND datetime >= :start_date AND datetime <= :end_date
                          AND symbol = ANY(:symbols)
                        """
                    ), {"start_date": start_date, "end_date": end_date, "symbols": symbols, "interval": interval_minutes}).scalar() or 0
                self.logger.info("📏 Диагностика фильтров загрузки:")
                self.logger.info(f"   - Всего в таблице: {total_all:,}")
                self.logger.info(f"   - Интервал {interval_minutes} мин: {total_interval:,}")
                self.logger.info(f"   - С учётом дат [{start_date}..{end_date}]: {total_date:,}")
                self.logger.info(f"   - С учётом символов ({len(symbols)} шт.): {total_symbols:,}")
            except Exception as e_diag:
                self.logger.warning(f"⚠️ Не удалось выполнить диагностику фильтров: {e_diag}")

            # SQL запрос для SQLAlchemy (поддерживает разные интервалы и типы рынка)
            query = text("""
            SELECT 
                id,
                symbol,
                timestamp,
                datetime,
                open,
                high,
                low,
                close,
                volume,
                turnover
            FROM raw_market_data
            WHERE 
                symbol = ANY(:symbols)
                AND datetime >= :start_date
                AND datetime <= :end_date
                AND market_type IN ('futures', 'spot')
                AND interval_minutes = :interval_minutes
            ORDER BY symbol, datetime
            """)
            
            chunk_size = 100000
            chunks = []
            
            # Используем SQLAlchemy для всех операций с БД
            
            with self.engine.connect() as conn:
                # Подсчет общего количества записей
                count_query = text("""
                SELECT COUNT(*) 
                FROM raw_market_data 
                WHERE 
                    symbol = ANY(:symbols)
                    AND datetime >= :start_date
                    AND datetime <= :end_date
                    AND market_type IN ('futures', 'spot')
                    AND interval_minutes = :interval_minutes
                """)

                result = conn.execute(count_query, {
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'interval_minutes': interval_minutes
                })
                total_records = result.scalar()

                self.logger.info(f"Загрузка {total_records:,} записей (после всех фильтров)...")

                # Диагностика распределения по символам (топ-5)
                try:
                    per_symbol = conn.execute(text(
                        """
                        SELECT symbol, COUNT(*) cnt
                        FROM raw_market_data
                        WHERE market_type IN ('futures', 'spot') AND interval_minutes = :interval_minutes
                          AND datetime >= :start_date AND datetime <= :end_date
                          AND symbol = ANY(:symbols)
                        GROUP BY symbol
                        ORDER BY cnt DESC
                        LIMIT 5
                        """
                    ), {"start_date": start_date, "end_date": end_date, "symbols": symbols, "interval_minutes": interval_minutes}).fetchall()
                    if per_symbol:
                        top = ", ".join([f"{row.symbol}:{row.cnt:,}" for row in per_symbol])
                        self.logger.info(f"   - Топ-5 по символам: {top}")
                except Exception as e_sym:
                    self.logger.warning(f"⚠️ Диагностика по символам недоступна: {e_sym}")
                
                # Загрузка данных через pandas
                with tqdm(total=total_records, desc="Загрузка данных") as pbar:
                    for chunk in pd.read_sql(
                        query,
                        conn,
                        params={
                            "symbols": symbols,
                            "start_date": start_date,
                            "end_date": end_date,
                            "interval_minutes": interval_minutes
                        },
                        chunksize=chunk_size
                    ):
                        chunks.append(chunk)
                        pbar.update(len(chunk))
            
            df = pd.concat(chunks, ignore_index=True)
            
            # Валидация загруженных данных
            required_columns = ['id', 'symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']
            validate_dataframe(df, required_columns, "загруженные рыночные данные")
            
            # Обработка типов данных
            df['datetime'] = pd.to_datetime(df['datetime'])
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            df[numeric_columns] = df[numeric_columns].astype(np.float32)
            
            self._log_data_statistics(df)
            self._save_to_cache(df, cache_key)
            
            self.logger.end_stage("data_loading", records=len(df))
            
            return df
            
        except Exception as e:
            self.logger.log_error(e, "load_data")
            raise
    
    def _log_data_statistics(self, df: pd.DataFrame):
        """Логирование статистики по загруженным данным"""
        self.logger.info("📊 Статистика загруженных данных:")
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            date_range = (
                symbol_data['datetime'].min().strftime('%Y-%m-%d'),
                symbol_data['datetime'].max().strftime('%Y-%m-%d')
            )
            
            self.logger.log_data_info(
                symbol=symbol,
                records=len(symbol_data),
                date_range=date_range
            )
    
    def load_symbol_data(self, symbol: str, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """Загрузка данных для одного символа"""
        return self.load_data([symbol], start_date, end_date)
    
    def get_available_symbols(self) -> List[str]:
        """Получение списка доступных символов в БД"""
        query = """
        SELECT DISTINCT symbol 
        FROM raw_market_data 
        WHERE market_type = 'futures'
        ORDER BY symbol
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            symbols = [row[0] for row in result]
        
        self.logger.info(f"Найдено {len(symbols)} доступных символов")
        return symbols
    
    def get_date_range(self, symbol: Optional[str] = None) -> Tuple[datetime, datetime]:
        """Получение диапазона дат для символа или всех данных"""
        if symbol:
            query = """
            SELECT MIN(datetime), MAX(datetime)
            FROM raw_market_data
            WHERE symbol = :symbol AND market_type = 'futures'
            """
            params = {"symbol": symbol}
        else:
            query = """
            SELECT MIN(datetime), MAX(datetime)
            FROM raw_market_data
            WHERE market_type = 'futures'
            """
            params = {}
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params).fetchone()
            
        return result[0], result[1]
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Проверка качества данных"""
        self.logger.start_stage("data_validation")
        
        quality_report = {}
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('datetime')
            
            report = {
                'total_records': len(symbol_data),
                'missing_values': symbol_data.isnull().sum().to_dict(),
                'duplicates': symbol_data.duplicated(subset=['datetime']).sum(),
                'gaps': 0,
                'anomalies': {}
            }
            
            # Проверка пропусков во времени
            expected_freq = pd.Timedelta(minutes=15)
            time_diff = symbol_data['datetime'].diff()
            gaps = time_diff[time_diff > expected_freq * 1.5]
            report['gaps'] = len(gaps)
            
            if len(gaps) > 0:
                self.logger.warning(
                    f"Обнаружено {len(gaps)} пропусков в данных {symbol}"
                )
            
            # Проверка нулевого объема
            zero_volume = (symbol_data['volume'] == 0).sum()
            if zero_volume > 0:
                report['anomalies']['zero_volume'] = zero_volume
            
            # Проверка отсутствия движения цены
            no_movement = (
                (symbol_data['open'] == symbol_data['high']) & 
                (symbol_data['high'] == symbol_data['low']) & 
                (symbol_data['low'] == symbol_data['close'])
            ).sum()
            if no_movement > 0:
                report['anomalies']['no_price_movement'] = no_movement
            
            # Проверка экстремальных изменений цены
            price_change = symbol_data['close'].pct_change()
            extreme_changes = (price_change.abs() > 0.2).sum()
            if extreme_changes > 0:
                report['anomalies']['extreme_price_changes'] = extreme_changes
            
            quality_report[symbol] = report
        
        self.logger.end_stage("data_validation", issues_found=sum(
            len(r['anomalies']) for r in quality_report.values()
        ))
        
        return quality_report
    
    def resample_data(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Изменение временного интервала данных"""
        self.logger.info(f"Ресемплирование данных к интервалу: {target_interval}")
        
        resampled_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data.set_index('datetime', inplace=True)
            
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'turnover': 'sum'
            }
            
            resampled = symbol_data.resample(target_interval).agg(agg_rules)
            resampled['symbol'] = symbol
            resampled.reset_index(inplace=True)
            
            resampled_dfs.append(resampled)
        
        return pd.concat(resampled_dfs, ignore_index=True)
    
    def _validate_symbols(self, symbols: List[str]):
        """Валидация символов"""
        if not symbols:
            raise ValueError("Список символов не может быть пустым")
        
        for symbol in symbols:
            if not isinstance(symbol, str) or not symbol.replace('USDT', '').replace('1000', '').isalnum():
                raise ValueError(f"Некорректный символ: {symbol}")
    
    def _validate_dates(self, start_date: str, end_date: str):
        """Валидация дат"""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if start >= end:
                raise ValueError("Начальная дата должна быть раньше конечной")
        except Exception as e:
            raise ValueError(f"Некорректный формат даты: {e}")
    
    def get_data_completeness(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Проверка полноты данных"""
        query = """
        SELECT 
            symbol,
            COUNT(*) as actual_records,
            MIN(datetime) as first_record,
            MAX(datetime) as last_record
        FROM raw_market_data 
        WHERE symbol = ANY(:symbols)
        AND datetime BETWEEN :start_date AND :end_date
        AND market_type = 'futures'
        GROUP BY symbol
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date
            })
            
            completeness = {}
            for row in result:
                symbol, actual, first, last = row
                
                # Вычисляем ожидаемое количество записей (15-минутные интервалы)
                time_diff = (last - first).total_seconds()
                expected_records = int(time_diff / (15 * 60)) + 1
                
                completeness[symbol] = {
                    'actual_records': actual,
                    'expected_records': expected_records,
                    'completeness_pct': (actual / expected_records) * 100 if expected_records > 0 else 0,
                    'first_record': first,
                    'last_record': last
                }
        
        return completeness
