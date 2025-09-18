#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для выгрузки исторических данных с Bybit в PostgreSQL
Сохраняет данные в БД для дальнейшего обучения модели
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError, wait, FIRST_COMPLETED
import threading
import signal
import sys
from contextlib import contextmanager
from psycopg2 import pool

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Глобальный флаг для graceful shutdown
shutdown_flag = threading.Event()

def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown"""
    logger.info("\n⚠️ Получен сигнал прерывания. Завершаем работу...")
    shutdown_flag.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class PostgreSQLManager:
    """
    Менеджер для работы с PostgreSQL с поддержкой многопоточности
    """

    def __init__(self, db_config: dict, max_connections: int = 30):
        """
        Инициализация подключения к PostgreSQL

        Args:
            db_config: Конфигурация подключения к БД
            max_connections: Максимальное количество соединений в пуле
        """
        self.db_config = db_config.copy()
        # Удаляем пустой пароль
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
        self.connection = None
        self.connection_pool = None
        self.max_connections = max_connections

    def _sanitize_db_config(self) -> dict:
        """Преобразует config.yaml -> psycopg2 dsn (dbname,user,password,host,port,...)"""
        cfg = self.db_config.copy()
        params = {}
        if 'dbname' in cfg:
            params['dbname'] = cfg['dbname']
        elif 'database' in cfg:
            params['dbname'] = cfg['database']
        for k in ('user', 'password', 'host', 'port', 'options', 'sslmode'):
            if k in cfg and cfg[k] not in (None, ''):
                params[k] = cfg[k]
        if 'port' in params:
            try:
                params['port'] = int(params['port'])
            except Exception:
                params.pop('port', None)
        return params

    def connect(self):
        """Создает пул подключений к БД для многопоточности"""
        try:
            # Создаем пул соединений
            conn_params = self._sanitize_db_config()
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                1,
                self.max_connections,
                **conn_params
            )
            
            # Основное соединение для однопоточных операций
            self.connection = psycopg2.connect(**conn_params)
            self.connection.autocommit = True
            
            logger.info(f"✅ Пул подключений к PostgreSQL создан (max: {self.max_connections})")
        except Exception as e:
            logger.error(f"❌ Ошибка создания пула подключений: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для получения соединения из пула"""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            conn.autocommit = True
            yield conn
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def disconnect(self):
        """Закрывает все подключения к БД"""
        if self.connection:
            self.connection.close()
        if self.connection_pool:
            self.connection_pool.closeall()
        logger.info("📤 Все подключения к PostgreSQL закрыты")

    def execute_query(self, query: str, params=None, fetch=False):
        """
        Выполняет SQL запрос

        Args:
            query: SQL запрос
            params: Параметры запроса
            fetch: Нужно ли возвращать результат

        Returns:
            Результат запроса или None
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса: {e}")
            logger.error(f"Запрос: {query}")
            raise

    def create_tables(self):
        """Создает необходимые таблицы"""

        logger.info("🔧 Создание таблиц в PostgreSQL...")

        # Таблица для сырых рыночных данных
        create_raw_data_table = """
        CREATE TABLE IF NOT EXISTS raw_market_data (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp BIGINT NOT NULL,
            datetime TIMESTAMP NOT NULL,
            open DECIMAL(20, 8) NOT NULL,
            high DECIMAL(20, 8) NOT NULL,
            low DECIMAL(20, 8) NOT NULL,
            close DECIMAL(20, 8) NOT NULL,
            volume DECIMAL(20, 8) NOT NULL,
            turnover DECIMAL(20, 8) DEFAULT 0,
            interval_minutes INTEGER NOT NULL DEFAULT 15,
            market_type VARCHAR(20) DEFAULT 'spot',  -- Тип рынка: spot или futures
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(symbol, timestamp, interval_minutes)
        );
        """

        # Индексы для оптимизации
        create_indexes = """
        CREATE INDEX IF NOT EXISTS idx_raw_market_data_symbol_timestamp 
        ON raw_market_data(symbol, timestamp);

        CREATE INDEX IF NOT EXISTS idx_raw_market_data_datetime 
        ON raw_market_data(datetime);

        CREATE INDEX IF NOT EXISTS idx_raw_market_data_symbol_datetime 
        ON raw_market_data(symbol, datetime);
        """

        # Таблица для обработанных данных с индикаторами
        create_processed_data_table = """
        CREATE TABLE IF NOT EXISTS processed_market_data (
            id BIGSERIAL PRIMARY KEY,
            raw_data_id BIGINT REFERENCES raw_market_data(id),
            symbol VARCHAR(20) NOT NULL,
            timestamp BIGINT NOT NULL,
            datetime TIMESTAMP NOT NULL,

            -- Базовые OHLCV
            open DECIMAL(20, 8) NOT NULL,
            high DECIMAL(20, 8) NOT NULL,
            low DECIMAL(20, 8) NOT NULL,
            close DECIMAL(20, 8) NOT NULL,
            volume DECIMAL(20, 8) NOT NULL,

            -- Технические индикаторы (JSON для гибкости)
            technical_indicators JSONB,

            -- Целевые переменные (метки)
            buy_profit_target INTEGER DEFAULT 0,
            buy_loss_target INTEGER DEFAULT 0,
            sell_profit_target INTEGER DEFAULT 0,
            sell_loss_target INTEGER DEFAULT 0,

            -- Метаданные
            processing_version VARCHAR(10) DEFAULT '1.0',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),

            UNIQUE(symbol, timestamp)
        );
        """

        # Индексы для processed_data
        create_processed_indexes = """
        CREATE INDEX IF NOT EXISTS idx_processed_market_data_symbol_timestamp 
        ON processed_market_data(symbol, timestamp);

        CREATE INDEX IF NOT EXISTS idx_processed_market_data_targets 
        ON processed_market_data(buy_profit_target, buy_loss_target, sell_profit_target, sell_loss_target);

        CREATE INDEX IF NOT EXISTS idx_processed_technical_indicators 
        ON processed_market_data USING GIN (technical_indicators);
        """

        # Таблица для метаданных модели
        create_model_metadata_table = """
        CREATE TABLE IF NOT EXISTS model_metadata (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            version VARCHAR(20) NOT NULL,
            feature_columns JSONB,
            training_config JSONB,
            performance_metrics JSONB,
            file_path VARCHAR(500),
            created_at TIMESTAMP DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE
        );
        """

        # Таблица для хранения последовательностей
        create_sequences_table = """
        CREATE TABLE IF NOT EXISTS training_sequences (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            sequence_start_timestamp BIGINT NOT NULL,
            sequence_end_timestamp BIGINT NOT NULL,
            sequence_length INTEGER NOT NULL,
            features JSONB NOT NULL,
            buy_profit_target INTEGER DEFAULT 0,
            buy_loss_target INTEGER DEFAULT 0,
            sell_profit_target INTEGER DEFAULT 0,
            sell_loss_target INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """

        # Выполняем создание таблиц
        queries = [
            create_raw_data_table,
            create_indexes,
            create_processed_data_table,
            create_processed_indexes,
            create_model_metadata_table,
            create_sequences_table
        ]

        for query in queries:
            self.execute_query(query)

        logger.info("✅ Все таблицы созданы успешно")


class BybitDataDownloader:
    """
    Класс для загрузки исторических данных с Bybit в PostgreSQL
    Поддерживает загрузку как спотовых, так и фьючерсных данных
    """

    def __init__(self, db_manager: PostgreSQLManager, market_type='futures'):
        self.base_url = "https://api.bybit.com"
        self.session = requests.Session()
        # Настройка таймаутов для requests
        self.session.timeout = 30  # 30 секунд таймаут
        self.db = db_manager
        self.market_type = market_type  # 'spot' или 'futures'
        # Thread-local storage для прогресс-баров
        self.thread_local = threading.local()

    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000):
        """
        Получает исторические свечи с Bybit

        Args:
            symbol: Торговая пара (например, BTCUSDT)
            interval: Таймфрейм (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            start_time: Время начала в миллисекундах
            end_time: Время окончания в миллисекундах
            limit: Максимальное количество свечей за запрос (макс 1000)

        Returns:
            list: Список свечей
        """

        url = f"{self.base_url}/v5/market/kline"
        params = {
            'category': 'linear' if self.market_type == 'futures' else 'spot',  # linear для USDT perpetual фьючерсов
            'symbol': symbol,
            'interval': interval,
            'start': start_time,
            'end': end_time,
            'limit': limit
        }

        try:
            # Проверяем флаг остановки
            if shutdown_flag.is_set():
                return []
                
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('retCode') == 0:
                return data.get('result', {}).get('list', [])
            else:
                logger.error(f"Ошибка API: {data.get('retMsg', 'Unknown error')}")
                return []

        except requests.exceptions.Timeout:
            logger.error(f"Таймаут запроса для {symbol}")
            return []
        except Exception as e:
            logger.error(f"Ошибка запроса: {e}")
            return []

    def insert_raw_data_batch(self, symbol: str, klines_data: list, interval_minutes: int = 15):
        """
        Вставляет данные свечей в БД батчом

        Args:
            symbol: Символ торговой пары
            klines_data: Список данных свечей
            interval_minutes: Интервал в минутах
        """

        if not klines_data:
            return 0

        # Подготавливаем данные для вставки
        values_to_insert = []
        for kline in klines_data:
            timestamp = int(kline[0])
            dt = datetime.fromtimestamp(timestamp / 1000)

            values_to_insert.append((
                symbol,
                timestamp,
                dt,
                float(kline[1]),  # open
                float(kline[2]),  # high
                float(kline[3]),  # low
                float(kline[4]),  # close
                float(kline[5]),  # volume
                float(kline[6]) if len(kline) > 6 else 0.0,  # turnover
                interval_minutes,
                self.market_type  # Добавляем тип рынка
            ))

        # SQL запрос для вставки с ON CONFLICT
        insert_query = """
        INSERT INTO raw_market_data 
        (symbol, timestamp, datetime, open, high, low, close, volume, turnover, interval_minutes, market_type)
        VALUES %s
        ON CONFLICT (symbol, timestamp, interval_minutes) 
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            turnover = EXCLUDED.turnover,
            market_type = EXCLUDED.market_type
        """

        try:
            # Используем соединение из пула для многопоточной безопасности
            with self.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    execute_values(cursor, insert_query, values_to_insert)
            return len(values_to_insert)
        except Exception as e:
            logger.error(f"Ошибка вставки данных для {symbol}: {e}")
            return 0

    def download_historical_data(self, symbol: str, interval: str = '15', days: int = 365 * 3, check_existing=True):
        """
        Загружает исторические данные за указанный период в PostgreSQL

        Args:
            symbol: Торговая пара
            interval: Таймфрейм (15 для 15-минутных свечей)
            days: Количество дней для загрузки (по умолчанию 3 года)
            check_existing: Проверять ли существующие данные

        Returns:
            dict: Статистика загрузки
        """

        logger.info(f"📊 Начинаем загрузку данных {symbol} ({interval} интервал) за {days} дней")

        # Проверяем, есть ли уже данные в БД
        existing_data = self.db.execute_query(
            """SELECT COUNT(*), MIN(datetime), MAX(datetime) 
               FROM raw_market_data 
               WHERE symbol = %s AND interval_minutes = %s AND market_type = %s""",
            (symbol, int(interval), self.market_type),
            fetch=True
        )[0]
        
        existing_count = existing_data[0]
        existing_start = existing_data[1]
        existing_end = existing_data[2]

        if existing_count > 0:
            logger.info(f"⚠️ Найдено {existing_count} существующих записей для {symbol}")
            if existing_start and existing_end:
                logger.info(f"   📅 Период: {existing_start} - {existing_end}")
                
                # Если проверка включена и данные актуальны, пропускаем
                if check_existing:
                    days_since_update = (datetime.now() - existing_end).days
                    # Пропускаем если данные не старше 7 дней
                    if days_since_update < 7:
                        logger.info(f"✅ Данные {symbol} достаточно свежие (обновлены {days_since_update} дней назад), пропускаем")
                        
                        # Получаем статистику для пропущенного символа
                        stats_query = """
                        SELECT MIN(close), MAX(close), AVG(volume)
                        FROM raw_market_data 
                        WHERE symbol = %s AND interval_minutes = %s
                        """
                        stats_data = self.db.execute_query(stats_query, (symbol, int(interval)), fetch=True)[0]
                        
                        return {
                            'symbol': symbol,
                            'interval': interval,
                            'total_records': existing_count,
                            'newly_inserted': 0,
                            'start_date': existing_start,
                            'end_date': existing_end,
                            'min_price': float(stats_data[0]) if stats_data[0] else 0,
                            'max_price': float(stats_data[1]) if stats_data[1] else 0,
                            'avg_volume': float(stats_data[2]) if stats_data[2] else 0,
                            'skipped': True
                        }
                    else:
                        # Загружаем только недостающие данные
                        logger.info(f"📥 Докачиваем последние {days_since_update} дней для {symbol}")
                        # Показываем примерное количество новых записей
                        approx_new_records = days_since_update * 24 * 4  # 4 записи в час для 15-мин интервала
                        logger.info(f"   📈 Ожидается примерно {approx_new_records} новых записей")

        # Вычисляем временные рамки
        end_time = int(datetime.now().timestamp() * 1000)
        
        # Если есть существующие данные, начинаем с последней даты
        if existing_end and existing_count > 0:
            # Добавляем 1 минуту к последней дате чтобы не дублировать
            start_time = int((existing_end + timedelta(minutes=1)).timestamp() * 1000)
            logger.info(f"📍 Продолжаем загрузку с {existing_end + timedelta(minutes=1)}")
        else:
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        # Определяем интервал в миллисекундах
        interval_ms = self._get_interval_ms(interval)
        max_klines_per_request = 1000
        chunk_size_ms = interval_ms * max_klines_per_request

        # Считаем общее количество запросов (это приблизительная оценка)
        # Реальное количество может отличаться из-за особенностей API Bybit
        estimated_requests = max(1, (end_time - start_time) // chunk_size_ms + 1)
        total_requests = estimated_requests

        total_inserted = 0
        current_start = start_time

        # Используем простой счетчик вместо tqdm для потоков
        request_count = 0
        last_log_time = time.time()
        
        while current_start < end_time and not shutdown_flag.is_set():
            current_end = min(current_start + chunk_size_ms, end_time)

            # Получаем данные
            klines = self.get_klines(symbol, interval, current_start, current_end)

            if klines:
                # Вставляем в БД
                inserted = self.insert_raw_data_batch(symbol, klines, int(interval))
                total_inserted += inserted

                # Обновляем позицию
                last_timestamp = int(klines[-1][0])
                current_start = last_timestamp + interval_ms
            else:
                current_start = current_end

            request_count += 1
            
            # Логирование прогресса каждые 5 секунд или каждые 10 запросов
            current_time = time.time()
            if request_count % 10 == 0 or (current_time - last_log_time) > 5:
                # Вычисляем реальный прогресс на основе времени
                time_progress = ((current_start - start_time) / (end_time - start_time)) * 100
                time_progress = min(time_progress, 100.0)
                
                # Если запросов больше чем ожидалось, показываем реальное количество
                if request_count > total_requests:
                    logger.info(f"📊 {symbol}: {time_progress:.1f}% по времени ({request_count} запросов, ожидалось ~{total_requests})")
                else:
                    logger.info(f"📊 {symbol}: {time_progress:.1f}% ({request_count}/{total_requests})")
                    
                last_log_time = current_time
                
            time.sleep(0.1)  # Пауза между запросами

        # Финальная статистика
        final_count = self.db.execute_query(
            "SELECT COUNT(*) FROM raw_market_data WHERE symbol = %s AND interval_minutes = %s",
            (symbol, int(interval)),
            fetch=True
        )[0][0]

        # Получаем период данных
        period_query = """
        SELECT MIN(datetime), MAX(datetime), MIN(close), MAX(close), AVG(volume)
        FROM raw_market_data 
        WHERE symbol = %s AND interval_minutes = %s
        """
        period_data = self.db.execute_query(period_query, (symbol, int(interval)), fetch=True)[0]

        stats = {
            'symbol': symbol,
            'interval': interval,
            'total_records': final_count,
            'newly_inserted': total_inserted,
            'start_date': period_data[0],
            'end_date': period_data[1],
            'min_price': float(period_data[2]) if period_data[2] else 0,
            'max_price': float(period_data[3]) if period_data[3] else 0,
            'avg_volume': float(period_data[4]) if period_data[4] else 0,
            'skipped': False
        }

        # Проверяем, была ли остановка
        if shutdown_flag.is_set():
            logger.warning(f"⚠️ Загрузка {symbol} прервана пользователем")
        else:
            if total_inserted > 0:
                logger.info(f"✅ {symbol}: загружено {total_inserted} новых записей за {request_count} запросов (всего: {final_count})")
            else:
                logger.info(f"✅ {symbol}: данные актуальны (всего: {final_count} записей)")

        return stats

    def _get_interval_ms(self, interval: str) -> int:
        """Конвертирует интервал в миллисекунды"""

        interval_map = {
            '1': 60 * 1000,
            '3': 3 * 60 * 1000,
            '5': 5 * 60 * 1000,
            '15': 15 * 60 * 1000,
            '30': 30 * 60 * 1000,
            '60': 60 * 60 * 1000,
            '120': 120 * 60 * 1000,
            '240': 240 * 60 * 1000,
            'D': 24 * 60 * 60 * 1000,
        }

        return interval_map.get(interval, 15 * 60 * 1000)

    def download_multiple_symbols(self, symbols: list, interval: str = '15', days: int = 365 * 3, max_workers: int = 5):
        """
        Загружает данные для нескольких символов используя многопоточность

        Args:
            symbols: Список торговых пар
            interval: Таймфрейм
            days: Количество дней
            max_workers: Максимальное количество потоков

        Returns:
            dict: Результаты загрузки для каждого символа
        """

        results = {}
        lock = threading.Lock()
        
        def download_symbol(symbol):
            """Функция загрузки для одного символа"""
            thread_name = threading.current_thread().name
            try:
                logger.info(f"🔄 [{thread_name}] Начинаем обработку {symbol}")
                
                stats = self.download_historical_data(
                    symbol=symbol,
                    interval=interval,
                    days=days
                )
                
                with lock:
                    results[symbol] = {'success': True, 'stats': stats}
                    
                return symbol, True
            except Exception as e:
                logger.error(f"❌ [{thread_name}] Ошибка загрузки {symbol}: {e}")
                with lock:
                    results[symbol] = {'success': False, 'error': str(e)}
                return symbol, False

        # Создаем прогресс бар для общего прогресса
        completed_count = 0
        with tqdm(total=len(symbols), desc="Общий прогресс загрузки", position=0) as pbar:
            # Используем ThreadPoolExecutor для параллельной загрузки
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Запускаем задачи
                futures = {executor.submit(download_symbol, symbol): symbol for symbol in symbols}
                
                # Обрабатываем результаты по мере готовности с таймаутом
                remaining_futures = list(futures.keys())
                
                while remaining_futures and not shutdown_flag.is_set():
                    # Ждем завершения с таймаутом
                    done, remaining_futures = wait(
                        remaining_futures, 
                        timeout=60,  # 60 секунд таймаут
                        return_when=FIRST_COMPLETED
                    )
                    
                    for future in done:
                        symbol = futures[future]
                        try:
                            symbol_result, success = future.result(timeout=5)
                            completed_count += 1
                            pbar.update(1)
                            if success:
                                logger.info(f"✅ {symbol_result} загружен успешно ({completed_count}/{len(symbols)})")
                            else:
                                logger.warning(f"⚠️ {symbol_result} не удалось загрузить")
                        except TimeoutError:
                            logger.error(f"❌ Таймаут для {symbol}")
                            with lock:
                                results[symbol] = {'success': False, 'error': 'Timeout'}
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"❌ Критическая ошибка для {symbol}: {e}")
                            with lock:
                                results[symbol] = {'success': False, 'error': str(e)}
                            pbar.update(1)
                
                # Отменяем оставшиеся задачи при shutdown
                if shutdown_flag.is_set():
                    logger.info("🛑 Отмена оставшихся задач...")
                    for future in remaining_futures:
                        future.cancel()
                    executor.shutdown(wait=False)

        # Подсчитываем статистику
        successful = sum(1 for r in results.values() if r.get('success', False))
        skipped = sum(1 for r in results.values() if r.get('success') and r.get('stats', {}).get('skipped', False))
        failed = len(results) - successful
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 ИТОГИ МНОГОПОТОЧНОЙ ЗАГРУЗКИ:")
        logger.info(f"   ✅ Успешно: {successful} символов")
        logger.info(f"   ⏭️ Пропущено (актуальные): {skipped} символов")
        logger.info(f"   ❌ Ошибки: {failed} символов")
        logger.info(f"   🚀 Использовано потоков: {max_workers}")
        logger.info(f"{'=' * 60}")

        return results

    def get_database_stats(self):
        """Возвращает статистику по данным в БД"""

        stats_query = """
        SELECT 
            symbol,
            COUNT(*) as total_records,
            MIN(datetime) as start_date,
            MAX(datetime) as end_date,
            MIN(close) as min_price,
            MAX(close) as max_price,
            AVG(volume) as avg_volume
        FROM raw_market_data 
        WHERE interval_minutes = 15
        GROUP BY symbol
        ORDER BY symbol
        """

        results = self.db.execute_query(stats_query, fetch=True)

        total_records = 0
        stats = {}

        for row in results:
            symbol_stats = {
                'total_records': row[1],
                'start_date': row[2],
                'end_date': row[3],
                'min_price': float(row[4]),
                'max_price': float(row[5]),
                'avg_volume': float(row[6])
            }
            stats[row[0]] = symbol_stats
            total_records += row[1]

        logger.info(f"\n📊 СТАТИСТИКА БАЗЫ ДАННЫХ:")
        logger.info(f"   🗃️ Всего символов: {len(stats)}")
        logger.info(f"   📈 Всего записей: {total_records:,}")

        return stats, total_records


def main():
    """Основная функция для запуска загрузки данных"""

    # Загружаем конфигурацию
    import yaml
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        alt = 'config/config.yaml'
        if os.path.exists(alt):
            config_path = alt
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']

    # Инициализируем менеджер БД
    db_manager = PostgreSQLManager(db_config)

    try:
        # Подключаемся к БД
        db_manager.connect()

        # Создаем таблицы
        db_manager.create_tables()

        # Инициализируем загрузчик для фьючерсов
        market_type = config['data_download'].get('market_type', 'futures')
        downloader = BybitDataDownloader(db_manager, market_type=market_type)

        # Загружаем параметры из конфигурации
        top_symbols = config['data_download']['symbols']
        # Исключаем TESTUSDT
        top_symbols = [s for s in top_symbols if 'TEST' not in s]
        interval = config['data_download']['interval']
        days = config['data_download']['days']

        logger.info(f"🚀 Начинаем загрузку данных для {len(top_symbols)} символов в PostgreSQL")
        logger.info(f"📊 Таймфрейм: {interval} минут")
        logger.info(f"📅 Период: {days} дней")
        db_name = db_config.get('dbname') or db_config.get('database')
        logger.info(f"🗃️ База данных: {db_name}")
        logger.info(f"📈 Тип рынка: {market_type.upper()}")
        
        # Определяем количество потоков (по умолчанию 25)
        max_workers = config['data_download'].get('max_workers', 25)
        logger.info(f"🔧 Количество потоков: {max_workers}")

        # Загружаем все монеты за 3 года с многопоточностью
        results = downloader.download_multiple_symbols(top_symbols, interval, days, max_workers=max_workers)

        # Итоговая статистика
        logger.info(f"\n{'=' * 50}")
        logger.info(f"📊 ИТОГОВАЯ СТАТИСТИКА")
        logger.info(f"{'=' * 50}")

        successful = sum(1 for r in results.values() if r.get('success', False))
        skipped = sum(1 for r in results.values() if r.get('success') and r.get('stats', {}).get('skipped', False))
        newly_downloaded = successful - skipped
        failed = len(results) - successful

        logger.info(f"✅ Успешно обработано: {successful}")
        logger.info(f"   📥 Новых загрузок: {newly_downloaded}")
        logger.info(f"   ⏭️ Пропущено (актуальные): {skipped}")
        logger.info(f"❌ Ошибок: {failed}")
        
        # Подсчет общего количества новых записей
        total_new_records = sum(
            r.get('stats', {}).get('newly_inserted', 0) 
            for r in results.values() 
            if r.get('success', False)
        )
        logger.info(f"📈 Всего новых записей добавлено: {total_new_records:,}")

        if failed > 0:
            logger.info(f"\n❌ Не удалось загрузить:")
            for symbol, result in results.items():
                if not result.get('success', False):
                    logger.info(f"   {symbol}: {result.get('error', 'Unknown error')}")

        # Показываем статистику БД
        stats, total_records = downloader.get_database_stats()
        logger.info(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА БД:")
        logger.info(f"   💾 Всего записей в БД: {total_records:,}")
        logger.info(f"   📈 Символов с данными: {len(stats)}")

    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()