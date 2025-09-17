#!/usr/bin/env python3
"""
Загрузчик исторических данных с Bybit для криптовалютных фьючерсов.
Поддерживает загрузку OHLCV данных и сохранение в PostgreSQL.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt
import psycopg2
from psycopg2.extras import execute_batch
import yaml
from tqdm import tqdm
import argparse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BybitDataDownloader:
    """Загрузчик данных с биржи Bybit."""

    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Инициализация загрузчика.

        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config = self._load_config(config_path)
        self.exchange = self._init_exchange()
        self.db_conn = self._init_database()

    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Конфигурация загружена из {config_path}")
            return config
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            sys.exit(1)

    def _init_exchange(self) -> ccxt.Exchange:
        """Инициализация подключения к Bybit."""
        try:
            exchange = ccxt.bybit({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Используем фьючерсы
                    'defaultSubType': 'linear'  # USDT perpetual
                }
            })

            # Загружаем рынки
            exchange.load_markets()
            logger.info(f"Подключено к Bybit. Доступно {len(exchange.markets)} рынков")
            return exchange

        except Exception as e:
            logger.error(f"Ошибка подключения к Bybit: {e}")
            sys.exit(1)

    def _init_database(self) -> psycopg2.extensions.connection:
        """Инициализация подключения к базе данных."""
        try:
            db_config = self.config['database']
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['name'],
                user=db_config['user'],
                password=db_config['password']
            )
            logger.info("Подключено к PostgreSQL")

            # Создаем таблицу если не существует
            self._create_tables(conn)
            return conn

        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            sys.exit(1)

    def _create_tables(self, conn):
        """Создание необходимых таблиц в БД."""
        cursor = conn.cursor()

        # Таблица для сырых данных
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timestamp BIGINT NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_raw_market_symbol_timestamp
            ON raw_market_data(symbol, timestamp);
        """)

        conn.commit()
        logger.info("Таблицы БД готовы")

    def download_symbol_data(
        self,
        symbol: str,
        timeframe: str = '15m',
        days: int = 365,
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """
        Загрузка данных для одного символа.

        Args:
            symbol: Торговый символ (например, 'BTC/USDT:USDT')
            timeframe: Временной интервал
            days: Количество дней для загрузки
            batch_size: Размер батча для одного запроса

        Returns:
            DataFrame с OHLCV данными
        """
        all_data = []

        # Конвертируем символ в формат Bybit
        market_symbol = self._convert_symbol(symbol)

        # Проверяем доступность символа
        if market_symbol not in self.exchange.markets:
            logger.warning(f"Символ {market_symbol} не найден на Bybit")
            return pd.DataFrame()

        # Вычисляем временные границы
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Конвертируем в миллисекунды
        since = int(start_time.timestamp() * 1000)
        until = int(end_time.timestamp() * 1000)

        logger.info(f"Загрузка {symbol} с {start_time} по {end_time}")

        # Загружаем данные батчами
        with tqdm(desc=f"Загрузка {symbol}", unit="candles") as pbar:
            while since < until:
                try:
                    # Получаем свечи
                    ohlcv = self.exchange.fetch_ohlcv(
                        market_symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=batch_size
                    )

                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)
                    pbar.update(len(ohlcv))

                    # Обновляем since для следующего батча
                    since = ohlcv[-1][0] + 1

                    # Небольшая задержка для rate limit
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Ошибка загрузки {symbol}: {e}")
                    time.sleep(1)
                    continue

        # Конвертируем в DataFrame
        if all_data:
            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Удаляем дубликаты
            df = df.drop_duplicates(subset=['timestamp'])

            # Сортируем по времени
            df = df.sort_values('timestamp')

            # Добавляем символ
            df['symbol'] = symbol.replace('/', '').replace(':', '')

            logger.info(f"Загружено {len(df)} свечей для {symbol}")
            return df
        else:
            logger.warning(f"Нет данных для {symbol}")
            return pd.DataFrame()

    def _convert_symbol(self, symbol: str) -> str:
        """
        Конвертация символа в формат Bybit.

        Args:
            symbol: Символ в формате 'BTCUSDT' или 'BTC/USDT'

        Returns:
            Символ в формате Bybit
        """
        # Удаляем слэши если есть
        clean_symbol = symbol.replace('/', '')

        # Для фьючерсов Bybit использует формат BTC/USDT:USDT
        if 'USDT' in clean_symbol:
            base = clean_symbol.replace('USDT', '')
            return f"{base}/USDT:USDT"
        else:
            return symbol

    def save_to_database(self, df: pd.DataFrame) -> int:
        """
        Сохранение данных в БД.

        Args:
            df: DataFrame с данными

        Returns:
            Количество сохраненных записей
        """
        if df.empty:
            return 0

        cursor = self.db_conn.cursor()

        # Подготавливаем данные для вставки
        records = []
        for _, row in df.iterrows():
            records.append((
                row['symbol'],
                int(row['timestamp']),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))

        # Batch insert с обработкой конфликтов
        query = """
            INSERT INTO raw_market_data (symbol, timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, timestamp) DO NOTHING
        """

        execute_batch(cursor, query, records, page_size=1000)
        self.db_conn.commit()

        inserted = cursor.rowcount
        logger.info(f"Сохранено {inserted} записей в БД")

        return inserted

    def download_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = '15m',
        days: int = 365
    ) -> Dict[str, int]:
        """
        Загрузка данных для всех символов.

        Args:
            symbols: Список символов (если None, берется из конфига)
            timeframe: Временной интервал
            days: Количество дней

        Returns:
            Словарь {символ: количество_записей}
        """
        if symbols is None:
            # Берем из конфига
            symbols = self.config.get('data_download', {}).get('symbols', [])

        if not symbols:
            logger.error("Список символов пуст")
            return {}

        logger.info(f"Начинаем загрузку {len(symbols)} символов")
        results = {}

        for symbol in symbols:
            try:
                # Загружаем данные
                df = self.download_symbol_data(symbol, timeframe, days)

                # Сохраняем в БД
                if not df.empty:
                    count = self.save_to_database(df)
                    results[symbol] = count
                else:
                    results[symbol] = 0

                # Задержка между символами
                time.sleep(1)

            except Exception as e:
                logger.error(f"Ошибка обработки {symbol}: {e}")
                results[symbol] = 0
                continue

        # Выводим статистику
        logger.info("=" * 50)
        logger.info("Результаты загрузки:")
        total = 0
        for symbol, count in results.items():
            logger.info(f"  {symbol}: {count} записей")
            total += count
        logger.info(f"Всего загружено: {total} записей")

        return results

    def get_available_symbols(self) -> List[str]:
        """
        Получение списка доступных USDT perpetual символов.

        Returns:
            Список символов
        """
        symbols = []

        for market_id, market in self.exchange.markets.items():
            # Фильтруем только USDT perpetual фьючерсы
            if (market['quote'] == 'USDT' and
                market['type'] == 'future' and
                market['linear'] and
                market['active']):

                # Конвертируем в наш формат
                base = market['base']
                symbol = f"{base}USDT"
                symbols.append(symbol)

        symbols.sort()
        logger.info(f"Найдено {len(symbols)} USDT perpetual символов")

        return symbols

    def check_data_quality(self, symbol: str) -> Dict:
        """
        Проверка качества загруженных данных.

        Args:
            symbol: Символ для проверки

        Returns:
            Словарь со статистикой
        """
        cursor = self.db_conn.cursor()

        # Получаем статистику
        cursor.execute("""
            SELECT
                COUNT(*) as total_records,
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp,
                COUNT(DISTINCT DATE(to_timestamp(timestamp/1000))) as days_count
            FROM raw_market_data
            WHERE symbol = %s
        """, (symbol,))

        result = cursor.fetchone()

        if result[0] == 0:
            return {
                'symbol': symbol,
                'status': 'NO_DATA'
            }

        # Проверяем пропуски
        cursor.execute("""
            WITH time_series AS (
                SELECT
                    timestamp,
                    LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
                FROM raw_market_data
                WHERE symbol = %s
            )
            SELECT COUNT(*)
            FROM time_series
            WHERE timestamp - prev_timestamp > 900000  -- Больше 15 минут
        """, (symbol,))

        gaps = cursor.fetchone()[0]

        return {
            'symbol': symbol,
            'total_records': result[0],
            'first_date': datetime.fromtimestamp(result[1]/1000),
            'last_date': datetime.fromtimestamp(result[2]/1000),
            'days_count': result[3],
            'gaps_count': gaps,
            'status': 'OK' if gaps < 10 else 'HAS_GAPS'
        }

    def close(self):
        """Закрытие соединений."""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Соединение с БД закрыто")


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(description='Загрузка данных с Bybit')
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Список символов для загрузки'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Количество дней для загрузки (по умолчанию 365)'
    )
    parser.add_argument(
        '--timeframe',
        default='15m',
        help='Временной интервал (по умолчанию 15m)'
    )
    parser.add_argument(
        '--list-symbols',
        action='store_true',
        help='Показать доступные символы'
    )
    parser.add_argument(
        '--check-quality',
        help='Проверить качество данных для символа'
    )

    args = parser.parse_args()

    # Инициализируем загрузчик
    downloader = BybitDataDownloader()

    try:
        if args.list_symbols:
            # Показываем доступные символы
            symbols = downloader.get_available_symbols()
            print("\nДоступные USDT perpetual символы:")
            for i, symbol in enumerate(symbols, 1):
                print(f"{i:3}. {symbol}")

        elif args.check_quality:
            # Проверяем качество данных
            stats = downloader.check_data_quality(args.check_quality)
            print(f"\nКачество данных для {args.check_quality}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        else:
            # Загружаем данные
            symbols = args.symbols

            if not symbols:
                # Берем популярные криптовалюты
                symbols = [
                    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
                    'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
                    'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'ATOMUSDT', 'NEARUSDT',
                    'FTMUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT'
                ]

            results = downloader.download_all_symbols(
                symbols=symbols,
                timeframe=args.timeframe,
                days=args.days
            )

            # Проверяем качество
            print("\nПроверка качества данных:")
            for symbol in symbols:
                stats = downloader.check_data_quality(symbol)
                print(f"{symbol}: {stats['status']} ({stats.get('total_records', 0)} записей)")

    except KeyboardInterrupt:
        logger.info("Загрузка прервана пользователем")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")

    finally:
        downloader.close()


if __name__ == "__main__":
    main()