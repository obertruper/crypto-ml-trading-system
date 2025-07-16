#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Загрузка исторических данных с Bybit FUTURES (деривативы)
"""

import time
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
from pybit.unified_trading import HTTP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """Менеджер для работы с PostgreSQL"""

    def __init__(self, db_config: dict):
        self.db_config = db_config.copy()
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
        self.connection = None

    def connect(self):
        """Создает подключение к БД"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
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

    def execute_query(self, query: str, params=None, fetch=False):
        """Выполняет SQL запрос"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса: {e}")
            raise

    def execute_batch_insert(self, query: str, data: list):
        """Выполняет пакетную вставку данных"""
        try:
            with self.connection.cursor() as cursor:
                execute_batch(cursor, query, data, page_size=1000)
                return cursor.rowcount
        except Exception as e:
            logger.error(f"❌ Ошибка пакетной вставки: {e}")
            raise


class BybitFuturesDownloader:
    """Класс для загрузки данных с Bybit Futures API"""

    def __init__(self, db_manager: PostgreSQLManager):
        self.db = db_manager
        self.session = HTTP(testnet=False)

    def get_futures_symbols(self):
        """Получает список всех доступных фьючерсных символов"""
        try:
            response = self.session.get_instruments_info(category="linear")
            
            if response['retCode'] == 0:
                symbols = []
                for instrument in response['result']['list']:
                    symbol = instrument['symbol']
                    status = instrument['status']
                    # Берем только активные USDT perpetual контракты
                    if symbol.endswith('USDT') and status == 'Trading':
                        symbols.append(symbol)
                
                # Исключаем TESTUSDT и другие тестовые символы
                symbols = [s for s in symbols if not any(test in s for test in ['TEST', 'DEMO', 'MOCK'])]
                
                return sorted(symbols)
            else:
                logger.error(f"Ошибка API: {response['retMsg']}")
                return []
                
        except Exception as e:
            logger.error(f"Ошибка получения символов: {e}")
            return []

    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> list:
        """
        Получает исторические данные (klines) для фьючерсного символа

        Args:
            symbol: Символ (например, BTCUSDT)
            interval: Интервал (15, 30, 60 и т.д.)
            start_time: Начальное время в миллисекундах
            end_time: Конечное время в миллисекундах

        Returns:
            list: Список свечей [[timestamp, open, high, low, close, volume, turnover]]
        """
        
        try:
            response = self.session.get_kline(
                category="linear",  # Используем linear futures
                symbol=symbol,
                interval=interval,
                start=start_time,
                end=end_time,
                limit=1000
            )

            if response['retCode'] == 0:
                klines = response['result']['list']
                # Возвращаем в правильном порядке (от старых к новым)
                return klines[::-1]
            else:
                logger.error(f"Ошибка API: {response['retMsg']}")
                return []

        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol}: {e}")
            return []

    def check_existing_data(self, symbol: str, interval: int) -> tuple:
        """
        Проверяет существующие данные в БД

        Returns:
            tuple: (has_data, last_timestamp)
        """
        
        query = """
        SELECT MAX(timestamp) 
        FROM raw_market_data 
        WHERE symbol = %s AND interval_minutes = %s AND market_type = 'futures'
        """
        
        result = self.db.execute_query(query, (symbol, interval), fetch=True)
        
        if result and result[0][0]:
            return True, result[0][0]
        
        return False, None

    def insert_raw_data_batch(self, symbol: str, klines: list, interval: int) -> int:
        """
        Вставляет данные пакетом в БД

        Args:
            symbol: Символ
            klines: Список свечей
            interval: Интервал в минутах

        Returns:
            int: Количество вставленных записей
        """
        
        if not klines:
            return 0

        # Подготавливаем данные для вставки
        insert_data = []
        
        for kline in klines:
            timestamp = int(kline[0])
            datetime_obj = datetime.fromtimestamp(timestamp / 1000)
            
            insert_data.append((
                symbol,
                timestamp,
                datetime_obj,
                float(kline[1]),  # open
                float(kline[2]),  # high
                float(kline[3]),  # low
                float(kline[4]),  # close
                float(kline[5]),  # volume
                float(kline[6]),  # turnover
                interval,
                'futures'  # Добавляем тип рынка
            ))

        # SQL запрос для вставки с обработкой дубликатов
        insert_query = """
        INSERT INTO raw_market_data 
        (symbol, timestamp, datetime, open, high, low, close, volume, turnover, interval_minutes, market_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, timestamp, interval_minutes) DO NOTHING
        """

        # Добавляем колонку market_type если её нет
        try:
            self.db.execute_query("""
                ALTER TABLE raw_market_data 
                ADD COLUMN IF NOT EXISTS market_type VARCHAR(20) DEFAULT 'spot'
            """)
        except:
            pass

        inserted = self.db.execute_batch_insert(insert_query, insert_data)
        
        logger.info(f"💾 Вставлено {inserted} записей для {symbol}")
        
        return inserted

    def download_historical_data(self, symbol: str, interval: str = '15', days: int = 1095) -> dict:
        """
        Загружает исторические данные для фьючерсного символа

        Args:
            symbol: Символ для загрузки
            interval: Интервал в минутах (по умолчанию 15)
            days: Количество дней истории (по умолчанию 3 года)

        Returns:
            dict: Статистика загрузки
        """
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Загрузка фьючерсных данных для {symbol}")
        logger.info(f"{'='*50}")
        logger.info(f"📊 Начинаем загрузку данных {symbol} (интервал {interval}m) за {days} дней")

        # Проверяем существующие данные
        has_data, last_timestamp = self.check_existing_data(symbol, int(interval))
        
        # Определяем временной диапазон
        end_time = int(time.time() * 1000)
        
        if has_data and last_timestamp:
            # Начинаем с последней записи
            start_time = last_timestamp + (int(interval) * 60 * 1000)
            logger.info(f"📊 Найдены существующие данные. Загружаем с {datetime.fromtimestamp(start_time/1000)}")
        else:
            # Начинаем с указанного периода
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            logger.info(f"📊 Новый символ. Загружаем с {datetime.fromtimestamp(start_time/1000)}")

        # Параметры для загрузки
        interval_ms = int(interval) * 60 * 1000
        max_klines_per_request = 1000
        chunk_size_ms = interval_ms * max_klines_per_request

        # Считаем общее количество запросов
        total_requests = (end_time - start_time) // chunk_size_ms + 1

        total_inserted = 0
        current_start = start_time

        with tqdm(total=total_requests, desc=f"Загрузка {symbol} (futures)") as pbar:
            while current_start < end_time:
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

                pbar.update(1)
                time.sleep(0.1)  # Пауза между запросами

        # Финальная статистика
        final_count = self.db.execute_query(
            "SELECT COUNT(*) FROM raw_market_data WHERE symbol = %s AND interval_minutes = %s AND market_type = 'futures'",
            (symbol, int(interval)),
            fetch=True
        )[0][0]

        # Получаем период данных
        period_query = """
        SELECT MIN(datetime), MAX(datetime), MIN(close), MAX(close), AVG(volume)
        FROM raw_market_data 
        WHERE symbol = %s AND interval_minutes = %s AND market_type = 'futures'
        """
        period_data = self.db.execute_query(period_query, (symbol, int(interval)), fetch=True)[0]

        stats = {
            'symbol': symbol,
            'market_type': 'futures',
            'interval': interval,
            'total_records': final_count,
            'newly_inserted': total_inserted,
            'start_date': period_data[0],
            'end_date': period_data[1],
            'min_price': float(period_data[2]) if period_data[2] else 0,
            'max_price': float(period_data[3]) if period_data[3] else 0,
            'avg_volume': float(period_data[4]) if period_data[4] else 0
        }

        logger.info(f"\n✅ Загрузка завершена для {symbol}:")
        logger.info(f"   📊 Всего записей: {stats['total_records']:,}")
        logger.info(f"   📈 Новых записей: {stats['newly_inserted']:,}")
        logger.info(f"   📅 Период: {stats['start_date']} - {stats['end_date']}")
        logger.info(f"   💰 Цена: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}")

        return stats

    def download_multiple_symbols(self, symbols: list = None, interval: str = '15', days: int = 1095) -> dict:
        """
        Загружает данные для нескольких фьючерсных символов

        Args:
            symbols: Список символов (если None - загружает все доступные)
            interval: Интервал
            days: Количество дней

        Returns:
            dict: Результаты загрузки для каждого символа
        """
        
        if symbols is None:
            # Получаем все доступные фьючерсные символы
            symbols = self.get_futures_symbols()
            logger.info(f"📊 Найдено {len(symbols)} фьючерсных символов")
        
        results = {}
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n[{i}/{len(symbols)}] Обработка {symbol}")
            
            try:
                stats = self.download_historical_data(symbol, interval, days)
                results[symbol] = {
                    'success': True,
                    'stats': stats
                }
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {symbol}: {e}")
                results[symbol] = {'success': False, 'error': str(e)}

        return results


def main():
    """Основная функция"""
    import yaml
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    data_config = config['data_download']
    
    # Убираем TESTUSDT из списка
    symbols = [s for s in data_config['symbols'] if 'TEST' not in s]
    
    # Инициализация
    db_manager = PostgreSQLManager(db_config)
    downloader = BybitFuturesDownloader(db_manager)
    
    try:
        db_manager.connect()
        
        # Получаем список доступных фьючерсных символов
        available_futures = downloader.get_futures_symbols()
        logger.info(f"📊 Доступно {len(available_futures)} фьючерсных символов")
        
        # Фильтруем только те, которые есть в нашем списке и доступны на фьючерсах
        symbols_to_download = [s for s in symbols if s in available_futures]
        
        logger.info(f"✅ Будет загружено {len(symbols_to_download)} символов из вашего списка")
        logger.info(f"❌ Не доступны на фьючерсах: {len(symbols) - len(symbols_to_download)} символов")
        
        # Показываем недоступные
        not_available = [s for s in symbols if s not in available_futures]
        if not_available:
            logger.info(f"   Недоступные: {', '.join(not_available[:10])}...")
        
        # Загружаем данные
        interval = data_config['interval']
        days = data_config['days']
        
        results = downloader.download_multiple_symbols(symbols_to_download, interval, days)
        
        # Статистика
        success_count = sum(1 for r in results.values() if r.get('success', False))
        logger.info(f"\n✅ Успешно загружено: {success_count}/{len(symbols_to_download)} символов")
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()