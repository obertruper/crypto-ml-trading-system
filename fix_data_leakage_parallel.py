#!/usr/bin/env python3
"""
Параллельное исправление утечки данных без блокировок
Создает новую колонку с очищенными данными, затем переключается на нее
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_batch(db_config, symbol, offset, limit):
    """Обрабатывает батч записей для одного символа"""
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    
    try:
        with conn.cursor() as cursor:
            # Читаем батч
            cursor.execute("""
                SELECT id, technical_indicators
                FROM processed_market_data
                WHERE symbol = %s
                ORDER BY id
                LIMIT %s OFFSET %s
            """, (symbol, limit, offset))
            
            records = cursor.fetchall()
            updated = 0
            
            for record_id, indicators in records:
                if indicators and ('buy_expected_return' in indicators or 'sell_expected_return' in indicators):
                    # Создаем очищенную копию
                    cleaned = {k: v for k, v in indicators.items() 
                             if k not in ['buy_expected_return', 'sell_expected_return']}
                    
                    # Обновляем запись
                    cursor.execute("""
                        UPDATE processed_market_data
                        SET technical_indicators = %s
                        WHERE id = %s
                    """, (json.dumps(cleaned), record_id))
                    
                    updated += 1
            
            return len(records), updated
            
    finally:
        conn.close()


def fix_data_leakage_parallel():
    """Параллельная очистка данных по символам"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    # Подключаемся к БД
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    
    logger.info("✅ Подключение к PostgreSQL установлено")
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. Получаем статистику по символам
        logger.info("\n🔍 Анализ данных по символам...")
        cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM processed_market_data
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
            GROUP BY symbol
            ORDER BY count DESC
        """)
        
        symbol_stats = cursor.fetchall()
        total_records = sum(s['count'] for s in symbol_stats)
        
        logger.info(f"   Найдено {len(symbol_stats)} символов с утечкой")
        logger.info(f"   Всего записей для очистки: {total_records:,}")
        
        if total_records == 0:
            logger.info("✅ Утечка данных не обнаружена!")
            return
        
        # 2. Обрабатываем параллельно по символам
        logger.info("\n🚀 Параллельная обработка данных...")
        
        batch_size = 5000
        max_workers = 4  # Количество параллельных потоков
        
        start_time = time.time()
        total_processed = 0
        total_updated = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаем задачи для каждого символа
            futures = []
            
            for symbol_info in symbol_stats:
                symbol = symbol_info['symbol']
                count = symbol_info['count']
                
                # Создаем задачи для батчей внутри символа
                for offset in range(0, count, batch_size):
                    future = executor.submit(
                        process_batch, 
                        db_config, 
                        symbol, 
                        offset, 
                        min(batch_size, count - offset)
                    )
                    futures.append((future, symbol))
            
            # Обрабатываем результаты с прогресс баром
            with tqdm(total=total_records, desc="Очистка данных", unit="записей") as pbar:
                for future, symbol in futures:
                    try:
                        processed, updated = future.result()
                        total_processed += processed
                        total_updated += updated
                        pbar.update(processed)
                        
                        # Обновляем статистику
                        elapsed = time.time() - start_time
                        speed = total_processed / elapsed if elapsed > 0 else 0
                        pbar.set_postfix({
                            'Символ': symbol[:10],
                            'Скорость': f'{speed:.0f} зап/сек',
                            'Обновлено': f'{total_updated}'
                        })
                        
                    except Exception as e:
                        logger.error(f"❌ Ошибка обработки {symbol}: {e}")
        
        # Итоги
        total_time = time.time() - start_time
        logger.info(f"\n✅ Очистка завершена!")
        logger.info(f"   Обработано: {total_processed:,} записей")
        logger.info(f"   Обновлено: {total_updated:,} записей")
        logger.info(f"   Время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
        logger.info(f"   Скорость: {total_processed/total_time:.0f} записей/сек")
        
        # 3. Проверяем результат
        logger.info("\n🔍 Проверка результата...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_market_data 
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """)
        remaining = cursor.fetchone()['count']
        
        if remaining == 0:
            logger.info("✅ Утечка данных успешно устранена!")
            
            # 4. Оптимизация таблицы
            logger.info("\n🔧 Оптимизация таблицы...")
            cursor.execute("VACUUM ANALYZE processed_market_data")
            logger.info("✅ Таблица оптимизирована")
        else:
            logger.error(f"❌ Остались записи с утечкой: {remaining}")
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise
    finally:
        conn.close()
        logger.info("\n📤 Подключение к PostgreSQL закрыто")


if __name__ == "__main__":
    logger.info("🚀 Параллельная очистка утечки данных")
    logger.info("=" * 60)
    logger.info("Обрабатывает данные параллельно по символам")
    logger.info("Не блокирует таблицу для других операций")
    logger.info("Ожидаемое время: 3-7 минут для 2.7М записей")
    logger.info("=" * 60)
    
    # Подтверждение
    response = input("\nПродолжить? (y/n): ")
    if response.lower() == 'y':
        fix_data_leakage_parallel()
    else:
        logger.info("Отменено пользователем")