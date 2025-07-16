#!/usr/bin/env python3
"""
Эффективное исправление утечки данных через временную таблицу
Использует оптимизированный подход для быстрой обработки миллионов записей
"""

import psycopg2
import yaml
import logging
import time
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_data_leakage_efficient():
    """Удаляет expected_returns из technical_indicators используя временную таблицу"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    # Подключаемся к БД
    conn = psycopg2.connect(**db_config)
    conn.autocommit = False
    
    logger.info("✅ Подключение к PostgreSQL установлено")
    
    try:
        cursor = conn.cursor()
        
        # 1. Проверяем масштаб проблемы
        logger.info("\n🔍 Анализ данных...")
        cursor.execute("""
            SELECT COUNT(*) as total,
                   COUNT(CASE WHEN technical_indicators ? 'buy_expected_return' 
                              OR technical_indicators ? 'sell_expected_return' 
                         THEN 1 END) as with_leak
            FROM processed_market_data
        """)
        total, with_leak = cursor.fetchone()
        
        logger.info(f"   Всего записей: {total:,}")
        logger.info(f"   С утечкой: {with_leak:,} ({with_leak/total*100:.1f}%)")
        
        if with_leak == 0:
            logger.info("✅ Утечка данных не обнаружена!")
            return
        
        # 2. Создаем временную таблицу с ID записей для обновления
        logger.info("\n🔧 Создание временной таблицы с ID...")
        cursor.execute("DROP TABLE IF EXISTS temp_ids_to_fix")
        cursor.execute("""
            CREATE TEMP TABLE temp_ids_to_fix AS
            SELECT id 
            FROM processed_market_data 
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """)
        
        cursor.execute("SELECT COUNT(*) FROM temp_ids_to_fix")
        ids_count = cursor.fetchone()[0]
        logger.info(f"   Найдено {ids_count:,} записей для обновления")
        
        # 3. Создаем индекс на временной таблице для ускорения JOIN
        logger.info("   Создание индекса на временной таблице...")
        cursor.execute("CREATE INDEX idx_temp_ids ON temp_ids_to_fix(id)")
        
        # 4. Анализируем временную таблицу для оптимизации
        cursor.execute("ANALYZE temp_ids_to_fix")
        
        # 5. Выполняем обновление батчами через JOIN
        logger.info("\n🚀 Выполнение очистки оптимизированным способом...")
        
        batch_size = 10000
        processed = 0
        start_time = time.time()
        
        # Создаем прогресс бар
        with tqdm(total=ids_count, desc="Очистка данных", unit="записей") as pbar:
            while processed < ids_count:
                batch_start_time = time.time()
                
                # Обновляем батч через эффективный JOIN
                cursor.execute("""
                    UPDATE processed_market_data p
                    SET technical_indicators = p.technical_indicators 
                        - 'buy_expected_return'::text 
                        - 'sell_expected_return'::text
                    FROM (
                        SELECT id 
                        FROM temp_ids_to_fix 
                        ORDER BY id
                        LIMIT %s
                        OFFSET %s
                    ) t
                    WHERE p.id = t.id
                """, (batch_size, processed))
                
                batch_updated = cursor.rowcount
                processed += batch_updated
                
                # Коммитим каждые 50,000 записей для надежности
                if processed % 50000 == 0:
                    conn.commit()
                    logger.info(f"   💾 Сохранено {processed:,} записей")
                
                # Обновляем прогресс
                pbar.update(batch_updated)
                
                # Рассчитываем скорость и ETA
                batch_time = time.time() - batch_start_time
                total_elapsed = time.time() - start_time
                speed = processed / total_elapsed if total_elapsed > 0 else 0
                eta_seconds = (ids_count - processed) / speed if speed > 0 else 0
                
                pbar.set_postfix({
                    'Скорость': f'{speed:.0f} зап/сек',
                    'Батч': f'{batch_time:.1f}с',
                    'ETA': f'{eta_seconds/60:.1f} мин'
                })
                
                # Если обновлено меньше batch_size, значит закончили
                if batch_updated < batch_size:
                    break
        
        # Финальный коммит
        conn.commit()
        
        total_time = time.time() - start_time
        logger.info(f"\n✅ Очистка завершена!")
        logger.info(f"   Обработано: {processed:,} записей")
        logger.info(f"   Время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
        logger.info(f"   Скорость: {processed/total_time:.0f} записей/сек")
        
        # 6. Проверяем результат
        logger.info("\n🔍 Проверка результата...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_market_data 
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """)
        remaining = cursor.fetchone()[0]
        
        if remaining == 0:
            logger.info("✅ Утечка данных успешно устранена!")
            
            # 7. Показываем статистику по размеру
            cursor.execute("""
                SELECT 
                    pg_size_pretty(AVG(pg_column_size(technical_indicators))::bigint) as avg_size,
                    pg_size_pretty(SUM(pg_column_size(technical_indicators))::bigint) as total_size
                FROM processed_market_data
                LIMIT 10000
            """)
            avg_size, total_size_sample = cursor.fetchone()
            logger.info(f"\n📊 Статистика после очистки:")
            logger.info(f"   Средний размер JSON: {avg_size}")
            logger.info(f"   Примерная экономия: ~{with_leak * 100 / 1024 / 1024:.0f} МБ")
            
            # 8. Оптимизация таблицы
            logger.info("\n🔧 Оптимизация таблицы (VACUUM ANALYZE)...")
            conn.autocommit = True
            cursor.execute("VACUUM ANALYZE processed_market_data")
            logger.info("✅ Таблица оптимизирована")
            
        else:
            logger.error(f"❌ Остались записи с утечкой: {remaining}")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Операция прервана пользователем")
        conn.rollback()
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.info("\n📤 Подключение к PostgreSQL закрыто")


if __name__ == "__main__":
    logger.info("🚀 Эффективное исправление утечки данных")
    logger.info("=" * 60)
    logger.info("Использует временную таблицу и оптимизированные JOIN")
    logger.info("Ожидаемое время: 5-10 минут для 2.7М записей")
    logger.info("=" * 60)
    
    # Подтверждение
    response = input("\nПродолжить? (y/n): ")
    if response.lower() == 'y':
        try:
            fix_data_leakage_efficient()
        except KeyboardInterrupt:
            logger.info("\n❌ Операция отменена")
    else:
        logger.info("Отменено пользователем")