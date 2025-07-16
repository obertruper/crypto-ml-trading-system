#!/usr/bin/env python3
"""
Прямое SQL решение для максимально быстрой очистки утечки данных
Выполняет операцию одной командой без батчей
"""

import psycopg2
import yaml
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_data_leakage_direct():
    """Удаляет expected_returns одной SQL командой"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    # Подключаемся к БД
    conn = psycopg2.connect(**db_config)
    
    logger.info("✅ Подключение к PostgreSQL установлено")
    
    try:
        cursor = conn.cursor()
        
        # 1. Проверяем масштаб проблемы
        logger.info("\n🔍 Анализ данных...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_market_data 
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """)
        with_leak = cursor.fetchone()[0]
        
        logger.info(f"   Записей с утечкой: {with_leak:,}")
        
        if with_leak == 0:
            logger.info("✅ Утечка данных не обнаружена!")
            return
        
        # 2. Увеличиваем параметры для больших операций
        logger.info("\n🔧 Настройка параметров PostgreSQL...")
        cursor.execute("SET work_mem = '1GB'")
        cursor.execute("SET maintenance_work_mem = '2GB'")
        cursor.execute("SET max_parallel_workers_per_gather = 4")
        cursor.execute("SET max_parallel_workers = 8")
        cursor.execute("SET parallel_tuple_cost = 0.01")
        cursor.execute("SET parallel_setup_cost = 100")
        
        # 3. Выполняем очистку одной командой
        logger.info("\n🚀 Выполнение очистки (это может занять 2-5 минут)...")
        logger.info("   Использую параллельное выполнение и GIN индекс...")
        
        start_time = time.time()
        
        # Прямой UPDATE с использованием GIN индекса
        cursor.execute("""
            UPDATE processed_market_data
            SET technical_indicators = technical_indicators 
                - 'buy_expected_return'::text 
                - 'sell_expected_return'::text
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """)
        
        updated_count = cursor.rowcount
        elapsed = time.time() - start_time
        
        # Коммитим изменения
        conn.commit()
        
        logger.info(f"\n✅ Очистка завершена!")
        logger.info(f"   Обновлено: {updated_count:,} записей")
        logger.info(f"   Время: {elapsed:.1f} сек ({elapsed/60:.1f} мин)")
        logger.info(f"   Скорость: {updated_count/elapsed:.0f} записей/сек")
        
        # 4. Проверяем результат
        logger.info("\n🔍 Проверка результата...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_market_data 
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """)
        remaining = cursor.fetchone()[0]
        
        if remaining == 0:
            logger.info("✅ Утечка данных успешно устранена!")
            
            # 5. Оптимизация таблицы
            logger.info("\n🔧 Оптимизация таблицы...")
            conn.autocommit = True
            cursor.execute("VACUUM (ANALYZE, VERBOSE) processed_market_data")
            logger.info("✅ Таблица оптимизирована")
        else:
            logger.error(f"❌ Остались записи с утечкой: {remaining}")
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.info("\n📤 Подключение к PostgreSQL закрыто")


if __name__ == "__main__":
    logger.info("🚀 Прямое SQL решение для очистки утечки данных")
    logger.info("=" * 60)
    logger.info("ВНИМАНИЕ: Выполняет UPDATE всех записей одной командой!")
    logger.info("Использует параллельное выполнение PostgreSQL")
    logger.info("Ожидаемое время: 2-5 минут для 2.7М записей")
    logger.info("=" * 60)
    
    # Предупреждение
    logger.warning("\n⚠️  Эта операция заблокирует таблицу на время выполнения!")
    logger.warning("   Убедитесь, что нет других активных процессов")
    
    # Подтверждение
    response = input("\nПродолжить? (y/n): ")
    if response.lower() == 'y':
        fix_data_leakage_direct()
    else:
        logger.info("Отменено пользователем")