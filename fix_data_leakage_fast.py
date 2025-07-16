#!/usr/bin/env python3
"""
Быстрое исправление утечки данных через SQL
"""

import psycopg2
import yaml
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_data_leakage_fast():
    """Удаляет expected_returns из technical_indicators используя SQL"""
    
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
        with conn.cursor() as cursor:
            # Проверяем количество записей с утечкой
            logger.info("🔍 Проверка записей с утечкой...")
            cursor.execute("""
                SELECT COUNT(*) 
                FROM processed_market_data 
                WHERE technical_indicators ? 'buy_expected_return' 
                   OR technical_indicators ? 'sell_expected_return'
            """)
            leak_count = cursor.fetchone()[0]
            logger.info(f"   Найдено записей с утечкой: {leak_count:,}")
            
            if leak_count == 0:
                logger.info("✅ Утечка данных не обнаружена!")
                return
            
            # Создаем резервную копию нескольких записей для проверки
            logger.info("\n📋 Пример данных ДО очистки:")
            cursor.execute("""
                SELECT id, technical_indicators 
                FROM processed_market_data 
                WHERE technical_indicators ? 'buy_expected_return' 
                LIMIT 1
            """)
            sample = cursor.fetchone()
            if sample:
                logger.info(f"   ID: {sample[0]}")
                logger.info(f"   Ключи в technical_indicators: {list(sample[1].keys())[:10]}...")
            
            # Выполняем очистку через SQL
            logger.info("\n🔧 Выполнение очистки через SQL...")
            start_time = time.time()
            
            cursor.execute("""
                UPDATE processed_market_data
                SET technical_indicators = technical_indicators - 'buy_expected_return' - 'sell_expected_return'
                WHERE technical_indicators ? 'buy_expected_return' 
                   OR technical_indicators ? 'sell_expected_return'
            """)
            
            updated_count = cursor.rowcount
            elapsed_time = time.time() - start_time
            
            logger.info(f"✅ Обновлено {updated_count:,} записей за {elapsed_time:.2f} секунд")
            
            # Проверяем результат
            logger.info("\n🔍 Проверка результата...")
            cursor.execute("""
                SELECT COUNT(*) 
                FROM processed_market_data 
                WHERE technical_indicators ? 'buy_expected_return' 
                   OR technical_indicators ? 'sell_expected_return'
            """)
            remaining = cursor.fetchone()[0]
            
            if remaining == 0:
                logger.info("✅ Утечка данных успешно устранена!")
                
                # Показываем пример очищенных данных
                logger.info("\n📋 Пример данных ПОСЛЕ очистки:")
                cursor.execute("""
                    SELECT id, technical_indicators 
                    FROM processed_market_data 
                    WHERE id = %s
                """, (sample[0],) if sample else (1,))
                cleaned = cursor.fetchone()
                if cleaned:
                    logger.info(f"   ID: {cleaned[0]}")
                    logger.info(f"   Ключи в technical_indicators: {list(cleaned[1].keys())[:10]}...")
                
                # Фиксируем изменения
                conn.commit()
                logger.info("\n💾 Изменения сохранены в базе данных")
            else:
                logger.error(f"❌ Остались записи с утечкой: {remaining}")
                conn.rollback()
                
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Ошибка: {e}")
        raise
    finally:
        conn.close()
        logger.info("\n📤 Подключение к PostgreSQL закрыто")


if __name__ == "__main__":
    logger.info("🚀 Быстрое исправление утечки данных через SQL")
    logger.info("⚠️  Это изменит данные в базе!")
    
    # Подтверждение
    response = input("\nПродолжить? (y/n): ")
    if response.lower() == 'y':
        fix_data_leakage_fast()
    else:
        logger.info("Отменено пользователем")