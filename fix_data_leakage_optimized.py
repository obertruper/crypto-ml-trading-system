#!/usr/bin/env python3
"""
Оптимизированное исправление утечки данных
"""

import psycopg2
from psycopg2.extras import Json
import yaml
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_data_leakage_optimized():
    """Удаляет expected_returns из technical_indicators оптимизированным способом"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    # Подключаемся к БД с autocommit=True изначально
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    
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
        
        # 2. Показываем пример данных
        cursor.execute("""
            SELECT id, 
                   array_length(array(SELECT jsonb_object_keys(technical_indicators)), 1) as keys_count,
                   pg_column_size(technical_indicators) as json_size
            FROM processed_market_data 
            WHERE technical_indicators ? 'buy_expected_return'
            LIMIT 1
        """)
        sample = cursor.fetchone()
        if sample:
            logger.info(f"\n📊 Пример записи с утечкой:")
            logger.info(f"   ID: {sample[0]}")
            logger.info(f"   Количество ключей: {sample[1]}")
            logger.info(f"   Размер JSON: {sample[2]} байт")
        
        # 3. Выполняем очистку через прямой SQL UPDATE
        logger.info("\n🔧 Выполнение очистки...")
        logger.info("   Используем оптимизированный SQL запрос...")
        
        start_time = time.time()
        
        # Начинаем транзакцию
        cursor.execute("BEGIN")
        
        # Увеличиваем work_mem для ускорения операции
        cursor.execute("SET work_mem = '256MB'")
        
        # Выполняем UPDATE одним запросом
        update_query = """
            UPDATE processed_market_data
            SET technical_indicators = technical_indicators 
                - 'buy_expected_return'::text 
                - 'sell_expected_return'::text
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """
        
        logger.info("   Запуск UPDATE (это может занять 1-2 минуты)...")
        cursor.execute(update_query)
        
        updated_count = cursor.rowcount
        elapsed = time.time() - start_time
        
        logger.info(f"\n✅ Обновлено {updated_count:,} записей за {elapsed:.1f} секунд")
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
            
            # Проверяем пример очищенных данных
            if sample:
                cursor.execute("""
                    SELECT array_length(array(SELECT jsonb_object_keys(technical_indicators)), 1) as keys_count,
                           pg_column_size(technical_indicators) as json_size
                    FROM processed_market_data 
                    WHERE id = %s
                """, (sample[0],))
                cleaned = cursor.fetchone()
                if cleaned:
                    logger.info(f"\n📊 После очистки:")
                    logger.info(f"   ID: {sample[0]}")
                    logger.info(f"   Количество ключей: {cleaned[0]} (было {sample[1]})")
                    logger.info(f"   Размер JSON: {cleaned[1]} байт (было {sample[2]})")
                    logger.info(f"   Экономия: {sample[2] - cleaned[1]} байт на запись")
                    logger.info(f"   Общая экономия: ~{(sample[2] - cleaned[1]) * updated_count / 1024 / 1024:.0f} МБ")
            
            # Коммитим изменения
            logger.info("\n💾 Сохранение изменений...")
            cursor.execute("COMMIT")
            logger.info("✅ Изменения успешно сохранены в базе данных")
            
            # 5. Выполняем VACUUM для оптимизации таблицы
            logger.info("\n🔧 Оптимизация таблицы (VACUUM)...")
            cursor.execute("VACUUM ANALYZE processed_market_data")
            logger.info("✅ Таблица оптимизирована")
            
        else:
            logger.error(f"❌ Остались записи с утечкой: {remaining}")
            cursor.execute("ROLLBACK")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Операция прервана пользователем")
        cursor.execute("ROLLBACK")
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        cursor.execute("ROLLBACK")
        raise
    finally:
        conn.close()
        logger.info("\n📤 Подключение к PostgreSQL закрыто")


if __name__ == "__main__":
    logger.info("🚀 Оптимизированное исправление утечки данных")
    logger.info("=" * 60)
    logger.info("ВНИМАНИЕ: Эта операция изменит данные в базе!")
    logger.info("Рекомендуется сделать резервную копию перед запуском")
    logger.info("=" * 60)
    
    # Подтверждение
    response = input("\nПродолжить? (y/n): ")
    if response.lower() == 'y':
        try:
            fix_data_leakage_optimized()
        except KeyboardInterrupt:
            logger.info("\n❌ Операция отменена")
    else:
        logger.info("Отменено пользователем")