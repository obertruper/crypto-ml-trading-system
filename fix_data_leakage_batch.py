#!/usr/bin/env python3
"""
Исправление утечки данных батчами с прогрессом
"""

import psycopg2
from psycopg2.extras import Json
import yaml
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_data_leakage_batch():
    """Удаляет expected_returns из technical_indicators батчами"""
    
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
        cursor = conn.cursor()
        
        # 1. Проверяем масштаб проблемы
        logger.info("\n🔍 Анализ данных...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_market_data 
            WHERE technical_indicators ? 'buy_expected_return' 
               OR technical_indicators ? 'sell_expected_return'
        """)
        total_with_leak = cursor.fetchone()[0]
        
        logger.info(f"   Записей с утечкой: {total_with_leak:,}")
        
        if total_with_leak == 0:
            logger.info("✅ Утечка данных не обнаружена!")
            return
        
        # 2. Обработка батчами
        batch_size = 50000  # Размер батча
        processed = 0
        start_time = time.time()
        
        logger.info(f"\n🔧 Начинаем очистку батчами по {batch_size:,} записей...")
        
        with tqdm(total=total_with_leak, desc="Очистка данных", unit="записей") as pbar:
            while processed < total_with_leak:
                # Обновляем батч записей
                cursor.execute("""
                    UPDATE processed_market_data
                    SET technical_indicators = technical_indicators 
                        - 'buy_expected_return'::text 
                        - 'sell_expected_return'::text
                    WHERE id IN (
                        SELECT id 
                        FROM processed_market_data 
                        WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
                        LIMIT %s
                    )
                """, (batch_size,))
                
                batch_updated = cursor.rowcount
                processed += batch_updated
                pbar.update(batch_updated)
                
                # Показываем скорость
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                remaining_time = (total_with_leak - processed) / speed if speed > 0 else 0
                
                pbar.set_postfix({
                    'Скорость': f'{speed:.0f} зап/сек',
                    'Осталось': f'{remaining_time/60:.1f} мин'
                })
                
                # Если обновлено меньше batch_size, значит закончили
                if batch_updated < batch_size:
                    break
        
        total_time = time.time() - start_time
        logger.info(f"\n✅ Очистка завершена!")
        logger.info(f"   Обработано: {processed:,} записей")
        logger.info(f"   Время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
        logger.info(f"   Скорость: {processed/total_time:.0f} записей/сек")
        
        # 3. Финальная проверка
        logger.info("\n🔍 Проверка результата...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_market_data 
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """)
        remaining = cursor.fetchone()[0]
        
        if remaining == 0:
            logger.info("✅ Утечка данных успешно устранена!")
            
            # 4. Оптимизация таблицы
            logger.info("\n🔧 Оптимизация таблицы (VACUUM ANALYZE)...")
            cursor.execute("VACUUM ANALYZE processed_market_data")
            logger.info("✅ Таблица оптимизирована")
        else:
            logger.error(f"❌ Остались записи с утечкой: {remaining}")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Операция прервана пользователем")
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise
    finally:
        conn.close()
        logger.info("\n📤 Подключение к PostgreSQL закрыто")


if __name__ == "__main__":
    logger.info("🚀 Исправление утечки данных батчами")
    logger.info("=" * 60)
    logger.info("Эта версия обрабатывает данные порциями")
    logger.info("и показывает прогресс в реальном времени")
    logger.info("=" * 60)
    
    # Подтверждение
    response = input("\nПродолжить? (y/n): ")
    if response.lower() == 'y':
        try:
            fix_data_leakage_batch()
        except KeyboardInterrupt:
            logger.info("\n❌ Операция отменена")
    else:
        logger.info("Отменено пользователем")