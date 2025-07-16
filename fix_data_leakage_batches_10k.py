#!/usr/bin/env python3
"""
Исправление утечки данных батчами по 10,000 записей
Оптимизировано для быстрой работы с коммитами после каждого батча
"""

import psycopg2
import yaml
import logging
import time
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_data_leakage_batches():
    """Удаляет expected_returns из technical_indicators батчами по 10k"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    # Подключаемся к БД
    conn = psycopg2.connect(**db_config)
    conn.autocommit = False  # Управляем транзакциями вручную
    
    logger.info("✅ Подключение к PostgreSQL установлено")
    
    try:
        cursor = conn.cursor()
        
        # 1. Получаем общее количество записей с утечкой
        logger.info("\n🔍 Анализ данных...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_market_data 
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """)
        total_with_leak = cursor.fetchone()[0]
        
        logger.info(f"   Записей с утечкой: {total_with_leak:,}")
        
        if total_with_leak == 0:
            logger.info("✅ Утечка данных не обнаружена!")
            return
        
        # 2. Настройка параметров для оптимизации
        logger.info("\n⚙️ Настройка параметров PostgreSQL...")
        cursor.execute("SET work_mem = '256MB'")
        cursor.execute("SET maintenance_work_mem = '512MB'")
        conn.commit()
        
        # 3. Создаем временную таблицу с ID для обработки
        logger.info("\n📋 Создание списка ID для обработки...")
        cursor.execute("DROP TABLE IF EXISTS temp_leak_ids")
        cursor.execute("""
            CREATE TEMP TABLE temp_leak_ids AS
            SELECT id 
            FROM processed_market_data 
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
            ORDER BY id
        """)
        cursor.execute("CREATE INDEX idx_temp_leak_ids ON temp_leak_ids(id)")
        cursor.execute("ANALYZE temp_leak_ids")
        conn.commit()
        
        # 3. Обработка батчами по 10,000
        batch_size = 10000
        processed = 0
        start_time = time.time()
        failed_batches = 0
        last_logged = 0  # Для отслеживания логирования
        
        logger.info(f"\n🚀 Начинаем очистку батчами по {batch_size:,} записей...")
        
        with tqdm(total=total_with_leak, desc="Очистка данных", unit="записей") as pbar:
            while processed < total_with_leak:
                batch_start = time.time()
                
                try:
                    # Обновляем батч через JOIN с временной таблицей (без ORDER BY для скорости)
                    cursor.execute("""
                        UPDATE processed_market_data p
                        SET technical_indicators = p.technical_indicators 
                            - 'buy_expected_return'::text 
                            - 'sell_expected_return'::text
                        FROM (
                            SELECT id FROM temp_leak_ids
                            LIMIT %s OFFSET %s
                        ) t
                        WHERE p.id = t.id
                    """, (batch_size, processed))
                    
                    batch_updated = cursor.rowcount
                    
                    # Если ничего не обновлено, но мы не в конце - проблема
                    if batch_updated == 0 and processed < total_with_leak:
                        logger.warning(f"⚠️ Батч {processed//batch_size + 1} вернул 0 обновлений")
                        # Пропускаем этот батч
                        processed += batch_size
                        continue
                    
                    # Коммитим после каждого батча
                    conn.commit()
                    
                    processed += batch_updated
                    pbar.update(batch_updated)
                    
                    # Статистика
                    batch_time = time.time() - batch_start
                    total_elapsed = time.time() - start_time
                    speed = processed / total_elapsed if total_elapsed > 0 else 0
                    remaining_records = total_with_leak - processed
                    eta = remaining_records / speed if speed > 0 else 0
                    
                    pbar.set_postfix({
                        'Батч': f'{batch_time:.1f}с',
                        'Скорость': f'{speed:.0f} зап/с',
                        'ETA': f'{eta/60:.1f} мин'
                    })
                    
                    # Логирование прогресса каждые 100k записей
                    if processed - last_logged >= 100000:
                        logger.info(f"   ✅ Обработано {processed:,} записей ({processed/total_with_leak*100:.1f}%)")
                        last_logged = processed
                    
                    # Если обновлено меньше batch_size, значит закончили
                    if batch_updated < batch_size:
                        break
                        
                except Exception as e:
                    failed_batches += 1
                    logger.error(f"❌ Ошибка в батче {processed//batch_size + 1}: {e}")
                    conn.rollback()
                    
                    if failed_batches > 3:
                        logger.error("❌ Слишком много ошибок, прерываем обработку")
                        raise
                    
                    # НЕ увеличиваем processed при ошибке - попробуем этот батч снова
                    time.sleep(1)  # Пауза перед повторной попыткой
                    continue
        
        # Итоговая статистика
        total_time = time.time() - start_time
        logger.info(f"\n✅ Очистка завершена!")
        logger.info(f"   Обработано: {processed:,} записей")
        logger.info(f"   Время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
        logger.info(f"   Средняя скорость: {processed/total_time:.0f} записей/сек")
        
        if failed_batches > 0:
            logger.warning(f"   ⚠️ Было {failed_batches} ошибок при обработке")
        
        # 4. Финальная проверка
        logger.info("\n🔍 Проверка результата...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM processed_market_data 
            WHERE technical_indicators ?| array['buy_expected_return', 'sell_expected_return']
        """)
        remaining = cursor.fetchone()[0]
        
        if remaining == 0:
            logger.info("✅ Утечка данных успешно устранена!")
            
            # 5. Очистка и оптимизация
            logger.info("\n🧹 Очистка временных данных...")
            cursor.execute("DROP TABLE IF EXISTS temp_leak_ids")
            conn.commit()
            
            logger.info("🔧 Оптимизация таблицы (VACUUM ANALYZE)...")
            conn.autocommit = True
            cursor.execute("VACUUM ANALYZE processed_market_data")
            logger.info("✅ Таблица оптимизирована")
            
        else:
            logger.warning(f"⚠️ Остались записи с утечкой: {remaining:,}")
            logger.info("   Возможно, нужно запустить скрипт повторно")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Операция прервана пользователем")
        conn.rollback()
        raise
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        logger.info("\n📤 Подключение к PostgreSQL закрыто")


if __name__ == "__main__":
    logger.info("🚀 Исправление утечки данных батчами по 10,000")
    logger.info("=" * 60)
    logger.info("Оптимизированная версия с коммитами после каждого батча")
    logger.info("Использует временную таблицу для эффективности")
    logger.info("Ожидаемое время: 3-5 минут для 2.5М записей")
    logger.info("=" * 60)
    
    # Подтверждение
    response = input("\nПродолжить? (y/n): ")
    if response.lower() == 'y':
        try:
            fix_data_leakage_batches()
        except KeyboardInterrupt:
            logger.info("\n❌ Операция отменена пользователем")
        except Exception as e:
            logger.error(f"\n❌ Ошибка выполнения: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("Отменено пользователем")