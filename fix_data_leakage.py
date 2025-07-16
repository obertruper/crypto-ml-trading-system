#!/usr/bin/env python3
"""
Исправление утечки данных - удаление expected_returns из technical_indicators
"""

import psycopg2
from psycopg2.extras import Json, execute_values
import json
import yaml
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_data_leakage():
    """Удаляет expected_returns из technical_indicators во всех записях"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    # Подключаемся к БД
    conn = psycopg2.connect(**db_config)
    conn.autocommit = False  # Используем транзакции
    
    logger.info("✅ Подключение к PostgreSQL установлено")
    
    try:
        with conn.cursor() as cursor:
            # Получаем количество записей для обработки
            cursor.execute("SELECT COUNT(*) FROM processed_market_data")
            total_count = cursor.fetchone()[0]
            logger.info(f"📊 Всего записей для обработки: {total_count:,}")
            
            # Обрабатываем батчами
            batch_size = 10000
            offset = 0
            fixed_count = 0
            
            with tqdm(total=total_count, desc="Исправление утечки данных") as pbar:
                while offset < total_count:
                    # Читаем батч записей
                    cursor.execute("""
                        SELECT id, technical_indicators 
                        FROM processed_market_data 
                        ORDER BY id
                        LIMIT %s OFFSET %s
                    """, (batch_size, offset))
                    
                    records = cursor.fetchall()
                    updates = []
                    
                    for record_id, indicators in records:
                        if indicators:
                            # Проверяем наличие утечки
                            if 'buy_expected_return' in indicators or 'sell_expected_return' in indicators:
                                # Удаляем целевые переменные
                                cleaned_indicators = {k: v for k, v in indicators.items() 
                                                    if k not in ['buy_expected_return', 'sell_expected_return']}
                                updates.append((Json(cleaned_indicators), record_id))
                                fixed_count += 1
                    
                    # Обновляем записи если есть что обновлять
                    if updates:
                        # Используем обычный executemany для UPDATE
                        cursor.executemany(
                            "UPDATE processed_market_data SET technical_indicators = %s WHERE id = %s",
                            updates
                        )
                        
                        # Фиксируем изменения каждые 10 батчей
                        if (offset // batch_size) % 10 == 0:
                            conn.commit()
                            logger.info(f"   💾 Сохранено {fixed_count} исправлений...")
                    
                    offset += batch_size
                    pbar.update(len(records))
            
            # Финальный commit
            conn.commit()
            
            logger.info(f"\n✅ Исправление завершено!")
            logger.info(f"   📊 Обработано записей: {total_count:,}")
            logger.info(f"   🔧 Исправлено записей: {fixed_count:,}")
            
            # Проверяем результат
            logger.info("\n🔍 Проверка результата...")
            cursor.execute("""
                SELECT COUNT(*) 
                FROM processed_market_data 
                WHERE technical_indicators::text LIKE '%expected_return%'
            """)
            remaining = cursor.fetchone()[0]
            
            if remaining == 0:
                logger.info("   ✅ Утечка данных успешно устранена!")
            else:
                logger.error(f"   ❌ Остались записи с утечкой: {remaining}")
                
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Ошибка: {e}")
        raise
    finally:
        conn.close()
        logger.info("\n📤 Подключение к PostgreSQL закрыто")


if __name__ == "__main__":
    logger.info("🚀 Запуск исправления утечки данных...")
    logger.info("⚠️  Это может занять несколько минут для больших датасетов")
    
    # Подтверждение
    response = input("\nПродолжить? (y/n): ")
    if response.lower() == 'y':
        fix_data_leakage()
    else:
        logger.info("Отменено пользователем")