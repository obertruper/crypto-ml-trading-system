#!/usr/bin/env python3
"""
Очистка таблицы processed_market_data для пересчета с новой логикой
"""

import psycopg2
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_processed_data():
    """Очищает таблицу processed_market_data"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    # Удаляем пустой пароль
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    try:
        # Подключаемся к БД
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Проверяем количество записей до очистки
        cursor.execute("SELECT COUNT(*) FROM processed_market_data")
        count_before = cursor.fetchone()[0]
        logger.info(f"📊 Записей в таблице до очистки: {count_before:,}")
        
        if count_before > 0:
            # Спрашиваем подтверждение
            response = input(f"\n⚠️  Вы уверены, что хотите удалить {count_before:,} записей? (yes/no): ")
            
            if response.lower() == 'yes':
                # Очищаем таблицу
                logger.info("🗑️  Очищаю таблицу processed_market_data...")
                cursor.execute("TRUNCATE TABLE processed_market_data")
                logger.info("✅ Таблица успешно очищена!")
                
                # Проверяем результат
                cursor.execute("SELECT COUNT(*) FROM processed_market_data")
                count_after = cursor.fetchone()[0]
                logger.info(f"📊 Записей в таблице после очистки: {count_after}")
                
                # Также очищаем метаданные о признаках
                cursor.execute("DELETE FROM model_metadata WHERE model_name = 'feature_extraction'")
                logger.info("✅ Метаданные о признаках также очищены")
            else:
                logger.info("❌ Операция отменена")
        else:
            logger.info("ℹ️  Таблица уже пуста")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Ошибка при работе с БД: {e}")
        raise

if __name__ == "__main__":
    clear_processed_data()
    print("\n💡 Теперь можно запустить пересчет:")
    print("   python prepare_dataset.py")