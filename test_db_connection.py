#!/usr/bin/env python3
"""
Скрипт для проверки подключения к PostgreSQL и инициализации БД
"""

import psycopg2
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connection():
    """Проверка подключения к БД"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    
    logger.info("🔍 Проверка подключения к PostgreSQL...")
    logger.info(f"   Хост: {db_config['host']}")
    logger.info(f"   Порт: {db_config['port']}")
    logger.info(f"   База: {db_config['dbname']}")
    logger.info(f"   Пользователь: {db_config['user']}")
    
    try:
        # Пытаемся подключиться
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        # Проверяем версию
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"✅ Подключение успешно!")
            logger.info(f"   PostgreSQL версия: {version}")
            
            # Проверяем существующие таблицы
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            
            if tables:
                logger.info(f"\n📋 Существующие таблицы:")
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                    count = cursor.fetchone()[0]
                    logger.info(f"   - {table[0]}: {count:,} записей")
            else:
                logger.info("\n⚠️  Таблицы не найдены. Запустите init_database.py для создания структуры БД")
        
        conn.close()
        
        logger.info("\n✅ Все проверки пройдены успешно!")
        logger.info("\n📋 Следующие шаги:")
        logger.info("   1. python init_database.py  # Создание таблиц")
        logger.info("   2. python download_data.py  # Загрузка данных (25 потоков)")
        
    except psycopg2.OperationalError as e:
        logger.error(f"❌ Ошибка подключения: {e}")
        logger.info("\n💡 Проверьте:")
        logger.info("   1. PostgreSQL запущен на порту 5555")
        logger.info("   2. Пользователь 'ruslan' существует")
        logger.info("   3. База данных 'crypto_trading' создана")
        logger.info("   4. Пароль в config.yaml корректный")
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}")

if __name__ == "__main__":
    test_connection()