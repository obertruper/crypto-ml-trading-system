#!/usr/bin/env python3
import psycopg2
import yaml

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database']

print(f"Подключение к БД: {db_config['dbname']} на {db_config['host']}:{db_config['port']}")

try:
    # Подключаемся
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Проверяем таблицы
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
    """)
    
    tables = cursor.fetchall()
    
    if tables:
        print(f"\nНайдено таблиц: {len(tables)}")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"  - {table[0]}: {count} записей")
    else:
        print("\nТаблицы не найдены!")
        
        # Создаем таблицы
        print("\nСоздаем таблицы...")
        
        # Импортируем и запускаем init_tables
        from init_database import init_tables
        init_tables(db_config)
        
        print("\nТаблицы созданы!")
        
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()