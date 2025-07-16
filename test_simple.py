#!/usr/bin/env python3
import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")

try:
    import psycopg2
    print("✅ psycopg2 imported successfully")
    
    import yaml
    print("✅ yaml imported successfully")
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    print(f"\n📋 DB Config:")
    print(f"   Host: {db_config['host']}")
    print(f"   Port: {db_config['port']}")
    print(f"   DB: {db_config['dbname']}")
    print(f"   User: {db_config['user']}")
    
    # Пытаемся подключиться
    conn = psycopg2.connect(**db_config)
    print("\n✅ Successfully connected to PostgreSQL!")
    
    with conn.cursor() as cursor:
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"   PostgreSQL version: {version}")
        
        # Проверяем таблицы
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_count = cursor.fetchone()[0]
        print(f"   Tables in database: {table_count}")
    
    conn.close()
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()