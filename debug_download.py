#!/usr/bin/env python3
import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print("\nПроверка импортов:")

try:
    import requests
    print("✅ requests")
except Exception as e:
    print(f"❌ requests: {e}")

try:
    import pandas
    print("✅ pandas")
except Exception as e:
    print(f"❌ pandas: {e}")

try:
    import psycopg2
    print("✅ psycopg2")
except Exception as e:
    print(f"❌ psycopg2: {e}")

try:
    import yaml
    print("✅ yaml")
except Exception as e:
    print(f"❌ yaml: {e}")

print("\nЗапуск download_data.py...")
try:
    # Читаем конец файла чтобы проверить if __name__ == "__main__"
    with open('download_data.py', 'r') as f:
        lines = f.readlines()
        last_lines = lines[-5:]
        print("Последние строки download_data.py:")
        for line in last_lines:
            print(f"  {line.rstrip()}")
    
    # Импортируем и запускаем
    import download_data
    if hasattr(download_data, 'main'):
        print("\nВызываем main()...")
        download_data.main()
    else:
        print("\n❌ Функция main() не найдена!")
        
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()