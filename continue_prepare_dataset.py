#!/usr/bin/env python3
"""
Скрипт для продолжения расчета индикаторов с места остановки
"""

import psycopg2
import yaml
import subprocess
import sys
import pickle
import os
from datetime import datetime

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

print("="*80)
print("🔄 ПРОДОЛЖЕНИЕ РАСЧЕТА ИНДИКАТОРОВ")
print("="*80)

# Подключение к БД
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Получаем список всех символов из конфига
all_symbols = config['data_download']['symbols']

# Получаем обработанные символы
cursor.execute('SELECT DISTINCT symbol FROM processed_market_data ORDER BY symbol')
processed_symbols = set([row[0] for row in cursor.fetchall()])

# Получаем символы с сырыми данными
cursor.execute("""
    SELECT symbol, COUNT(*) as count 
    FROM raw_market_data 
    WHERE market_type = 'futures'
    GROUP BY symbol 
    ORDER BY symbol
""")
raw_data = {row[0]: row[1] for row in cursor.fetchall()}

print(f"\n📊 Статус обработки:")
print(f"Всего символов в конфиге: {len(all_symbols)}")
print(f"Символов с сырыми данными: {len(raw_data)}")
print(f"Обработано символов: {len(processed_symbols)}")

# Определяем необработанные символы
unprocessed_symbols = []
for symbol in all_symbols:
    if symbol in raw_data and symbol not in processed_symbols:
        unprocessed_symbols.append(symbol)

if not unprocessed_symbols:
    print("\n✅ Все символы уже обработаны!")
    cursor.close()
    conn.close()
    sys.exit(0)

print(f"\n📋 Необработанные символы ({len(unprocessed_symbols)}):")
for i, symbol in enumerate(unprocessed_symbols[:10], 1):
    print(f"   {i}. {symbol}: {raw_data.get(symbol, 0):,} записей")
if len(unprocessed_symbols) > 10:
    print(f"   ... и еще {len(unprocessed_symbols) - 10} символов")

# Проверяем checkpoint
checkpoint_file = 'prepare_dataset_checkpoint.pkl'
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"\n📌 Найден checkpoint от {checkpoint['timestamp']}")
    print(f"   Обработано: {len(checkpoint['processed'])} символов")

cursor.close()
conn.close()

# Спрашиваем подтверждение
print(f"\n🚀 Готов обработать {len(unprocessed_symbols)} символов")
response = input("Продолжить? (y/n): ")

if response.lower() != 'y':
    print("❌ Отменено пользователем")
    sys.exit(0)

# Запускаем prepare_dataset.py
print("\n🔄 Запускаем prepare_dataset.py...")
print("="*80)

try:
    # Запускаем с выводом в реальном времени
    process = subprocess.Popen(
        [sys.executable, 'prepare_dataset.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Читаем и выводим вывод в реальном времени
    for line in process.stdout:
        print(line, end='')
    
    # Ждем завершения
    return_code = process.wait()
    
    if return_code == 0:
        print("\n✅ Обработка успешно завершена!")
    else:
        print(f"\n❌ Ошибка при обработке (код возврата: {return_code})")
        
except KeyboardInterrupt:
    print("\n\n⚠️ Прервано пользователем (Ctrl+C)")
    print("Процесс сохранил checkpoint и может быть продолжен позже")
    process.terminate()
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    sys.exit(1)