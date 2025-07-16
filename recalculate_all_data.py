#!/usr/bin/env python3
"""
Скрипт для полного пересчета всех данных с улучшенной логикой
"""

import psycopg2
import yaml
import subprocess
import os
import time

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

print("="*80)
print("🔄 ПОЛНЫЙ ПЕРЕСЧЕТ ДАННЫХ")
print("="*80)

# Подключение к БД
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# 1. Проверяем текущее состояние
print("\n📊 Текущее состояние БД:")
cursor.execute("SELECT COUNT(*) FROM processed_market_data")
current_count = cursor.fetchone()[0]
print(f"   Записей в processed_market_data: {current_count:,}")

cursor.execute("SELECT COUNT(DISTINCT symbol) FROM processed_market_data")
current_symbols = cursor.fetchone()[0]
print(f"   Обработано символов: {current_symbols}")

# 2. Резервное копирование статистики
print("\n📊 Статистика текущих данных:")
cursor.execute("""
    SELECT 
        COUNT(DISTINCT buy_expected_return) as unique_buy,
        COUNT(DISTINCT sell_expected_return) as unique_sell,
        COUNT(*) as total
    FROM processed_market_data
""")
result = cursor.fetchone()
if result:
    unique_buy, unique_sell, total = result
    print(f"   Уникальность buy_expected_return: {unique_buy/total*100:.1f}%")
    print(f"   Уникальность sell_expected_return: {unique_sell/total*100:.1f}%")

# 3. Подтверждение
print("\n⚠️ ВНИМАНИЕ!")
print("Это действие:")
print("1. Удалит все данные из processed_market_data")
print("2. Пересчитает все индикаторы и expected_return с улучшенной логикой")
print("3. Процесс может занять несколько часов")

response = input("\nПродолжить? (yes/no): ")
if response.lower() != 'yes':
    print("❌ Операция отменена")
    conn.close()
    exit()

# 4. Удаляем чекпоинт если есть
checkpoint_file = 'prepare_dataset_checkpoint.pkl'
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print(f"\n✅ Удален старый чекпоинт: {checkpoint_file}")

# 5. Очищаем таблицу
print("\n🗑️ Очистка таблицы processed_market_data...")
cursor.execute("TRUNCATE TABLE processed_market_data")
conn.commit()
print("✅ Таблица очищена")

# 6. Закрываем соединение перед запуском prepare_dataset
conn.close()

# 7. Запускаем пересчет
print("\n🚀 Запускаем пересчет данных...")
print("   Это может занять несколько часов...")
print("   Следите за прогрессом ниже:\n")

start_time = time.time()

try:
    # Запускаем prepare_dataset.py
    process = subprocess.Popen(
        ['python', 'prepare_dataset.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Выводим вывод в реальном времени
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
    
    process.wait()
    
    if process.returncode == 0:
        print("\n✅ Пересчет данных успешно завершен!")
    else:
        print(f"\n❌ Ошибка при пересчете данных (код: {process.returncode})")
        
except KeyboardInterrupt:
    print("\n\n⚠️ Прервано пользователем")
    print("💡 Используйте 'python prepare_dataset.py --resume' для продолжения")
    process.terminate()
    
elapsed_time = time.time() - start_time
print(f"\n⏱️ Общее время: {elapsed_time/60:.1f} минут")

# 8. Проверяем результаты
print("\n🔍 Проверка результатов...")
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM processed_market_data")
new_count = cursor.fetchone()[0]
print(f"   Новых записей: {new_count:,}")

cursor.execute("SELECT COUNT(DISTINCT symbol) FROM processed_market_data")
new_symbols = cursor.fetchone()[0]
print(f"   Обработано символов: {new_symbols}")

# Проверяем уникальность
cursor.execute("""
    SELECT 
        COUNT(DISTINCT buy_expected_return) as unique_buy,
        COUNT(DISTINCT sell_expected_return) as unique_sell,
        COUNT(*) as total
    FROM processed_market_data
""")
result = cursor.fetchone()
if result:
    unique_buy, unique_sell, total = result
    buy_uniqueness = unique_buy/total*100
    sell_uniqueness = unique_sell/total*100
    print(f"\n📊 Новая статистика:")
    print(f"   Уникальность buy_expected_return: {buy_uniqueness:.1f}%")
    print(f"   Уникальность sell_expected_return: {sell_uniqueness:.1f}%")
    
    if buy_uniqueness > 50 and sell_uniqueness > 50:
        print("\n✅ Данные готовы для обучения!")
        print("   Запустите: python train_universal_transformer.py --task regression")
    else:
        print("\n⚠️ Уникальность все еще низкая. Проверьте логику расчета.")

conn.close()

print("\n" + "="*80)