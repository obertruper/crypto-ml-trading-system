#!/usr/bin/env python3
"""
Скрипт для полного пересчета всех данных с реалистичными точками входа
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
print("🔄 ПОЛНЫЙ ПЕРЕСЧЕТ ДАННЫХ С РЕАЛИСТИЧНЫМИ ТОЧКАМИ ВХОДА")
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

# 2. Анализ текущих данных
print("\n📊 Анализ текущих данных:")
cursor.execute("""
    SELECT 
        COUNT(*) as total_bars,
        SUM(CASE WHEN buy_expected_return != 0 THEN 1 ELSE 0 END) as buy_entries,
        SUM(CASE WHEN sell_expected_return != 0 THEN 1 ELSE 0 END) as sell_entries,
        COUNT(DISTINCT buy_expected_return) as unique_buy,
        COUNT(DISTINCT sell_expected_return) as unique_sell
    FROM processed_market_data
""")
result = cursor.fetchone()
if result and result[0] > 0:
    total, buy_entries, sell_entries, unique_buy, unique_sell = result
    print(f"   Всего баров: {total:,}")
    print(f"   BUY точек входа: {buy_entries:,} ({buy_entries/total*100:.1f}% от всех баров)")
    print(f"   SELL точек входа: {sell_entries:,} ({sell_entries/total*100:.1f}% от всех баров)")
    print(f"   Уникальность buy_expected_return: {unique_buy/total*100:.1f}%")
    print(f"   Уникальность sell_expected_return: {unique_sell/total*100:.1f}%")

# 3. Подтверждение
print("\n⚠️ ВНИМАНИЕ!")
print("Это действие:")
print("1. Обновит схему БД для новых полей (is_long_entry, is_short_entry и т.д.)")
print("2. Удалит все данные из processed_market_data")
print("3. Пересчитает все с РЕАЛИСТИЧНЫМИ точками входа (~2% баров)")
print("4. Процесс может занять несколько часов")

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

# 7. Запускаем пересчет с реалистичными точками входа
print("\n🚀 Запускаем пересчет данных с реалистичными точками входа...")
print("   Ожидаемое время: 2-4 часа")
print("   Следите за прогрессом ниже:\n")

start_time = time.time()

try:
    # Запускаем prepare_dataset_realistic.py
    process = subprocess.Popen(
        ['python', 'prepare_dataset_realistic.py'],
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

# Проверяем статистику точек входа
cursor.execute("""
    SELECT 
        COUNT(*) as total_bars,
        SUM(CASE WHEN is_long_entry THEN 1 ELSE 0 END) as long_entries,
        SUM(CASE WHEN is_short_entry THEN 1 ELSE 0 END) as short_entries,
        COUNT(DISTINCT buy_expected_return) as unique_buy,
        COUNT(DISTINCT sell_expected_return) as unique_sell
    FROM processed_market_data
""")
result = cursor.fetchone()
if result:
    total, long_entries, short_entries, unique_buy, unique_sell = result
    buy_uniqueness = unique_buy/total*100
    sell_uniqueness = unique_sell/total*100
    
    print(f"\n📊 Новая статистика:")
    print(f"   Всего баров: {total:,}")
    print(f"   LONG точек входа: {long_entries:,} ({long_entries/total*100:.2f}% от всех баров)")
    print(f"   SHORT точек входа: {short_entries:,} ({short_entries/total*100:.2f}% от всех баров)")
    print(f"   Уникальность buy_expected_return: {buy_uniqueness:.1f}%")
    print(f"   Уникальность sell_expected_return: {sell_uniqueness:.1f}%")
    
    # Анализ по типам входов
    cursor.execute("""
        SELECT 
            long_entry_type,
            COUNT(*) as count,
            AVG(buy_expected_return) as avg_return
        FROM processed_market_data
        WHERE is_long_entry = TRUE
        GROUP BY long_entry_type
        ORDER BY avg_return DESC
    """)
    
    print("\n📊 Анализ LONG входов по типам:")
    for entry_type, count, avg_return in cursor.fetchall():
        print(f"   {entry_type}: {count} входов, средний return: {avg_return:.2f}%")
    
    cursor.execute("""
        SELECT 
            short_entry_type,
            COUNT(*) as count,
            AVG(sell_expected_return) as avg_return
        FROM processed_market_data
        WHERE is_short_entry = TRUE
        GROUP BY short_entry_type
        ORDER BY avg_return DESC
    """)
    
    print("\n📊 Анализ SHORT входов по типам:")
    for entry_type, count, avg_return in cursor.fetchall():
        print(f"   {entry_type}: {count} входов, средний return: {avg_return:.2f}%")
    
    if buy_uniqueness > 50 and sell_uniqueness > 50:
        print("\n✅ Данные готовы для обучения!")
        print("   Запустите: python train_universal_transformer.py --task regression")
    else:
        print("\n⚠️ Уникальность все еще низкая. Проверьте логику расчета.")

conn.close()

print("\n" + "="*80)