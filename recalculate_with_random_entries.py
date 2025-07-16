#!/usr/bin/env python3
"""
Скрипт для полного пересчета данных со случайными точками входа
Expected return рассчитывается для ВСЕХ баров
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
print("🔄 ПОЛНЫЙ ПЕРЕСЧЕТ ДАННЫХ СО СЛУЧАЙНЫМИ ТОЧКАМИ ВХОДА")
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
    print(f"   BUY expected_return != 0: {buy_entries:,} ({buy_entries/total*100:.1f}%)")
    print(f"   SELL expected_return != 0: {sell_entries:,} ({sell_entries/total*100:.1f}%)")
    print(f"   Уникальность buy_expected_return: {unique_buy/total*100:.1f}%")
    print(f"   Уникальность sell_expected_return: {unique_sell/total*100:.1f}%")

# 3. Подтверждение
print("\n⚠️ ВНИМАНИЕ!")
print("Новый подход:")
print("1. Expected return рассчитывается для ВСЕХ баров (100%)")
print("2. ~15% баров случайно помечаются как точки входа (для статистики)")
print("3. Модель будет обучаться на ВСЕХ барах")
print("4. Удалит все данные из processed_market_data")
print("5. Процесс может занять несколько часов")

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

# 7. Запускаем пересчет с обновленным prepare_dataset.py
print("\n🚀 Запускаем пересчет данных...")
print("   Expected return будет рассчитан для ВСЕХ баров")
print("   ~15% баров будут помечены как случайные входы")
print("   Следите за прогрессом ниже:\n")

start_time = time.time()

try:
    # Запускаем обновленный prepare_dataset.py
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

# Проверяем новую статистику
cursor.execute("""
    SELECT 
        COUNT(*) as total_bars,
        SUM(CASE WHEN is_long_entry THEN 1 ELSE 0 END) as long_entries,
        SUM(CASE WHEN is_short_entry THEN 1 ELSE 0 END) as short_entries,
        COUNT(DISTINCT buy_expected_return) as unique_buy,
        COUNT(DISTINCT sell_expected_return) as unique_sell,
        AVG(CASE WHEN is_long_entry THEN buy_expected_return END) as avg_random_long,
        AVG(CASE WHEN is_short_entry THEN sell_expected_return END) as avg_random_short
    FROM processed_market_data
""")
result = cursor.fetchone()
if result:
    total, long_entries, short_entries, unique_buy, unique_sell, avg_long, avg_short = result
    buy_uniqueness = unique_buy/total*100
    sell_uniqueness = unique_sell/total*100
    
    print(f"\n📊 Новая статистика:")
    print(f"   Всего баров: {total:,}")
    print(f"   Случайных LONG входов: {long_entries:,} ({long_entries/total*100:.1f}%)")
    print(f"   Случайных SHORT входов: {short_entries:,} ({short_entries/total*100:.1f}%)")
    print(f"   Уникальность buy_expected_return: {buy_uniqueness:.1f}%")
    print(f"   Уникальность sell_expected_return: {sell_uniqueness:.1f}%")
    
    if avg_long is not None and avg_short is not None:
        print(f"\n📈 Средние returns случайных входов:")
        print(f"   LONG: {avg_long:.3f}%")
        print(f"   SHORT: {avg_short:.3f}%")
    
    # Win rate случайных входов
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN is_long_entry AND buy_expected_return > 0 THEN 1 ELSE 0 END)::FLOAT / 
                NULLIF(SUM(CASE WHEN is_long_entry THEN 1 ELSE 0 END), 0) * 100 as long_wr,
            SUM(CASE WHEN is_short_entry AND sell_expected_return > 0 THEN 1 ELSE 0 END)::FLOAT / 
                NULLIF(SUM(CASE WHEN is_short_entry THEN 1 ELSE 0 END), 0) * 100 as short_wr
        FROM processed_market_data
    """)
    wr_result = cursor.fetchone()
    if wr_result:
        long_wr, short_wr = wr_result
        if long_wr and short_wr:
            print(f"\n🎯 Win Rate случайных входов:")
            print(f"   LONG: {long_wr:.1f}%")
            print(f"   SHORT: {short_wr:.1f}%")
            
            avg_wr = (long_wr + short_wr) / 2
            if avg_wr < 45:
                print(f"\n✅ Реалистичные данные! Win rate ~{avg_wr:.0f}%")
            else:
                print(f"\n⚠️ Win rate {avg_wr:.0f}% высоковат для случайных входов")
    
    if buy_uniqueness > 80 and sell_uniqueness > 80:
        print("\n✅ Данные готовы для обучения!")
        print("   - Expected return рассчитан для всех баров")
        print("   - Высокая уникальность значений")
        print("   - Случайные входы для валидации")
        print("\n📊 Анализ данных: python analyze_random_entries.py")
        print("🚀 Обучение: python train_universal_transformer.py --task regression")
    else:
        print("\n⚠️ Уникальность все еще низкая. Проверьте логику расчета.")

conn.close()

print("\n" + "="*80)