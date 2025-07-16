#!/usr/bin/env python3
"""
Оценка времени завершения подготовки датасета
"""

import psycopg2
import yaml
import time
from datetime import datetime, timedelta

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

# История прогресса для расчета скорости
progress_history = []

def get_progress():
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM processed_market_data")
    total = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) 
        FROM processed_market_data 
        WHERE technical_indicators->>'buy_expected_return' IS NOT NULL
    """)
    completed = cursor.fetchone()[0]
    
    conn.close()
    return total, completed

# Собираем данные за 30 секунд
print("🔄 Измеряем скорость обработки...")
for i in range(4):
    total, completed = get_progress()
    progress_history.append((time.time(), completed))
    print(f"   {i+1}/4: {completed:,} записей обработано")
    if i < 3:
        time.sleep(10)

# Расчет скорости
if len(progress_history) >= 2:
    time_diff = progress_history[-1][0] - progress_history[0][0]
    records_diff = progress_history[-1][1] - progress_history[0][1]
    
    if time_diff > 0 and records_diff > 0:
        speed = records_diff / time_diff  # записей в секунду
        
        total, completed = get_progress()
        remaining = total - completed
        
        if speed > 0:
            eta_seconds = remaining / speed
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            
            print("\n="*60)
            print("📊 ОЦЕНКА ВРЕМЕНИ ЗАВЕРШЕНИЯ")
            print("="*60)
            print(f"Всего записей: {total:,}")
            print(f"Обработано: {completed:,} ({completed/total*100:.1f}%)")
            print(f"Осталось: {remaining:,}")
            print(f"Скорость: {speed:.0f} записей/сек")
            print(f"Примерное время завершения: {eta.strftime('%H:%M:%S')}")
            print(f"Осталось времени: {int(eta_seconds//60)} минут")
            print("="*60)
        else:
            print("⚠️ Процесс не движется")
    else:
        print("⚠️ Недостаточно данных для оценки")
else:
    print("❌ Не удалось собрать данные")