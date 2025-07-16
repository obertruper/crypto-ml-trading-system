#!/usr/bin/env python3
"""
Мониторинг прогресса подготовки датасета
"""

import psycopg2
import yaml
import time
import os

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

def check_progress():
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Общий прогресс
    cursor.execute("SELECT COUNT(*) FROM processed_market_data")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) 
        FROM processed_market_data 
        WHERE technical_indicators->>'buy_expected_return' IS NOT NULL
    """)
    records_with_labels = cursor.fetchone()[0]
    
    # По символам
    cursor.execute("""
        SELECT 
            symbol,
            COUNT(*) as total,
            COUNT(CASE WHEN technical_indicators->>'buy_expected_return' IS NOT NULL THEN 1 END) as with_labels
        FROM processed_market_data
        GROUP BY symbol
        ORDER BY symbol
    """)
    
    symbol_stats = cursor.fetchall()
    
    conn.close()
    
    return total_records, records_with_labels, symbol_stats

def main():
    print("🔄 Мониторинг прогресса подготовки датасета...")
    print("Нажмите Ctrl+C для выхода\n")
    
    try:
        while True:
            total, with_labels, symbols = check_progress()
            
            # Очистка экрана
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("="*60)
            print("📊 ПРОГРЕСС ПОДГОТОВКИ ДАТАСЕТА")
            print("="*60)
            
            overall_progress = (with_labels / total * 100) if total > 0 else 0
            print(f"\n📈 Общий прогресс: {with_labels:,} / {total:,} ({overall_progress:.1f}%)")
            
            # Прогресс-бар
            bar_length = 40
            filled = int(bar_length * overall_progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"[{bar}] {overall_progress:.1f}%")
            
            print(f"\n📊 По символам:")
            print(f"{'Символ':<15} {'Всего':<10} {'Готово':<10} {'Прогресс':<10}")
            print("-"*50)
            
            for symbol, total_sym, with_labels_sym in symbols:
                progress = (with_labels_sym / total_sym * 100) if total_sym > 0 else 0
                status = "✅" if progress == 100 else "⏳"
                print(f"{status} {symbol:<13} {total_sym:<10,} {with_labels_sym:<10,} {progress:<10.1f}%")
            
            # Оценка времени
            if with_labels > 0 and overall_progress < 100:
                rate = with_labels / 60  # примерная скорость в минуту
                remaining = total - with_labels
                eta_minutes = remaining / rate if rate > 0 else 0
                print(f"\n⏱️  Примерное время до завершения: {eta_minutes:.0f} мин")
            
            print(f"\n🕒 Обновление через 5 секунд...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n✋ Мониторинг остановлен")
        print("Подготовка датасета продолжается в фоне")

if __name__ == "__main__":
    main()