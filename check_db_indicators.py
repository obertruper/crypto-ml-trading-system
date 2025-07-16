#!/usr/bin/env python3
"""
Проверка наличия технических индикаторов в базе данных
"""

import pandas as pd
import psycopg2
import json
import yaml

def check_db_indicators():
    # Загрузка конфигурации
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    
    # Подключение к БД
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['dbname'],
        user=db_config['user']
    )
    
    # Получаем примеры данных
    query = """
    SELECT symbol, timestamp, technical_indicators
    FROM processed_market_data
    WHERE symbol IN ('BTCUSDT', 'ETHUSDT')
    ORDER BY timestamp DESC
    LIMIT 10
    """
    
    df = pd.read_sql_query(query, conn)
    
    print(f"📊 Загружено {len(df)} записей")
    print(f"Символы: {df['symbol'].unique()}")
    
    # Анализируем technical_indicators
    for idx, row in df.iterrows():
        print(f"\n{'='*60}")
        print(f"Символ: {row['symbol']}, Timestamp: {row['timestamp']}")
        
        indicators = row['technical_indicators']
        if indicators:
            print(f"Количество индикаторов: {len(indicators)}")
            
            # Проверяем ключевые индикаторы
            key_indicators = ['rsi_val', 'macd_hist', 'adx_val', 'bb_upper', 'bb_lower']
            print("\nКлючевые индикаторы:")
            for ind in key_indicators:
                if ind in indicators:
                    value = indicators[ind]
                    if value is not None:
                        print(f"  {ind}: {value:.4f}")
                    else:
                        print(f"  {ind}: None")
                else:
                    print(f"  {ind}: ОТСУТСТВУЕТ")
            
            # Проверяем временные признаки
            time_features = ['hour', 'day_of_week', 'is_weekend']
            print("\nВременные признаки:")
            for feat in time_features:
                if feat in indicators:
                    print(f"  {feat}: {indicators[feat]}")
                else:
                    print(f"  {feat}: ОТСУТСТВУЕТ")
                    
            # Показываем все доступные индикаторы
            if idx == 0:  # Только для первой записи
                print(f"\nВсе доступные индикаторы ({len(indicators)}):")
                for i, (key, value) in enumerate(sorted(indicators.items())):
                    if i < 20:  # Показываем первые 20
                        print(f"  - {key}")
                if len(indicators) > 20:
                    print(f"  ... и еще {len(indicators) - 20} индикаторов")
        else:
            print("⚠️ technical_indicators пустой!")
            
        if idx >= 2:  # Показываем только 3 примера
            break
    
    conn.close()
    print("\n✅ Проверка завершена")

if __name__ == "__main__":
    check_db_indicators()