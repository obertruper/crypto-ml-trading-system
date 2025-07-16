#!/usr/bin/env python3
"""
Проверка расчета от точки входа в таблице processed_market_data
"""

import psycopg2
import yaml
import pandas as pd

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

conn = psycopg2.connect(**db_config)

print("="*80)
print("🔍 ПРОВЕРКА РАСЧЕТА ОТ ТОЧКИ ВХОДА")
print("="*80)

# 1. Проверяем структуру данных
query = """
SELECT 
    p.symbol,
    p.datetime,
    r.close as entry_price,
    p.buy_expected_return,
    p.sell_expected_return,
    p.buy_profit_target,
    p.buy_loss_target,
    p.sell_profit_target,
    p.sell_loss_target
FROM processed_market_data p
JOIN raw_market_data r ON p.symbol = r.symbol AND p.datetime = r.datetime
WHERE p.symbol = %s
ORDER BY p.datetime
LIMIT 100
"""

# Берем первый доступный символ
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT symbol FROM processed_market_data LIMIT 1")
symbol = cursor.fetchone()

if not symbol:
    print("⚠️ Таблица processed_market_data пуста!")
    conn.close()
    exit()

symbol = symbol[0]
print(f"\n📊 Анализируем символ: {symbol}")

df = pd.read_sql(query, conn, params=(symbol,))

if len(df) > 0:
    print(f"\n✅ Найдено {len(df)} записей")
    
    # Показываем первые 20 записей
    print("\n📋 Первые 20 записей (каждая строка = потенциальная точка входа):")
    print("-"*120)
    print(f"{'Время':<20} {'Цена входа':>12} {'Buy Return':>12} {'Sell Return':>12} {'Buy P/L':>15} {'Sell P/L':>15}")
    print("-"*120)
    
    for i in range(min(20, len(df))):
        row = df.iloc[i]
        buy_pl = "Profit" if row['buy_profit_target'] == 1 else ("Loss" if row['buy_loss_target'] == 1 else "-")
        sell_pl = "Profit" if row['sell_profit_target'] == 1 else ("Loss" if row['sell_loss_target'] == 1 else "-")
        
        print(f"{str(row['datetime']):<20} {row['entry_price']:>12.2f} {row['buy_expected_return']:>12.3f}% {row['sell_expected_return']:>12.3f}% {buy_pl:>15} {sell_pl:>15}")
    
    # 2. Проверяем соседние записи
    print("\n\n🔍 ПРОВЕРКА: Каждая строка должна иметь РАЗНЫЕ expected_returns")
    print("(т.к. каждая строка = своя точка входа с анализом следующих 100 баров)")
    
    # Проверяем, есть ли одинаковые значения подряд
    duplicates = 0
    for i in range(1, len(df)):
        if (df.iloc[i]['buy_expected_return'] == df.iloc[i-1]['buy_expected_return'] and 
            df.iloc[i]['sell_expected_return'] == df.iloc[i-1]['sell_expected_return']):
            duplicates += 1
    
    if duplicates > 0:
        print(f"\n⚠️ ПРОБЛЕМА: Найдено {duplicates} одинаковых значений подряд!")
        print("Это означает, что расчет может быть неверным.")
    else:
        print(f"\n✅ Все значения уникальны - расчет корректен!")
    
    # 3. Визуальная проверка логики
    print("\n\n📊 ЛОГИКА РАСЧЕТА:")
    print("Каждая строка в таблице представляет:")
    print("1. ТОЧКУ ВХОДА = close цена этого бара")
    print("2. АНАЛИЗ = следующие 100 баров (25 часов)")
    print("3. РЕЗУЛЬТАТ = expected_return для входа именно в этот момент")
    
    # 4. Проверка распределения
    print(f"\n\n📈 РАСПРЕДЕЛЕНИЕ РЕЗУЛЬТАТОВ для {symbol}:")
    stats = df['buy_expected_return'].describe()
    print(f"Count: {stats['count']:.0f}")
    print(f"Mean:  {stats['mean']:.3f}%")
    print(f"Std:   {stats['std']:.3f}%")
    print(f"Min:   {stats['min']:.3f}%")
    print(f"25%:   {stats['25%']:.3f}%")
    print(f"50%:   {stats['50%']:.3f}%")
    print(f"75%:   {stats['75%']:.3f}%")
    print(f"Max:   {stats['max']:.3f}%")
    
    # 5. Проверяем конкретный пример
    print("\n\n🔍 ДЕТАЛЬНАЯ ПРОВЕРКА ОДНОЙ ТОЧКИ:")
    example_idx = 10
    example = df.iloc[example_idx]
    
    print(f"\nТочка входа: {example['datetime']}")
    print(f"Цена входа: ${example['entry_price']:.2f}")
    print(f"Buy Expected Return: {example['buy_expected_return']:.3f}%")
    print(f"Sell Expected Return: {example['sell_expected_return']:.3f}%")
    
    # Получаем будущие бары для этой точки
    future_query = """
    SELECT datetime, high, low, close
    FROM raw_market_data
    WHERE symbol = %s 
    AND datetime > %s
    ORDER BY datetime
    LIMIT 5
    """
    
    future_df = pd.read_sql(future_query, conn, params=(symbol, example['datetime']))
    
    if len(future_df) > 0:
        print(f"\n📊 Следующие 5 баров после входа:")
        print(f"{'Время':<20} {'High':>10} {'Low':>10} {'Close':>10}")
        print("-"*55)
        for _, bar in future_df.iterrows():
            print(f"{str(bar['datetime']):<20} {bar['high']:>10.2f} {bar['low']:>10.2f} {bar['close']:>10.2f}")
    
else:
    print(f"❌ Нет данных для символа {symbol}")

conn.close()

print("\n\n💡 ВЫВОД:")
print("Если expected_returns разные для каждой строки - расчет ПРАВИЛЬНЫЙ")
print("Каждая строка = отдельная симуляция входа в этот момент времени")