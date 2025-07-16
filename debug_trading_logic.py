#!/usr/bin/env python3
"""
Детальная проверка логики торговли
"""

import psycopg2
import yaml
import pandas as pd
import numpy as np

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

conn = psycopg2.connect(**db_config)

print("="*80)
print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ТОРГОВОЙ ЛОГИКИ")
print("="*80)

# Берем пример данных DOTUSDT
query = """
SELECT 
    r.datetime,
    r.open, r.high, r.low, r.close,
    p.buy_expected_return,
    p.sell_expected_return
FROM raw_market_data r
JOIN processed_market_data p ON r.id = p.raw_data_id
WHERE r.symbol = 'DOTUSDT'
ORDER BY r.timestamp
LIMIT 200
"""

df = pd.read_sql(query, conn)

# Симуляция торговли
print("\n📊 СИМУЛЯЦИЯ ТОРГОВЛИ (первые 10 сделок):")
print("-"*80)

risk_profile = config['risk_profile']
buy_sl_pct = risk_profile.get('stop_loss_pct_buy', 0.989)  # -1.1%
buy_tp_pct = risk_profile.get('take_profit_pct_buy', 1.058)  # +5.8%

partial_levels = [
    {'percent': 1.2, 'close_ratio': 0.20},
    {'percent': 2.4, 'close_ratio': 0.30},
    {'percent': 3.5, 'close_ratio': 0.30}
]

trades_shown = 0
i = 0

while trades_shown < 10 and i < len(df) - 100:
    entry_price = df.iloc[i]['close']
    entry_time = df.iloc[i]['datetime']
    expected_return = df.iloc[i]['buy_expected_return']
    
    # Пропускаем если нет expected_return
    if pd.isna(expected_return) or expected_return == 0:
        i += 1
        continue
    
    print(f"\n🎯 СДЕЛКА #{trades_shown + 1}:")
    print(f"   Время входа: {entry_time}")
    print(f"   Цена входа: ${entry_price:.4f}")
    print(f"   Expected return: {expected_return:.2f}%")
    
    # Уровни
    sl_price = entry_price * buy_sl_pct
    tp_price = entry_price * buy_tp_pct
    tp1_price = entry_price * 1.012  # +1.2%
    tp2_price = entry_price * 1.024  # +2.4%
    tp3_price = entry_price * 1.035  # +3.5%
    
    print(f"\n   📍 Уровни:")
    print(f"      Stop Loss: ${sl_price:.4f} (-1.1%)")
    print(f"      TP1: ${tp1_price:.4f} (+1.2%)")
    print(f"      TP2: ${tp2_price:.4f} (+2.4%)")
    print(f"      TP3: ${tp3_price:.4f} (+3.5%)")
    print(f"      Final TP: ${tp_price:.4f} (+5.8%)")
    
    # Симуляция сделки
    position_size = 1.0
    realized_pnl = 0.0
    current_sl = sl_price
    
    print(f"\n   📈 Движение цены:")
    print(f"   {'Бар':>5} {'Время':>20} {'High':>10} {'Low':>10} {'Close':>10} {'Событие':>30}")
    print(f"   {'-'*85}")
    
    for j in range(i+1, min(i+11, len(df))):  # Показываем первые 10 баров
        bar = df.iloc[j]
        high = bar['high']
        low = bar['low']
        close = bar['close']
        time = bar['datetime']
        
        event = ""
        
        # Проверка стоп-лосса
        if low <= current_sl and position_size > 0:
            event = f"🛑 STOP LOSS! Цена {low:.4f} <= {current_sl:.4f}"
            break
            
        # Проверка частичных профитов
        if high >= tp1_price and position_size == 1.0:
            event = f"✅ TP1 достигнут! {high:.4f} >= {tp1_price:.4f}"
        elif high >= tp2_price and position_size > 0.8:
            event = f"✅ TP2 достигнут! {high:.4f} >= {tp2_price:.4f}"
        elif high >= tp3_price and position_size > 0.5:
            event = f"✅ TP3 достигнут! {high:.4f} >= {tp3_price:.4f}"
        elif high >= tp_price and position_size > 0:
            event = f"🎯 FINAL TP! {high:.4f} >= {tp_price:.4f}"
            break
            
        print(f"   {j-i:>5} {str(time):>20} {high:>10.4f} {low:>10.4f} {close:>10.4f} {event:>30}")
        
        if event and "STOP" in event:
            break
    
    trades_shown += 1
    i += 20  # Пропускаем несколько баров для следующей сделки

# Анализ проблемы
print("\n\n🔍 АНАЛИЗ ПРОБЛЕМЫ С ЧАСТИЧНЫМИ ЗАКРЫТИЯМИ:")
print("-"*80)

# Проверяем сколько раз цена достигает уровней до стоп-лосса
query2 = """
WITH entry_points AS (
    SELECT 
        r.id,
        r.timestamp,
        r.close as entry_price,
        r.symbol
    FROM raw_market_data r
    WHERE r.symbol = 'DOTUSDT'
    AND r.market_type = 'futures'
    ORDER BY r.timestamp
    LIMIT 1000
),
price_movements AS (
    SELECT 
        e.id as entry_id,
        e.entry_price,
        MAX(CASE 
            WHEN f.high >= e.entry_price * 1.012 
            AND f.timestamp > e.timestamp 
            AND f.timestamp <= e.timestamp + 90000 -- 25 часов
            THEN 1 ELSE 0 
        END) as reached_tp1,
        MAX(CASE 
            WHEN f.low <= e.entry_price * 0.989 
            AND f.timestamp > e.timestamp 
            AND f.timestamp <= e.timestamp + 90000
            THEN 1 ELSE 0 
        END) as hit_sl,
        MIN(CASE 
            WHEN f.low <= e.entry_price * 0.989 
            AND f.timestamp > e.timestamp 
            THEN f.timestamp - e.timestamp 
        END) as time_to_sl,
        MIN(CASE 
            WHEN f.high >= e.entry_price * 1.012 
            AND f.timestamp > e.timestamp 
            THEN f.timestamp - e.timestamp 
        END) as time_to_tp1
    FROM entry_points e
    JOIN raw_market_data f ON f.symbol = e.symbol
    WHERE f.timestamp > e.timestamp 
    AND f.timestamp <= e.timestamp + 90000
    GROUP BY e.id, e.entry_price
)
SELECT 
    COUNT(*) as total_entries,
    SUM(reached_tp1) as reached_tp1_count,
    SUM(hit_sl) as hit_sl_count,
    SUM(CASE WHEN time_to_sl < time_to_tp1 OR (hit_sl = 1 AND reached_tp1 = 0) THEN 1 ELSE 0 END) as sl_before_tp1,
    AVG(time_to_sl / 900.0) as avg_bars_to_sl,
    AVG(time_to_tp1 / 900.0) as avg_bars_to_tp1
FROM price_movements
"""

result = pd.read_sql(query2, conn)
print("\nСтатистика движения цены после входа (первые 1000 точек):")
print(result.to_string())

print("\n💡 ВЫВОДЫ:")
print("1. Если SL срабатывает раньше TP1 в большинстве случаев - частичные закрытия невозможны")
print("2. Нужно проверить, правильно ли рассчитываются уровни")
print("3. Возможно, стоп-лосс слишком близкий для волатильной крипты")

conn.close()