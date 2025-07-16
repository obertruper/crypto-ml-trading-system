#!/usr/bin/env python3
"""
Детальная проверка работы частичных закрытий
"""

import psycopg2
import yaml
import pandas as pd
import numpy as np
from prepare_dataset import MarketDatasetPreparator, PostgreSQLManager

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

print("="*80)
print("🔍 ДЕТАЛЬНАЯ ПРОВЕРКА ЧАСТИЧНЫХ ЗАКРЫТИЙ")
print("="*80)

# Инициализация
db_manager = PostgreSQLManager(db_config)
db_manager.connect()
preparator = MarketDatasetPreparator(db_manager, config['risk_profile'])

# Берем один символ для детального анализа
symbol = 'BTCUSDT'
print(f"\n📊 Анализ символа: {symbol}")

# Загружаем данные
df = preparator.load_raw_data(symbol, limit=1000)
if len(df) == 0:
    print("❌ Нет данных")
    exit()

# Параметры из конфига
buy_sl_pct = 0.989  # -1.1%
buy_tp_pct = 1.058  # +5.8%
partial_tp_levels = [
    {'percent': 1.2, 'close_ratio': 0.20},
    {'percent': 2.4, 'close_ratio': 0.30},
    {'percent': 3.5, 'close_ratio': 0.30}
]
profit_protection = {
    'breakeven_percent': 1.2,
    'breakeven_offset': 0.3,
    'lock_levels': [
        {'trigger': 2.4, 'lock': 1.2},
        {'trigger': 3.5, 'lock': 2.4},
        {'trigger': 4.6, 'lock': 3.5}
    ]
}

# Статистика
stats = {
    'total': 0,
    'stop_loss': 0,
    'take_profit': 0,
    'timeout': 0,
    'partial_only': 0,
    'partial_then_sl': 0,
    'partial_then_tp': 0,
    'partial_then_timeout': 0,
    'no_partial': 0
}

detailed_trades = []

# Анализируем первые 100 точек входа
for i in range(min(100, len(df) - 100)):
    entry_price = df.iloc[i]['close']
    entry_time = df.iloc[i]['datetime']
    
    # Получаем будущие бары
    future_bars = []
    for j in range(i + 1, min(i + 101, len(df))):
        future_bars.append({
            'open': df.iloc[j]['open'],
            'high': df.iloc[j]['high'],
            'low': df.iloc[j]['low'],
            'close': df.iloc[j]['close']
        })
    
    # Рассчитываем результат
    result = preparator._calculate_enhanced_result(
        entry_price, future_bars, 'buy',
        buy_sl_pct, buy_tp_pct, partial_tp_levels, profit_protection
    )
    
    stats['total'] += 1
    
    # Детальный анализ
    partial_count = 0
    if result['realized_pnl'] > 0:
        partial_count = int(result['realized_pnl'] / (result['final_return'] - result['realized_pnl'] + 0.0001))
    
    trade_info = {
        'entry_price': entry_price,
        'entry_time': entry_time,
        'final_return': result['final_return'],
        'realized_pnl': result['realized_pnl'],
        'exit_reason': result['exit_reason'],
        'exit_bar': result['exit_bar'],
        'partial_count': partial_count
    }
    
    # Классификация
    if 'Stop Loss' in result['exit_reason']:
        if result['realized_pnl'] > 0:
            stats['partial_then_sl'] += 1
            trade_info['type'] = 'Partial→SL'
        else:
            stats['stop_loss'] += 1
            stats['no_partial'] += 1
            trade_info['type'] = 'Direct SL'
    elif 'Take Profit' in result['exit_reason']:
        if result['realized_pnl'] > 0:
            stats['partial_then_tp'] += 1
            trade_info['type'] = 'Partial→TP'
        else:
            stats['take_profit'] += 1
            stats['no_partial'] += 1
            trade_info['type'] = 'Direct TP'
    elif 'Timeout' in result['exit_reason']:
        if result['realized_pnl'] > 0:
            stats['partial_then_timeout'] += 1
            trade_info['type'] = 'Partial→Timeout'
        else:
            stats['timeout'] += 1
            stats['no_partial'] += 1
            trade_info['type'] = 'Direct Timeout'
    
    detailed_trades.append(trade_info)

# Выводим статистику
print(f"\n📊 СТАТИСТИКА ({stats['total']} сделок):")
print(f"\n🚫 БЕЗ частичных закрытий: {stats['no_partial']} ({stats['no_partial']/stats['total']*100:.1f}%)")
print(f"   - Прямой Stop Loss: {stats['stop_loss']} ({stats['stop_loss']/stats['total']*100:.1f}%)")
print(f"   - Прямой Take Profit: {stats['take_profit']} ({stats['take_profit']/stats['total']*100:.1f}%)")
print(f"   - Прямой Timeout: {stats['timeout']} ({stats['timeout']/stats['total']*100:.1f}%)")

partial_total = stats['partial_then_sl'] + stats['partial_then_tp'] + stats['partial_then_timeout']
print(f"\n✅ С частичными закрытиями: {partial_total} ({partial_total/stats['total']*100:.1f}%)")
print(f"   - Partial → Stop Loss: {stats['partial_then_sl']} ({stats['partial_then_sl']/stats['total']*100:.1f}%)")
print(f"   - Partial → Take Profit: {stats['partial_then_tp']} ({stats['partial_then_tp']/stats['total']*100:.1f}%)")
print(f"   - Partial → Timeout: {stats['partial_then_timeout']} ({stats['partial_then_timeout']/stats['total']*100:.1f}%)")

# Примеры сделок с частичными закрытиями
print("\n📋 ПРИМЕРЫ СДЕЛОК С ЧАСТИЧНЫМИ ЗАКРЫТИЯМИ:")
partial_examples = [t for t in detailed_trades if 'Partial' in t['type']][:5]
for i, trade in enumerate(partial_examples):
    print(f"\nПример {i+1}:")
    print(f"   Вход: ${trade['entry_price']:.2f} в {trade['entry_time']}")
    print(f"   Тип: {trade['type']}")
    print(f"   Realized PnL: {trade['realized_pnl']:.2f}%")
    print(f"   Final Return: {trade['final_return']:.2f}%")
    print(f"   Выход: {trade['exit_reason']}")

# Анализ проблемы
print("\n" + "="*80)
print("🔍 АНАЛИЗ ПРОБЛЕМЫ:")
print("="*80)

if partial_total == 0:
    print("\n❌ ЧАСТИЧНЫЕ ЗАКРЫТИЯ НЕ РАБОТАЮТ!")
    print("\nВозможные причины:")
    print("1. Слишком быстрое движение к SL (не успевает достичь TP1 +1.2%)")
    print("2. Высокая волатильность - цена сразу проскакивает уровни")
    print("3. Проблема в коде расчета")
    
    # Проверим сколько раз цена вообще достигает TP1
    tp1_reached = 0
    for i in range(min(100, len(df) - 100)):
        entry_price = df.iloc[i]['close']
        tp1_price = entry_price * 1.012  # +1.2%
        sl_price = entry_price * 0.989   # -1.1%
        
        reached_tp1 = False
        hit_sl_first = False
        
        for j in range(i + 1, min(i + 101, len(df))):
            if df.iloc[j]['high'] >= tp1_price and not hit_sl_first:
                reached_tp1 = True
                break
            if df.iloc[j]['low'] <= sl_price:
                hit_sl_first = True
                if not reached_tp1:
                    break
        
        if reached_tp1:
            tp1_reached += 1
    
    print(f"\n📊 Анализ достижимости TP1 (+1.2%):")
    print(f"   Достигли TP1: {tp1_reached} из 100 ({tp1_reached}%)")
    print(f"   НЕ достигли TP1: {100-tp1_reached} ({100-tp1_reached}%)")
    
    if tp1_reached < 30:
        print("\n💡 ВЫВОД: Уровни TP слишком далеко для текущей волатильности!")
        print("   Рекомендации:")
        print("   - Уменьшить первый уровень TP с 1.2% до 0.8%")
        print("   - Или увеличить SL с -1.1% до -1.5%")
else:
    print(f"\n✅ Частичные закрытия работают, но учитываются в статистике неправильно")
    print(f"   Реальных частичных закрытий: {partial_total} ({partial_total/stats['total']*100:.1f}%)")
    print("\n   Проблема в подсчете статистики - если после частичных закрытий")
    print("   срабатывает SL, то вся сделка считается как 'Stop Loss'")

db_manager.disconnect()
print("\n" + "="*80)