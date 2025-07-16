#!/usr/bin/env python3
"""
Анализ достижения уровней частичных закрытий по годам
"""

import psycopg2
import yaml
import pandas as pd
import numpy as np
from prepare_dataset import MarketDatasetPreparator, PostgreSQLManager
from datetime import datetime

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)

print("="*80)
print("🎯 АНАЛИЗ ДОСТИЖЕНИЯ УРОВНЕЙ ЧАСТИЧНЫХ ЗАКРЫТИЙ ПО ГОДАМ")
print("="*80)

# Инициализация
db_manager = PostgreSQLManager(db_config)
db_manager.connect()
preparator = MarketDatasetPreparator(db_manager, config['risk_profile'])

# Анализируем ALGOUSDT
symbol = 'ALGOUSDT'
print(f"\n📊 Анализ символа: {symbol}")

# Загружаем все данные
df = preparator.load_raw_data(symbol)
if len(df) == 0:
    print("❌ Нет данных")
    exit()

# Добавляем год
df['year'] = pd.to_datetime(df['datetime']).dt.year
years = sorted(df['year'].unique())
print(f"Доступные годы: {years}")

# Параметры
buy_sl_pct = 0.989  # -1.1%
buy_tp_pct = 1.058  # +5.8%
partial_tp_levels = [
    {'percent': 1.2, 'close_ratio': 0.20},  # TP1
    {'percent': 2.4, 'close_ratio': 0.30},  # TP2
    {'percent': 3.5, 'close_ratio': 0.30}   # TP3
]

# Общая статистика по всем годам
all_years_stats = {}

# Анализируем каждый год
for year in years:
    year_df = df[df['year'] == year].reset_index(drop=True)
    
    # Пропускаем годы с недостаточным количеством данных
    if len(year_df) < 1100:  # Нужно минимум 1100 записей (1000 + lookahead)
        print(f"\n⚠️ {year}: недостаточно данных ({len(year_df)} записей)")
        continue
    
    print(f"\n📅 Анализ {year} года ({len(year_df)} записей)...")
    
    # Статистика по уровням для года
    stats = {
        'total_positions': 0,
        'reached_tp1': 0,
        'reached_tp2': 0,
        'reached_tp3': 0,
        'reached_final_tp': 0,
        'hit_sl_direct': 0,
        'hit_sl_after_tp1': 0,
        'hit_sl_after_tp2': 0,
        'hit_sl_after_tp3': 0,
        'timeout_no_tp': 0,
        'timeout_after_tp1': 0,
        'timeout_after_tp2': 0,
        'timeout_after_tp3': 0
    }
    
    # Детальный анализ по барам
    lookahead_limit = 100
    positions_to_analyze = min(1000, len(year_df) - lookahead_limit)
    
    for i in range(positions_to_analyze):  # Анализируем до 1000 позиций
        entry_price = year_df.iloc[i]['close']
        stats['total_positions'] += 1
        
        # Уровни цен
        sl_price = entry_price * buy_sl_pct
        tp1_price = entry_price * 1.012  # +1.2%
        tp2_price = entry_price * 1.024  # +2.4%
        tp3_price = entry_price * 1.035  # +3.5%
        final_tp_price = entry_price * buy_tp_pct  # +5.8%
        
        # Анализ будущих баров
        reached_levels = []
        hit_sl = False
        hit_final_tp = False
        exit_bar = None
        
        for j in range(i + 1, min(i + lookahead_limit + 1, len(year_df))):
            high = year_df.iloc[j]['high']
            low = year_df.iloc[j]['low']
            
            # Проверяем достижение уровней (в порядке от низшего к высшему)
            if not hit_sl and not hit_final_tp:
                # Проверка SL
                if low <= sl_price:
                    hit_sl = True
                    exit_bar = j - i
                    
                    # Определяем после какого TP был SL
                    if 'tp3' in reached_levels:
                    stats['hit_sl_after_tp3'] += 1
                elif 'tp2' in reached_levels:
                    stats['hit_sl_after_tp2'] += 1
                elif 'tp1' in reached_levels:
                    stats['hit_sl_after_tp1'] += 1
                else:
                    stats['hit_sl_direct'] += 1
                break
            
            # Проверка TP1
            if high >= tp1_price and 'tp1' not in reached_levels:
                reached_levels.append('tp1')
                stats['reached_tp1'] += 1
                # После TP1 стоп переносится на +0.3%
                sl_price = entry_price * 1.003
            
            # Проверка TP2
            if high >= tp2_price and 'tp2' not in reached_levels:
                reached_levels.append('tp2')
                stats['reached_tp2'] += 1
                # После TP2 стоп переносится на +1.2%
                sl_price = entry_price * 1.012
            
            # Проверка TP3
            if high >= tp3_price and 'tp3' not in reached_levels:
                reached_levels.append('tp3')
                stats['reached_tp3'] += 1
                # После TP3 стоп переносится на +2.4%
                sl_price = entry_price * 1.024
            
            # Проверка финального TP
            if high >= final_tp_price:
                hit_final_tp = True
                stats['reached_final_tp'] += 1
                exit_bar = j - i
                break
    
    # Если не вышли ни по SL, ни по TP - timeout
    if not hit_sl and not hit_final_tp:
        if 'tp3' in reached_levels:
            stats['timeout_after_tp3'] += 1
        elif 'tp2' in reached_levels:
            stats['timeout_after_tp2'] += 1
        elif 'tp1' in reached_levels:
            stats['timeout_after_tp1'] += 1
        else:
            stats['timeout_no_tp'] += 1

# Выводим результаты
print(f"\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА ({stats['total_positions']} позиций):")
print("\n🎯 Достижение уровней TP:")
print(f"   TP1 (+1.2%): {stats['reached_tp1']} ({stats['reached_tp1']/stats['total_positions']*100:.1f}%)")
print(f"   TP2 (+2.4%): {stats['reached_tp2']} ({stats['reached_tp2']/stats['total_positions']*100:.1f}%)")
print(f"   TP3 (+3.5%): {stats['reached_tp3']} ({stats['reached_tp3']/stats['total_positions']*100:.1f}%)")
print(f"   Final TP (+5.8%): {stats['reached_final_tp']} ({stats['reached_final_tp']/stats['total_positions']*100:.1f}%)")

print("\n❌ Выходы по Stop Loss:")
print(f"   Прямой SL (без TP): {stats['hit_sl_direct']} ({stats['hit_sl_direct']/stats['total_positions']*100:.1f}%)")
print(f"   SL после TP1: {stats['hit_sl_after_tp1']} ({stats['hit_sl_after_tp1']/stats['total_positions']*100:.1f}%)")
print(f"   SL после TP2: {stats['hit_sl_after_tp2']} ({stats['hit_sl_after_tp2']/stats['total_positions']*100:.1f}%)")
print(f"   SL после TP3: {stats['hit_sl_after_tp3']} ({stats['hit_sl_after_tp3']/stats['total_positions']*100:.1f}%)")

print("\n⏱️ Выходы по Timeout:")
print(f"   Timeout без TP: {stats['timeout_no_tp']} ({stats['timeout_no_tp']/stats['total_positions']*100:.1f}%)")
print(f"   Timeout после TP1: {stats['timeout_after_tp1']} ({stats['timeout_after_tp1']/stats['total_positions']*100:.1f}%)")
print(f"   Timeout после TP2: {stats['timeout_after_tp2']} ({stats['timeout_after_tp2']/stats['total_positions']*100:.1f}%)")
print(f"   Timeout после TP3: {stats['timeout_after_tp3']} ({stats['timeout_after_tp3']/stats['total_positions']*100:.1f}%)")

# Итоговая статистика
print("\n📈 ИТОГОВАЯ СТАТИСТИКА:")
total_with_partials = stats['reached_tp1']
print(f"   Позиций с частичными закрытиями: {total_with_partials} ({total_with_partials/stats['total_positions']*100:.1f}%)")
print(f"   Позиций без частичных закрытий: {stats['total_positions'] - total_with_partials} ({(stats['total_positions'] - total_with_partials)/stats['total_positions']*100:.1f}%)")

# Средняя глубина достижения
avg_depth = (stats['reached_tp1'] * 1 + 
             (stats['reached_tp2'] - stats['reached_tp1']) * 2 + 
             (stats['reached_tp3'] - stats['reached_tp2']) * 3 + 
             (stats['reached_final_tp'] - stats['reached_tp3']) * 4) / stats['reached_tp1'] if stats['reached_tp1'] > 0 else 0

print(f"\n   Средняя глубина достижения TP: {avg_depth:.2f}")
print("   (1 = только TP1, 2 = до TP2, 3 = до TP3, 4 = до финального)")

db_manager.disconnect()
print("\n" + "="*80)