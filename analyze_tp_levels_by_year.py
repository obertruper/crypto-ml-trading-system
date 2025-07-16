#!/usr/bin/env python3
"""
Анализ достижения уровней частичных закрытий по годам
"""

import psycopg2
import yaml
import pandas as pd
import numpy as np
from prepare_dataset import MarketDatasetPreparator, PostgreSQLManager
from collections import defaultdict

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
lookahead_limit = 100

# Анализируем каждый год
for year in years:
    year_df = df[df['year'] == year].reset_index(drop=True)
    
    # Пропускаем годы с недостаточным количеством данных
    if len(year_df) < 1100:  # Нужно минимум 1100 записей (1000 + lookahead)
        print(f"\n⚠️ {year}: недостаточно данных ({len(year_df)} записей)")
        continue
    
    print(f"\n📅 Анализ {year} года ({len(year_df)} записей)...")
    
    # Статистика по уровням для года
    stats = defaultdict(int)
    
    # Детальный анализ по барам
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
        
        for j in range(i + 1, min(i + lookahead_limit + 1, len(year_df))):
            high = year_df.iloc[j]['high']
            low = year_df.iloc[j]['low']
            
            # Проверяем достижение уровней
            if not hit_sl and not hit_final_tp:
                # Проверка SL
                if low <= sl_price:
                    hit_sl = True
                    
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
    
    # Сохраняем статистику года
    all_years_stats[year] = dict(stats)
    
    # Выводим результаты для года
    print(f"\n📊 РЕЗУЛЬТАТЫ {year} года ({stats['total_positions']} позиций):")
    print("\n🎯 Достижение уровней TP:")
    print(f"   TP1 (+1.2%): {stats['reached_tp1']} ({stats['reached_tp1']/stats['total_positions']*100:.1f}%)")
    print(f"   TP2 (+2.4%): {stats['reached_tp2']} ({stats['reached_tp2']/stats['total_positions']*100:.1f}%)")
    print(f"   TP3 (+3.5%): {stats['reached_tp3']} ({stats['reached_tp3']/stats['total_positions']*100:.1f}%)")
    print(f"   Final TP (+5.8%): {stats['reached_final_tp']} ({stats['reached_final_tp']/stats['total_positions']*100:.1f}%)")
    
    # Позиций с частичными закрытиями
    total_with_partials = stats['reached_tp1']
    print(f"\n💰 Позиций с частичными закрытиями: {total_with_partials} ({total_with_partials/stats['total_positions']*100:.1f}%)")

# Сравнительная таблица по годам
print("\n" + "="*80)
print("📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА ПО ГОДАМ")
print("="*80)
print(f"{'Год':<10} {'Позиций':<10} {'TP1 %':<10} {'TP2 %':<10} {'TP3 %':<10} {'Final %':<10} {'Partials %':<12}")
print("-"*72)

for year, stats in sorted(all_years_stats.items()):
    total = stats['total_positions']
    tp1_pct = stats['reached_tp1'] / total * 100
    tp2_pct = stats['reached_tp2'] / total * 100
    tp3_pct = stats['reached_tp3'] / total * 100
    final_pct = stats['reached_final_tp'] / total * 100
    partials_pct = stats['reached_tp1'] / total * 100
    
    print(f"{year:<10} {total:<10} {tp1_pct:<10.1f} {tp2_pct:<10.1f} {tp3_pct:<10.1f} {final_pct:<10.1f} {partials_pct:<12.1f}")

# Средние значения
if all_years_stats:
    print("-"*72)
    avg_tp1 = np.mean([s['reached_tp1']/s['total_positions']*100 for s in all_years_stats.values()])
    avg_tp2 = np.mean([s['reached_tp2']/s['total_positions']*100 for s in all_years_stats.values()])
    avg_tp3 = np.mean([s['reached_tp3']/s['total_positions']*100 for s in all_years_stats.values()])
    avg_final = np.mean([s['reached_final_tp']/s['total_positions']*100 for s in all_years_stats.values()])
    avg_partials = np.mean([s['reached_tp1']/s['total_positions']*100 for s in all_years_stats.values()])
    
    print(f"{'СРЕДНЕЕ':<10} {'-':<10} {avg_tp1:<10.1f} {avg_tp2:<10.1f} {avg_tp3:<10.1f} {avg_final:<10.1f} {avg_partials:<12.1f}")

# Анализ трендов
print("\n📈 АНАЛИЗ ТРЕНДОВ:")
if len(all_years_stats) > 1:
    years_sorted = sorted(all_years_stats.keys())
    first_year = years_sorted[0]
    last_year = years_sorted[-1]
    
    first_partials = all_years_stats[first_year]['reached_tp1'] / all_years_stats[first_year]['total_positions'] * 100
    last_partials = all_years_stats[last_year]['reached_tp1'] / all_years_stats[last_year]['total_positions'] * 100
    
    trend = last_partials - first_partials
    if trend > 0:
        print(f"   📈 Частичные закрытия выросли с {first_partials:.1f}% ({first_year}) до {last_partials:.1f}% ({last_year})")
    else:
        print(f"   📉 Частичные закрытия снизились с {first_partials:.1f}% ({first_year}) до {last_partials:.1f}% ({last_year})")

db_manager.disconnect()
print("\n" + "="*80)