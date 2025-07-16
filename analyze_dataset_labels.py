#!/usr/bin/env python3
"""
Анализ распределения меток в датасете
"""

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import yaml

def connect_to_db():
    """Подключение к PostgreSQL"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    conn = psycopg2.connect(**db_config)
    return conn

def analyze_labels():
    """Анализ меток в датасете"""
    print("="*80)
    print("АНАЛИЗ МЕТОК В ДАТАСЕТЕ КРИПТОТРЕЙДИНГА")
    print("="*80)
    print(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    conn = connect_to_db()
    
    # Загружаем данные
    print("📊 Загрузка данных из processed_market_data...")
    query = """
    SELECT 
        symbol, timestamp, datetime,
        buy_profit_target, buy_loss_target,
        sell_profit_target, sell_loss_target
    FROM processed_market_data
    WHERE technical_indicators IS NOT NULL
    ORDER BY symbol, timestamp
    """
    
    df = pd.read_sql(query, conn)
    print(f"✅ Загружено {len(df):,} записей\n")
    
    # 1. Общая статистика по меткам
    print("📈 СТАТИСТИКА ПО МЕТКАМ")
    print("-"*50)
    
    labels = ['buy_profit_target', 'buy_loss_target', 'sell_profit_target', 'sell_loss_target']
    
    for label in labels:
        total = len(df)
        positive = df[label].sum()
        percentage = (positive / total) * 100
        print(f"{label:20s}: {positive:8,} / {total:,} ({percentage:5.2f}%)")
    
    # Общие BUY и SELL сигналы
    buy_signals = df[(df['buy_profit_target'] == 1) | (df['buy_loss_target'] == 1)].shape[0]
    sell_signals = df[(df['sell_profit_target'] == 1) | (df['sell_loss_target'] == 1)].shape[0]
    
    print(f"\n📊 Общие сигналы:")
    print(f"BUY сигналов:  {buy_signals:,} ({buy_signals/len(df)*100:.2f}%)")
    print(f"SELL сигналов: {sell_signals:,} ({sell_signals/len(df)*100:.2f}%)")
    
    # Соотношение profit/loss
    buy_profit_ratio = df['buy_profit_target'].sum() / (df['buy_loss_target'].sum() + 1)
    sell_profit_ratio = df['sell_profit_target'].sum() / (df['sell_loss_target'].sum() + 1)
    
    print(f"\n💰 Соотношение profit/loss:")
    print(f"BUY:  {buy_profit_ratio:.2f} (profit/loss)")
    print(f"SELL: {sell_profit_ratio:.2f} (profit/loss)")
    
    # 2. Проверка корректности меток
    print("\n\n🔍 ПРОВЕРКА КОРРЕКТНОСТИ МЕТОК")
    print("-"*50)
    
    # Проверка конфликтов (одновременно profit и loss)
    buy_conflicts = df[(df['buy_profit_target'] == 1) & (df['buy_loss_target'] == 1)]
    sell_conflicts = df[(df['sell_profit_target'] == 1) & (df['sell_loss_target'] == 1)]
    
    print(f"❌ BUY конфликтов (profit + loss):  {len(buy_conflicts)}")
    print(f"❌ SELL конфликтов (profit + loss): {len(sell_conflicts)}")
    
    # Проверка значений (должны быть только 0 и 1)
    for label in labels:
        unique_vals = df[label].unique()
        if not all(v in [0, 1] for v in unique_vals):
            print(f"⚠️ {label} содержит недопустимые значения: {unique_vals}")
        else:
            print(f"✅ {label}: корректные значения")
    
    # 3. Распределение по символам
    print("\n\n📊 РАСПРЕДЕЛЕНИЕ ПО СИМВОЛАМ")
    print("-"*50)
    
    symbol_stats = []
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        total = len(symbol_df)
        
        buy_profit = symbol_df['buy_profit_target'].sum()
        buy_loss = symbol_df['buy_loss_target'].sum()
        sell_profit = symbol_df['sell_profit_target'].sum()
        sell_loss = symbol_df['sell_loss_target'].sum()
        
        total_signals = buy_profit + buy_loss + sell_profit + sell_loss
        
        symbol_stats.append({
            'symbol': symbol,
            'total_records': total,
            'buy_profit': buy_profit,
            'buy_loss': buy_loss,
            'sell_profit': sell_profit,
            'sell_loss': sell_loss,
            'total_signals': total_signals,
            'signal_percentage': (total_signals / total) * 100 if total > 0 else 0
        })
    
    symbol_df = pd.DataFrame(symbol_stats)
    symbol_df = symbol_df.sort_values('total_signals', ascending=False)
    
    print("\nТоп-20 символов по количеству сигналов:")
    print(f"{'Символ':<15} {'Записей':<10} {'BUY+':<8} {'BUY-':<8} {'SELL+':<8} {'SELL-':<8} {'Всего':<10} {'%':<5}")
    print("-"*80)
    
    for _, row in symbol_df.head(20).iterrows():
        print(f"{row['symbol']:<15} {row['total_records']:<10,} {row['buy_profit']:<8} {row['buy_loss']:<8} "
              f"{row['sell_profit']:<8} {row['sell_loss']:<8} {row['total_signals']:<10,} {row['signal_percentage']:<5.1f}")
    
    # Символы без сигналов
    no_signals = symbol_df[symbol_df['total_signals'] == 0]
    if len(no_signals) > 0:
        print(f"\n⚠️ Символов без торговых сигналов: {len(no_signals)}")
        print(f"   {', '.join(no_signals['symbol'].tolist())}")
    
    # 4. Временное распределение
    print("\n\n⏰ ВРЕМЕННОЕ РАСПРЕДЕЛЕНИЕ")
    print("-"*50)
    
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    
    # Сигналы по дням
    daily_signals = df.groupby('date').agg({
        'buy_profit_target': 'sum',
        'buy_loss_target': 'sum',
        'sell_profit_target': 'sum',
        'sell_loss_target': 'sum'
    })
    daily_signals['total'] = daily_signals.sum(axis=1)
    
    print(f"📅 Период данных: {daily_signals.index.min()} - {daily_signals.index.max()}")
    print(f"📊 Среднее количество сигналов в день: {daily_signals['total'].mean():.1f}")
    print(f"📈 Максимум сигналов в день: {daily_signals['total'].max()}")
    print(f"📉 Минимум сигналов в день: {daily_signals['total'].min()}")
    
    # Распределение по часам
    hourly_signals = df.groupby('hour').agg({
        'buy_profit_target': 'sum',
        'buy_loss_target': 'sum',
        'sell_profit_target': 'sum',
        'sell_loss_target': 'sum'
    })
    hourly_signals['total'] = hourly_signals.sum(axis=1)
    
    print("\n🕐 Распределение сигналов по часам (UTC):")
    print(f"{'Час':<5} {'BUY+':<8} {'BUY-':<8} {'SELL+':<8} {'SELL-':<8} {'Всего':<10}")
    print("-"*50)
    
    for hour, row in hourly_signals.iterrows():
        print(f"{hour:02d}:00 {row['buy_profit_target']:<8.0f} {row['buy_loss_target']:<8.0f} "
              f"{row['sell_profit_target']:<8.0f} {row['sell_loss_target']:<8.0f} {row['total']:<10.0f}")
    
    # 5. Итоговый отчет
    print("\n\n📋 ИТОГОВЫЙ ОТЧЕТ")
    print("="*80)
    
    total_records = len(df)
    total_symbols = df['symbol'].nunique()
    total_days = len(daily_signals)
    
    print(f"📊 Общие метрики датасета:")
    print(f"   - Всего записей: {total_records:,}")
    print(f"   - Уникальных символов: {total_symbols}")
    print(f"   - Период данных: {total_days} дней")
    print(f"   - Средняя частота: {total_records / total_days:.0f} записей/день")
    
    # Рассчитаем win rate
    buy_total = df['buy_profit_target'].sum() + df['buy_loss_target'].sum()
    sell_total = df['sell_profit_target'].sum() + df['sell_loss_target'].sum()
    
    buy_win_rate = (df['buy_profit_target'].sum() / buy_total * 100) if buy_total > 0 else 0
    sell_win_rate = (df['sell_profit_target'].sum() / sell_total * 100) if sell_total > 0 else 0
    
    print(f"\n💰 Успешность сигналов:")
    print(f"   - BUY Win Rate:  {buy_win_rate:.1f}%")
    print(f"   - SELL Win Rate: {sell_win_rate:.1f}%")
    
    # Дисбаланс классов
    overall_positive = (df[labels].sum().sum()) / (len(df) * 4) * 100
    print(f"\n⚖️ Баланс классов:")
    print(f"   - Положительных меток: {overall_positive:.1f}%")
    print(f"   - Отрицательных меток: {100 - overall_positive:.1f}%")
    
    if overall_positive < 20:
        print(f"\n⚠️ ВНИМАНИЕ: Сильный дисбаланс классов!")
        print(f"   Рекомендуется использовать техники балансировки при обучении.")
    
    conn.close()
    print("\n✅ Анализ завершен!")

if __name__ == "__main__":
    analyze_labels()