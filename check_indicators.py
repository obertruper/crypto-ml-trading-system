#!/usr/bin/env python3
"""
Скрипт для проверки корректности рассчитанных индикаторов
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import json

def connect_to_db():
    """Подключение к базе данных."""
    conn = psycopg2.connect(
        host='localhost',
        port=5555,
        database='crypto_trading',
        user='ruslan',
        password='ruslan'
    )
    return conn

def check_indicators():
    """Проверка рассчитанных индикаторов"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    print("🔍 ПРОВЕРКА ИНДИКАТОРОВ\n")
    
    # 1. Получаем список индикаторов из метаданных
    cursor.execute("""
        SELECT feature_columns FROM model_metadata 
        WHERE model_type = 'preprocessing'
        ORDER BY created_at DESC LIMIT 1
    """)
    result = cursor.fetchone()
    
    if not result:
        print("❌ Метаданные не найдены!")
        return
    
    feature_names = result[0]  # Уже JSON массив
    print(f"📊 Найдено {len(feature_names)} индикаторов:")
    
    # Группируем индикаторы по типам
    groups = {
        'Ценовые': ['close', 'high', 'low', 'open', 'volume'],
        'Скользящие средние': [f for f in feature_names if 'ema' in f or 'sma' in f],
        'Трендовые': [f for f in feature_names if any(x in f for x in ['adx', 'macd', 'aroon', 'sar'])],
        'Осцилляторы': [f for f in feature_names if any(x in f for x in ['rsi', 'stoch', 'cci', 'williams'])],
        'Волатильность': [f for f in feature_names if any(x in f for x in ['atr', 'bb_', 'donchian'])],
        'Объемные': [f for f in feature_names if any(x in f for x in ['obv', 'cmf', 'mfi'])],
        'Ichimoku': [f for f in feature_names if 'ichimoku' in f],
        'Временные': [f for f in feature_names if any(x in f for x in ['hour', 'dayofweek', 'month'])],
        'Производные': [f for f in feature_names if any(x in f for x in ['_diff', '_pct_change', '_ratio'])]
    }
    
    for group_name, indicators in groups.items():
        if indicators:
            print(f"\n{group_name} ({len(indicators)}):")
            for ind in sorted(indicators)[:5]:  # Показываем первые 5
                print(f"  - {ind}")
            if len(indicators) > 5:
                print(f"  ... и еще {len(indicators) - 5}")
    
    # 2. Проверяем примеры значений
    print("\n\n📈 ПРОВЕРКА ЗНАЧЕНИЙ ИНДИКАТОРОВ")
    print("=" * 80)
    
    # Получаем примеры данных
    cursor.execute("""
        SELECT symbol, technical_indicators 
        FROM processed_market_data 
        LIMIT 10
    """)
    
    samples = cursor.fetchall()
    
    for symbol, technical_indicators in samples[:3]:  # Первые 3 записи
        features = technical_indicators  # Уже JSONB
        print(f"\n{symbol}:")
        
        # Проверяем основные индикаторы
        checks = {
            'RSI': features.get('rsi_val', None),
            'MACD': features.get('macd_val', None),
            'ADX': features.get('adx_val', None),
            'ATR': features.get('atr_val', None),
            'BB Upper': features.get('bb_upper', None),
            'OBV': features.get('obv', None),
            'EMA 15': features.get('ema_15', None),
            'SAR': features.get('sar', None)
        }
        
        for name, value in checks.items():
            if value is not None:
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: ❌ Не найден")
    
    # 3. Проверка на NaN и экстремальные значения
    print("\n\n🔍 ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
    print("=" * 80)
    
    cursor.execute("""
        SELECT COUNT(*) as total,
               COUNT(CASE WHEN technical_indicators::text LIKE '%null%' THEN 1 END) as with_nulls,
               COUNT(CASE WHEN technical_indicators::text LIKE '%NaN%' THEN 1 END) as with_nans
        FROM processed_market_data
    """)
    
    total, with_nulls, with_nans = cursor.fetchone()
    
    print(f"Всего записей: {total}")
    print(f"Записей с null значениями: {with_nulls} ({with_nulls/total*100:.1f}%)")
    print(f"Записей с NaN значениями: {with_nans} ({with_nans/total*100:.1f}%)")
    
    # 4. Проверка меток
    print("\n\n🎯 ПРОВЕРКА МЕТОК")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            symbol,
            SUM(buy_profit_target) as buy_profits,
            SUM(buy_loss_target) as buy_losses,
            SUM(sell_profit_target) as sell_profits,
            SUM(sell_loss_target) as sell_losses,
            COUNT(*) as total
        FROM processed_market_data
        GROUP BY symbol
    """)
    
    labels_stats = cursor.fetchall()
    
    for symbol, buy_p, buy_l, sell_p, sell_l, total in labels_stats:
        buy_wr = (buy_p / (buy_p + buy_l) * 100) if (buy_p + buy_l) > 0 else 0
        sell_wr = (sell_p / (sell_p + sell_l) * 100) if (sell_p + sell_l) > 0 else 0
        
        print(f"\n{symbol}:")
        print(f"  🟢 BUY:  {buy_p} прибыльных, {buy_l} убыточных (WR: {buy_wr:.1f}%)")
        print(f"  🔴 SELL: {sell_p} прибыльных, {sell_l} убыточных (WR: {sell_wr:.1f}%)")
        print(f"  📊 Всего: {total} записей")
    
    # 5. Проверка корреляций между индикаторами
    print("\n\n📊 ПРОВЕРКА КОРРЕЛЯЦИЙ")
    print("=" * 80)
    
    # Загружаем данные для анализа корреляций
    cursor.execute("""
        SELECT technical_indicators 
        FROM processed_market_data 
        WHERE symbol = 'BTCUSDT'
        LIMIT 100
    """)
    
    data_for_corr = []
    for row in cursor.fetchall():
        features = row[0]  # Уже JSONB
        data_for_corr.append(features)
    
    if data_for_corr:
        df = pd.DataFrame(data_for_corr)
        
        # Проверяем корреляции между похожими индикаторами
        check_pairs = [
            ('rsi_14', 'rsi_21'),
            ('ema_12', 'ema_26'),
            ('bb_upper', 'bb_lower'),
            ('stoch_k', 'stoch_d')
        ]
        
        print("Корреляции между связанными индикаторами:")
        for ind1, ind2 in check_pairs:
            if ind1 in df.columns and ind2 in df.columns:
                corr = df[ind1].corr(df[ind2])
                print(f"  {ind1} <-> {ind2}: {corr:.3f}")
    
    cursor.close()
    conn.close()
    
    print("\n✅ Проверка завершена!")

if __name__ == "__main__":
    check_indicators()