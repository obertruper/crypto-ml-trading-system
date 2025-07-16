#!/usr/bin/env python3
"""
Визуализация логики расчета expected_return
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*80)
print("📊 ВИЗУАЛИЗАЦИЯ ЛОГИКИ РАСЧЕТА EXPECTED_RETURN")
print("="*80)

# Создаем синтетические данные для примера
dates = pd.date_range(start='2024-01-01 00:00', periods=110, freq='15min')
prices = 100 + np.cumsum(np.random.randn(110) * 0.5)  # Случайное блуждание

df = pd.DataFrame({
    'datetime': dates,
    'close': prices,
    'high': prices + np.abs(np.random.randn(110) * 0.2),
    'low': prices - np.abs(np.random.randn(110) * 0.2)
})

print("\n📈 ПРИМЕР ДАННЫХ (первые 10 баров):")
print(df.head(10).to_string(index=False))

print("\n\n🎯 КАК РАБОТАЕТ РАСЧЕТ:")
print("="*60)

# Показываем расчет для 3 разных точек входа
entry_points = [0, 1, 2]

for entry_idx in entry_points:
    print(f"\n📍 ТОЧКА ВХОДА #{entry_idx + 1}:")
    print(f"   Время: {df.iloc[entry_idx]['datetime']}")
    print(f"   Цена входа: ${df.iloc[entry_idx]['close']:.2f}")
    
    # Симулируем анализ будущих баров
    future_start = entry_idx + 1
    future_end = min(entry_idx + 6, len(df))  # Показываем только 5 баров для примера
    
    print(f"\n   🔮 Анализируем следующие бары:")
    print(f"   {'Бар':>5} {'Время':>20} {'High':>10} {'Low':>10} {'Потенциал':>15}")
    print(f"   {'-'*65}")
    
    entry_price = df.iloc[entry_idx]['close']
    
    for j in range(future_start, future_end):
        bar = df.iloc[j]
        high_pct = ((bar['high'] - entry_price) / entry_price) * 100
        low_pct = ((bar['low'] - entry_price) / entry_price) * 100
        
        potential = ""
        if high_pct >= 1.2:
            potential = "🎯 TP +1.2%"
        elif low_pct <= -1.1:
            potential = "🛑 SL -1.1%"
        
        print(f"   {j-entry_idx:>5} {str(bar['datetime']):>20} {bar['high']:>10.2f} {bar['low']:>10.2f} {potential:>15}")
    
    # Симулируем результат
    np.random.seed(entry_idx)
    if np.random.rand() > 0.5:
        result = np.random.choice([-1.1, 0.48, 1.56, 2.49, 3.17])
    else:
        result = -1.1
    
    print(f"\n   📊 Expected Return: {result:.2f}%")
    print(f"   {'='*65}")

print("\n\n💡 КЛЮЧЕВЫЕ МОМЕНТЫ:")
print("1. Каждая строка в таблице = отдельная точка входа")
print("2. Для каждой точки анализируются СЛЕДУЮЩИЕ 100 баров")
print("3. Результат зависит от движения цены ПОСЛЕ входа")
print("4. Поэтому expected_return РАЗНЫЙ для каждой строки")

print("\n\n❌ ЕСЛИ ВСЕ ЗНАЧЕНИЯ ОДИНАКОВЫЕ:")
print("Это означает, что:")
print("- Либо рынок не двигался (все сделки закрылись по стопу)")
print("- Либо есть ошибка в расчете")
print("- Либо недостаточно данных для анализа")

print("\n\n✅ ПРАВИЛЬНЫЙ РЕЗУЛЬТАТ:")
print("- Большинство значений должны быть разными")
print("- Распределение: ~40% стопы (-1.1%), ~30% около нуля, ~30% прибыль")
print("- Среднее значение: небольшая положительная величина (~0.1-0.5%)")