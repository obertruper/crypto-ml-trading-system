#!/usr/bin/env python3
"""
Сравнение старого и нового подхода к созданию датасета
"""

import numpy as np
import matplotlib.pyplot as plt

def demonstrate_calculation():
    """Демонстрация расчетов прибыли/убытка"""
    
    print("="*80)
    print("СРАВНЕНИЕ ПОДХОДОВ К РАСЧЕТУ ПРИБЫЛИ")
    print("="*80)
    
    # Параметры позиции
    entry_price = 50000  # BTC
    position_size = 1.0  # 1 BTC
    
    print(f"\n📊 ИСХОДНЫЕ ДАННЫЕ:")
    print(f"Инструмент: BTCUSDT")
    print(f"Направление: BUY")
    print(f"Цена входа: ${entry_price:,}")
    print(f"Размер позиции: {position_size} BTC")
    
    # Старый подход
    print("\n" + "-"*60)
    print("🔴 СТАРЫЙ ПОДХОД (prepare_dataset.py):")
    print("-"*60)
    
    sl_old = entry_price * 0.989  # -1.1%
    tp_old = entry_price * 1.058  # +5.8%
    
    print(f"Stop Loss: ${sl_old:,.2f} (-1.1%)")
    print(f"Take Profit: ${tp_old:,.2f} (+5.8%)")
    print(f"\nРезультаты:")
    print(f"- При достижении SL: убыток ${(entry_price - sl_old):,.2f} (-1.1%)")
    print(f"- При достижении TP: прибыль ${(tp_old - entry_price):,.2f} (+5.8%)")
    print(f"- Бинарная метка: 0 (loss) или 1 (profit)")
    
    # Новый подход
    print("\n" + "-"*60)
    print("🟢 НОВЫЙ ПОДХОД (с частичными закрытиями):")
    print("-"*60)
    
    print(f"Начальный Stop Loss: ${sl_old:,.2f} (-1.1%)")
    print(f"Начальный Take Profit: ${tp_old:,.2f} (+5.8%)")
    
    print("\n📈 Сценарий движения цены:")
    
    # Симуляция движения цены
    price_levels = [
        (50600, 1.2),   # +1.2%
        (51200, 2.4),   # +2.4%
        (51750, 3.5),   # +3.5%
        (52300, 4.6),   # +4.6%
    ]
    
    position_remaining = 1.0
    realized_profit = 0
    current_sl = sl_old
    
    print("\n1️⃣ Цена достигает $50,600 (+1.2%):")
    print("   - Срабатывает частичное закрытие: 20% позиции")
    close_amount = 0.2
    profit_1 = close_amount * (50600 - entry_price)
    realized_profit += profit_1
    position_remaining -= close_amount
    print(f"   - Закрыто: {close_amount} BTC")
    print(f"   - Прибыль: ${profit_1:,.2f}")
    print(f"   - SL обновлен в безубыток: ${entry_price * 1.001:,.2f}")
    print(f"   - Осталось в позиции: {position_remaining} BTC")
    
    print("\n2️⃣ Цена достигает $51,200 (+2.4%):")
    print("   - Срабатывает частичное закрытие: 30% позиции")
    print("   - Срабатывает защита прибыли: фиксируем +1.2%")
    close_amount = 0.3
    profit_2 = close_amount * (51200 - entry_price)
    realized_profit += profit_2
    position_remaining -= close_amount
    current_sl = entry_price * 1.012  # Фиксируем +1.2%
    print(f"   - Закрыто: {close_amount} BTC")
    print(f"   - Прибыль: ${profit_2:,.2f}")
    print(f"   - SL обновлен: ${current_sl:,.2f} (защита +1.2%)")
    print(f"   - Осталось в позиции: {position_remaining} BTC")
    
    print("\n3️⃣ Цена достигает $51,750 (+3.5%):")
    print("   - Срабатывает частичное закрытие: 30% позиции")
    print("   - Срабатывает защита прибыли: фиксируем +2.4%")
    close_amount = 0.3
    profit_3 = close_amount * (51750 - entry_price)
    realized_profit += profit_3
    position_remaining -= close_amount
    current_sl = entry_price * 1.024  # Фиксируем +2.4%
    print(f"   - Закрыто: {close_amount} BTC")
    print(f"   - Прибыль: ${profit_3:,.2f}")
    print(f"   - SL обновлен: ${current_sl:,.2f} (защита +2.4%)")
    print(f"   - Осталось в позиции: {position_remaining} BTC")
    
    print("\n4️⃣ Цена падает до $51,200 (активируется защитный SL):")
    print("   - Закрывается оставшаяся позиция по защитному SL")
    profit_4 = position_remaining * (current_sl - entry_price)
    realized_profit += profit_4
    print(f"   - Закрыто: {position_remaining} BTC по цене ${current_sl:,.2f}")
    print(f"   - Прибыль: ${profit_4:,.2f}")
    
    print("\n" + "="*60)
    print("💰 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
    print("="*60)
    
    total_profit_percent = (realized_profit / entry_price) * 100
    
    print(f"\nРеализованная прибыль:")
    print(f"- Частичное закрытие 1: ${profit_1:,.2f}")
    print(f"- Частичное закрытие 2: ${profit_2:,.2f}")
    print(f"- Частичное закрытие 3: ${profit_3:,.2f}")
    print(f"- Финальное закрытие: ${profit_4:,.2f}")
    print(f"\nОБЩАЯ ПРИБЫЛЬ: ${realized_profit:,.2f} ({total_profit_percent:.2f}%)")
    
    print("\n" + "="*60)
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("="*60)
    
    print("\n🔴 Старый подход:")
    print("- Бинарный результат: 0 или 1")
    print("- Не учитывает частичную реализацию прибыли")
    print("- Результат: либо -1.1%, либо +5.8%")
    
    print("\n🟢 Новый подход:")
    print(f"- Точный результат: +{total_profit_percent:.2f}%")
    print("- Учитывает частичные закрытия")
    print("- Учитывает защиту прибыли")
    print("- Более реалистичная оценка")
    
    # Визуализация
    print("\n📈 Создание визуализации...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График 1: Распределение результатов (старый подход)
    ax1.bar(['Убыток\n(-1.1%)', 'Прибыль\n(+5.8%)'], [79.75, 10.73], 
            color=['red', 'green'], alpha=0.7)
    ax1.set_title('Старый подход: Бинарное распределение', fontsize=14)
    ax1.set_ylabel('Процент случаев (%)')
    ax1.set_ylim(0, 100)
    
    # График 2: Распределение результатов (новый подход)
    # Симулируем более реалистичное распределение
    np.random.seed(42)
    results = []
    
    # Генерируем примерное распределение
    # 60% - небольшие убытки (частичные закрытия смягчают)
    results.extend(np.random.normal(-0.5, 0.5, 600))
    # 25% - небольшая прибыль (частичные закрытия)
    results.extend(np.random.normal(1.5, 0.8, 250))
    # 10% - средняя прибыль
    results.extend(np.random.normal(2.5, 0.5, 100))
    # 5% - большая прибыль
    results.extend(np.random.normal(4.0, 1.0, 50))
    
    ax2.hist(results, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', label='Безубыток')
    ax2.axvline(x=total_profit_percent, color='green', linestyle='-', 
                linewidth=2, label=f'Пример: +{total_profit_percent:.1f}%')
    ax2.set_title('Новый подход: Непрерывное распределение', fontsize=14)
    ax2.set_xlabel('Результат (%)')
    ax2.set_ylabel('Количество случаев')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ График сохранен: dataset_comparison.png")
    
    # Влияние на обучение модели
    print("\n" + "="*60)
    print("🤖 ВЛИЯНИЕ НА ОБУЧЕНИЕ МОДЕЛИ:")
    print("="*60)
    
    print("\n🔴 Проблемы старого подхода:")
    print("1. Сильный дисбаланс классов (10% vs 90%)")
    print("2. Модель не видит промежуточные результаты")
    print("3. Не учитывается реальная торговая логика")
    print("4. Завышенные ожидания прибыли (+5.8% редко достижимы)")
    
    print("\n🟢 Преимущества нового подхода:")
    print("1. Более сбалансированное распределение")
    print("2. Модель учится на реальных сценариях")
    print("3. Учитывается частичная фиксация прибыли")
    print("4. Реалистичные целевые значения")
    
    print("\n💡 Рекомендации для модели:")
    print("1. Использовать регрессию вместо классификации")
    print("2. Предсказывать ожидаемую прибыль в процентах")
    print("3. Добавить предсказание оптимального времени удержания")
    print("4. Учитывать вероятность достижения разных уровней прибыли")


if __name__ == "__main__":
    demonstrate_calculation()