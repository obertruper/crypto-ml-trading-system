#!/usr/bin/env python3
"""
Отладка расчета SELL позиций
"""

import numpy as np

def debug_sell_calculation():
    """Демонстрация проблемы с расчетом SELL"""
    
    print("="*80)
    print("ОТЛАДКА РАСЧЕТА SELL ПОЗИЦИЙ")
    print("="*80)
    
    # Параметры
    entry_price = 100.0
    sell_sl_pct = 1.011  # +1.1% (для SELL это убыток)
    sell_tp_pct = 0.942  # -5.8% (для SELL это прибыль)
    
    print(f"\n📊 Параметры SELL позиции:")
    print(f"Цена входа: ${entry_price}")
    print(f"Stop Loss: {sell_sl_pct} (цена = ${entry_price * sell_sl_pct:.2f})")
    print(f"Take Profit: {sell_tp_pct} (цена = ${entry_price * sell_tp_pct:.2f})")
    
    # Симулируем движение цены
    test_prices = [
        (102, "Цена растет до $102"),
        (101.5, "Цена растет до $101.5"), 
        (101, "Цена на $101"),
        (99, "Цена падает до $99"),
        (95, "Цена падает до $95"),
        (94, "Цена падает до $94")
    ]
    
    print("\n🔍 Анализ различных сценариев:")
    print("-"*60)
    
    for price, description in test_prices:
        # Расчет прибыли для SELL
        profit_pct = ((entry_price - price) / entry_price) * 100
        
        # Проверка условий
        sl_hit = price >= entry_price * sell_sl_pct  # 101.1
        tp_hit = price <= entry_price * sell_tp_pct  # 94.2
        
        print(f"\n{description}:")
        print(f"  Прибыль/убыток: {profit_pct:+.2f}%")
        print(f"  Stop Loss сработал: {'ДА' if sl_hit else 'НЕТ'}")
        print(f"  Take Profit сработал: {'ДА' if tp_hit else 'НЕТ'}")
        
        if sl_hit:
            print(f"  Результат: УБЫТОК {profit_pct:.2f}%")
        elif tp_hit:
            print(f"  Результат: ПРИБЫЛЬ {profit_pct:.2f}%")
        else:
            print(f"  Результат: Позиция открыта")
    
    # Проблема с логикой
    print("\n" + "="*60)
    print("❌ ПРОБЛЕМА В КОДЕ:")
    print("="*60)
    
    print("""
В функции _calculate_enhanced_result есть ошибка для SELL позиций:

НЕПРАВИЛЬНО:
```python
# Проверка стоп-лосса для SELL
if price >= current_sl:
    remaining_loss = ((entry_price - current_sl) / entry_price) * 100 * position_size
    return {
        'final_return': realized_profit - remaining_loss,  # ❌ Вычитаем убыток
        ...
    }
```

ПРАВИЛЬНО:
```python
# Проверка стоп-лосса для SELL
if price >= current_sl:
    remaining_loss = -((current_sl - entry_price) / entry_price) * 100 * position_size
    return {
        'final_return': realized_profit + remaining_loss,  # ✅ Добавляем отрицательный результат
        ...
    }
```

Проблема: при срабатывании SL для SELL, убыток вычитается из realized_profit,
что дает положительный результат вместо отрицательного!
""")
    
    # Демонстрация ошибки
    print("\n📊 Пример ошибочного расчета:")
    entry = 100
    sl_price = 101.1
    current_price = 102
    
    # Неправильный расчет (как сейчас в коде)
    wrong_loss = ((entry - sl_price) / entry) * 100  # -1.1%
    wrong_result = 0 - wrong_loss  # 0 - (-1.1) = +1.1%
    
    # Правильный расчет
    correct_loss = -((sl_price - entry) / entry) * 100  # -1.1%
    correct_result = 0 + correct_loss  # 0 + (-1.1) = -1.1%
    
    print(f"\nЦена входа: ${entry}")
    print(f"Stop Loss: ${sl_price}")
    print(f"Текущая цена: ${current_price}")
    print(f"\n❌ Неправильный расчет: {wrong_result:+.2f}% (позитивный!)")
    print(f"✅ Правильный расчет: {correct_result:+.2f}% (негативный)")


if __name__ == "__main__":
    debug_sell_calculation()