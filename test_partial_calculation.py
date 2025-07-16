#!/usr/bin/env python3
"""
Проверка правильности расчета final_return для позиций с частичными закрытиями
"""

# Пример: вход по цене 100
entry_price = 100

# После достижения +1.2% (цена 101.2):
# - Закрываем 20% позиции с прибылью 1.2%
# - Стоп переносится на 100.3 (+0.3%)

# Realized PnL от частичного закрытия:
partial_close_ratio = 0.20
partial_profit_pct = 1.2
realized_pnl = partial_profit_pct * partial_close_ratio  # 0.24%

# Оставшаяся позиция: 80%
remaining_position = 1.0 - partial_close_ratio  # 0.80

# Если затем срабатывает стоп на 100.3:
new_sl_price = entry_price * 1.003  # 100.3
sl_profit_pct = ((new_sl_price - entry_price) / entry_price) * 100  # 0.3%
unrealized_pnl = sl_profit_pct * remaining_position  # 0.24%

# Final return:
final_return = realized_pnl + unrealized_pnl

print("=== Проверка расчета для позиции с частичным закрытием ===")
print(f"Вход: ${entry_price}")
print(f"\n1. Достигли +1.2%, закрыли 20% позиции:")
print(f"   Realized PnL: {realized_pnl:.2f}%")
print(f"   Оставшаяся позиция: {remaining_position*100:.0f}%")
print(f"   Стоп перенесен на: ${new_sl_price} (+{sl_profit_pct:.1f}%)")
print(f"\n2. Сработал стоп на +0.3%:")
print(f"   Unrealized PnL: {unrealized_pnl:.2f}% (на {remaining_position*100:.0f}% позиции)")
print(f"\n3. Итоговый результат:")
print(f"   Final return = {realized_pnl:.2f}% + {unrealized_pnl:.2f}% = {final_return:.2f}%")

# Проверим логику в коде prepare_dataset.py
print("\n=== Проверка кода ===")
print("Строка 795: loss_pct = ((current_sl - entry_price) / entry_price) * 100")
print(f"  При current_sl={new_sl_price}, entry_price={entry_price}")
print(f"  loss_pct = (({new_sl_price} - {entry_price}) / {entry_price}) * 100 = {sl_profit_pct:.1f}%")
print("\nСтрока 796: remaining_loss = loss_pct * position_size")
print(f"  remaining_loss = {sl_profit_pct:.1f}% * {remaining_position} = {unrealized_pnl:.2f}%")
print("\nСтрока 800: final_return = realized_pnl + remaining_loss")
print(f"  final_return = {realized_pnl:.2f}% + {unrealized_pnl:.2f}% = {final_return:.2f}%")

# Проверим другие сценарии
print("\n=== Другие сценарии ===")

# Если достигли 2 уровня TP
print("\n1. Если достигли TP2 (+2.4%):")
realized_2 = 1.2 * 0.20 + 2.4 * 0.30  # 0.96%
remaining_2 = 1.0 - 0.20 - 0.30  # 0.50
new_sl_2 = 1.2  # стоп на +1.2%
unrealized_2 = new_sl_2 * remaining_2  # 0.60%
final_2 = realized_2 + unrealized_2
print(f"   Realized: {realized_2:.2f}%, Unrealized: {unrealized_2:.2f}%, Final: {final_2:.2f}%")

# Если достигли все 3 уровня TP и финальный TP
print("\n2. Если достигли все TP и финальный (+5.8%):")
realized_all = 1.2 * 0.20 + 2.4 * 0.30 + 3.5 * 0.30  # 2.01%
remaining_all = 1.0 - 0.20 - 0.30 - 0.30  # 0.20
final_tp = 5.8 * remaining_all  # 1.16%
final_all = realized_all + final_tp
print(f"   Realized: {realized_all:.2f}%, Final TP: {final_tp:.2f}%, Total: {final_all:.2f}%")