#!/usr/bin/env python3
"""
Тест новой логики расчета expected_return с правильной торговой механикой
"""

import numpy as np
import pandas as pd
from prepare_dataset import MarketDatasetPreparator
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_trading_scenarios():
    """Тестирует различные торговые сценарии"""
    
    # Создаем экземпляр MarketDatasetPreparator с пустым риск-профилем
    risk_profile = {}  # Не используется в _calculate_enhanced_result
    preparer = MarketDatasetPreparator(None, risk_profile)
    
    # Тестовые сценарии
    scenarios = [
        {
            'name': 'BUY - Быстрый стоп-лосс',
            'entry_price': 100.0,
            'direction': 'buy',
            'bars': [
                {'high': 100.5, 'low': 98.8, 'close': 99.0},  # Стоп-лосс срабатывает
            ],
            'expected': -1.1  # Потеря -1.1%
        },
        {
            'name': 'BUY - Частичные профиты',
            'entry_price': 100.0,
            'direction': 'buy',
            'bars': [
                {'high': 101.3, 'low': 100.1, 'close': 101.2},  # +1.2% - закрываем 20%
                {'high': 102.5, 'low': 101.0, 'close': 102.4},  # +2.4% - закрываем 30%
                {'high': 103.6, 'low': 102.0, 'close': 103.5},  # +3.5% - закрываем 30%
                {'high': 106.0, 'low': 103.0, 'close': 105.8},  # +5.8% - закрываем 20%
            ],
            'expected': 1.2*0.2 + 2.4*0.3 + 3.5*0.3 + 5.8*0.2  # = 3.19%
        },
        {
            'name': 'BUY - Стоп после частичного профита',
            'entry_price': 100.0,
            'direction': 'buy',
            'bars': [
                {'high': 101.3, 'low': 100.1, 'close': 101.2},  # +1.2% - закрываем 20%
                {'high': 101.0, 'low': 99.5, 'close': 100.0},   # Откат
                {'high': 100.5, 'low': 100.2, 'close': 100.3},  # Стоп в безубытке срабатывает
            ],
            'expected': 1.2*0.2 + 0.3*0.8  # 20% закрыто с +1.2%, 80% с +0.3%
        },
        {
            'name': 'SELL - Полный тейк-профит',
            'entry_price': 100.0,
            'direction': 'sell',
            'bars': [
                {'high': 99.5, 'low': 94.0, 'close': 94.2},  # -5.8% - полный профит
            ],
            'expected': 3.17  # С частичными закрытиями результат будет таким же как у BUY
        },
        {
            'name': 'Таймаут с небольшой прибылью',
            'entry_price': 100.0,
            'direction': 'buy',
            'bars': [{'high': 100.5, 'low': 99.5, 'close': 100.5} for _ in range(100)],
            'expected': 0.5  # Закрытие по таймауту с +0.5%
        }
    ]
    
    # Параметры из prepare_dataset.py
    buy_sl_pct = 0.989   # -1.1%
    buy_tp_pct = 1.058   # +5.8%
    sell_sl_pct = 1.011  # +1.1%
    sell_tp_pct = 0.942  # -5.8%
    
    partial_levels = [
        {'percent': 1.2, 'close_ratio': 0.20},
        {'percent': 2.4, 'close_ratio': 0.30},
        {'percent': 3.5, 'close_ratio': 0.30}
    ]
    
    protection = {
        'breakeven_percent': 1.2,
        'breakeven_offset': 0.3,
        'lock_levels': [
            {'trigger': 2.4, 'lock': 1.2},
            {'trigger': 3.5, 'lock': 2.4},
            {'trigger': 4.6, 'lock': 3.5}
        ]
    }
    
    print("\n🧪 ТЕСТИРОВАНИЕ НОВОЙ ТОРГОВОЙ ЛОГИКИ")
    print("="*60)
    
    for scenario in scenarios:
        result = preparer._calculate_enhanced_result(
            scenario['entry_price'],
            scenario['bars'],
            scenario['direction'],
            buy_sl_pct if scenario['direction'] == 'buy' else sell_sl_pct,
            buy_tp_pct if scenario['direction'] == 'buy' else sell_tp_pct,
            partial_levels,
            protection
        )
        
        print(f"\n📋 Сценарий: {scenario['name']}")
        print(f"   Направление: {scenario['direction'].upper()}")
        print(f"   Цена входа: {scenario['entry_price']}")
        print(f"   Количество баров: {len(scenario['bars'])}")
        print(f"   Результат: {result['final_return']:.2f}%")
        print(f"   Ожидалось: {scenario['expected']:.2f}%")
        print(f"   Причина выхода: {result['exit_reason']}")
        print(f"   Realized PnL: {result['realized_pnl']:.2f}%")
        print(f"   Unrealized PnL: {result['unrealized_pnl']:.2f}%")
        
        # Проверка корректности
        if abs(result['final_return'] - scenario['expected']) < 0.01:
            print("   ✅ ТЕСТ ПРОЙДЕН")
        else:
            print("   ❌ ТЕСТ НЕ ПРОЙДЕН")

def test_real_data_simulation():
    """Симуляция на реальных данных"""
    print("\n\n📊 СИМУЛЯЦИЯ НА РЕАЛЬНЫХ ДАННЫХ")
    print("="*60)
    
    # Создаем синтетические данные похожие на крипторынок
    np.random.seed(42)
    n_bars = 1000
    
    # Генерируем волатильные данные
    returns = np.random.normal(0, 0.02, n_bars)  # 2% волатильность
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Добавляем шум для high/low
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    
    # Создаем DataFrame
    df = pd.DataFrame({
        'close': prices,
        'high': highs,
        'low': lows
    })
    
    # Тестируем стратегию
    preparer = MarketDatasetPreparator(None, {})
    
    wins = 0
    losses = 0
    total_return = 0
    
    # Симулируем входы каждые 10 баров
    for i in range(0, len(df)-100, 10):
        entry_price = df.iloc[i]['close']
        future_bars = df.iloc[i+1:i+101].to_dict('records')
        
        # Тестируем BUY
        buy_result = preparer._calculate_enhanced_result(
            entry_price, future_bars, 'buy',
            0.989, 1.058,
            [
                {'percent': 1.2, 'close_ratio': 0.20},
                {'percent': 2.4, 'close_ratio': 0.30},
                {'percent': 3.5, 'close_ratio': 0.30}
            ],
            {
                'breakeven_percent': 1.2,
                'breakeven_offset': 0.3,
                'lock_levels': [
                    {'trigger': 2.4, 'lock': 1.2},
                    {'trigger': 3.5, 'lock': 2.4}
                ]
            }
        )
        
        if buy_result['final_return'] > 0:
            wins += 1
        else:
            losses += 1
        total_return += buy_result['final_return']
    
    print(f"\nРезультаты симуляции (BUY):")
    print(f"   Всего сделок: {wins + losses}")
    print(f"   Прибыльных: {wins} ({wins/(wins+losses)*100:.1f}%)")
    print(f"   Убыточных: {losses} ({losses/(wins+losses)*100:.1f}%)")
    print(f"   Средний результат: {total_return/(wins+losses):.2f}%")
    print(f"   Общий результат: {total_return:.2f}%")

if __name__ == "__main__":
    test_trading_scenarios()
    test_real_data_simulation()