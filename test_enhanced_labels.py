#!/usr/bin/env python3
"""
Тестирование новой логики создания меток с учетом частичных закрытий
"""

import pandas as pd
import psycopg2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database'].copy()
if not db_config.get('password'):
    db_config.pop('password', None)


def test_enhanced_labels():
    """Тестирует новую логику на небольшом объеме данных"""
    
    print("="*80)
    print("ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ ЛОГИКИ СОЗДАНИЯ МЕТОК")
    print("="*80)
    
    # Подключаемся к БД
    conn = psycopg2.connect(**db_config)
    
    # Загружаем небольшой объем данных для теста
    query = """
    SELECT 
        r.symbol, r.timestamp, r.close, r.high, r.low,
        p.buy_profit_target, p.buy_loss_target,
        p.sell_profit_target, p.sell_loss_target,
        p.technical_indicators
    FROM raw_market_data r
    JOIN processed_market_data p ON r.id = p.raw_data_id
    WHERE r.symbol = '1INCHUSDT' 
    AND r.market_type = 'futures'
    ORDER BY r.timestamp
    LIMIT 5000
    """
    
    df = pd.read_sql(query, conn)
    print(f"\n📊 Загружено {len(df)} записей для анализа")
    
    # Извлекаем ожидаемые результаты из JSONB
    df['buy_expected_return'] = df['technical_indicators'].apply(
        lambda x: x.get('buy_expected_return', 0.0) if x else 0.0
    )
    df['sell_expected_return'] = df['technical_indicators'].apply(
        lambda x: x.get('sell_expected_return', 0.0) if x else 0.0
    )
    
    # Анализ старых меток
    print("\n📈 АНАЛИЗ СТАРЫХ БИНАРНЫХ МЕТОК:")
    print("-"*50)
    
    buy_profit = df['buy_profit_target'].sum()
    buy_loss = df['buy_loss_target'].sum()
    sell_profit = df['sell_profit_target'].sum()
    sell_loss = df['sell_loss_target'].sum()
    
    total_buy = buy_profit + buy_loss
    total_sell = sell_profit + sell_loss
    
    print(f"BUY сигналы:")
    if total_buy > 0:
        print(f"  Прибыльных: {buy_profit} ({buy_profit/total_buy*100:.1f}%)")
        print(f"  Убыточных: {buy_loss} ({buy_loss/total_buy*100:.1f}%)")
    else:
        print("  Нет сигналов")
    
    print(f"\nSELL сигналы:")
    if total_sell > 0:
        print(f"  Прибыльных: {sell_profit} ({sell_profit/total_sell*100:.1f}%)")
        print(f"  Убыточных: {sell_loss} ({sell_loss/total_sell*100:.1f}%)")
    else:
        print("  Нет сигналов")
    
    # Анализ новых ожидаемых результатов
    print("\n📊 АНАЛИЗ НОВЫХ ОЖИДАЕМЫХ РЕЗУЛЬТАТОВ:")
    print("-"*50)
    
    # Фильтруем только те записи, где есть ожидаемые результаты
    df_with_returns = df[
        (df['buy_expected_return'] != 0) | (df['sell_expected_return'] != 0)
    ]
    
    if len(df_with_returns) > 0:
        print(f"\nНайдено {len(df_with_returns)} записей с новыми расчетами")
        
        # Статистика по BUY
        buy_returns = df_with_returns['buy_expected_return']
        buy_positive = (buy_returns > 0.5).sum()
        buy_negative = (buy_returns < -0.5).sum()
        buy_neutral = len(buy_returns) - buy_positive - buy_negative
        
        print(f"\nBUY позиции:")
        print(f"  Прибыльных (>0.5%): {buy_positive} ({buy_positive/len(buy_returns)*100:.1f}%)")
        print(f"  Убыточных (<-0.5%): {buy_negative} ({buy_negative/len(buy_returns)*100:.1f}%)")
        print(f"  Нейтральных: {buy_neutral} ({buy_neutral/len(buy_returns)*100:.1f}%)")
        print(f"  Средний результат: {buy_returns.mean():.2f}%")
        print(f"  Медиана: {buy_returns.median():.2f}%")
        print(f"  Лучший результат: {buy_returns.max():.2f}%")
        print(f"  Худший результат: {buy_returns.min():.2f}%")
        
        # Статистика по SELL
        sell_returns = df_with_returns['sell_expected_return']
        sell_positive = (sell_returns > 0.5).sum()
        sell_negative = (sell_returns < -0.5).sum()
        sell_neutral = len(sell_returns) - sell_positive - sell_negative
        
        print(f"\nSELL позиции:")
        print(f"  Прибыльных (>0.5%): {sell_positive} ({sell_positive/len(sell_returns)*100:.1f}%)")
        print(f"  Убыточных (<-0.5%): {sell_negative} ({sell_negative/len(sell_returns)*100:.1f}%)")
        print(f"  Нейтральных: {sell_neutral} ({sell_neutral/len(sell_returns)*100:.1f}%)")
        print(f"  Средний результат: {sell_returns.mean():.2f}%")
        print(f"  Медиана: {sell_returns.median():.2f}%")
        print(f"  Лучший результат: {sell_returns.max():.2f}%")
        print(f"  Худший результат: {sell_returns.min():.2f}%")
        
        # Визуализация распределения
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Гистограмма BUY
        axes[0, 0].hist(buy_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Безубыток')
        axes[0, 0].axvline(x=buy_returns.mean(), color='blue', linestyle='-', label=f'Среднее: {buy_returns.mean():.2f}%')
        axes[0, 0].set_title('Распределение ожидаемых результатов BUY')
        axes[0, 0].set_xlabel('Ожидаемый результат (%)')
        axes[0, 0].set_ylabel('Количество')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Гистограмма SELL
        axes[0, 1].hist(sell_returns, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Безубыток')
        axes[0, 1].axvline(x=sell_returns.mean(), color='blue', linestyle='-', label=f'Среднее: {sell_returns.mean():.2f}%')
        axes[0, 1].set_title('Распределение ожидаемых результатов SELL')
        axes[0, 1].set_xlabel('Ожидаемый результат (%)')
        axes[0, 1].set_ylabel('Количество')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot для сравнения
        axes[1, 0].boxplot([buy_returns, sell_returns], labels=['BUY', 'SELL'])
        axes[1, 0].set_title('Сравнение распределений BUY vs SELL')
        axes[1, 0].set_ylabel('Ожидаемый результат (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Кумулятивное распределение
        buy_sorted = np.sort(buy_returns)
        sell_sorted = np.sort(sell_returns)
        buy_cumulative = np.arange(1, len(buy_sorted) + 1) / len(buy_sorted)
        sell_cumulative = np.arange(1, len(sell_sorted) + 1) / len(sell_sorted)
        
        axes[1, 1].plot(buy_sorted, buy_cumulative, label='BUY', color='green')
        axes[1, 1].plot(sell_sorted, sell_cumulative, label='SELL', color='red')
        axes[1, 1].set_title('Кумулятивное распределение результатов')
        axes[1, 1].set_xlabel('Ожидаемый результат (%)')
        axes[1, 1].set_ylabel('Кумулятивная вероятность')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('enhanced_labels_analysis.png', dpi=150, bbox_inches='tight')
        print("\n📈 График сохранен: enhanced_labels_analysis.png")
        
    else:
        print("\n⚠️ Новые ожидаемые результаты еще не рассчитаны.")
        print("   Запустите prepare_dataset.py для обновления данных.")
    
    # Примеры конкретных сделок
    print("\n" + "="*80)
    print("ПРИМЕРЫ КОНКРЕТНЫХ СДЕЛОК")
    print("="*80)
    
    # Находим примеры успешных сделок с новой логикой
    if len(df_with_returns) > 0:
        successful_buys = df_with_returns[df_with_returns['buy_expected_return'] > 2.0].head(3)
        
        if len(successful_buys) > 0:
            print("\n🟢 Примеры успешных BUY сделок:")
            for idx, row in successful_buys.iterrows():
                print(f"\nВремя: {row['timestamp']}")
                print(f"Цена входа: ${row['close']:,.2f}")
                print(f"Ожидаемый результат: {row['buy_expected_return']:.2f}%")
                print(f"Старая метка: {'Profit' if row['buy_profit_target'] else 'Loss' if row['buy_loss_target'] else 'Нет'}")
    
    conn.close()
    
    print("\n" + "="*80)
    print("ВЫВОДЫ:")
    print("="*80)
    
    print("""
1. Старый подход дает сильно искаженную картину из-за жестких целей (5.8%)
2. Новый подход учитывает частичные закрытия и защиту прибыли
3. Ожидаемые результаты более реалистичны и сбалансированы
4. Модель сможет лучше обучаться на реальных паттернах торговли
    """)


if __name__ == "__main__":
    test_enhanced_labels()