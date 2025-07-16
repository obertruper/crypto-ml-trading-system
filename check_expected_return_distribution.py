#!/usr/bin/env python3
"""
Диагностический скрипт для проверки распределения expected_return
и влияния разных порогов классификации
"""

import pandas as pd
import numpy as np
import psycopg2
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Настройка графиков
plt.style.use('dark_background')
sns.set_palette("husl")

def analyze_expected_returns():
    # Загрузка конфигурации
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    
    # Подключение к БД
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['dbname'],
        user=db_config['user']
    )
    
    # Загружаем expected_return для всех символов
    query = """
    SELECT 
        symbol,
        buy_expected_return,
        sell_expected_return
    FROM processed_market_data
    WHERE symbol IN ('BTCUSDT', 'ETHUSDT')
    ORDER BY timestamp DESC
    LIMIT 50000
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"📊 Загружено {len(df)} записей")
    print(f"Символы: {df['symbol'].unique()}")
    
    # Общая статистика
    print("\n📈 ОБЩАЯ СТАТИСТИКА EXPECTED_RETURN:")
    print(f"\nBuy Expected Return:")
    print(df['buy_expected_return'].describe())
    print(f"\nSell Expected Return:")
    print(df['sell_expected_return'].describe())
    
    # Анализ для разных порогов
    thresholds = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    print("\n🎯 АНАЛИЗ РАЗНЫХ ПОРОГОВ КЛАССИФИКАЦИИ:")
    print(f"{'Порог':<10} {'Buy Класс 1':<15} {'Sell Класс 1':<15} {'Buy/Sell Ratio':<15}")
    print("-" * 55)
    
    for threshold in thresholds:
        buy_class_1 = (df['buy_expected_return'] > threshold).mean() * 100
        sell_class_1 = (df['sell_expected_return'] > threshold).mean() * 100
        ratio = buy_class_1 / sell_class_1 if sell_class_1 > 0 else float('inf')
        print(f"{threshold:<10.1f} {buy_class_1:<15.1f} {sell_class_1:<15.1f} {ratio:<15.2f}")
    
    # Создаем визуализацию
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Анализ распределения Expected Return', fontsize=16)
    
    # 1. Гистограммы распределения
    ax = axes[0, 0]
    ax.hist(df['buy_expected_return'], bins=100, alpha=0.7, label='Buy', color='green')
    ax.hist(df['sell_expected_return'], bins=100, alpha=0.7, label='Sell', color='red')
    ax.axvline(0.5, color='yellow', linestyle='--', label='Порог 0.5%')
    ax.axvline(1.0, color='orange', linestyle='--', label='Порог 1.0%')
    ax.set_xlabel('Expected Return (%)')
    ax.set_ylabel('Количество')
    ax.set_title('Распределение Expected Return')
    ax.legend()
    ax.set_xlim(-5, 5)
    
    # 2. Box plots
    ax = axes[0, 1]
    data_to_plot = [df['buy_expected_return'], df['sell_expected_return']]
    box = ax.boxplot(data_to_plot, labels=['Buy', 'Sell'], patch_artist=True)
    box['boxes'][0].set_facecolor('green')
    box['boxes'][1].set_facecolor('red')
    ax.axhline(0.5, color='yellow', linestyle='--', label='Порог 0.5%')
    ax.axhline(1.0, color='orange', linestyle='--', label='Порог 1.0%')
    ax.set_ylabel('Expected Return (%)')
    ax.set_title('Box Plot Expected Return')
    ax.legend()
    
    # 3. Процент положительных классов для разных порогов
    ax = axes[1, 0]
    buy_percentages = [(df['buy_expected_return'] > t).mean() * 100 for t in thresholds]
    sell_percentages = [(df['sell_expected_return'] > t).mean() * 100 for t in thresholds]
    
    ax.plot(thresholds, buy_percentages, 'g-o', label='Buy', linewidth=2)
    ax.plot(thresholds, sell_percentages, 'r-o', label='Sell', linewidth=2)
    ax.axvline(0.5, color='yellow', linestyle='--', alpha=0.5)
    ax.axvline(1.0, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Порог классификации (%)')
    ax.set_ylabel('% Класс 1 (положительные)')
    ax.set_title('Влияние порога на баланс классов')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Кумулятивное распределение
    ax = axes[1, 1]
    sorted_buy = np.sort(df['buy_expected_return'])
    sorted_sell = np.sort(df['sell_expected_return'])
    
    ax.plot(sorted_buy, np.linspace(0, 100, len(sorted_buy)), 'g-', label='Buy', linewidth=2)
    ax.plot(sorted_sell, np.linspace(0, 100, len(sorted_sell)), 'r-', label='Sell', linewidth=2)
    ax.axvline(0.5, color='yellow', linestyle='--', alpha=0.5, label='Порог 0.5%')
    ax.axvline(1.0, color='orange', linestyle='--', alpha=0.5, label='Порог 1.0%')
    ax.set_xlabel('Expected Return (%)')
    ax.set_ylabel('Кумулятивный %')
    ax.set_title('Кумулятивное распределение')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    
    plt.tight_layout()
    
    # Сохраняем график
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'expected_return_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 График сохранен: {filename}")
    
    # Дополнительный анализ
    print("\n🔍 ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
    print(f"Отрицательные buy returns: {(df['buy_expected_return'] < 0).sum()} ({(df['buy_expected_return'] < 0).mean()*100:.1f}%)")
    print(f"Отрицательные sell returns: {(df['sell_expected_return'] < 0).sum()} ({(df['sell_expected_return'] < 0).mean()*100:.1f}%)")
    
    # Квантили
    print("\n📊 КВАНТИЛИ:")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    for q in quantiles:
        buy_q = df['buy_expected_return'].quantile(q)
        sell_q = df['sell_expected_return'].quantile(q)
        print(f"Q{int(q*100):02d}: Buy={buy_q:6.2f}%, Sell={sell_q:6.2f}%")

if __name__ == "__main__":
    analyze_expected_returns()