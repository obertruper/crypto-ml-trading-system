#!/usr/bin/env python3
"""
Анализ проблемы с метками классов и порогами
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from data.loader import DataLoader
from config.settings import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_labels():
    """Анализ распределения меток и expected returns"""
    
    # Загружаем конфигурацию
    config = Config()
    
    # Создаем загрузчик
    loader = DataLoader(config)
    
    # Загружаем тестовые данные
    logger.info("📥 Загрузка данных...")
    df = loader.load_data(test_mode=True)
    
    # Анализируем expected returns
    logger.info("\n📊 АНАЛИЗ EXPECTED RETURNS:")
    logger.info(f"Количество записей: {len(df):,}")
    
    # Статистика по buy_expected_return
    logger.info("\n🎯 BUY Expected Returns:")
    logger.info(f"   Среднее: {df['buy_expected_return'].mean():.4f}%")
    logger.info(f"   Медиана: {df['buy_expected_return'].median():.4f}%")
    logger.info(f"   Мин: {df['buy_expected_return'].min():.4f}%")
    logger.info(f"   Макс: {df['buy_expected_return'].max():.4f}%")
    logger.info(f"   Std: {df['buy_expected_return'].std():.4f}%")
    
    # Статистика по sell_expected_return
    logger.info("\n🎯 SELL Expected Returns:")
    logger.info(f"   Среднее: {df['sell_expected_return'].mean():.4f}%")
    logger.info(f"   Медиана: {df['sell_expected_return'].median():.4f}%")
    logger.info(f"   Мин: {df['sell_expected_return'].min():.4f}%")
    logger.info(f"   Макс: {df['sell_expected_return'].max():.4f}%")
    logger.info(f"   Std: {df['sell_expected_return'].std():.4f}%")
    
    # Анализ распределения для разных порогов
    logger.info("\n📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ ДЛЯ РАЗНЫХ ПОРОГОВ:")
    
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for threshold in thresholds:
        buy_positive = (df['buy_expected_return'] > threshold).sum()
        buy_negative = (df['buy_expected_return'] <= threshold).sum()
        buy_ratio = buy_positive / len(df) * 100
        
        sell_positive = (df['sell_expected_return'] > threshold).sum()
        sell_negative = (df['sell_expected_return'] <= threshold).sum()
        sell_ratio = sell_positive / len(df) * 100
        
        logger.info(f"\n🔍 Порог = {threshold}%:")
        logger.info(f"   BUY:  {buy_positive:,} положительных ({buy_ratio:.1f}%), {buy_negative:,} отрицательных")
        logger.info(f"   SELL: {sell_positive:,} положительных ({sell_ratio:.1f}%), {sell_negative:,} отрицательных")
        
        # Проверка баланса классов
        buy_balance = min(buy_positive, buy_negative) / max(buy_positive, buy_negative)
        sell_balance = min(sell_positive, sell_negative) / max(sell_positive, sell_negative)
        
        logger.info(f"   Баланс классов: BUY={buy_balance:.3f}, SELL={sell_balance:.3f}")
        
        if buy_balance < 0.1 or sell_balance < 0.1:
            logger.warning(f"   ⚠️ СИЛЬНЫЙ ДИСБАЛАНС КЛАССОВ!")
    
    # Анализ квантилей
    logger.info("\n📊 КВАНТИЛИ EXPECTED RETURNS:")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    logger.info("\nBUY квантили:")
    for q in quantiles:
        value = df['buy_expected_return'].quantile(q)
        logger.info(f"   {q*100:.0f}%: {value:.4f}%")
    
    logger.info("\nSELL квантили:")
    for q in quantiles:
        value = df['sell_expected_return'].quantile(q)
        logger.info(f"   {q*100:.0f}%: {value:.4f}%")
    
    # Проверка на аномалии
    logger.info("\n🔍 ПРОВЕРКА НА АНОМАЛИИ:")
    
    # Проверка на нулевые значения
    buy_zeros = (df['buy_expected_return'] == 0).sum()
    sell_zeros = (df['sell_expected_return'] == 0).sum()
    
    logger.info(f"   Нулевые значения: BUY={buy_zeros:,} ({buy_zeros/len(df)*100:.1f}%), SELL={sell_zeros:,} ({sell_zeros/len(df)*100:.1f}%)")
    
    # Проверка на очень маленькие значения
    very_small_threshold = 0.01  # 0.01%
    buy_very_small = (df['buy_expected_return'].abs() < very_small_threshold).sum()
    sell_very_small = (df['sell_expected_return'].abs() < very_small_threshold).sum()
    
    logger.info(f"   Очень маленькие значения (<{very_small_threshold}%): BUY={buy_very_small:,} ({buy_very_small/len(df)*100:.1f}%), SELL={sell_very_small:,} ({sell_very_small/len(df)*100:.1f}%)")
    
    # Визуализация распределения
    logger.info("\n📊 ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ:")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Buy expected returns
    axes[0, 0].hist(df['buy_expected_return'], bins=100, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].axvline(x=1.5, color='red', linestyle='--', label='Порог 1.5%')
    axes[0, 0].set_title('Buy Expected Returns')
    axes[0, 0].set_xlabel('Expected Return (%)')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].legend()
    
    # Sell expected returns
    axes[0, 1].hist(df['sell_expected_return'], bins=100, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].axvline(x=1.5, color='green', linestyle='--', label='Порог 1.5%')
    axes[0, 1].set_title('Sell Expected Returns')
    axes[0, 1].set_xlabel('Expected Return (%)')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].legend()
    
    # Buy - логарифмическая шкала
    axes[1, 0].hist(df['buy_expected_return'], bins=100, alpha=0.7, color='green', edgecolor='black', log=True)
    axes[1, 0].axvline(x=1.5, color='red', linestyle='--', label='Порог 1.5%')
    axes[1, 0].set_title('Buy Expected Returns (log scale)')
    axes[1, 0].set_xlabel('Expected Return (%)')
    axes[1, 0].set_ylabel('Частота (log)')
    axes[1, 0].legend()
    
    # Sell - логарифмическая шкала
    axes[1, 1].hist(df['sell_expected_return'], bins=100, alpha=0.7, color='red', edgecolor='black', log=True)
    axes[1, 1].axvline(x=1.5, color='green', linestyle='--', label='Порог 1.5%')
    axes[1, 1].set_title('Sell Expected Returns (log scale)')
    axes[1, 1].set_xlabel('Expected Return (%)')
    axes[1, 1].set_ylabel('Частота (log)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('expected_returns_distribution.png', dpi=150)
    logger.info("✅ График сохранен: expected_returns_distribution.png")
    
    # Рекомендации
    logger.info("\n💡 РЕКОМЕНДАЦИИ:")
    
    # Оптимальный порог на основе баланса классов
    best_threshold = None
    best_balance = 0
    
    for threshold in thresholds:
        buy_positive = (df['buy_expected_return'] > threshold).sum()
        buy_negative = (df['buy_expected_return'] <= threshold).sum()
        balance = min(buy_positive, buy_negative) / max(buy_positive, buy_negative)
        
        if balance > best_balance and balance > 0.2:  # Минимум 20% баланс
            best_balance = balance
            best_threshold = threshold
    
    if best_threshold:
        logger.info(f"   ✅ Рекомендуемый порог: {best_threshold}% (баланс классов: {best_balance:.3f})")
    else:
        logger.info(f"   ⚠️ Не найден хороший порог с балансом классов > 20%")
        
    # Проверка масштаба данных
    if df['buy_expected_return'].std() < 0.1:
        logger.warning("   ⚠️ Очень маленькая дисперсия expected returns! Возможно, данные в долях (0.015), а не процентах (1.5)")
        logger.info("   💡 Попробуйте умножить expected returns на 100")
    
    # Финальная проверка текущего порога
    current_threshold = config.training.classification_threshold
    logger.info(f"\n🔍 ПРОВЕРКА ТЕКУЩЕГО ПОРОГА ({current_threshold}%):")
    
    buy_positive = (df['buy_expected_return'] > current_threshold).sum()
    buy_ratio = buy_positive / len(df) * 100
    
    sell_positive = (df['sell_expected_return'] > current_threshold).sum()
    sell_ratio = sell_positive / len(df) * 100
    
    logger.info(f"   BUY:  {buy_positive:,} положительных ({buy_ratio:.1f}%)")
    logger.info(f"   SELL: {sell_positive:,} положительных ({sell_ratio:.1f}%)")
    
    if buy_ratio < 5 or sell_ratio < 5:
        logger.error("   ❌ КРИТИЧНО: Менее 5% положительных примеров! Модель не сможет обучиться!")
    elif buy_ratio < 10 or sell_ratio < 10:
        logger.warning("   ⚠️ ВНИМАНИЕ: Менее 10% положительных примеров. Сильный дисбаланс классов!")
    else:
        logger.info("   ✅ Баланс классов приемлемый")


if __name__ == "__main__":
    analyze_labels()