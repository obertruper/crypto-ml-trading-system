#!/usr/bin/env python3
"""
Тестирование исправлений feature selection и весов категорий
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
from config import Config
from data.feature_engineer import FeatureEngineer
from utils.feature_selector import FeatureSelector
from config.feature_mapping import get_feature_category, get_category_targets

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_feature_categories():
    """Тест категоризации признаков"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ 1: Категоризация признаков")
    logger.info("="*60)
    
    # Тестовые признаки
    test_features = [
        ('rsi_val', 'technical'),
        ('rsi_val_ma_10', 'technical'),
        ('hour_sin', 'temporal'),
        ('dow_cos', 'temporal'),
        ('btc_correlation_20', 'btc_related'),
        ('market_regime_low_vol', 'technical'),  # Теперь в technical!
        ('consecutive_hh', 'technical'),  # Паттерн свечей
        ('is_hammer', 'technical'),  # Паттерн свечей
        ('is_weekend', 'temporal'),
        ('is_btc', 'symbol'),
        ('volume_ratio_ma_20', 'technical'),
        ('random_feature', 'other')
    ]
    
    passed = 0
    failed = 0
    
    for feature, expected_cat in test_features:
        actual_cat = get_feature_category(feature)
        if actual_cat == expected_cat:
            logger.info(f"✅ {feature}: {actual_cat} (правильно)")
            passed += 1
        else:
            logger.error(f"❌ {feature}: {actual_cat} (ожидалось: {expected_cat})")
            failed += 1
    
    logger.info(f"\nРезультат: {passed} пройдено, {failed} провалено")
    return failed == 0


def test_feature_selection_quotas():
    """Тест соблюдения квот при отборе признаков"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ 2: Соблюдение квот признаков")
    logger.info("="*60)
    
    # Создаем тестовый DataFrame с разными признаками
    n_samples = 1000
    
    # Генерируем признаки разных категорий
    df = pd.DataFrame({
        # Технические (должно быть 80%)
        'rsi_val': np.random.uniform(20, 80, n_samples),
        'rsi_val_ma_10': np.random.uniform(20, 80, n_samples),
        'macd_hist': np.random.randn(n_samples),
        'adx_val': np.random.uniform(10, 50, n_samples),
        'volume_ratio': np.random.exponential(1, n_samples),
        'bb_position': np.random.uniform(0, 1, n_samples),
        'atr': np.random.exponential(0.5, n_samples),
        'stoch_k': np.random.uniform(0, 100, n_samples),
        'ema_15': np.random.randn(n_samples).cumsum() + 100,
        'market_regime_low_vol': np.random.randint(0, 2, n_samples),
        'consecutive_hh': np.random.randint(0, 4, n_samples),
        'is_hammer': np.random.randint(0, 2, n_samples),
        
        # Временные (должно быть 5%)
        'hour_sin': np.sin(np.random.uniform(0, 2*np.pi, n_samples)),
        'hour_cos': np.cos(np.random.uniform(0, 2*np.pi, n_samples)),
        'dow_sin': np.sin(np.random.uniform(0, 2*np.pi, n_samples)),
        'dow_cos': np.cos(np.random.uniform(0, 2*np.pi, n_samples)),
        'is_weekend': np.random.randint(0, 2, n_samples),
        
        # BTC related (должно быть 10%)
        'btc_correlation_20': np.random.uniform(-1, 1, n_samples),
        'btc_correlation_60': np.random.uniform(-1, 1, n_samples),
        'btc_volatility': np.random.exponential(0.02, n_samples),
        
        # Символы (должно быть 5%)
        'is_btc': np.random.randint(0, 2, n_samples),
        'is_eth': np.random.randint(0, 2, n_samples),
        
        # Другие
        'unknown_feature': np.random.randn(n_samples)
    })
    
    # Целевая переменная
    y = (np.random.randn(n_samples) > 0).astype(int)
    
    # Тестируем отбор признаков
    selector = FeatureSelector(method="hierarchical", top_k=20)
    X_selected, selected_features = selector.select_features(df, pd.Series(y))
    
    logger.info(f"\n📊 Отобрано {len(selected_features)} признаков из {len(df.columns)}")
    
    # Проверяем распределение
    category_counts = {}
    for feature in selected_features:
        cat = get_feature_category(feature)
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Целевые проценты
    targets = get_category_targets()
    
    logger.info("\n📊 Распределение отобранных признаков:")
    all_ok = True
    for category, target_percent in targets.items():
        count = category_counts.get(category, 0)
        actual_percent = count / len(selected_features) * 100
        deviation = abs(actual_percent - target_percent)
        
        if deviation > 10:
            status = "❌ FAIL"
            all_ok = False
        else:
            status = "✅ OK"
            
        logger.info(f"   {category}: {count} ({actual_percent:.0f}%) {status} [цель: {target_percent}%]")
    
    # Проверяем что temporal не превышает 5%
    temporal_count = category_counts.get('temporal', 0)
    max_temporal = int(20 * 0.05)  # 5% от 20 = 1
    
    if temporal_count > max_temporal:
        logger.error(f"\n❌ Слишком много временных признаков: {temporal_count} > {max_temporal}")
        all_ok = False
    else:
        logger.info(f"\n✅ Временные признаки в пределах нормы: {temporal_count} <= {max_temporal}")
    
    return all_ok


def test_duplicate_features():
    """Тест на создание дублирующих признаков"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ 3: Проверка дублирующих признаков")
    logger.info("="*60)
    
    config = Config()
    config.training.test_mode = True
    feature_engineer = FeatureEngineer(config)
    
    # Создаем DataFrame с уже существующими rolling признаками
    df = pd.DataFrame({
        'rsi_val': np.random.uniform(20, 80, 100),
        'rsi_val_ma_10': np.random.uniform(20, 80, 100),  # Уже существует!
        'adx_val': np.random.uniform(10, 50, 100),
        'volume_ratio': np.random.exponential(1, 100)
    })
    
    # Вызываем создание rolling features
    df_with_features = feature_engineer._create_rolling_features(df.copy())
    
    # Проверяем что не создались дубликаты
    if 'rsi_val_ma_10' in df.columns:
        # Проверяем что значения не изменились
        if np.array_equal(df['rsi_val_ma_10'], df_with_features['rsi_val_ma_10']):
            logger.info("✅ Дублирующий признак rsi_val_ma_10 НЕ был пересоздан")
        else:
            logger.error("❌ Признак rsi_val_ma_10 был перезаписан!")
            return False
    
    # Проверяем что новые признаки создались
    new_features = set(df_with_features.columns) - set(df.columns)
    logger.info(f"\n📊 Создано {len(new_features)} новых признаков:")
    for feat in sorted(new_features):
        logger.info(f"   - {feat}")
    
    return True


def main():
    """Запуск всех тестов"""
    logger.info("""
    ╔══════════════════════════════════════════╗
    ║  Тестирование Feature Selection Fix v3   ║
    ╚══════════════════════════════════════════╝
    """)
    
    tests = [
        ("Категоризация признаков", test_feature_categories),
        ("Соблюдение квот", test_feature_selection_quotas),
        ("Дублирующие признаки", test_duplicate_features)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"\n❌ Ошибка в тесте '{test_name}': {e}")
            failed += 1
    
    logger.info("\n" + "="*60)
    logger.info("📊 ИТОГОВЫЙ РЕЗУЛЬТАТ")
    logger.info("="*60)
    logger.info(f"✅ Пройдено: {passed}")
    logger.info(f"❌ Провалено: {failed}")
    
    if failed == 0:
        logger.info("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        logger.info("   Веса категорий 80/5/10/5 теперь работают правильно!")
        logger.info("   Временные признаки ограничены до 5%!")
        logger.info("   Дублирующие признаки не создаются!")
    else:
        logger.error("\n⚠️ Некоторые тесты провалены, требуется доработка")


if __name__ == "__main__":
    main()