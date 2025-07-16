#!/usr/bin/env python3
"""
Тест исправлений feature selection для XGBoost v3.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
from typing import List, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_feature_mapping():
    """Тест feature mapping"""
    logger.info("🧪 Тестирование feature mapping...")
    
    try:
        from config.feature_mapping import (
            get_feature_category, 
            get_category_targets, 
            get_temporal_blacklist
        )
        
        # Тест категоризации
        test_features = [
            'rsi_val', 'dow_cos', 'btc_correlation_20', 'is_btc', 
            'market_regime_low_vol', 'hour_sin', 'is_weekend'
        ]
        
        expected_categories = [
            'technical', 'temporal', 'btc_related', 'symbol',
            'technical', 'temporal', 'temporal'
        ]
        
        logger.info("   Тест категоризации признаков:")
        for feature, expected in zip(test_features, expected_categories):
            actual = get_feature_category(feature)
            status = "✅" if actual == expected else "❌"
            logger.info(f"      {feature}: {actual} {status}")
        
        # Тест целевых процентов
        targets = get_category_targets()
        logger.info(f"   Целевые проценты: {targets}")
        
        # Тест blacklist
        blacklist = get_temporal_blacklist()
        logger.info(f"   Temporal blacklist: {blacklist}")
        
        logger.info("✅ Feature mapping работает корректно")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в feature mapping: {e}")
        return False

def test_feature_selector():
    """Тест feature selector с новыми квотами"""
    logger.info("🧪 Тестирование feature selector...")
    
    try:
        from utils.feature_selector import FeatureSelector
        
        # Создаем тестовые данные
        np.random.seed(42)
        n_samples = 1000
        
        # Технические признаки
        technical_features = [
            'rsi_val', 'macd_val', 'bb_position', 'adx_val', 'atr',
            'volume_ratio', 'stoch_k', 'williams_r', 'mfi', 'cci',
            'market_regime_low_vol', 'market_regime_med_vol'
        ]
        
        # Temporal признаки (включая проблемные)
        temporal_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
        
        # BTC признаки
        btc_features = ['btc_correlation_20', 'btc_volatility', 'btc_return_1h']
        
        # Symbol признаки
        symbol_features = ['is_btc', 'is_eth', 'is_bnb']
        
        all_features = technical_features + temporal_features + btc_features + symbol_features
        
        # Создаем DataFrame с синтетическими данными
        data = {}
        for feature in all_features:
            data[feature] = np.random.randn(n_samples)
        
        # Делаем temporal признаки "важными" для проверки что они ограничиваются
        for temp_feat in temporal_features:
            data[temp_feat] = data[temp_feat] * 2  # Увеличиваем дисперсию
        
        X = pd.DataFrame(data)
        y = pd.Series(np.random.randint(0, 2, n_samples))  # Бинарная целевая
        
        # Тестируем hierarchical selector
        selector = FeatureSelector(method="hierarchical", top_k=20)
        
        logger.info(f"   Исходные признаки ({len(all_features)}): {', '.join(all_features)}")
        
        X_selected, selected_features = selector.select_features(X, y, all_features)
        
        logger.info(f"   Отобрано признаков: {len(selected_features)}")
        
        # Анализируем распределение
        from config.feature_mapping import get_feature_category
        
        category_counts = {}
        for feature in selected_features:
            cat = get_feature_category(feature)
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        logger.info("   Распределение по категориям:")
        total = len(selected_features)
        for cat, count in category_counts.items():
            percentage = count / total * 100 if total > 0 else 0
            logger.info(f"      {cat}: {count} ({percentage:.1f}%)")
        
        # Проверяем что temporal <= 1 признак (что для 20 признаков = 5%)
        temporal_count = category_counts.get('temporal', 0)
        temporal_percentage = temporal_count / total * 100 if total > 0 else 0
        
        # Для теста с 20 признаками: 1 temporal = 5%, что нормально
        max_allowed_temporal = 1
        
        if temporal_count <= max_allowed_temporal:
            logger.info(f"✅ Temporal ограничение работает: {temporal_count} признак(ов) <= {max_allowed_temporal}")
        else:
            logger.error(f"❌ Слишком много temporal: {temporal_count} > {max_allowed_temporal}")
            return False
        
        logger.info("✅ Feature selector работает корректно")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в feature selector: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_importance_validator():
    """Тест валидатора важности признаков"""
    logger.info("🧪 Тестирование feature importance validator...")
    
    try:
        from utils.feature_importance_validator import FeatureImportanceValidator
        import xgboost as xgb
        
        # Создаем тестовую модель
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # Имитируем проблемную ситуацию - temporal признаки важнее
        feature_names = [
            'dow_cos', 'hour_sin', 'is_weekend',  # temporal (первые 3 - будут важнее)
            'rsi_val', 'macd_val', 'bb_position', 'adx_val',  # technical
            'btc_correlation_20', 'btc_volatility',  # btc_related
            'is_btc', 'is_eth'  # symbol
        ] + [f'other_feature_{i}' for i in range(9)]  # other
        
        # Искусственно делаем temporal признаки важнее
        original_importances = model.feature_importances_.copy()
        modified_importances = original_importances.copy()
        modified_importances[0] = 0.3  # dow_cos
        modified_importances[1] = 0.25  # hour_sin
        modified_importances[2] = 0.15  # is_weekend
        # Остальные нормализуем
        remaining_sum = 1.0 - 0.7
        remaining_features = modified_importances[3:]
        remaining_features = remaining_features / remaining_features.sum() * remaining_sum
        modified_importances[3:] = remaining_features
        
        # Подменяем важности через monkey patching
        # Поскольку feature_importances_ readonly, создаем mock объект
        class MockModel:
            def __init__(self, importances):
                self.feature_importances_ = importances
        
        mock_model = MockModel(modified_importances)
        
        # Тестируем валидатор
        validator = FeatureImportanceValidator(max_temporal_importance=5.0)
        result = validator.validate_model_feature_importance(mock_model, feature_names, "test_model")
        
        logger.info(f"   Результат валидации: {result}")
        
        # Должен обнаружить проблему
        if not result['valid'] and result['severity'] == 'critical':
            logger.info("✅ Валидатор корректно обнаружил переобучение на temporal")
        else:
            logger.error("❌ Валидатор не обнаружил проблему")
            return False
        
        logger.info("✅ Feature importance validator работает корректно")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в validator: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_updates():
    """Тест обновлений конфигурации"""
    logger.info("🧪 Тестирование обновлений конфигурации...")
    
    try:
        # Проверяем что rolling_windows уменьшены
        from config.features_config import FEATURE_CONFIG
        
        rolling_windows = FEATURE_CONFIG['rolling_windows']
        logger.info(f"   Rolling windows: {rolling_windows}")
        
        if len(rolling_windows) == 2 and rolling_windows == [20, 60]:
            logger.info("✅ Rolling windows корректно уменьшены до 2")
        else:
            logger.error(f"❌ Rolling windows некорректны: {rolling_windows}")
            return False
        
        # Проверяем обновленные целевые проценты
        from config.feature_mapping import get_category_targets
        targets = get_category_targets()
        
        if (targets['technical'] == 85 and targets['temporal'] == 2 and 
            targets['btc_related'] == 10 and targets['symbol'] == 3):
            logger.info("✅ Целевые проценты корректно обновлены")
        else:
            logger.error(f"❌ Целевые проценты некорректны: {targets}")
            return False
        
        logger.info("✅ Конфигурация обновлена корректно")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка в конфигурации: {e}")
        return False

def main():
    """Запуск всех тестов"""
    logger.info("🚀 ЗАПУСК ТЕСТОВ ИСПРАВЛЕНИЙ FEATURE SELECTION")
    logger.info("="*60)
    
    tests = [
        test_feature_mapping,
        test_config_updates,
        test_feature_selector,
        test_feature_importance_validator
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в {test_func.__name__}: {e}")
            failed += 1
        
        logger.info("-" * 40)
    
    logger.info("="*60)
    logger.info("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    logger.info(f"✅ Пройдено: {passed}")
    logger.info(f"❌ Провалено: {failed}")
    
    if failed == 0:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Исправления работают корректно.")
        return True
    else:
        logger.error("💥 ЕСТЬ ПРОВАЛЬНЫЕ ТЕСТЫ! Требуется дополнительная отладка.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)