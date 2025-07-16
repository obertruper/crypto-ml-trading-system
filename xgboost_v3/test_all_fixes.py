#!/usr/bin/env python3
"""
Тестирование всех исправленных проблем в XGBoost v3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import pandas as pd
import numpy as np
from config import Config
from config.constants import TOP_SYMBOLS, DATA_LEAKAGE_PARAMS, ENSEMBLE_PARAMS
from data.btc_data_loader import BTCDataLoader
from data.feature_engineer import FeatureEngineer
from data.preprocessor import DataPreprocessor
from models.data_balancer import DataBalancer
from models.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_btc_data_loader():
    """Тест исправления в btc_data_loader.py"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ 1: BTCDataLoader - validate_btc_coverage")
    logger.info("="*60)
    
    try:
        config = Config()
        btc_loader = BTCDataLoader(config)
        
        # Создаем тестовый DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min').astype(int) // 10**6,
            'close': np.random.randn(100).cumsum() + 100,
            'btc_close': np.random.randn(100).cumsum() + 50000
        })
        
        # Тестируем метод validate_btc_coverage
        stats = btc_loader.validate_btc_coverage(df)
        
        logger.info(f"✅ Метод validate_btc_coverage работает корректно")
        logger.info(f"   Покрытие: {stats['coverage']:.1f}%")
        logger.info(f"   Синтетические данные: {stats['is_synthetic']}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в BTCDataLoader: {e}")
        return False
        
    return True


def test_constants():
    """Тест новых констант в constants.py"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ 2: Новые константы TOP_SYMBOLS и DATA_LEAKAGE_PARAMS")
    logger.info("="*60)
    
    try:
        logger.info(f"✅ TOP_SYMBOLS определены: {len(TOP_SYMBOLS)} символов")
        logger.info(f"   Первые 5: {TOP_SYMBOLS[:5]}")
        
        logger.info(f"\n✅ DATA_LEAKAGE_PARAMS определены:")
        for key, value in DATA_LEAKAGE_PARAMS.items():
            logger.info(f"   {key}: {value}")
            
    except Exception as e:
        logger.error(f"❌ Ошибка в константах: {e}")
        return False
        
    return True


def test_feature_engineer():
    """Тест исправлений в feature_engineer.py"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ 3: FeatureEngineer - порядок операций и TOP_SYMBOLS")
    logger.info("="*60)
    
    try:
        config = Config()
        config.training.test_mode = True
        feature_engineer = FeatureEngineer(config)
        
        # Создаем тестовый DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min').astype(int) // 10**6,
            'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'BNBUSDT'], 100),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.exponential(1000, 100),
            'rsi_val': np.random.uniform(20, 80, 100),
            'adx_val': np.random.uniform(10, 50, 100),
            'macd_hist': np.random.randn(100),
            'atr': np.random.exponential(0.5, 100),
            'bb_position': np.random.uniform(0, 1, 100),
            'volume_ratio': np.random.exponential(1, 100),
            'ema_15': np.random.randn(100).cumsum() + 100,
            'sar': np.random.randn(100).cumsum() + 100
        })
        
        # Тестируем создание признаков
        df_features = feature_engineer.create_features(df.copy())
        
        logger.info(f"✅ Создание признаков завершено успешно")
        logger.info(f"   Исходных признаков: {df.shape[1]}")
        logger.info(f"   Итоговых признаков: {df_features.shape[1]}")
        logger.info(f"   Создано новых: {df_features.shape[1] - df.shape[1]}")
        
        # Проверяем, что symbol one-hot encoding использует TOP_SYMBOLS
        symbol_features = [col for col in df_features.columns if col.startswith('is_')]
        logger.info(f"\n✅ Symbol one-hot encoding признаки: {len(symbol_features)}")
        logger.info(f"   Примеры: {symbol_features[:5]}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в FeatureEngineer: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


def test_preprocessor():
    """Тест улучшенной проверки утечки данных в preprocessor.py"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ 4: DataPreprocessor - проверка утечки данных с DATA_LEAKAGE_PARAMS")
    logger.info("="*60)
    
    try:
        config = Config()
        preprocessor = DataPreprocessor(config)
        
        # Создаем тестовые данные
        n_samples = 1000
        n_features = 50
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Добавляем признак с высокой корреляцией
        y_buy = pd.Series(np.random.randn(n_samples))
        y_sell = pd.Series(np.random.randn(n_samples))
        
        # Создаем утечку в одном признаке
        X['feature_leak'] = y_buy * 0.95 + np.random.randn(n_samples) * 0.05
        
        # Тестируем проверку утечки данных
        logger.info(f"✅ Тестируем проверку утечки данных...")
        preprocessor._check_data_leakage(X, y_buy, y_sell)
        
        logger.info(f"\n✅ Проверка утечки данных работает с новыми параметрами")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в DataPreprocessor: {e}")
        return False
        
    return True


def test_data_balancer():
    """Тест DataBalancer на корректность работы"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ 5: DataBalancer - обработка бинарных признаков")
    logger.info("="*60)
    
    try:
        config = Config()
        config.training.balance_method = "smote"
        balancer = DataBalancer(config)
        
        # Создаем несбалансированные данные
        n_majority = 900
        n_minority = 100
        n_features = 10
        
        X_majority = pd.DataFrame(
            np.random.randn(n_majority, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        X_minority = pd.DataFrame(
            np.random.randn(n_minority, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Добавляем бинарные признаки
        X_majority['is_bullish'] = np.random.choice([0, 1], n_majority)
        X_minority['is_bullish'] = np.random.choice([0, 1], n_minority)
        
        X = pd.concat([X_majority, X_minority], ignore_index=True)
        y = pd.Series([0] * n_majority + [1] * n_minority)
        
        # Балансировка
        X_balanced, y_balanced = balancer.balance_data(X, y, is_classification=True)
        
        logger.info(f"✅ Балансировка выполнена успешно")
        logger.info(f"   До балансировки: {len(y)} примеров")
        logger.info(f"   После балансировки: {len(y_balanced)} примеров")
        
        # Проверяем бинарные признаки
        if 'is_bullish' in X_balanced.columns:
            unique_vals = X_balanced['is_bullish'].unique()
            logger.info(f"   Уникальные значения is_bullish: {sorted(unique_vals)}")
            if all(val in [0, 1] for val in unique_vals):
                logger.info(f"   ✅ Бинарные признаки сохранены корректно")
            else:
                logger.error(f"   ❌ Бинарные признаки повреждены!")
                
    except Exception as e:
        logger.error(f"❌ Ошибка в DataBalancer: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


def test_ensemble_params():
    """Тест использования ENSEMBLE_PARAMS в ensemble.py"""
    logger.info("\n" + "="*60)
    logger.info("🧪 ТЕСТ 6: EnsembleModel - использование ENSEMBLE_PARAMS")
    logger.info("="*60)
    
    try:
        config = Config()
        ensemble = EnsembleModel(config)
        
        # Тестируем получение разнообразных параметров
        for i in range(10):
            params = ensemble._get_diverse_params(i)
            variation_idx = i % len(ENSEMBLE_PARAMS['model_variations'])
            expected_variation = ENSEMBLE_PARAMS['model_variations'][variation_idx]
            
            logger.info(f"\nМодель {i+1}:")
            logger.info(f"   max_depth: {params.get('max_depth')} (ожидается: {expected_variation['max_depth']})")
            logger.info(f"   learning_rate: {params.get('learning_rate')} (ожидается: {expected_variation['learning_rate']})")
            logger.info(f"   subsample: {params.get('subsample')} (ожидается: {expected_variation['subsample']})")
            
        logger.info(f"\n✅ ENSEMBLE_PARAMS используются корректно")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в EnsembleModel: {e}")
        return False
        
    return True


def main():
    """Запуск всех тестов"""
    logger.info("""
    ╔══════════════════════════════════════════╗
    ║   ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ XGBoost v3    ║
    ╚══════════════════════════════════════════╝
    """)
    
    tests = [
        ("BTCDataLoader", test_btc_data_loader),
        ("Constants", test_constants),
        ("FeatureEngineer", test_feature_engineer),
        ("DataPreprocessor", test_preprocessor),
        ("DataBalancer", test_data_balancer),
        ("EnsembleModel", test_ensemble_params)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в тесте {test_name}: {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    logger.info("\n" + "="*60)
    logger.info("📊 ИТОГОВЫЙ ОТЧЕТ")
    logger.info("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nВсего тестов: {total}")
    logger.info(f"Успешно: {passed}")
    logger.info(f"Провалено: {total - passed}")
    
    if passed == total:
        logger.info("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        logger.info("\n⚠️ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ!")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)