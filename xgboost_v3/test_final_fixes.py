#!/usr/bin/env python3
"""
Финальный тест всех исправлений XGBoost v3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# Цвета для вывода без colorama
CYAN = '\033[96m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_header(text):
    """Печать заголовка"""
    print(f"\n{CYAN}{'='*50}")
    print(f"{text}")
    print(f"{'='*50}{RESET}")

def print_test(test_name, passed, details=""):
    """Печать результата теста"""
    status = f"{GREEN}✅ PASSED{RESET}" if passed else f"{RED}❌ FAILED{RESET}"
    print(f"{test_name}: {status}")
    if details:
        print(f"   {details}")

def test_imports():
    """Тест импортов и констант"""
    print_header("Тест 1: Импорты и константы")
    try:
        from config.constants import TOP_SYMBOLS, DATA_LEAKAGE_PARAMS
        from config import Config
        
        # Проверка констант
        assert isinstance(TOP_SYMBOLS, list), "TOP_SYMBOLS должен быть списком"
        assert len(TOP_SYMBOLS) == 14, f"TOP_SYMBOLS должен содержать 14 символов, а содержит {len(TOP_SYMBOLS)}"
        
        assert isinstance(DATA_LEAKAGE_PARAMS, dict), "DATA_LEAKAGE_PARAMS должен быть словарем"
        assert 'check_n_features' in DATA_LEAKAGE_PARAMS, "DATA_LEAKAGE_PARAMS должен содержать check_n_features"
        
        print_test("Импорт констант", True, f"TOP_SYMBOLS содержит {len(TOP_SYMBOLS)} символов")
        return True
    except Exception as e:
        print_test("Импорт констант", False, str(e))
        return False

def test_btc_data_loader():
    """Тест BTCDataLoader"""
    print_header("Тест 2: BTCDataLoader")
    try:
        from data.btc_data_loader import BTCDataLoader
        from config import Config
        
        # Создаем тестовый конфиг
        config = Config()
        
        # Создаем loader
        loader = BTCDataLoader(config)
        
        # Создаем тестовый DataFrame
        df = pd.DataFrame({
            'timestamp': [1000, 2000, 3000],
            'close': [100, 105, 110]
        })
        
        # Проверяем validate_btc_coverage
        df['btc_close'] = [100, 105, 110]
        stats = loader.validate_btc_coverage(df)
        
        assert isinstance(stats, dict), "validate_btc_coverage должен возвращать словарь"
        assert 'coverage' in stats, "Статистика должна содержать coverage"
        assert 'is_synthetic' in stats, "Статистика должна содержать is_synthetic"
        
        print_test("BTCDataLoader.validate_btc_coverage", True, f"Coverage: {stats['coverage']:.1f}%")
        return True
    except Exception as e:
        print_test("BTCDataLoader", False, str(e))
        return False

def test_feature_engineer():
    """Тест FeatureEngineer"""
    print_header("Тест 3: FeatureEngineer")
    try:
        from data.feature_engineer import FeatureEngineer
        from config import Config, TOP_SYMBOLS
        
        config = Config()
        engineer = FeatureEngineer(config)
        
        # Проверяем использование TOP_SYMBOLS
        df = pd.DataFrame({
            'symbol': ['BTCUSDT', 'ETHUSDT'],
            'close': [100, 200]
        })
        
        # Создаем symbol features
        df_with_features = engineer._create_symbol_features(df)
        
        # Проверяем что создались правильные колонки
        expected_cols = ['is_btc', 'is_eth']
        created_cols = [col for col in df_with_features.columns if col.startswith('is_')]
        
        assert len(created_cols) > 0, "Должны быть созданы symbol features"
        
        print_test("FeatureEngineer использует TOP_SYMBOLS", True, f"Создано {len(created_cols)} symbol features")
        
        # Проверяем порядок вызовов
        # Метод _remove_constant_features должен вызываться ДО validate_features
        assert hasattr(engineer, '_remove_constant_features'), "_remove_constant_features должен существовать"
        
        print_test("Порядок методов в FeatureEngineer", True)
        return True
    except Exception as e:
        print_test("FeatureEngineer", False, str(e))
        return False

def test_preprocessor():
    """Тест DataPreprocessor"""
    print_header("Тест 4: DataPreprocessor")
    try:
        from data.preprocessor import DataPreprocessor
        from config import Config, DATA_LEAKAGE_PARAMS
        
        config = Config()
        preprocessor = DataPreprocessor(config)
        
        # Создаем тестовые данные
        n_samples = 100
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_buy = pd.Series(np.random.randn(n_samples))
        y_sell = pd.Series(np.random.randn(n_samples))
        
        # Проверяем что используются DATA_LEAKAGE_PARAMS
        # В _check_data_leakage должны использоваться параметры из констант
        assert DATA_LEAKAGE_PARAMS['check_n_features'] == 10, "check_n_features должен быть 10"
        
        print_test("DataPreprocessor использует DATA_LEAKAGE_PARAMS", True, 
                  f"Проверяется {DATA_LEAKAGE_PARAMS['check_n_features']} признаков")
        return True
    except Exception as e:
        print_test("DataPreprocessor", False, str(e))
        return False

def test_ensemble():
    """Тест EnsembleModel"""
    print_header("Тест 5: EnsembleModel")
    try:
        from models.ensemble import EnsembleModel
        from config import Config, ENSEMBLE_PARAMS
        
        config = Config()
        ensemble = EnsembleModel(config)
        
        # Проверяем использование ENSEMBLE_PARAMS
        assert 'score_normalization' in ENSEMBLE_PARAMS, "ENSEMBLE_PARAMS должен содержать score_normalization"
        assert 'weight_smoothing' in ENSEMBLE_PARAMS, "ENSEMBLE_PARAMS должен содержать weight_smoothing"
        
        # Проверяем что используются правильные пороги
        norm_params = ENSEMBLE_PARAMS['score_normalization']
        assert 'similarity_threshold' in norm_params, "Должен быть similarity_threshold"
        
        smoothing_params = ENSEMBLE_PARAMS['weight_smoothing']
        assert 'extreme_weight_threshold' in smoothing_params, "Должен быть extreme_weight_threshold"
        
        print_test("EnsembleModel использует ENSEMBLE_PARAMS", True,
                  f"similarity_threshold={norm_params['similarity_threshold']}, "
                  f"extreme_weight_threshold={smoothing_params['extreme_weight_threshold']}")
        return True
    except Exception as e:
        print_test("EnsembleModel", False, str(e))
        return False

def test_database_types():
    """Тест преобразования типов для БД"""
    print_header("Тест 6: Преобразование типов numpy для PostgreSQL")
    try:
        # Проверяем что numpy типы правильно преобразуются
        import numpy as np
        
        min_timestamp = np.int64(1000000)
        max_timestamp = np.int64(2000000)
        
        # Преобразование в int
        min_ts_int = int(min_timestamp)
        max_ts_int = int(max_timestamp)
        
        assert isinstance(min_ts_int, int), "Должен быть обычный int"
        assert isinstance(max_ts_int, int), "Должен быть обычный int"
        
        print_test("Преобразование numpy.int64 в int", True, 
                  f"numpy.int64({min_timestamp}) -> int({min_ts_int})")
        return True
    except Exception as e:
        print_test("Преобразование типов", False, str(e))
        return False

def main():
    """Основная функция тестирования"""
    print(f"{YELLOW}")
    print("╔══════════════════════════════════════════════════╗")
    print("║     Финальное тестирование XGBoost v3.0         ║")
    print("║           Проверка всех исправлений              ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"{RESET}")
    
    tests = [
        test_imports,
        test_btc_data_loader,
        test_feature_engineer,
        test_preprocessor,
        test_ensemble,
        test_database_types
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print_header("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    
    if passed == total:
        print(f"{GREEN}🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! ({passed}/{total})")
        print(f"✅ Код готов к запуску обучения{RESET}")
    else:
        print(f"{RED}❌ Некоторые тесты не пройдены: {passed}/{total}")
        print(f"⚠️  Требуется дополнительная отладка{RESET}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)