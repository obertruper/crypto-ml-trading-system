#!/usr/bin/env python3
"""
Главный скрипт для обучения XGBoost v3.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

from config import Config
from data import DataLoader, DataPreprocessor, FeatureEngineer
from data.cacher import CacheManager
from models import XGBoostTrainer, EnsembleModel, OptunaOptimizer, DataBalancer
from utils import LoggingManager, ReportGenerator
from utils.feature_selector import FeatureSelector
from utils.feature_importance_validator import FeatureImportanceValidator

logger = logging.getLogger(__name__)


def analyze_feature_importance(models: Dict, feature_names: List[str]) -> Dict:
    """Анализ важности признаков по категориям"""
    # Категоризация признаков
    def get_category(feature):
        feature_lower = feature.lower()
        
        # Технические индикаторы
        technical_patterns = [
            'rsi', 'macd', 'bb_', 'bollinger', 'adx', 'atr', 'stoch', 'williams', 
            'mfi', 'cci', 'cmf', 'obv', 'ema', 'sma', 'vwap', 'sar', 'ich_',
            'aroon', 'kc_', 'dc_', 'volume_ratio', 'price_momentum', 'volatility',
            'hl_spread', 'close_ratio', 'upper_shadow', 'lower_shadow', 'body_size',
            'open_ratio', 'high_ratio', 'low_ratio', 'log_return', 'log_volume',
            'price_to_', 'volume_position', 'cumulative_volume', 'higher_high',
            'lower_low', 'consecutive_', 'inside_bar', 'pin_bar', 'spread_approximation',
            'price_efficiency', 'gk_volatility', 'position_in_', 'trend_'
        ]
        for pattern in technical_patterns:
            if pattern in feature_lower:
                return 'Технические индикаторы'
        
        # BTC корреляция
        btc_patterns = ['btc_', 'bitcoin', 'relative_strength_btc']
        for pattern in btc_patterns:
            if pattern in feature_lower:
                return 'BTC корреляции'
        
        # Временные признаки
        time_patterns = ['hour', 'dow', 'day', 'week', 'month', 'time', 'weekend']
        for pattern in time_patterns:
            if pattern in feature_lower:
                return 'Временные признаки'
        
        # Рыночные режимы
        if 'market_regime' in feature_lower:
            return 'Рыночные режимы'
            
        # Символы
        symbol_patterns = ['is_btc', 'is_eth', 'is_bnb', 'is_xrp', 'is_ada', 'is_doge', 
                          'is_sol', 'is_dot', 'is_matic', 'is_shib']
        for pattern in symbol_patterns:
            if pattern in feature_lower:
                return 'Символы'
        
        return 'Другие'
    
    # Собираем важности со всех моделей
    all_importances = {}
    for direction in ['buy', 'sell']:
        ensemble = models[direction]['ensemble']
        for i, model in enumerate(ensemble.models):
            if hasattr(model, 'feature_importances_'):
                for feat, imp in zip(feature_names, model.feature_importances_):
                    if feat not in all_importances:
                        all_importances[feat] = []
                    all_importances[feat].append(imp)
    
    # Усредняем важности
    avg_importances = {feat: np.mean(imps) for feat, imps in all_importances.items()}
    
    # Группируем по категориям
    category_analysis = {}
    for feat, imp in avg_importances.items():
        cat = get_category(feat)
        if cat not in category_analysis:
            category_analysis[cat] = {
                'features': [],
                'importances': [],
                'count': 0,
                'total_importance': 0
            }
        category_analysis[cat]['features'].append(feat)
        category_analysis[cat]['importances'].append(imp)
        category_analysis[cat]['count'] += 1
        category_analysis[cat]['total_importance'] += imp
    
    # Форматируем результаты
    results = {}
    total_features = sum(cat['count'] for cat in category_analysis.values())
    
    for cat, data in category_analysis.items():
        # Сортируем признаки по важности
        sorted_features = sorted(zip(data['features'], data['importances']), 
                               key=lambda x: x[1], reverse=True)
        
        results[cat] = {
            'count': data['count'],
            'percentage': (data['count'] / total_features * 100) if total_features > 0 else 0,
            'total_importance': data['total_importance'],
            'top_features': [f[0] for f in sorted_features[:5]]
        }
    
    return results


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="XGBoost v3.0 Training")
    
    parser.add_argument('--config', type=str, default=None,
                       help='Путь к файлу конфигурации (YAML)')
    
    parser.add_argument('--task', type=str, default='classification_binary',
                       choices=['classification_binary', 'classification_multi', 'regression'],
                       help='Тип задачи')
    
    parser.add_argument('--test-mode', action='store_true',
                       help='Режим тестирования (только 2 символа)')
    
    parser.add_argument('--no-cache', action='store_true',
                       help='Не использовать кэш')
    
    parser.add_argument('--optimize', action='store_true',
                       help='Запустить оптимизацию гиперпараметров')
    
    parser.add_argument('--ensemble-size', type=int, default=None,
                       help='Количество моделей в ансамбле')
    
    parser.add_argument('--gpu', action='store_true',
                       help='Использовать GPU для обучения')
    
    return parser.parse_args()




def main():
    """Основная функция обучения"""
    # Парсим аргументы
    args = parse_args()
    
    # Загружаем конфигурацию
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
        
    # Обновляем конфигурацию из аргументов
    config.training.task_type = args.task
    config.training.test_mode = args.test_mode
    config.training.use_cache = not args.no_cache
    
    if args.ensemble_size:
        config.training.ensemble_size = args.ensemble_size
        
    # Настройка GPU
    if args.gpu:
        logger.info("🚀 GPU режим активирован")
        # Проверяем доступность GPU
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ GPU обнаружен и доступен")
                config.model.tree_method = "gpu_hist"
                config.model.predictor = "gpu_predictor"
                # Добавляем gpu_id в параметры
                config.model.gpu_id = 0
            else:
                logger.warning("⚠️ GPU недоступен, используется CPU")
                config.model.tree_method = "hist"
                config.model.predictor = "cpu_predictor"
        except:
            logger.warning("⚠️ nvidia-smi не найден, используется CPU")
            config.model.tree_method = "hist"
            config.model.predictor = "cpu_predictor"
    else:
        logger.info("💻 CPU режим")
        config.model.tree_method = "hist"
        config.model.predictor = "cpu_predictor"
        
    # ВАЖНО: Принудительно используем иерархический отбор для продакшена
    if not config.training.test_mode:
        logger.warning("⚠️ ПРОДАКШН РЕЖИМ: Принудительно используем иерархический отбор признаков!")
        config.training.feature_selection_method = "hierarchical"
        
    # Валидируем конфигурацию
    config.validate()
    
    # Настройка логирования
    log_dir = config.get_log_dir()
    logging_manager = LoggingManager(log_dir)
    logging_manager.setup_logging()
    
    logger.info("""
    ╔══════════════════════════════════════════╗
    ║        XGBoost v3.0 - ML Trading         ║
    ╚══════════════════════════════════════════╝
    """)
    
    logger.info(config)
    
    # Сохраняем конфигурацию
    config.save(log_dir / "config.yaml")
    
    try:
        # 1. Загрузка данных
        logger.info("\n" + "="*60)
        logger.info("📥 ШАГ 1: ЗАГРУЗКА ДАННЫХ")
        logger.info("="*60)
        
        cacher = CacheManager(config)
        data_loader = DataLoader(config)
        
        # Пробуем загрузить из кэша
        df = None
        if config.training.use_cache:
            df = cacher.load_from_cache()
            
        if df is None:
            data_loader.connect()
            df = data_loader.load_data()
            data_loader.validate_data(df)
            data_loader.disconnect()
            
            # Сохраняем в кэш
            if config.training.use_cache:
                cacher.save_to_cache(df)
                
        # 2. Предобработка данных
        logger.info("\n" + "="*60)
        logger.info("🔧 ШАГ 2: ПРЕДОБРАБОТКА ДАННЫХ")
        logger.info("="*60)
        
        preprocessor = DataPreprocessor(config)
        
        # 3. Инженерия признаков (до разделения на X и y)
        logger.info("\n" + "="*60)
        logger.info("🔬 ШАГ 3: ИНЖЕНЕРИЯ ПРИЗНАКОВ")
        logger.info("="*60)
        
        feature_engineer = FeatureEngineer(config)
        df = feature_engineer.create_features(df)
        
        # Теперь извлекаем признаки и целевые переменные
        X, y_buy, y_sell = preprocessor.preprocess(df)
        
        # ВАЖНО: Проверяем наличие expected returns
        logger.info("\n📊 ПРОВЕРКА EXPECTED RETURNS:")
        logger.info(f"   Buy Expected Return - min: {y_buy.min():.2f}%, max: {y_buy.max():.2f}%, mean: {y_buy.mean():.2f}%")
        logger.info(f"   Sell Expected Return - min: {y_sell.min():.2f}%, max: {y_sell.max():.2f}%, mean: {y_sell.mean():.2f}%")
        logger.info(f"   Buy > 0%: {(y_buy > 0).mean()*100:.1f}%, Buy > 0.5%: {(y_buy > 0.5).mean()*100:.1f}%")
        logger.info(f"   Sell > 0%: {(y_sell > 0).mean()*100:.1f}%, Sell > 0.5%: {(y_sell > 0.5).mean()*100:.1f}%")
        
        if y_buy.isna().any() or y_sell.isna().any():
            logger.error("❌ Обнаружены NaN в expected returns!")
            logger.error(f"   Buy NaN: {y_buy.isna().sum()}, Sell NaN: {y_sell.isna().sum()}")
            raise ValueError("Expected returns содержат NaN значения!")
        
        # 4. Разделение данных
        logger.info("\n" + "="*60)
        logger.info("📊 ШАГ 4: РАЗДЕЛЕНИЕ ДАННЫХ")
        logger.info("="*60)
        
        data_splits = preprocessor.split_data(X, y_buy, y_sell)
        
        # Преобразование в метки для классификации
        if config.training.task_type != "regression":
            for direction in ['buy', 'sell']:
                y_train_binary, y_test_binary = preprocessor.transform_to_classification_labels(
                    data_splits[direction]['y_train'],
                    data_splits[direction]['y_test']
                )
                data_splits[direction]['y_train'] = y_train_binary
                data_splits[direction]['y_test'] = y_test_binary
                
        # 5. Нормализация признаков
        logger.info("\n" + "="*60)
        logger.info("📏 ШАГ 5: НОРМАЛИЗАЦИЯ ПРИЗНАКОВ")
        logger.info("="*60)
        
        for direction in ['buy', 'sell']:
            X_train_norm, X_test_norm = preprocessor.normalize_features(
                data_splits[direction]['X_train'],
                data_splits[direction]['X_test']
            )
            data_splits[direction]['X_train'] = X_train_norm
            data_splits[direction]['X_test'] = X_test_norm
        
        # 5.5. Отбор признаков
        logger.info("\n" + "="*60)
        logger.info("🎯 ШАГ 5.5: ОТБОР ПРИЗНАКОВ")
        logger.info("="*60)
        
        # Используем feature selection для выбора лучших признаков
        # ВСЕГДА используем иерархический метод для правильного распределения 60/20/10/10
        if config.training.test_mode:
            # Тест режим - используем меньше признаков для скорости
            logger.info("🎯 Тест режим: используем иерархический отбор признаков (80 признаков)")
            feature_selector = FeatureSelector(method="hierarchical", top_k=80)
            
            # Объединяем данные для отбора признаков
            X_all_train = pd.concat([data_splits['buy']['X_train'], data_splits['sell']['X_train']])
            y_all_train = pd.concat([data_splits['buy']['y_train'], data_splits['sell']['y_train']])
            
            # Отбираем признаки
            _, selected_features = feature_selector.select_features(X_all_train, y_all_train)
        else:
            # Продакшен режим - используем больше признаков для лучшего качества
            logger.info("🎯 ПРОДАКШЕН режим: используем иерархический отбор признаков (120 признаков)")
            logger.info("   Расширенный набор признаков для максимального качества")
            feature_selector = FeatureSelector(method="hierarchical", top_k=120)
            
            # Объединяем данные для отбора признаков
            X_all_train = pd.concat([data_splits['buy']['X_train'], data_splits['sell']['X_train']])
            y_all_train = pd.concat([data_splits['buy']['y_train'], data_splits['sell']['y_train']])
            
            # Отбираем признаки
            _, selected_features = feature_selector.select_features(X_all_train, y_all_train)
        
        # Применяем отбор ко всем данным
        for direction in ['buy', 'sell']:
            data_splits[direction]['X_train'] = data_splits[direction]['X_train'][selected_features]
            data_splits[direction]['X_test'] = data_splits[direction]['X_test'][selected_features]
        
        # Обновляем список признаков
        preprocessor.feature_names = selected_features
        
        # НОВОЕ: Детальное логирование отобранных признаков
        logger.info(f"\n📋 ДЕТАЛИЗАЦИЯ ОТОБРАННЫХ {len(selected_features)} ПРИЗНАКОВ:")
        
        # Группируем по категориям для детального логирования
        try:
            from config.feature_mapping import get_feature_category
            category_features = {}
            for feature in selected_features:
                cat = get_feature_category(feature)
                if cat not in category_features:
                    category_features[cat] = []
                category_features[cat].append(feature)
            
            for category, features in category_features.items():
                logger.info(f"   {category} ({len(features)}): {', '.join(features[:5])}"
                          + (f" ...и еще {len(features)-5}" if len(features) > 5 else ""))
        except ImportError:
            logger.info(f"   Всего признаков: {len(selected_features)}")
            logger.info(f"   Первые 10: {', '.join(selected_features[:10])}")
        
        # НОВОЕ: Валидация распределения признаков по категориям
        logger.info("\n" + "="*60)
        logger.info("🔍 ВАЛИДАЦИЯ РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ")
        logger.info("="*60)
        
        # Импортируем функции для категоризации
        try:
            from config.feature_mapping import get_feature_category, get_category_targets
            
            # Подсчитываем распределение
            category_counts = {}
            for feature in selected_features:
                cat = get_feature_category(feature)
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Получаем целевые проценты
            targets = get_category_targets()
            
            # Проверяем соответствие
            all_ok = True
            for category, target_percent in targets.items():
                count = category_counts.get(category, 0)
                actual_percent = count / len(selected_features) * 100 if selected_features else 0
                deviation = abs(actual_percent - target_percent)
                
                if deviation > 10:  # Допустимое отклонение 10%
                    status = "❌ КРИТИЧЕСКОЕ ОТКЛОНЕНИЕ!"
                    all_ok = False
                elif deviation > 5:
                    status = "⚠️ Превышено отклонение"
                else:
                    status = "✅"
                    
                logger.info(f"   {category}: {count} ({actual_percent:.1f}%) {status} [цель: {target_percent}%]")
            
            if not all_ok:
                logger.warning("\n⚠️ ВНИМАНИЕ: Распределение признаков значительно отклоняется от целевого!")
                logger.warning("   Это может привести к переобучению на временных паттернах!")
                
        except ImportError:
            logger.warning("⚠️ Не удалось проверить распределение признаков (feature_mapping.py не найден)")
            
        # 6. Балансировка данных
        logger.info("\n" + "="*60)
        logger.info("⚖️ ШАГ 6: БАЛАНСИРОВКА ДАННЫХ")
        logger.info("="*60)
        
        balancer = DataBalancer(config)
        
        for direction in ['buy', 'sell']:
            logger.info(f"\n🎯 Балансировка для {direction.upper()}")
            
            X_balanced, y_balanced = balancer.balance_data(
                data_splits[direction]['X_train'],
                data_splits[direction]['y_train'],
                is_classification=(config.training.task_type != "regression")
            )
            
            data_splits[direction]['X_train'] = X_balanced
            data_splits[direction]['y_train'] = y_balanced
            
        # 7. Обучение моделей
        logger.info("\n" + "="*60)
        logger.info("🚀 ШАГ 7: ОБУЧЕНИЕ МОДЕЛЕЙ")
        logger.info("="*60)
        
        models = {}
        
        for direction in ['buy', 'sell']:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 ОБУЧЕНИЕ МОДЕЛИ: {direction.upper()}")
            logger.info(f"{'='*60}")
            
            # Оптимизация гиперпараметров
            # В тест режиме всегда используем оптимизацию для лучших результатов
            if args.optimize or config.training.test_mode:
                logger.info("\n🔍 Запуск оптимизации гиперпараметров...")
                optimizer = OptunaOptimizer(config)
                
                # Для тест режима используем больше попыток для лучшей оптимизации
                # В продакшн режиме используем еще больше попыток
                if config.training.test_mode:
                    n_trials = 50  # Увеличено с 20 до 50
                else:
                    # Продакшен - еще больше попыток
                    n_trials = 200 if config.training.ensemble_size > 5 else config.training.optuna_trials
                
                best_params = optimizer.optimize(
                    data_splits[direction]['X_train'],
                    data_splits[direction]['y_train'],
                    n_trials=n_trials,
                    model_type=direction
                )
                
                # Обновляем конфигурацию
                for key, value in best_params.items():
                    if hasattr(config.model, key):
                        setattr(config.model, key, value)
                        
                # Сохраняем график оптимизации
                if config.training.save_plots:
                    plot_path = log_dir / f"{direction}_optuna_history.png"
                    optimizer.plot_optimization_history(str(plot_path))
                        
            # Обучение ансамбля
            ensemble = EnsembleModel(config)
            ensemble_models = ensemble.train_ensemble(
                data_splits[direction]['X_train'],
                data_splits[direction]['y_train'],
                data_splits[direction]['X_test'],
                data_splits[direction]['y_test']
            )
            
            # Оценка на тестовых данных
            test_metrics = ensemble.evaluate(
                data_splits[direction]['X_test'],
                data_splits[direction]['y_test'],
                f"Test ({direction})"
            )
            
            # Сохранение моделей
            if config.training.save_models:
                model_dir = log_dir / f"{direction}_models"
                model_dir.mkdir(exist_ok=True)
                ensemble.save_ensemble(str(model_dir))
                
            models[direction] = {
                'ensemble': ensemble,
                'test_metrics': test_metrics
            }
        
        # 7.5. НОВОЕ: Валидация важности признаков
        logger.info("\n" + "="*60)
        logger.info("🔍 ШАГ 7.5: ВАЛИДАЦИЯ ВАЖНОСТИ ПРИЗНАКОВ")
        logger.info("="*60)
        
        validator = FeatureImportanceValidator(max_temporal_importance=3.0)
        validation_results = validator.validate_ensemble_importance(models, selected_features)
        
        # Проверяем результаты валидации
        if not validation_results['ensemble_validation']['valid']:
            severity = validation_results['ensemble_validation']['severity']
            if severity == 'critical':
                logger.error("❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: Модель переобучается на временных паттернах!")
                logger.error("   Рекомендуется прервать использование этой модели в продакшене!")
            else:
                logger.warning("⚠️ Обнаружены проблемы с важностью признаков")
            
            # Выводим рекомендации
            recommendations = validator.get_recommendations()
            logger.info("\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec}")
        else:
            logger.info("✅ Валидация пройдена: модель не переобучается на временных паттернах")
            
        # 8. Генерация отчета
        logger.info("\n" + "="*60)
        logger.info("📝 ШАГ 8: ГЕНЕРАЦИЯ ОТЧЕТА")
        logger.info("="*60)
        
        report_generator = ReportGenerator(config, log_dir)
        
        # Анализ важности признаков по категориям
        feature_importance_analysis = analyze_feature_importance(
            models, preprocessor.feature_names
        )
        
        # Добавляем результаты валидации в анализ
        feature_importance_analysis['validation_results'] = validation_results
        
        # Формируем результаты для отчета
        results = {
            'buy': models['buy']['test_metrics'],
            'sell': models['sell']['test_metrics'],
            'config': config,
            'n_features': X.shape[1],
            'n_samples': len(X),
            'feature_names': preprocessor.feature_names[:20],  # Топ-20 признаков
            'feature_importance_analysis': feature_importance_analysis
        }
        
        report_generator.generate_report(results)
        
        # Визуализация важности признаков по категориям
        if config.training.save_plots:
            try:
                from utils.visualization import plot_feature_importance_by_category
                plot_path = log_dir / "feature_importance_by_category.png"
                plot_feature_importance_by_category(feature_importance_analysis, plot_path)
                logger.info(f"📊 График важности по категориям: {plot_path}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось создать график важности: {e}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info(f"📁 Результаты сохранены в: {log_dir}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ ОШИБКА: {e}", exc_info=True)
        raise
        

if __name__ == "__main__":
    main()