#!/usr/bin/env python3
"""
Улучшенный главный скрипт для обучения XGBoost v3.0
Использует упрощенные таргеты для лучшего качества предсказаний
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import gc

from config import Config
from data import DataLoader, DataPreprocessor, FeatureEngineer
from data.target_calculator import TargetCalculator
from data.cacher import CacheManager
from models import XGBoostTrainer, EnsembleModel, OptunaOptimizer, DataBalancer
from utils import LoggingManager, ReportGenerator
from utils.feature_selector import FeatureSelector

logger = logging.getLogger(__name__)


class ImprovedXGBoostPipeline:
    """Улучшенный пайплайн с упрощенными таргетами"""
    
    def __init__(self, config: Config):
        self.config = config
        self.log_manager = LoggingManager(config)
        self.cache_manager = CacheManager(config)
        self.report_generator = ReportGenerator(config)
        self.target_calculator = TargetCalculator(
            lookahead_bars=4,  # 1 час для 15-мин данных
            price_threshold=0.5  # порог 0.5% для threshold_binary
        )
        
    def run(self, target_type: str = "threshold_binary", optimize: bool = True, 
            ensemble_size: int = 5):
        """
        Запуск полного пайплайна обучения
        
        Args:
            target_type: Тип таргета - simple_binary, threshold_binary, 
                        direction_multiclass, simple_regression
            optimize: Использовать Optuna для оптимизации
            ensemble_size: Размер ансамбля моделей
        """
        start_time = time.time()
        
        try:
            # 1. Загрузка данных
            logger.info("=" * 60)
            logger.info("🚀 ЗАПУСК УЛУЧШЕННОГО XGBOOST v3.0")
            logger.info(f"📊 Тип таргета: {target_type}")
            logger.info("=" * 60)
            
            data_loader = DataLoader(self.config)
            df = data_loader.load_data()
            
            logger.info(f"✅ Загружено данных: {len(df)} строк")
            logger.info(f"📈 Символы: {df['symbol'].nunique()}")
            
            # 2. Создание признаков
            logger.info("\n🔧 Создание признаков...")
            feature_engineer = FeatureEngineer(self.config)
            df_features = feature_engineer.create_features(df)
            
            # 3. Расчет упрощенных таргетов
            logger.info(f"\n🎯 Расчет таргетов типа: {target_type}")
            df_with_targets = self.target_calculator.calculate_all_targets(df_features)
            
            # Выбираем нужный таргет
            target_column = f"target_{target_type}"
            if target_column not in df_with_targets.columns:
                raise ValueError(f"Неизвестный тип таргета: {target_type}")
            
            # Для мультикласса и регрессии используем другую логику
            if target_type == "direction_multiclass":
                return self._train_multiclass(df_with_targets, target_column, 
                                            optimize, ensemble_size)
            elif target_type == "simple_regression":
                return self._train_regression(df_with_targets, target_column,
                                            optimize, ensemble_size)
            else:
                # Для бинарной классификации
                return self._train_binary(df_with_targets, target_column,
                                        optimize, ensemble_size)
                
        except Exception as e:
            logger.error(f"❌ Ошибка в пайплайне: {e}", exc_info=True)
            raise
        finally:
            total_time = time.time() - start_time
            logger.info(f"\n⏱️ Общее время выполнения: {total_time/60:.1f} минут")
            
    def _train_binary(self, df: pd.DataFrame, target_column: str,
                     optimize: bool, ensemble_size: int) -> Dict:
        """Обучение для бинарной классификации"""
        
        # Препроцессинг
        preprocessor = DataPreprocessor(self.config)
        X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(
            df, target_column=target_column
        )
        
        # Логируем статистику
        logger.info("\n📊 Статистика данных:")
        logger.info(f"   Train: {len(X_train)} примеров")
        logger.info(f"   Test: {len(X_test)} примеров")
        logger.info(f"   Признаков: {len(feature_names)}")
        logger.info(f"   Положительных в train: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
        logger.info(f"   Положительных в test: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")
        
        # Отбор признаков
        feature_selector = FeatureSelector(
            method=self.config.training.feature_selection_method,
            top_k=100  # Используем топ-100 признаков
        )
        
        X_train_selected, selected_features = feature_selector.select_features(
            X_train, y_train, feature_names
        )
        X_test_selected = X_test[selected_features]
        
        # Балансировка данных
        balancer = DataBalancer(self.config)
        X_train_balanced, y_train_balanced = balancer.balance_data(
            X_train_selected, y_train
        )
        
        # Оптимизация гиперпараметров
        if optimize:
            optimizer = OptunaOptimizer(self.config)
            best_params = optimizer.optimize(
                X_train_balanced, y_train_balanced,
                n_trials=self.config.training.optuna_trials
            )
            logger.info(f"\n🎯 Лучшие параметры: {best_params}")
        else:
            best_params = self.config.model.to_dict()
            
        # Обучение ансамбля
        ensemble = EnsembleModel(
            base_params=best_params,
            ensemble_size=ensemble_size,
            config=self.config
        )
        
        logger.info(f"\n🚀 Обучение ансамбля из {ensemble_size} моделей...")
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Оценка
        metrics = ensemble.evaluate(X_test_selected, y_test)
        
        # Анализ важности признаков
        feature_importance = self._analyze_feature_importance(
            ensemble, selected_features
        )
        
        # Сохранение результатов
        results = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'selected_features': selected_features,
            'best_params': best_params,
            'target_type': target_column,
            'ensemble': ensemble
        }
        
        # Генерация отчета
        self.report_generator.generate_final_report(results, self.log_manager.log_dir)
        
        # Сохранение модели
        model_path = self.log_manager.log_dir / f"model_{target_column}.pkl"
        ensemble.save(model_path)
        logger.info(f"\n💾 Модель сохранена: {model_path}")
        
        return results
        
    def _train_multiclass(self, df: pd.DataFrame, target_column: str,
                         optimize: bool, ensemble_size: int) -> Dict:
        """Обучение для мультиклассовой классификации"""
        logger.info("\n🎯 Мультиклассовая классификация (5 классов)")
        
        # Препроцессинг
        preprocessor = DataPreprocessor(self.config)
        X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(
            df, target_column=target_column
        )
        
        # Статистика по классам
        logger.info("\n📊 Распределение классов:")
        for class_id in range(5):
            train_count = (y_train == class_id).sum()
            test_count = (y_test == class_id).sum()
            logger.info(f"   Класс {class_id}: Train {train_count} ({train_count/len(y_train)*100:.1f}%), "
                       f"Test {test_count} ({test_count/len(y_test)*100:.1f}%)")
        
        # Отбор признаков
        feature_selector = FeatureSelector(
            method=self.config.training.feature_selection_method,
            top_k=100
        )
        
        X_train_selected, selected_features = feature_selector.select_features(
            X_train, y_train, feature_names
        )
        X_test_selected = X_test[selected_features]
        
        # Для мультикласса изменяем параметры модели
        multiclass_params = self.config.model.to_dict()
        multiclass_params['objective'] = 'multi:softprob'
        multiclass_params['num_class'] = 5
        multiclass_params['eval_metric'] = 'mlogloss'
        
        # Обучение
        trainer = XGBoostTrainer(self.config)
        trainer.params = multiclass_params
        
        model = trainer.train(
            X_train_selected, y_train,
            X_test_selected, y_test
        )
        
        # Оценка
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred = model.predict(X_test_selected)
        
        logger.info("\n📊 Отчет по классификации:")
        logger.info(classification_report(y_test, y_pred))
        
        results = {
            'model': model,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'selected_features': selected_features,
            'target_type': target_column
        }
        
        return results
        
    def _train_regression(self, df: pd.DataFrame, target_column: str,
                         optimize: bool, ensemble_size: int) -> Dict:
        """Обучение для регрессии"""
        logger.info("\n📈 Регрессия (предсказание процентного изменения)")
        
        # Препроцессинг
        preprocessor = DataPreprocessor(self.config)
        X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(
            df, target_column=target_column
        )
        
        # Статистика таргетов
        logger.info("\n📊 Статистика таргетов:")
        logger.info(f"   Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        logger.info(f"   Test: mean={y_test.mean():.4f}, std={y_test.std():.4f}")
        
        # Отбор признаков
        feature_selector = FeatureSelector(
            method=self.config.training.feature_selection_method,
            top_k=100
        )
        
        X_train_selected, selected_features = feature_selector.select_features(
            X_train, y_train, feature_names
        )
        X_test_selected = X_test[selected_features]
        
        # Для регрессии изменяем параметры
        regression_params = self.config.model.to_dict()
        regression_params['objective'] = 'reg:squarederror'
        regression_params['eval_metric'] = 'rmse'
        
        # Обучение
        trainer = XGBoostTrainer(self.config)
        trainer.params = regression_params
        
        model = trainer.train(
            X_train_selected, y_train,
            X_test_selected, y_test
        )
        
        # Оценка
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = model.predict(X_test_selected)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info("\n📊 Метрики регрессии:")
        logger.info(f"   MAE: {metrics['mae']:.6f}")
        logger.info(f"   RMSE: {metrics['rmse']:.6f}")
        logger.info(f"   R²: {metrics['r2']:.4f}")
        
        results = {
            'model': model,
            'metrics': metrics,
            'selected_features': selected_features,
            'target_type': target_column,
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred
            }
        }
        
        return results
        
    def _analyze_feature_importance(self, ensemble: EnsembleModel, 
                                   feature_names: List[str]) -> Dict:
        """Анализ важности признаков с категоризацией"""
        
        # Получаем усредненную важность
        importances = []
        for model in ensemble.models:
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
                
        avg_importance = np.mean(importances, axis=0)
        
        # Создаем DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        # Категоризация
        def get_category(feature):
            feature_lower = feature.lower()
            
            if any(p in feature_lower for p in ['rsi', 'macd', 'bb_', 'adx', 'atr', 
                                                'stoch', 'williams', 'mfi', 'cci']):
                return 'Технические индикаторы'
            elif 'btc_' in feature_lower:
                return 'BTC корреляции'
            elif any(p in feature_lower for p in ['hour', 'dow', 'day', 'week']):
                return 'Временные признаки'
            elif 'volume' in feature_lower:
                return 'Объемные признаки'
            else:
                return 'Другие'
                
        importance_df['category'] = importance_df['feature'].apply(get_category)
        
        # Статистика по категориям
        category_stats = importance_df.groupby('category')['importance'].agg([
            'sum', 'mean', 'count'
        ]).sort_values('sum', ascending=False)
        
        logger.info("\n📊 Важность признаков по категориям:")
        for cat, row in category_stats.iterrows():
            logger.info(f"   {cat}: {row['sum']*100:.1f}% "
                       f"(среднее: {row['mean']*100:.2f}%, количество: {row['count']})")
            
        logger.info("\n🔝 Топ-10 важных признаков:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']*100:.2f}%")
            
        return {
            'importance_df': importance_df,
            'category_stats': category_stats
        }


def main():
    """Главная функция с аргументами командной строки"""
    parser = argparse.ArgumentParser(description='Улучшенный XGBoost v3.0')
    
    parser.add_argument('--target-type', type=str, default='threshold_binary',
                       choices=['simple_binary', 'threshold_binary', 
                               'direction_multiclass', 'simple_regression'],
                       help='Тип таргета для обучения')
    parser.add_argument('--optimize', action='store_true',
                       help='Использовать Optuna для оптимизации')
    parser.add_argument('--ensemble-size', type=int, default=5,
                       help='Размер ансамбля моделей')
    parser.add_argument('--test-mode', action='store_true',
                       help='Тестовый режим (только BTC/ETH)')
    parser.add_argument('--gpu', action='store_true',
                       help='Использовать GPU')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(str(config_path))
    else:
        config = Config()
        
    # Применение аргументов
    config.training.test_mode = args.test_mode
    
    if args.gpu:
        config.model.tree_method = 'gpu_hist'
        config.model.predictor = 'gpu_predictor'
        config.model.gpu_id = 0
        
    # Запуск пайплайна
    pipeline = ImprovedXGBoostPipeline(config)
    results = pipeline.run(
        target_type=args.target_type,
        optimize=args.optimize,
        ensemble_size=args.ensemble_size
    )
    
    logger.info("\n✅ Обучение завершено успешно!")
    

if __name__ == "__main__":
    main()