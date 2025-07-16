#!/usr/bin/env python3
"""
XGBoost v3.0 - Главный модуль запуска
Чистая модульная архитектура для ML криптотрейдинга
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к модулю в sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Импортируем наши модули
from config import Config
from data import DataLoader, DataPreprocessor, FeatureEngineer
from models import XGBoostTrainer, EnsembleModel, OptunaOptimizer
from utils import Visualizer, CacheManager

# Настройка логирования
def setup_logging(log_dir: Path):
    """Настройка системы логирования"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Главная функция запуска"""
    # Парсинг аргументов
    parser = argparse.ArgumentParser(description='XGBoost v3.0 для криптотрейдинга')
    
    # Основные параметры
    parser.add_argument('--task', type=str, default='classification_binary',
                       choices=['classification_binary', 'classification_multi', 'regression'],
                       help='Тип задачи')
    parser.add_argument('--config', type=str, default=None,
                       help='Путь к конфигурационному файлу')
    
    # Режимы работы
    parser.add_argument('--test-mode', action='store_true',
                       help='Тестовый режим (2 символа)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Не использовать кэш')
    
    # Параметры обучения
    parser.add_argument('--ensemble-size', type=int, default=2,
                       help='Количество моделей в ансамбле')
    parser.add_argument('--balance-method', type=str, default='smote',
                       choices=['none', 'smote', 'adasyn', 'class_weight'],
                       help='Метод балансировки классов')
    parser.add_argument('--optuna-trials', type=int, default=50,
                       help='Количество попыток Optuna')
    
    # Дополнительные параметры
    parser.add_argument('--save-models', action='store_true', default=True,
                       help='Сохранить обученные модели')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Создать визуализации')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
        
    # Обновляем конфигурацию из аргументов
    config.training.task_type = args.task
    config.training.test_mode = args.test_mode
    config.training.use_cache = not args.no_cache
    config.training.ensemble_size = args.ensemble_size
    config.training.balance_method = args.balance_method
    config.training.optuna_trials = args.optuna_trials
    
    # Валидация конфигурации
    config.validate()
    
    # Создание директории для логов
    log_dir = config.get_log_dir()
    logger = setup_logging(log_dir)
    
    # Сохраняем конфигурацию
    config.save(log_dir / 'config.yaml')
    
    logger.info("="*80)
    logger.info("🚀 XGBoost v3.0 - Начало работы")
    logger.info("="*80)
    logger.info(str(config))
    
    try:
        # 1. Инициализация компонентов
        logger.info("\n📦 Инициализация компонентов...")
        
        data_loader = DataLoader(config)
        preprocessor = DataPreprocessor(config)
        feature_engineer = FeatureEngineer(config)
        cache_manager = CacheManager(cache_dir=".cache/xgboost_v3")
        visualizer = Visualizer(save_dir=log_dir / "plots")
        
        # 2. Загрузка данных
        logger.info("\n📊 Загрузка данных...")
        
        # Проверяем кэш
        cache_key = f"data_{config.training.test_mode}_{config.training.task_type}"
        df = None
        
        if config.training.use_cache:
            df = cache_manager.load_dataframe(cache_key)
            
        if df is None:
            # Загружаем из БД
            data_loader.connect()
            df = data_loader.load_data()
            
            # Валидация данных
            if not data_loader.validate_data(df):
                raise ValueError("Данные не прошли валидацию")
                
            # Сохраняем в кэш
            if config.training.use_cache:
                cache_manager.cache_dataframe(df, cache_key)
                
            data_loader.disconnect()
        
        # 3. Feature Engineering
        logger.info("\n🔧 Создание признаков...")
        df = feature_engineer.create_features(df)
        
        # 4. Предобработка данных
        logger.info("\n📐 Предобработка данных...")
        X, y_buy, y_sell = preprocessor.preprocess(df)
        
        # Преобразование для классификации если нужно
        if config.training.task_type != "regression":
            y_buy, y_sell = preprocessor.transform_to_classification_labels(y_buy, y_sell)
        
        # 5. Разделение данных
        data_splits = preprocessor.split_data(X, y_buy, y_sell)
        
        # Результаты будут храниться здесь
        results = {
            'buy': {},
            'sell': {}
        }
        
        # 6. Обучение моделей для Buy и Sell
        for target_type in ['buy', 'sell']:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 Обучение моделей для {target_type.upper()}")
            logger.info(f"{'='*60}")
            
            # Получаем данные для текущего типа
            X_train = data_splits[target_type]['X_train']
            X_test = data_splits[target_type]['X_test']
            y_train = data_splits[target_type]['y_train']
            y_test = data_splits[target_type]['y_test']
            
            # Нормализация
            X_train_scaled, X_test_scaled = preprocessor.normalize_features(X_train, X_test)
            
            # 7. Оптимизация гиперпараметров
            if config.training.optuna_trials > 0:
                logger.info("\n🔍 Оптимизация гиперпараметров...")
                
                optimizer = OptunaOptimizer(config)
                best_params = optimizer.optimize(
                    X_train_scaled, y_train,
                    n_trials=config.training.optuna_trials,
                    model_type=target_type
                )
                
                # Визуализация оптимизации
                if config.training.save_plots:
                    optimizer.plot_optimization_history(
                        save_path=log_dir / f"plots/{target_type}_optuna_history.png"
                    )
                    
                # Обновляем параметры модели
                model_params = config.model.to_dict()
                model_params.update(best_params)
            else:
                model_params = None
                
            # 8. Обучение ансамбля
            if config.training.ensemble_size > 1:
                logger.info(f"\n🎲 Обучение ансамбля из {config.training.ensemble_size} моделей...")
                
                ensemble = EnsembleModel(config)
                models = ensemble.train_ensemble(
                    X_train_scaled, y_train,
                    X_test_scaled, y_test,
                    n_models=config.training.ensemble_size
                )
                
                # Оценка ансамбля
                test_metrics = ensemble.evaluate(X_test_scaled, y_test, "Test")
                
                # Сохранение ансамбля
                if args.save_models:
                    ensemble.save_ensemble(log_dir / "models" / target_type)
                    
                results[target_type]['ensemble'] = ensemble
                results[target_type]['metrics'] = test_metrics
                
            else:
                # Обучение одиночной модели
                logger.info("\n🎯 Обучение одиночной модели...")
                
                trainer = XGBoostTrainer(config, model_name=f"{target_type}_model")
                model = trainer.train(
                    X_train_scaled, y_train,
                    X_test_scaled, y_test,
                    model_params=model_params
                )
                
                # Оценка модели
                test_metrics = trainer.evaluate(X_test_scaled, y_test, "Test")
                
                # Сохранение модели
                if args.save_models:
                    trainer.save_model(log_dir / "models" / target_type)
                    
                results[target_type]['model'] = trainer
                results[target_type]['metrics'] = test_metrics
                
            # 9. Визуализация результатов
            if args.visualize:
                logger.info("\n📊 Создание визуализаций...")
                
                # Feature importance
                if hasattr(results[target_type].get('model', results[target_type].get('ensemble')), 'get_feature_importance'):
                    feature_importance = results[target_type]['model'].get_feature_importance()
                    if not feature_importance.empty:
                        visualizer.plot_feature_importance(
                            feature_importance,
                            model_name=f"{target_type}_model"
                        )
                
                # Предсказания для визуализации
                if 'ensemble' in results[target_type]:
                    y_pred_proba = results[target_type]['ensemble'].predict(X_test_scaled, return_proba=True)
                else:
                    y_pred_proba = results[target_type]['model'].predict(X_test_scaled, return_proba=True)
                
                # ROC кривая для классификации
                if config.training.task_type != "regression":
                    visualizer.plot_roc_curve(y_test, y_pred_proba, model_name=f"{target_type}_model")
                    
                    # Confusion matrix
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    visualizer.plot_confusion_matrix(y_test, y_pred, model_name=f"{target_type}_model")
                
                # Распределение предсказаний
                visualizer.plot_prediction_distribution(
                    y_test, y_pred_proba,
                    model_name=f"{target_type}_model",
                    task_type=config.training.task_type
                )
        
        # 10. Итоговый отчет
        logger.info("\n" + "="*80)
        logger.info("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        logger.info("="*80)
        
        for target_type in ['buy', 'sell']:
            logger.info(f"\n{target_type.upper()} модель:")
            metrics = results[target_type]['metrics']
            
            if config.training.task_type == "regression":
                logger.info(f"  MAE: {metrics.get('mae', 0):.4f}")
                logger.info(f"  RMSE: {metrics.get('rmse', 0):.4f}")
                logger.info(f"  R²: {metrics.get('r2', 0):.4f}")
                logger.info(f"  Direction Accuracy: {metrics.get('direction_accuracy', 0)*100:.1f}%")
            else:
                logger.info(f"  Accuracy: {metrics.get('accuracy', 0)*100:.1f}%")
                logger.info(f"  Precision: {metrics.get('precision', 0)*100:.1f}%")
                logger.info(f"  Recall: {metrics.get('recall', 0)*100:.1f}%")
                logger.info(f"  F1-Score: {metrics.get('f1', 0):.3f}")
                logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
        
        logger.info("\n✅ Обучение успешно завершено!")
        logger.info(f"📁 Результаты сохранены в: {log_dir}")
        
        # Информация о кэше
        cache_info = cache_manager.get_cache_info()
        logger.info(f"\n💾 Кэш: {cache_info['total_items']} файлов, {cache_info['total_size_mb']:.1f} MB")
        
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {e}", exc_info=True)
        raise
        
    finally:
        # Очистка
        if 'data_loader' in locals() and data_loader.connection:
            data_loader.disconnect()


if __name__ == "__main__":
    main()