#!/usr/bin/env python3
"""
Главный pipeline для Transformer v3
Полностью модульная архитектура по примеру xgboost_v3
"""

import argparse
import logging
import time
import warnings
import numpy as np
import tensorflow as tf
from pathlib import Path

# Настройка окружения
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# Настройка GPU для эффективного использования памяти
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Настроено {len(gpus)} GPU с динамическим выделением памяти")
    except RuntimeError as e:
        print(f"❌ Ошибка настройки GPU: {e}")

# Импорты из модулей
from config import Config
from data import DataLoader, DataPreprocessor, SequenceCreator, CacheManager
from models import TFTTrainer, TFTEnsemble
from utils import LoggingManager, ReportGenerator

logger = logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Transformer v3.0 Training")
    
    parser.add_argument('--config', type=str, default=None,
                       help='Путь к конфигурационному файлу')
    
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification_binary'],
                       help='Тип задачи')
    
    parser.add_argument('--test-mode', action='store_true',
                       help='Режим тестирования (быстрое обучение)')
    
    parser.add_argument('--test-symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Символы для тестового режима')
    
    parser.add_argument('--no-cache', action='store_true',
                       help='Не использовать кеш')
    
    parser.add_argument('--ensemble-size', type=int, default=3,
                       help='Количество моделей в ансамбле')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Количество эпох обучения')
    
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Размер батча')
    
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
    
    if args.test_mode:
        logger.info("🧪 Активирован тестовый режим")
        config.training.epochs = 20           # Увеличили для лучшего обучения
        config.training.batch_size = 64       # Увеличенный batch_size из config
        config.model.sequence_length = 100    # Полная длина последовательности
        config.training.test_mode = True      # Флаг тестового режима
        
    if args.epochs:
        config.training.epochs = args.epochs
        
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    # Валидируем конфигурацию
    config.validate()
    
    # Настройка логирования
    log_dir = config.get_log_dir()
    logging_manager = LoggingManager(log_dir)
    logging_manager.setup_logging()
    
    logger.info("""
    ╔══════════════════════════════════════════╗
    ║      Transformer v3.0 - ML Trading       ║
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
        
        # Пробуем загрузить из кеша
        df = None
        if not args.no_cache:
            df = cacher.load_processed_data('raw')
            if df is not None:
                logger.info("📂 Использую кэшированные сырые данные")
        
        if df is None:
            with DataLoader(config) as data_loader:
                if args.test_mode:
                    df = data_loader.load_data(symbols=args.test_symbols, limit=10000)
                else:
                    df = data_loader.load_data()
                
                # Проверка качества данных
                quality_report = data_loader.check_data_quality(df)
                logger.info(f"📊 Качество данных: {quality_report['total_records']:,} записей")
            
            # Сохраняем в кеш
            if not args.test_mode:
                cacher.save_processed_data(df, 'raw')
        
        # 2. Предобработка данных
        logger.info("\n" + "="*60)
        logger.info("🔧 ШАГ 2: ПРЕДОБРАБОТКА ДАННЫХ")
        logger.info("="*60)
        
        preprocessor = DataPreprocessor(config)
        
        # Пробуем загрузить обработанные признаки из кеша
        features_df = None
        if not args.no_cache:
            features_df = cacher.load_processed_data('features')
            if features_df is not None:
                logger.info("📂 Использую кэшированные признаки")
                # Восстанавливаем feature_columns
                preprocessor.feature_columns = features_df.columns.drop(['symbol', 'timestamp', 'datetime', 'buy_expected_return', 'sell_expected_return']).tolist()
        
        if features_df is None:
            # Извлечение признаков
            features_df = preprocessor.extract_features(df)
            # Сохраняем в кеш
            if not args.test_mode:
                cacher.save_processed_data(features_df, 'features')
        
        # Пробуем загрузить нормализованные данные из кеша
        normalized_data = None
        if not args.no_cache:
            normalized_data = cacher.load_data('normalized_splits')
            if normalized_data is not None:
                logger.info("📂 Использую кэшированные нормализованные данные")
                train_norm = normalized_data['train']
                val_norm = normalized_data['val'] 
                test_norm = normalized_data['test']
                preprocessor.scaler = normalized_data['scaler']
                preprocessor.feature_columns = normalized_data['feature_columns']
        
        if normalized_data is None:
            # Временное разделение
            data_splits = preprocessor.split_data_temporal(features_df)
            
            # 3. Нормализация
            logger.info("\n" + "="*60)
            logger.info("📏 ШАГ 3: НОРМАЛИЗАЦИЯ ДАННЫХ")
            logger.info("="*60)
            
            train_norm, val_norm, test_norm = preprocessor.normalize_features(
                data_splits['train'],
                data_splits['val'],
                data_splits['test']
            )
            
            # Сохраняем в кеш
            if not args.test_mode:
                normalized_cache = {
                    'train': train_norm,
                    'val': val_norm,
                    'test': test_norm,
                    'scaler': preprocessor.scaler,
                    'feature_columns': preprocessor.feature_columns
                }
                cacher.save_data(normalized_cache, 'normalized_splits')
        
        # 4. Создание последовательностей
        logger.info("\n" + "="*60)
        logger.info("🔄 ШАГ 4: СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
        logger.info("="*60)
        
        sequence_creator = SequenceCreator(config)
        
        # Словарь для хранения результатов
        models = {}
        
        # Обучаем модели для buy и sell
        for direction in ['buy', 'sell']:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 ОБРАБОТКА НАПРАВЛЕНИЯ: {direction.upper()}")
            logger.info(f"{'='*60}")
            
            # Пробуем загрузить последовательности из кеша
            sequences = None
            if not args.no_cache:
                sequences = cacher.load_sequences(direction, config.training.task_type)
                if sequences is not None:
                    logger.info("📂 Использую кэшированные последовательности")
                    
                    # Проверяем данные на NaN
                    for split_name, split_data in sequences.items():
                        if 'X' in split_data and 'y' in split_data:
                            X_nan = np.isnan(split_data['X']).sum()
                            y_nan = np.isnan(split_data['y']).sum()
                            if X_nan > 0 or y_nan > 0:
                                logger.warning(f"⚠️ Найдено NaN в {split_name}: X={X_nan}, y={y_nan}")
                                sequences = None  # Пересоздаем если есть NaN
                                break
                            
                            # Выводим статистику
                            logger.info(f"📊 {split_name} - X: {split_data['X'].shape}, "
                                       f"y: mean={np.mean(split_data['y']):.4f}, "
                                       f"std={np.std(split_data['y']):.4f}")
            
            if sequences is None:
                # Создаем последовательности
                logger.info("🔄 Создание последовательностей...")
                sequences = sequence_creator.create_sequences_for_splits(
                    train_norm, val_norm, test_norm,
                    feature_columns=preprocessor.feature_columns,
                    target_type=direction
                )
                
                # Сохраняем в кеш
                if not args.test_mode:
                    cacher.save_sequences(sequences, direction, config.training.task_type)
            
            # Конвертируем метки для классификации если нужно
            if config.training.task_type == 'classification_binary':
                for split in ['train', 'val', 'test']:
                    if split in sequences:
                        sequences[split]['y'] = preprocessor.convert_to_binary_labels(
                            sequences[split]['y']
                        )
            
            # Статистика последовательностей
            stats = sequence_creator.get_sequence_statistics(
                sequences['train']['X'],
                sequences['train']['y']
            )
            logger.info(f"📊 Статистика последовательностей: {stats}")
            
            # 5. Обучение моделей
            logger.info("\n" + "="*60)
            logger.info(f"🚀 ШАГ 5: ОБУЧЕНИЕ МОДЕЛЕЙ ({direction})")
            logger.info("="*60)
            
            # Создаем ансамбль
            ensemble = TFTEnsemble(config, base_name=f"tft_{direction}")
            
            # Обучаем ансамбль
            ensemble_models = ensemble.train_ensemble(
                sequences['train']['X'],
                sequences['train']['y'],
                sequences['val']['X'],
                sequences['val']['y'],
                n_models=args.ensemble_size,
                feature_columns=preprocessor.feature_columns
            )
            
            # 6. Оценка на тестовых данных
            logger.info("\n" + "="*60)
            logger.info(f"📊 ШАГ 6: ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ ({direction})")
            logger.info("="*60)
            
            test_metrics = ensemble.evaluate(
                sequences['test']['X'],
                sequences['test']['y'],
                f"Test ({direction})"
            )
            
            # Сохраняем результаты
            models[direction] = {
                'ensemble': ensemble,
                'test_metrics': test_metrics,
                'sequences_stats': stats
            }
            
            # Сохраняем модели
            if config.training.save_models:
                model_dir = log_dir / 'models' / direction
                ensemble.save_ensemble(model_dir)
        
        # 7. Генерация отчета
        logger.info("\n" + "="*60)
        logger.info("📝 ШАГ 7: ГЕНЕРАЦИЯ ОТЧЕТА")
        logger.info("="*60)
        
        report_generator = ReportGenerator(config, log_dir)
        
        # Собираем всю информацию для отчета
        results = {
            'models': models,
            'data_info': {
                'total_records': len(df),
                'n_symbols': df['symbol'].nunique(),
                'n_features': len(preprocessor.feature_columns),
                'train_size': len(sequences['train']['X']) if 'train' in sequences else 0,
                'val_size': len(sequences['val']['X']) if 'val' in sequences else 0,
                'test_size': len(sequences['test']['X']) if 'test' in sequences else 0
            }
        }
        
        report_generator.generate_report(results)
        
        logger.info("\n" + "="*60)
        logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info(f"📁 Результаты сохранены в: {log_dir}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ ОШИБКА: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()