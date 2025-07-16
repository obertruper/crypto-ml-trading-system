#!/usr/bin/env python3
"""
Упрощенный main для Transformer v3 с оптимизированной подготовкой данных
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_v3.config import Config
from transformer_v3.data.loader import DataLoader
from transformer_v3.data.data_processor import DataProcessor
from transformer_v3.data.noise_filter import NoiseFilter
from transformer_v3.data.sequence_creator import SequenceCreator
from transformer_v3.data.cacher import CacheManager
from transformer_v3.models.ensemble import TFTEnsemble
from transformer_v3.utils.logging_manager import LoggingManager
from transformer_v3.utils.report_generator import ReportGenerator

# Подавляем некритичные предупреждения
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Настройка TensorFlow
tf.config.optimizer.set_jit(False)  # Отключаем XLA для стабильности
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Transformer v3.0 - ML Trading (Simplified)')
    
    # Основные параметры
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification_binary'],
                       help='Тип задачи')
    parser.add_argument('--test-mode', action='store_true',
                       help='Тестовый режим (только 2 символа)')
    parser.add_argument('--test-symbols', nargs='+', 
                       default=['BTCUSDT', 'ETHUSDT'],
                       help='Символы для тестового режима')
    
    # Параметры обучения
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Размер батча (по умолчанию из конфига)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (по умолчанию из конфига)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Количество эпох (по умолчанию из конфига)')
    
    # Настройки модели
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Отключить mixed precision')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Другие параметры
    parser.add_argument('--no-cache', action='store_true',
                       help='Не использовать кэш')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    return parser.parse_args()


def main():
    """Основная функция"""
    args = parse_arguments()
    
    # Загрузка конфигурации
    config_path = Path(args.config) if args.config else None
    if config_path and config_path.exists():
        config = Config.from_yaml(str(config_path))
    else:
        config = Config()
    
    # Применяем аргументы командной строки
    if args.task:
        config.training.task_type = args.task
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.epochs:
        config.training.epochs = args.epochs
    if args.no_mixed_precision:
        config.model.use_mixed_precision = False
        
    # Тестовый режим обрабатывается через аргументы
    
    # Валидация конфигурации
    config.validate()
    
    # Настройка логирования
    log_dir = config.get_log_dir()
    logging_manager = LoggingManager(log_dir)
    logging_manager.setup_logging()
    
    logger.info("""
    ╔══════════════════════════════════════════╗
    ║   Transformer v3.0 - Simplified Trading  ║
    ╚══════════════════════════════════════════╝
    """)
    
    logger.info(f"🔧 Конфигурация:")
    logger.info(f"   Task: {config.training.task_type}")
    logger.info(f"   Batch size: {config.training.batch_size}")
    logger.info(f"   Learning rate: {config.training.learning_rate}")
    logger.info(f"   Optimizer: {config.training.optimizer}")
    logger.info(f"   Test mode: {args.test_mode}")
    
    # Сохраняем конфигурацию
    config.save(log_dir / "config.yaml")
    
    try:
        # 1. Загрузка данных
        logger.info("\n" + "="*60)
        logger.info("📥 ШАГ 1: ЗАГРУЗКА ДАННЫХ")
        logger.info("="*60)
        
        cacher = CacheManager(config)
        
        # Пробуем загрузить из кэша
        df = None
        if not args.no_cache:
            df = cacher.load_processed_data('processed')
            if df is not None:
                logger.info("📂 Использую кэшированные обработанные данные")
        
        if df is None:
            # Загружаем сырые данные
            raw_df = cacher.load_processed_data('raw')
            if raw_df is None:
                with DataLoader(config) as data_loader:
                    if args.test_mode:
                        raw_df = data_loader.load_data(symbols=args.test_symbols, limit=50000)
                    else:
                        raw_df = data_loader.load_data()
                
                # Сохраняем сырые данные
                if not args.test_mode:
                    cacher.save_processed_data(raw_df, 'raw')
            
            # 2. Обработка данных
            logger.info("\n" + "="*60)
            logger.info("🔧 ШАГ 2: УПРОЩЕННАЯ ОБРАБОТКА ДАННЫХ")
            logger.info("="*60)
            
            processor = DataProcessor(config)
            df = processor.process_data(raw_df)
            
            # Применяем дополнительную фильтрацию шума
            logger.info("🔇 Применение ансамблевой фильтрации шума...")
            noise_filter = NoiseFilter(method='ensemble')
            
            numeric_cols = [col for col in df.columns 
                          if col not in ['timestamp', 'buy_expected_return', 'sell_expected_return']]
            
            for col in numeric_cols:
                if df[col].std() > 0:
                    df[col] = noise_filter.filter_series(df[col].values)
            
            # Сохраняем обработанные данные
            if not args.test_mode:
                cacher.save_processed_data(df, 'processed')
        
        # 3. Разделение данных
        logger.info("\n" + "="*60)
        logger.info("✂️ ШАГ 3: РАЗДЕЛЕНИЕ ДАННЫХ")
        logger.info("="*60)
        
        processor = DataProcessor(config)
        processor.feature_columns = [col for col in df.columns 
                                   if col not in ['timestamp', 'buy_expected_return', 'sell_expected_return']]
        
        data_splits = processor.split_data(df)
        
        logger.info(f"📊 Количество признаков: {len(processor.feature_columns)}")
        logger.info(f"📊 Топ-10 признаков: {processor.feature_columns[:10]}")
        
        # 4. Создание последовательностей
        logger.info("\n" + "="*60)
        logger.info("🔄 ШАГ 4: СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
        logger.info("="*60)
        
        sequence_creator = SequenceCreator(config)
        
        # Обрабатываем каждое направление
        for direction in ['buy', 'sell']:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 ОБРАБОТКА НАПРАВЛЕНИЯ: {direction.upper()}")
            logger.info(f"{'='*60}")
            
            # Создаем последовательности
            target_col = f'{direction}_expected_return'
            
            X_train_seq, y_train_seq = sequence_creator.create_sequences(
                data_splits[direction]['X_train'],
                data_splits[direction]['y_train'],
                target_column=target_col
            )
            
            X_val_seq, y_val_seq = sequence_creator.create_sequences(
                data_splits[direction]['X_val'],
                data_splits[direction]['y_val'],
                target_column=target_col
            )
            
            X_test_seq, y_test_seq = sequence_creator.create_sequences(
                data_splits[direction]['X_test'],
                data_splits[direction]['y_test'],
                target_column=target_col
            )
            
            logger.info(f"📊 Размеры последовательностей:")
            logger.info(f"   Train: {X_train_seq.shape}")
            logger.info(f"   Val: {X_val_seq.shape}")
            logger.info(f"   Test: {X_test_seq.shape}")
            
            # 5. Обучение моделей
            logger.info("\n" + "="*60)
            logger.info(f"🚀 ШАГ 5: ОБУЧЕНИЕ МОДЕЛЕЙ ({direction})")
            logger.info("="*60)
            
            # Создаем ансамбль
            ensemble = TFTEnsemble(
                config=config,
                model_name=f"tft_{direction}_model",
                n_models=3  # Уменьшено для упрощения
            )
            
            # Обучаем ансамбль
            ensemble.train_ensemble(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                feature_columns=processor.feature_columns
            )
            
            # 6. Оценка на тестовой выборке
            logger.info("\n" + "="*60)
            logger.info("📊 ШАГ 6: ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
            logger.info("="*60)
            
            test_metrics = ensemble.evaluate(X_test_seq, y_test_seq, "Test")
            
            # Сохраняем модели
            save_dir = log_dir / f"models_{direction}"
            ensemble.save_models(save_dir)
            
            # Генерируем отчет
            report_gen = ReportGenerator(log_dir)
            report_gen.generate_model_report(
                model_name=f"ensemble_{direction}",
                metrics={
                    'train': ensemble.train_metrics,
                    'val': ensemble.val_metrics,
                    'test': test_metrics
                },
                config=config,
                feature_importance=ensemble.get_feature_importance()
            )
        
        logger.info("\n" + "="*60)
        logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info("="*60)
        logger.info(f"📊 Результаты сохранены в: {log_dir}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Обучение прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()