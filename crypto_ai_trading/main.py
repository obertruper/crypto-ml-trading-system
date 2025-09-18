#!/usr/bin/env python3
"""
Crypto AI Trading System - Универсальная точка входа
Защита от переобучения встроена в архитектуру
Поддержка production режима с расширенной валидацией
"""

import argparse
import yaml
from pathlib import Path
import torch
import pandas as pd
from datetime import datetime
import warnings
import sys
import os
import json
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

from utils.logger import get_logger

# Конфигурация CUDA/TF32 согласно конфигу
def configure_device_settings(config: dict, logger) -> None:
    try:
        perf = config.get('performance', {})
        device_cfg = perf.get('device', 'cuda')
        use_cuda = (device_cfg == 'cuda') and torch.cuda.is_available()
        if not use_cuda:
            logger.info("🖥️ Работа на CPU или CUDA недоступна — пропускаю CUDA-настройки")
            return
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        use_tf32 = config.get('model', {}).get('use_tf32', True)
        torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(use_tf32)
        torch.set_float32_matmul_precision('high' if use_tf32 else 'medium')
        logger.info(f"⚙️ TF32: {'on' if use_tf32 else 'off'} | cudnn.benchmark: on")
    except Exception as e:
        try:
            logger.warning(f"⚠️ Не удалось применить CUDA-настройки: {e}")
        except Exception:
            pass

from utils.config import load_config as load_config_util

# Версия системы

# Версия системы
__version__ = "3.0.0"

class ProductionConfig:
    """Управление production конфигурацией"""
    def __init__(self, config_path: str, production_mode: bool = False):
        # Единый загрузчик конфигурации
        self.config = load_config_util(config_path)
        if production_mode:
            self.validate_config()
            self.apply_production_settings()

    def load_config(self, config_path: str) -> dict:
        # Поддержка старого интерфейса
        return load_config_util(config_path)

    def validate_config(self):
        """Лёгкая валидация критических разделов"""
        required_keys = ['model', 'loss', 'data', 'performance', 'database', 'risk_management']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Отсутствует обязательный раздел конфигурации: {key}")
        # Мягкие предупреждения
        try:
            if self.config['model'].get('learning_rate', 1.0) < 0.0001:
                print("⚠️ Предупреждение: очень низкий learning rate может замедлить обучение")
            if self.config['loss'].get('task_weights', {}).get('directions', 5.0) < 5.0:
                print("⚠️ Предупреждение: низкий вес direction loss может привести к плохим предсказаниям направления")
        except Exception:
            pass

    def apply_production_settings(self):
        """Применение production-настроек по умолчанию"""
        self.config['logging'] = self.config.get('logging', {})
        self.config['logging']['level'] = 'INFO'
        self.config['logging']['save_to_file'] = True
        self.config['validation'] = {
            'check_data_quality': True,
            'check_model_performance': True,
            'minimum_direction_accuracy': 0.6,
            'minimum_win_rate': 0.45,
            'maximum_flat_predictions': 0.7
        }
        self.config['model']['early_stopping_patience'] = max(
            25, self.config['model'].get('early_stopping_patience', 10)
        )
        self.config['model']['min_delta'] = self.config['model'].get('min_delta', 0.0001)
        return self.config

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Crypto AI Trading System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['data', 'train', 'backtest', 'full', 'demo', 'interactive', 'production', 'inference', 'validate', 'monitor', 'staged'],
                       help='Режим работы')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Путь к сохраненной модели (для режима backtest)')
    parser.add_argument('--use-improved-model', action='store_true',
                       help='Использовать улучшенную версию модели с FeatureAttention')
    parser.add_argument('--validate-only', action='store_true',
                       help='Только валидация конфигурации без запуска')
    parser.add_argument('--prepare-data', action='store_true',
                       help='Автоматически запустить prepare_trading_data.py если нет кеша')
    
    # Новые параметры для расширенного обучения
    parser.add_argument('--target-focus', type=str, default='all',
                       choices=['all', 'returns', 'directions', 'long_profits', 'short_profits', 'risk_metrics'],
                       help='Фокус на конкретной группе целевых переменных')
    parser.add_argument('--loss-type', type=str, default='unified',
                       choices=['unified', 'directional', 'profit_aware', 'ensemble'],
                       help='Тип loss функции для оптимизации')
    parser.add_argument('--ensemble-count', type=int, default=1,
                       help='Количество моделей в ансамбле (1 = без ансамбля)')
    parser.add_argument('--direction-focus', action='store_true',
                       help='Специализация на предсказании направления движения цены')
    parser.add_argument('--large-movement-weight', type=float, default=1.0,
                       help='Коэффициент веса для крупных движений цены (1.0 = без веса)')
    parser.add_argument('--min-movement-threshold', type=float, default=0.005,
                       help='Минимальный порог движения для торговых сигналов (0.5%)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Путь к checkpoint для fine-tuning (например: models_saved/best_model_20250710_150018.pth)')
    
    # Новые аргументы для иерархической модели и curriculum learning
    parser.add_argument('--model-type', type=str, default='unified',
                       choices=['unified', 'hierarchical'],
                       help='Тип модели: unified (UnifiedPatchTST)')
    parser.add_argument('--curriculum-stage', type=int, default=0,
                       choices=[0, 1, 2, 3, 4, 5],
                       help='Этап curriculum learning (0 = без curriculum, 1-5 = этапы)')
    parser.add_argument('--task-focus', type=str, default=None,
                       choices=['market_regime', 'direction', 'targets', 'risk', 'all'],
                       help='Фокус задачи для curriculum learning')
    parser.add_argument('--use-focal-loss', action='store_true',
                       help='Использовать Focal Loss для несбалансированных классов')
    parser.add_argument('--class-weights', type=str, default=None,
                       help='Веса классов в формате "[w1,w2,w3]" для LONG,SHORT,FLAT')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Путь к checkpoint для продолжения curriculum обучения')
    
    args = parser.parse_args()
    
    # Определяем production режим и загружаем конфигурацию
    is_production_mode = args.mode in ['production', 'inference', 'validate']

    if is_production_mode:
        # Используем ProductionConfig для production режимов
        config_manager = ProductionConfig(args.config, production_mode=True)
        config = config_manager.config
        logger_name = "CryptoAI-Production"
    else:
        # Обычная загрузка конфигурации
        config = load_config_util(args.config)
        logger_name = "CryptoAI"
    
    # Создаем logger сразу
    logger = get_logger(logger_name)

    # Применяем настройки CUDA/TF32 из конфига
    configure_device_settings(config, logger)
    
    # Применяем флаг улучшенной модели к конфигурации
    if args.use_improved_model:
        config['model']['use_improvements'] = True
        config['model']['feature_attention'] = True
        config['model']['multi_scale_patches'] = True
    
    # Обработка новых параметров обучения
    # Создаем секцию training если её нет
    if 'training' not in config:
        config['training'] = {}
    
    if args.target_focus != 'all':
        config['training']['target_focus'] = args.target_focus
        logger.info(f"🎯 Фокус на целевых переменных: {args.target_focus}")
    
    if args.loss_type != 'unified':
        config['training']['loss_type'] = args.loss_type
        logger.info(f"🔧 Тип loss функции: {args.loss_type}")
    
    if args.ensemble_count > 1:
        config['training']['ensemble_count'] = args.ensemble_count
        config['model']['use_ensemble'] = True
        logger.info(f"🎭 Ансамбль из {args.ensemble_count} моделей")
    
    if args.direction_focus:
        config['training']['direction_focus'] = True
        config['model']['task_type'] = 'direction_prediction'
        logger.info("🎯 Специализация на предсказании направления движения")
    
    if args.large_movement_weight != 1.0:
        config['training']['large_movement_weight'] = args.large_movement_weight
        logger.info(f"⚖️ Вес крупных движений: {args.large_movement_weight}")
    
    if args.min_movement_threshold != 0.005:
        config['training']['min_movement_threshold'] = args.min_movement_threshold
        logger.info(f"📏 Минимальный порог движения: {args.min_movement_threshold:.3f} ({args.min_movement_threshold*100:.1f}%)")
    
    logger.info("="*80)
    logger.info("🚀 Запуск Crypto AI Trading System")
    logger.info(f"📋 Режим: {args.mode}")
    logger.info(f"⚙️ Конфигурация: {args.config}")
    if args.mode == 'production':
        logger.info("🏭 PRODUCTION MODE - Оптимизированные настройки для финального обучения")
        logger.info("📊 Особенности production режима:")
        logger.info("   - Уменьшенный batch size (512) для стабильности")
        logger.info("   - Усиленная регуляризация (dropout=0.5, weight_decay=0.01)")
        logger.info("   - Динамические веса классов для борьбы с дисбалансом")
        logger.info("   - Увеличенный вес direction loss (15.0)")
        logger.info("   - Focal Loss с агрессивными параметрами")
    if args.use_improved_model:
        logger.info("🔥 Используется улучшенная модель с FeatureAttention")
    logger.info("="*80)
    
    # Валидация конфигурации
    if args.validate_only:
        logger.info("🔍 Режим валидации конфигурации...")
        from utils.config_validator import validate_config
        is_valid = validate_config(config)
        if is_valid:
            logger.info("✅ Конфигурация валидна!")
        else:
            logger.error("❌ Конфигурация содержит ошибки!")
        return
    
    # Интерактивный режим
    if args.mode == 'interactive':
        logger.info("🎮 Запуск интерактивного режима...")
        from run_interactive import run_interactive_mode
        run_interactive_mode(config)
        return
    
    try:
        # Централизованная загрузка данных для всех режимов
        train_data, val_data, test_data, feature_cols, target_cols = None, None, None, None, None
        train_loader, val_loader, test_loader = None, None, None
        config_updated = config.copy()
        model = None
        model_path = None
        
        if args.mode in ['data', 'train', 'full', 'production', 'backtest']:
            # Production режим эквивалентен train с production конфигурацией
            if args.mode == 'production':
                logger.info("🏭 Production режим активирован - используем оптимизированные настройки")
                
            # Сначала проверяем наличие кэшированных данных
            train_data, val_data, test_data, feature_cols, target_cols = load_cached_data_if_exists(logger)
            
            if train_data is not None:
                # Используем кэшированные данные
                logger.info("🎯 Используем кэшированные данные для всех режимов")
                
                # Ограничиваем количество символов если указано в конфиге
                max_symbols = config.get('data', {}).get('max_symbols', None)
                if max_symbols:
                    logger.info(f"🎯 Ограничиваем данные до {max_symbols} символов")
                    unique_symbols = train_data['symbol'].unique()[:max_symbols]
                    train_data = train_data[train_data['symbol'].isin(unique_symbols)]
                    val_data = val_data[val_data['symbol'].isin(unique_symbols)]
                    test_data = test_data[test_data['symbol'].isin(unique_symbols)]
                    logger.info(f"📊 После ограничения: train={len(train_data):,}, val={len(val_data):,}, test={len(test_data):,}")
                
                train_loader, val_loader, test_loader, config_updated = create_unified_data_loaders(
                    train_data, val_data, test_data, feature_cols, target_cols, config, logger
                )
            elif args.mode in ['data', 'full']:
                # Создаем новые данные только если их нет и это режим data/full
                logger.info("🔄 Кэшированные данные не найдены, создаем новые...")
                train_loader, val_loader, test_loader = prepare_data(config, logger)
                config_updated = config  # используем оригинальную конфигурацию
            elif args.mode == 'backtest':
                # Для backtest пробуем загрузить существующие данные
                logger.info("🔍 Режим backtest - ищем существующие данные...")
                train_loader, val_loader, test_loader, config_updated = create_unified_data_loaders(
                    config, demo_mode=False
                )
            else:
                # Режим train без кэшированных данных
                logger.error("❌ Режим train требует наличия кэшированных данных!")
                
                if args.prepare_data:
                    logger.info("🔄 Запускаем prepare_trading_data.py для создания кеша...")
                    import subprocess
                    result = subprocess.run(
                        ["python", "prepare_trading_data.py", "--config", args.config],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        logger.info("✅ Данные успешно подготовлены!")
                        # Повторно пытаемся загрузить кеш
                        train_data, val_data, test_data, feature_cols, target_cols = load_cached_data_if_exists(logger)
                        if train_data is not None:
                            train_loader, val_loader, test_loader, config_updated = create_unified_data_loaders(
                                train_data, val_data, test_data, feature_cols, target_cols, config, logger
                            )
                        else:
                            logger.error("❌ Не удалось загрузить данные после подготовки")
                            return
                    else:
                        logger.error(f"❌ Ошибка при подготовке данных: {result.stderr}")
                        return
                else:
                    logger.error("Запустите: python prepare_trading_data.py")
                    logger.error("Или используйте флаг --prepare-data для автоматического запуска")
                    return
        
        if args.mode in ['train', 'full', 'production', 'staged']:
            # Проверяем, нужно ли делать fine-tuning
            if config_updated.get('fine_tuning', {}).get('enabled', False) and args.checkpoint:
                logger.info("🎯 Fine-tuning режим активирован")
                from training.fine_tuner import create_fine_tuner
                
                # Создаем FineTuner с существующим checkpoint
                fine_tuner = create_fine_tuner(config_updated, args.checkpoint)
                
                # Обновляем learning rate для fine-tuning
                fine_tuning_lr = config_updated.get('fine_tuning', {}).get('learning_rate', 0.00002)
                for param_group in fine_tuner.optimizer.param_groups:
                    param_group['lr'] = fine_tuning_lr
                
                # Запускаем fine-tuning
                fine_tuning_epochs = config_updated.get('fine_tuning', {}).get('epochs', 30)
                best_val_loss = float('inf')
                
                for epoch in range(fine_tuning_epochs):
                    fine_tuner.current_epoch = epoch
                    
                    # Train
                    train_metrics = fine_tuner.train_epoch(train_loader)
                    
                    # Validate
                    val_metrics = fine_tuner.validate(val_loader)
                    
                    # Scheduler step
                    if fine_tuner.scheduler:
                        fine_tuner.scheduler.step(val_metrics['loss'])
                    
                    # Save best model
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        model_path = fine_tuner._save_checkpoint(epoch, val_metrics['loss'], is_best=True)
                    
                    logger.info(f"Epoch {epoch+1}/{fine_tuning_epochs} - "
                              f"Train Loss: {train_metrics['loss']:.4f}, "
                              f"Val Loss: {val_metrics['loss']:.4f}, "
                              f"Direction Acc: {val_metrics.get('direction_accuracy', 0):.3f}")
                
                model = fine_tuner.model
                
            elif args.mode == 'staged':
                # Поэтапное обучение - ИСПРАВЛЕНО: используем подготовленные данные
                logger.info("🎯 Запуск поэтапного обучения (staged mode)...")
                
                # Загружаем кэшированные данные
                train_data, val_data, test_data, feature_cols, target_cols = load_cached_data_if_exists(logger)
                
                if train_data is None:
                    logger.error("❌ Кэшированные данные не найдены!")
                    logger.error("   Сначала запустите: python prepare_trading_data.py --test")
                    return
                
                # Создаем DataLoader'ы
                try:
                    train_loader, val_loader, test_loader, config_updated = create_unified_data_loaders(
                        train_data, val_data, test_data, feature_cols, target_cols, config_updated, logger
                    )
                    logger.info("✅ DataLoader'ы созданы для staged обучения")
                except Exception as e:
                    logger.error(f"❌ Ошибка создания DataLoader'ов: {e}")
                    return
                
                # Обучаем с OptimizedTrainer (staged как алиас на обычное обучение)
                model, model_path, _ = train_model(config_updated, train_loader, val_loader, logger, model_type='unified', args=args)
                logger.info("✅ Обучение (staged alias) завершено")
                # Сохраняем маркер режима staged
                try:
                    if model_path:
                        import torch
                        ckpt = torch.load(model_path, map_location='cpu') if Path(model_path).exists() else None
                        if isinstance(ckpt, dict):
                            ckpt['training_mode'] = 'staged'
                            torch.save(ckpt, model_path)
                except Exception:
                    pass
                
                # Удалён fallback повторной загрузки parquet и второй вызов staged_manager
                # РЕЗЕРВ: старый код для fallback (удалено)
                if train_loader is None or val_loader is None or test_loader is None:
                    logger.info("📊 Загрузка данных для поэтапного обучения...")
                    
                    # Сначала загружаем данные из файлов
                    import pandas as pd
                    from pathlib import Path
                    
                    data_dir = Path("data/processed")
                    if not data_dir.exists():
                        logger.error(f"❌ Директория {data_dir} не найдена. Сначала запустите: python prepare_trading_data.py")
                        return
                    
                    # Загружаем подготовленные данные
                    try:
                        logger.info("📂 Загрузка train_data.parquet...")
                        train_data = pd.read_parquet(data_dir / "train_data.parquet")
                        logger.info(f"   - Train: {len(train_data):,} записей")
                        
                        logger.info("📂 Загрузка val_data.parquet...")
                        val_data = pd.read_parquet(data_dir / "val_data.parquet")
                        logger.info(f"   - Val: {len(val_data):,} записей")
                        
                        logger.info("📂 Загрузка test_data.parquet...")
                        test_data = pd.read_parquet(data_dir / "test_data.parquet")
                        logger.info(f"   - Test: {len(test_data):,} записей")
                        
                        # Загружаем списки признаков из текстовых файлов
                        # Поддержка двух форматов: с нумерацией (123→column) и без
                        with open(data_dir / "feature_cols.txt", "r") as f:
                            lines = f.readlines()
                            feature_cols = []
                            for line in lines:
                                line = line.strip()
                                if not line:  # Пропускаем пустые строки
                                    continue
                                # Проверяем формат с нумерацией "123→column_name"
                                if '→' in line:
                                    col_name = line.split('→')[1].strip()
                                else:
                                    # Простой формат - просто имя колонки
                                    col_name = line
                                if col_name:  # Добавляем непустые имена
                                    feature_cols.append(col_name)
                                        
                        with open(data_dir / "target_cols.txt", "r") as f:
                            lines = f.readlines()
                            target_cols = []
                            for line in lines:
                                line = line.strip()
                                if not line:  # Пропускаем пустые строки
                                    continue
                                # Проверяем формат с нумерацией "123→column_name"
                                if '→' in line:
                                    col_name = line.split('→')[1].strip()
                                else:
                                    # Простой формат - просто имя колонки
                                    col_name = line
                                if col_name:  # Добавляем непустые имена
                                    target_cols.append(col_name)
                            
                        logger.info(f"📊 Признаки: {len(feature_cols)} входных, {len(target_cols)} целевых")
                        
                        # Создаем DataLoaders
                        from data.precomputed_dataset import create_precomputed_data_loaders
                        train_loader, val_loader, test_loader = create_precomputed_data_loaders(
                            train_data=train_data,
                            val_data=val_data, 
                            test_data=test_data,
                            config=config_updated,
                            feature_cols=feature_cols,
                            target_cols=target_cols
                        )
                        
                        if train_loader is None:
                            logger.error("❌ Не удалось создать DataLoader'ы")
                            return
                        
                        logger.info(f"✅ DataLoader'ы созданы:")
                        logger.info(f"   - Train: {len(train_loader)} батчей")
                        logger.info(f"   - Val: {len(val_loader)} батчей")
                        logger.info(f"   - Test: {len(test_loader)} батчей")
                        
                    except FileNotFoundError as e:
                        logger.error(f"❌ Файл не найден: {e}")
                        logger.error("Сначала запустите: python prepare_trading_data.py")
                        return
                    except Exception as e:
                        logger.error(f"❌ Ошибка загрузки данных: {e}")
                        return
            else:
                # Обычное обучение модели с унифицированной конфигурацией
                model, model_path, train_loader = train_model(config_updated, train_loader, val_loader, logger, 
                                                               model_type=args.model_type if hasattr(args, 'model_type') else 'unified',
                                                               args=args)
        
        if args.mode in ['backtest', 'full']:
            if args.mode == 'backtest':
                if not args.model_path:
                    logger.error("Необходимо указать --model-path для режима backtest")
                    return
                
                logger.info(f"📥 Загрузка модели: {args.model_path}")
                
                # Загрузка модели
                checkpoint = torch.load(args.model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
                
                # Создание модели с конфигурацией из checkpoint
                from models.patchtst_unified import UnifiedPatchTST
                
                # Всегда используем конфигурацию из checkpoint для backtest
                if 'config' in checkpoint:
                    checkpoint_config = checkpoint['config']['model']
                    # КРИТИЧНО: Обновляем input_size из checkpoint
                    if 'input_size' in checkpoint_config:
                        checkpoint_config['input_size'] = checkpoint_config.get('input_size', 248)
                    else:
                        checkpoint_config['input_size'] = 248  # Используем полный набор признаков после feature engineering
                    
                    logger.info(f"🔧 Используем конфигурацию из checkpoint: input_size={checkpoint_config['input_size']}")
                    
                    # Добавляем недостающие поля для совместимости
                    if 'seq_len' not in checkpoint_config:
                        checkpoint_config['seq_len'] = checkpoint_config.get('context_window', 96)
                    
                    # Обновляем config_updated для совместимости с DataLoader'ами
                    config_updated['model'] = checkpoint_config
                    
                    model = UnifiedPatchTST(checkpoint_config)
                else:
                    # В старых checkpoint'ах нет конфигурации - используем текущую с правильным input_size
                    logger.warning("⚠️ В checkpoint нет конфигурации! Используем текущую конфигурацию с input_size=248...")
                    config_updated['model']['input_size'] = 248  # Используем полный набор признаков после feature engineering
                    model = UnifiedPatchTST(config_updated['model'])
                
                # Загрузка весов
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                
                model.eval()
                logger.info("✅ Модель загружена успешно")
                
            results = backtest_strategy(config, model, test_loader, train_loader, logger)
            
            validation_passed = analyze_results(config, results, logger)
        
        if args.mode == 'demo':
            logger.info("🎯 Демонстрационный режим - только проверка подключения к БД")
            from data.data_loader import CryptoDataLoader
            
            data_loader = CryptoDataLoader(config)
            available_symbols = data_loader.get_available_symbols()
            
            logger.info(f"✅ Подключение к БД успешно")
            logger.info(f"📊 Найдено {len(available_symbols)} символов")
            logger.info(f"🔍 Первые 10 символов: {available_symbols[:10]}")
            
            # Загружаем небольшой образец данных
            sample_data = data_loader.load_data(
                symbols=available_symbols[:2],
                start_date="2025-06-01",
                end_date="2025-06-16"
            )
            
            logger.info(f"📈 Загружено {len(sample_data)} записей для демонстрации")
        
        # Production-специфичные режимы
        if args.mode == 'inference':
            # Production inference
            if not args.model_path:
                logger.error("❌ Необходимо указать --model-path для режима inference")
                return
            
            logger.info("🔮 Запуск production inference...")
            
            inference = ProductionInference(args.model_path, config, logger)
            
            # Здесь должна быть загрузка реальных данных
            # Для примера используем случайные данные
            test_data = torch.randn(1, config['model']['context_window'], config['model']['input_size'])
            
            results = inference.predict(test_data)
            
            if 'error' not in results:
                logger.info("✅ Предсказание выполнено успешно:")
                logger.info(f"   Future Returns: {results['future_returns'].numpy()}")
                if 'direction_classes' in results:
                    classes = ['LONG', 'SHORT', 'FLAT']
                    for i, cls in enumerate(results['direction_classes'][0]):
                        logger.info(f"   Direction {i+1}: {classes[cls]}")
            else:
                logger.error("❌ Использованы безопасные значения по умолчанию")
        
        if args.mode == 'validate':
            # Отдельная валидация существующей модели
            if not args.model_path:
                logger.error("❌ Необходимо указать --model-path для валидации")
                return
            
            logger.info("🔍 Запуск валидации модели...")
            
            # Загружаем модель
            from models.patchtst_unified import create_model as create_unified_model
            model = create_unified_model(config['model'])
            
            checkpoint = torch.load(args.model_path, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # Загружаем данные для валидации
            if val_loader is None:
                from data.precomputed_dataset import create_precomputed_loaders
                _, val_loader, _ = create_precomputed_loaders(config, logger)
            
            # Валидация
            validator = ModelValidator(config, logger)
            if validator.validate_model(model, val_loader):
                logger.info("✅ Модель прошла валидацию!")
            else:
                logger.error("❌ Модель не прошла валидацию!")
        
        if args.mode == 'monitor':
            # Мониторинг обучения
            logger.info("📊 Запуск мониторинга...")
            
            import subprocess
            subprocess.run(['python', 'monitor_training.py'])
        
        # Production режим с валидацией после обучения
        if args.mode == 'production' and model is not None:
            logger.info("✅ Запуск production валидации после обучения...")
            validator = ModelValidator(config, logger)
            
            if validator.validate_model(model, val_loader):
                logger.info("🎉 Модель прошла production валидацию!")
                logger.info(f"📦 Модель готова к использованию: {model_path}")
            else:
                logger.error("❌ Модель не прошла production валидацию!")
                logger.error("Необходимо дополнительное обучение или изменение параметров")
        
        logger.info("="*80)
        logger.info("✅ Выполнение завершено успешно!")
        logger.info("="*80)
        
    except Exception as e:
        logger.log_error(e, "main")
        logger.critical("❌ Критическая ошибка! Выполнение прервано.")
        raise

if __name__ == "__main__":
    main()
