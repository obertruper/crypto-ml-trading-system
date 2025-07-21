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
import numpy as np
from datetime import datetime
import warnings
import sys
import os
import json
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

from utils.logger import get_logger

# Оптимизация GPU если доступен
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # Установка float32 matmul precision для ускорения на новых GPU
    torch.set_float32_matmul_precision('high')
    # Дополнительные оптимизации для Ampere+ архитектуры (RTX 5090)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Версия системы
__version__ = "3.0.0"

def load_config(config_path: str) -> dict:
    """Загрузка конфигурации"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class ProductionConfig:
    """Управление production конфигурацией"""
    
    def __init__(self, config_path: str, production_mode: bool = False):
        self.config = self.load_config(config_path)
        if production_mode:
            self.validate_config()
            self.apply_production_settings()
    
    def load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации с валидацией"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def validate_config(self):
        """Валидация критических параметров"""
        required_keys = [
            'model', 'loss', 'data', 'performance', 
            'database', 'risk_management'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Отсутствует обязательный раздел конфигурации: {key}")
        
        # Проверка критических параметров для production
        if self.config['model']['learning_rate'] < 0.0001:
            print("⚠️ Предупреждение: очень низкий learning rate может замедлить обучение")
        
        if self.config['loss']['task_weights']['directions'] < 5.0:
            print("⚠️ Предупреждение: низкий вес direction loss может привести к плохим предсказаниям направления")
    
    def apply_production_settings(self):
        """Применение production-специфичных настроек"""
        # Увеличиваем логирование
        self.config['logging'] = self.config.get('logging', {})
        self.config['logging']['level'] = 'INFO'
        self.config['logging']['save_to_file'] = True
        
        # Включаем все проверки
        self.config['validation'] = {
            'check_data_quality': True,
            'check_model_performance': True,
            'minimum_direction_accuracy': 0.6,
            'minimum_win_rate': 0.45,
            'maximum_flat_predictions': 0.7
        }
        
        # Защита от переобучения
        self.config['model']['early_stopping_patience'] = 25
        self.config['model']['min_delta'] = 0.0001
        
        return self.config


class ModelValidator:
    """Валидация модели перед использованием в production"""
    
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.validation_results = {}
    
    def validate_model(self, model: torch.nn.Module, val_loader) -> bool:
        """Полная валидация модели"""
        self.logger.info("🔍 Запуск production валидации модели...")
        
        # 1. Проверка архитектуры
        if not self._validate_architecture(model):
            return False
        
        # 2. Проверка производительности
        if not self._validate_performance(model, val_loader):
            return False
        
        # 3. Проверка разнообразия предсказаний
        if not self._validate_prediction_diversity(model, val_loader):
            return False
        
        # 4. Проверка устойчивости
        if not self._validate_robustness(model, val_loader):
            return False
        
        self._save_validation_report()
        return True
    
    def _validate_architecture(self, model: torch.nn.Module) -> bool:
        """Проверка корректности архитектуры"""
        self.logger.info("  📐 Проверка архитектуры...")
        
        # Проверка наличия необходимых компонентов
        required_modules = ['direction_head', 'future_returns_head', 'long_levels_head']
        
        for module_name in required_modules:
            if not hasattr(model, module_name):
                self.logger.error(f"    ❌ Отсутствует обязательный модуль: {module_name}")
                return False
        
        # Проверка размерностей
        try:
            batch_size = 32
            seq_len = self.config['model']['context_window']
            n_features = self.config['model']['input_size']
            
            dummy_input = torch.randn(batch_size, seq_len, n_features).to(next(model.parameters()).device)
            with torch.no_grad():
                output = model(dummy_input)
            
            expected_output_size = self.config['model']['output_size']
            if output.shape[-1] != expected_output_size:
                self.logger.error(f"    ❌ Неверный размер выхода: {output.shape[-1]} != {expected_output_size}")
                return False
            
            self.logger.info("    ✅ Архитектура корректна")
            return True
            
        except Exception as e:
            self.logger.error(f"    ❌ Ошибка при проверке архитектуры: {e}")
            return False
    
    def _validate_performance(self, model: torch.nn.Module, val_loader) -> bool:
        """Проверка производительности модели"""
        self.logger.info("  📊 Проверка производительности...")
        
        from training.optimized_trainer import OptimizedTrainer
        
        # Создаем временный trainer для оценки
        trainer = OptimizedTrainer(model, self.config, device=next(model.parameters()).device)
        
        # Получаем метрики
        metrics = trainer.validate_with_enhanced_metrics(val_loader)
        
        # Проверяем минимальные требования
        min_requirements = self.config['validation']
        
        direction_accuracy = metrics.get('direction_accuracy_overall', 0)
        win_rate = metrics.get('win_rate_overall', 0)
        
        self.validation_results['direction_accuracy'] = direction_accuracy
        self.validation_results['win_rate'] = win_rate
        
        if direction_accuracy < min_requirements['minimum_direction_accuracy']:
            self.logger.error(f"    ❌ Direction accuracy слишком низкая: {direction_accuracy:.3f} < {min_requirements['minimum_direction_accuracy']}")
            return False
        
        if win_rate < min_requirements['minimum_win_rate']:
            self.logger.error(f"    ❌ Win rate слишком низкий: {win_rate:.3f} < {min_requirements['minimum_win_rate']}")
            return False
        
        self.logger.info(f"    ✅ Производительность достаточна (Accuracy: {direction_accuracy:.3f}, Win Rate: {win_rate:.3f})")
        return True
    
    def _validate_prediction_diversity(self, model: torch.nn.Module, val_loader) -> bool:
        """Проверка разнообразия предсказаний"""
        self.logger.info("  🎲 Проверка разнообразия предсказаний...")
        
        from training.optimized_trainer import OptimizedTrainer
        
        trainer = OptimizedTrainer(model, self.config, device=next(model.parameters()).device)
        
        # Получаем первый батч для анализа
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)
            break
        
        with torch.no_grad():
            outputs = model(inputs)
        
        # Анализируем direction предсказания
        direction_metrics = trainer.compute_direction_metrics(outputs, targets)
        
        pred_entropy = direction_metrics.get('pred_entropy_overall', 0)
        flat_ratio = direction_metrics.get('pred_flat_ratio_overall', 1.0)
        
        self.validation_results['prediction_entropy'] = pred_entropy
        self.validation_results['flat_prediction_ratio'] = flat_ratio
        
        max_flat = self.config['validation']['maximum_flat_predictions']
        
        if flat_ratio > max_flat:
            self.logger.error(f"    ❌ Слишком много FLAT предсказаний: {flat_ratio:.1%} > {max_flat:.1%}")
            return False
        
        if pred_entropy < 0.3:
            self.logger.error(f"    ❌ Слишком низкая энтропия предсказаний: {pred_entropy:.3f}")
            return False
        
        self.logger.info(f"    ✅ Разнообразие предсказаний достаточно (Entropy: {pred_entropy:.3f}, FLAT: {flat_ratio:.1%})")
        return True
    
    def _validate_robustness(self, model: torch.nn.Module, val_loader) -> bool:
        """Проверка устойчивости модели к шуму"""
        self.logger.info("  🛡️ Проверка устойчивости...")
        
        # Получаем первый батч
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(next(model.parameters()).device)
            break
        
        with torch.no_grad():
            # Обычные предсказания
            outputs_normal = model(inputs)
            
            # Предсказания с небольшим шумом
            noise = torch.randn_like(inputs) * 0.01
            outputs_noisy = model(inputs + noise)
            
            # Сравниваем direction предсказания
            if hasattr(outputs_normal, '_direction_logits'):
                pred_normal = torch.argmax(outputs_normal._direction_logits, dim=-1)
                pred_noisy = torch.argmax(outputs_noisy._direction_logits, dim=-1)
                
                consistency = (pred_normal == pred_noisy).float().mean().item()
                
                self.validation_results['noise_robustness'] = consistency
                
                if consistency < 0.9:
                    self.logger.error(f"    ❌ Низкая устойчивость к шуму: {consistency:.3f}")
                    return False
                
                self.logger.info(f"    ✅ Модель устойчива к шуму (consistency: {consistency:.3f})")
                return True
            else:
                self.logger.warning("    ⚠️ Не удалось проверить устойчивость (нет direction_logits)")
                return True
    
    def _save_validation_report(self):
        """Сохранение отчета о валидации"""
        report_path = Path("validation_reports") / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
            "validation_results": self.validation_results,
            "passed": True
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"  💾 Отчет сохранен: {report_path}")


class ProductionInference:
    """Класс для production inference с защитой от ошибок"""
    
    def __init__(self, model_path: str, config: dict, logger):
        self.config = config
        self.logger = logger
        self.model = self._load_model(model_path)
        self.device = next(self.model.parameters()).device
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Безопасная загрузка модели"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        from models.patchtst_unified import create_unified_model
        
        # Загружаем checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Обновляем конфигурацию из checkpoint если есть
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            saved_config = checkpoint['config']
            if 'model' in saved_config:
                self.config['model'].update(saved_config['model'])
        
        # Создаем модель
        model = create_unified_model(self.config)
        
        # Загружаем веса
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Переносим на нужное устройство
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        self.logger.info(f"✅ Модель загружена: {model_path}")
        return model
    
    def predict(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Безопасное предсказание с обработкой ошибок"""
        try:
            self.model.eval()
            with torch.no_grad():
                # Проверка размерности
                if len(data.shape) == 2:
                    data = data.unsqueeze(0)  # Добавляем batch dimension
                
                # Перенос на устройство
                data = data.to(self.device)
                
                # Предсказание
                outputs = self.model(data)
                
                # Парсинг результатов
                results = self._parse_outputs(outputs)
                
                # Валидация результатов
                if self._validate_predictions(results):
                    return results
                else:
                    raise ValueError("Предсказания не прошли валидацию")
                    
        except Exception as e:
            self.logger.error(f"❌ Ошибка при предсказании: {e}")
            # Возвращаем безопасные значения по умолчанию
            return self._get_safe_defaults()
    
    def _parse_outputs(self, outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Парсинг выходов модели в удобный формат"""
        results = {
            'future_returns': outputs[:, 0:4].cpu(),
            'directions': outputs[:, 4:8].cpu(),
            'long_levels': torch.sigmoid(outputs[:, 8:12]).cpu(),
            'short_levels': torch.sigmoid(outputs[:, 12:16]).cpu(),
            'risk_metrics': outputs[:, 16:20].cpu()
        }
        
        # Добавляем классы direction если есть логиты
        if hasattr(outputs, '_direction_logits'):
            direction_probs = torch.softmax(outputs._direction_logits, dim=-1)
            direction_classes = torch.argmax(direction_probs, dim=-1)
            results['direction_classes'] = direction_classes.cpu()
            results['direction_probs'] = direction_probs.cpu()
        
        return results
    
    def _validate_predictions(self, results: Dict[str, torch.Tensor]) -> bool:
        """Валидация предсказаний на разумность"""
        # Проверяем future returns в разумных пределах (-50%, +50%)
        returns = results['future_returns']
        if torch.abs(returns).max() > 0.5:
            self.logger.warning("⚠️ Обнаружены экстремальные значения returns")
            return False
        
        # Проверяем вероятности в [0, 1]
        for key in ['long_levels', 'short_levels']:
            probs = results[key]
            if probs.min() < 0 or probs.max() > 1:
                self.logger.warning(f"⚠️ Недопустимые вероятности в {key}")
                return False
        
        return True
    
    def _get_safe_defaults(self) -> Dict[str, torch.Tensor]:
        """Безопасные значения по умолчанию при ошибке"""
        batch_size = 1
        return {
            'future_returns': torch.zeros(batch_size, 4),
            'directions': torch.full((batch_size, 4), 2),  # FLAT
            'long_levels': torch.zeros(batch_size, 4),
            'short_levels': torch.zeros(batch_size, 4),
            'risk_metrics': torch.zeros(batch_size, 4),
            'direction_classes': torch.full((batch_size, 4), 2),  # FLAT
            'error': True
        }


def load_cached_data_if_exists(logger) -> tuple:
    """Централизованная загрузка кэшированных данных
    
    Returns:
        tuple: (train_data, val_data, test_data, feature_cols, target_cols) или (None, None, None, None, None)
    """
    logger.info("🔍 Проверка наличия кэшированных данных...")
    
    processed_dir = Path("data/processed")
    train_file = processed_dir / "train_data.parquet"
    val_file = processed_dir / "val_data.parquet"
    test_file = processed_dir / "test_data.parquet"
    
    if all(f.exists() for f in [train_file, val_file, test_file]):
        logger.info("✅ Найдены кэшированные данные, загружаем...")
        
        train_data = pd.read_parquet(train_file)
        val_data = pd.read_parquet(val_file)
        test_data = pd.read_parquet(test_file)
        
        logger.info(f"📊 Размеры кэшированных данных:")
        logger.info(f"   - Train: {len(train_data):,} записей")
        logger.info(f"   - Val: {len(val_data):,} записей")
        logger.info(f"   - Test: {len(test_data):,} записей")
        
        # Определяем признаки и целевые переменные из кэшированных данных
        from data.constants import (
            get_feature_columns, get_target_columns, 
            validate_data_structure, TRADING_TARGET_VARIABLES
        )
        
        try:
            data_info = validate_data_structure(train_data)
            feature_cols = data_info['feature_cols']
            target_cols = data_info['target_cols']
            
            logger.info(f"📈 Структура кэшированных данных:")
            logger.info(f"   - Всего колонок: {len(train_data.columns)}")
            logger.info(f"   - Признаков для модели: {len(feature_cols)}")
            logger.info(f"   - Целевых переменных: {len(target_cols)}")
            logger.info(f"   - Служебных колонок: {len(train_data.columns) - len(feature_cols) - len(target_cols)}")
            
            return train_data, val_data, test_data, feature_cols, target_cols
            
        except ValueError as e:
            logger.error(f"❌ Ошибка структуры кэшированных данных: {e}")
            return None, None, None, None, None
    else:
        logger.info("❌ Кэшированные данные не найдены")
        missing_files = [f.name for f in [train_file, val_file, test_file] if not f.exists()]
        logger.info(f"   Отсутствуют файлы: {missing_files}")
        return None, None, None, None, None

def create_unified_data_loaders(train_data, val_data, test_data, feature_cols, target_cols, config, logger):
    """Унифицированное создание DataLoader'ов для всех режимов
    
    Args:
        train_data, val_data, test_data: DataFrame'ы с данными
        feature_cols, target_cols: списки колонок
        config: конфигурация
        logger: логгер
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info("🏗️ Создание унифицированных DataLoader'ов...")
    
    from data.dataset import create_data_loaders
    from data.precomputed_dataset import create_precomputed_data_loaders
    
    # Используем PrecomputedDataset для максимальной производительности
    use_precomputed = config.get('performance', {}).get('use_precomputed_dataset', True)
    
    # Обновляем конфигурацию чтобы соответствовать реальным данным
    config_updated = config.copy()
    config_updated['model']['input_features'] = len(feature_cols)
    config_updated['model']['n_features'] = len(feature_cols)
    
    # Проверяем совместимость данных с конфигурацией модели
    task_type = config['model'].get('task_type', 'regression')
    
    if task_type == 'trading':
        # Используем ВСЕ доступные торговые целевые переменные из кэша
        config_updated['model']['target_variables'] = target_cols
        logger.info(f"✅ Торговая модель: используем все {len(target_cols)} целевых переменных")
        logger.info(f"   Первые 5 переменных: {target_cols[:5]}")
    else:
        # Для регрессии выбираем основную целевую переменную
        main_target = [col for col in target_cols if col.startswith('future_return_')]
        if main_target:
            config_updated['model']['target_variable'] = main_target[0]
            logger.info(f"✅ Регрессия: используем целевую переменную {main_target[0]}")
        else:
            logger.error("❌ Не найдена целевая переменная для регрессии!")
            raise ValueError("Нет подходящей целевой переменной для регрессии")
    
    # Создание DataLoader'ов с правильными параметрами
    if use_precomputed:
        logger.info("🚀 Используем PrecomputedDataset для максимальной скорости")
        train_loader, val_loader, test_loader = create_precomputed_data_loaders(
            train_data=train_data,
            val_data=val_data, 
            test_data=test_data,
            config=config_updated,
            feature_cols=feature_cols,
            target_cols=target_cols
        )
    else:
        logger.info("📊 Используем стандартный Dataset")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data=train_data,
            val_data=val_data, 
            test_data=test_data,
            config=config_updated,
            feature_cols=feature_cols,
            target_cols=target_cols
        )
    
    logger.info("✅ DataLoader'ы созданы успешно")
    return train_loader, val_loader, test_loader, config_updated

def prepare_data(config: dict, logger):
    """Подготовка данных для обучения с защитой от data leakage"""
    logger.start_stage("data_preparation")
    
    logger.info("📥 Загрузка данных из PostgreSQL...")
    
    # Импорт здесь для избежания циклических импортов
    from data.data_loader import CryptoDataLoader
    from data.feature_engineering import FeatureEngineer
    from data.dataset import create_data_loaders, TradingDataset
    
    data_loader = CryptoDataLoader(config)
    
    # Получаем список символов
    if config['data']['symbols'] == 'all':
        available_symbols = data_loader.get_available_symbols()
        # Ограничиваем количество символов для демо
        max_symbols = config.get('data', {}).get('max_symbols', 10)
        symbols_to_load = available_symbols[:max_symbols]
        logger.info(f"📊 Загружаем первые {max_symbols} символов из {len(available_symbols)}: {symbols_to_load}")
    else:
        symbols_to_load = config['data']['symbols']
        logger.info(f"📊 Загружаем указанные символы: {symbols_to_load}")
    
    raw_data = data_loader.load_data(
        symbols=symbols_to_load,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    logger.info("🔍 Проверка качества данных...")
    quality_report = data_loader.validate_data_quality(raw_data)
    
    for symbol, report in quality_report.items():
        if report['anomalies']:
            logger.warning(f"Аномалии в данных {symbol}: {report['anomalies']}")
    
    logger.info("🛠️ Создание признаков с защитой от data leakage...")
    feature_engineer = FeatureEngineer(config)
    
    # Используем новый метод с защитой от data leakage
    train_data, val_data, test_data = feature_engineer.create_features_with_train_split(
        raw_data,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )
    
    logger.info("🏗️ Создание datasets...")
    
    # Создание DataLoader'ов через унифицированную систему
    from data.constants import get_feature_columns, get_target_columns, validate_data_structure
    
    # Определяем структуру данных
    data_info = validate_data_structure(train_data)
    feature_cols = data_info['feature_cols']
    target_cols = data_info['target_cols']
    
    train_loader, val_loader, test_loader, _ = create_unified_data_loaders(
        train_data, val_data, test_data, feature_cols, target_cols, config, logger
    )
    
    logger.info(f"📊 Размеры datasets:")
    logger.info(f"   - Train: {len(train_data)} записей")
    logger.info(f"   - Val: {len(val_data)} записей")
    logger.info(f"   - Test: {len(test_data)} записей")
    
    logger.end_stage("data_preparation", 
                    train_size=len(train_data),
                    val_size=len(val_data),
                    test_size=len(test_data))
    
    return train_loader, val_loader, test_loader

def train_model(config: dict, train_loader, val_loader, logger):
    """Обучение модели"""
    import time
    logger.start_stage("model_training")
    
    logger.info("🏗️ Создание модели PatchTST...")
    
    # ВРЕМЕННОЕ РЕШЕНИЕ: используем известные размеры вместо загрузки батча
    # TODO: исправить медленную загрузку первого батча с HDF5
    logger.info("📊 Используем предопределенные размеры данных...")
    n_features = 240  # Известно из конфигурации
    n_targets = 20    # Известно из конфигурации
    
    # Закомментировано из-за проблемы с медленной загрузкой
    # import time
    # start_time = time.time()
    # logger.info("📊 Получение первого батча для анализа...")
    # sample_batch = next(iter(train_loader))
    # logger.info(f"✅ Первый батч получен за {time.time() - start_time:.2f} секунд")
    # X_sample, y_sample, _ = sample_batch
    # 
    # n_features = X_sample.shape[-1]  # Последняя размерность
    # n_targets = y_sample.shape[-1] if y_sample is not None else 1
    
    logger.info(f"📊 Входные признаки: {n_features}, Целевые переменные: {n_targets}")
    
    # Проверяем соответствие с конфигурацией
    config_input_size = config['model'].get('input_size', 100)
    config_output_size = config['model'].get('output_size', 1)
    task_type = config['model'].get('task_type', 'regression')
    
    if n_features != config_input_size:
        logger.warning(f"⚠️ Размерность признаков не совпадает: данные={n_features}, конфиг={config_input_size}")
        logger.info(f"🔧 Автоматически обновляем input_size в конфигурации")
        config['model']['input_size'] = n_features
    
    if task_type == 'trading':
        # Для торговой модели с большим количеством целей используем базовую архитектуру
        if config['model']['name'] == 'UnifiedPatchTST':  # Используем унифицированную модель
            logger.info(f"📊 Торговая модель: {n_targets} целевых переменных - используем гибкую архитектуру")
            config['model']['output_size'] = n_targets
        else:
            logger.info(f"📊 Торговая модель: используется PatchTSTForTrading с несколькими выходами")
    else:
        if n_targets != config_output_size:
            logger.warning(f"⚠️ Размерность целей не совпадает: данные={n_targets}, конфиг={config_output_size}")
            logger.info(f"🔧 Автоматически обновляем output_size в конфигурации")
            config['model']['output_size'] = n_targets
    
    # Используем фабрику для создания правильной модели
    from models.patchtst import create_patchtst_model
    from models.patchtst_unified import create_unified_model, UnifiedPatchTSTForTrading
    
    # Ensemble обучение
    ensemble_count = config.get('training', {}).get('ensemble_count', 1)
    
    if ensemble_count > 1:
        logger.info(f"🎭 Создание ансамбля из {ensemble_count} моделей")
        models = []
        
        for i in range(ensemble_count):
            logger.info(f"📊 Создание модели {i+1}/{ensemble_count}")
            
            # Вариативность для каждой модели в ансамбле
            model_config = config.copy()
            model_config['model']['random_seed'] = config.get('model', {}).get('random_seed', 42) + i
            
            # Небольшие вариации архитектуры для разнообразия
            if i > 0:
                # Вариация dropout
                original_dropout = model_config['model'].get('dropout', 0.1)
                model_config['model']['dropout'] = original_dropout + (i * 0.05)
                
                # Вариация learning rate  
                original_lr = model_config['model'].get('learning_rate', 2e-5)
                model_config['model']['learning_rate'] = original_lr * (1 + (i-1) * 0.2)
            
            # Создание модели
            if task_type == 'trading' and n_targets > 10:
                model_config['model']['name'] = 'UnifiedPatchTST'
                model_config['model']['output_size'] = n_targets
                model = create_unified_model(model_config)
            elif model_config['model']['name'] == 'UnifiedPatchTST':
                model = create_unified_model(model_config)
            else:
                model = create_patchtst_model(model_config)
            
            models.append(model)
        
        logger.info(f"✅ Ансамбль из {len(models)} моделей создан")
        
        # Для простоты, обучаем первую модель (в будущем можно расширить)
        model = models[0] 
        logger.info(f"🎯 Обучение первой модели из ансамбля (модель 1/{ensemble_count})")
        
    else:
        # Одиночная модель
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Всегда используем UnifiedPatchTST для 20 целевых переменных
        if task_type == 'trading' and n_targets > 10:
            logger.info(f"🎯 Обнаружено {n_targets} целевых переменных - используем UnifiedPatchTST")
            config['model']['name'] = 'UnifiedPatchTST'
            config['model']['output_size'] = n_targets
            
            model_start_time = time.time()
            logger.info("🔨 Вызов create_unified_model...")
            model = create_unified_model(config)
            logger.info(f"✅ Модель создана за {time.time() - model_start_time:.2f} секунд")
            logger.info(f"✅ UnifiedPatchTST создан с {n_targets} выходами для торговой модели")
            logger.info("⚠️ torch.compile отключен для RTX 5090 (sm_120) - требуется поддержка в будущих версиях PyTorch")
        elif config['model']['name'] == 'UnifiedPatchTST':
            model = create_unified_model(config)
            logger.info("📊 Используется UnifiedPatchTST с 20 выходами")
            logger.info("⚠️ torch.compile отключен для RTX 5090 (sm_120)")
        else:
            model = create_patchtst_model(config)
            # Логируем тип модели
            if hasattr(model, 'long_model'):
                logger.info("✅ Используется PatchTSTForTrading с поддержкой LONG/SHORT")
            else:
                logger.info("📊 Используется базовая PatchTSTForPrediction")
    
    # ВАЖНО: Явно перемещаем модель на GPU перед созданием трейнера
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        logger.info(f"🔥 Модель перемещена на GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU память доступна: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        logger.warning("⚠️ GPU не доступен, используется CPU")
    
    # Проверяем, нужно ли использовать поэтапное обучение
    if config.get('production', {}).get('staged_training', {}).get('enabled', False):
        logger.info("🎯 Используется поэтапное обучение (StagedTrainer)")
        from training.staged_trainer import StagedTrainer
        trainer = StagedTrainer(model, config, device=device)
    else:
        # Создание оптимизированного трейнера с явным указанием устройства
        from training.optimized_trainer import OptimizedTrainer
        trainer = OptimizedTrainer(model, config, device=device)
    
    # Проверка размещения модели
    logger.info(f"✅ Модель на устройстве: {next(model.parameters()).device}")
    logger.info(f"✅ Трейнер использует: {trainer.device}")
    
    # DataLoader'ы уже созданы, используем их напрямую
    
    # Обучение
    logger.info("🚀 Начало обучения...")
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Сохранение лучшей модели
    if hasattr(trainer, 'checkpoint_dir'):
        # Для OptimizedTrainer
        best_model_path = trainer.checkpoint_dir / "best_model.pth"
    else:
        # Для StagedTrainer - создаем путь вручную
        from pathlib import Path
        checkpoint_dir = Path(config['model'].get('checkpoint_dir', 'models_saved'))
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        best_model_path = checkpoint_dir / "best_model.pth"
        
        # Сохраняем модель
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'training_results': training_results
        }, best_model_path)
        logger.info(f"💾 Модель сохранена в {best_model_path}")
    
    logger.info(f"✅ Обучение завершено. Лучшая модель: {best_model_path}")
    
    logger.end_stage("model_training", model_path=str(best_model_path))
    
    return model, best_model_path, train_loader

def backtest_strategy(config: dict, model, test_loader, train_loader, logger):
    """Бэктестирование стратегии с UnifiedPatchTST"""
    logger.start_stage("backtesting")
    
    logger.info("💰 Запуск современного бэктестирования для UnifiedPatchTST...")
    
    # Используем новый UnifiedBacktester
    from trading.unified_backtester import UnifiedBacktester
    
    # Создаем бэктестер
    backtester = UnifiedBacktester(config)
    
    # Запускаем бэктестинг с реальными предсказаниями модели
    logger.info("🔮 Генерация предсказаний модели на тестовых данных...")
    
    try:
        # Запуск бэктестинга
        backtest_results = backtester.run_backtest(model, test_loader)
    except Exception as e:
        logger.error(f"❌ Ошибка в бэктестинге: {e}")
        # Возвращаем пустые результаты при ошибке
        backtest_results = {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'final_balance': config['backtesting']['initial_capital']
        }
    
    # Отображение результатов
    logger.info("📈 РЕЗУЛЬТАТЫ БЭКТЕСТИНГА:")
    logger.info(f"  Начальный капитал: ${config['backtesting']['initial_capital']:,.2f}")
    logger.info(f"  Финальный капитал: ${backtest_results.get('final_balance', config['backtesting']['initial_capital']):,.2f}")
    logger.info(f"  Общая доходность: {backtest_results.get('total_return', 0)*100:.2f}%")
    logger.info(f"  Коэффициент Шарпа: {backtest_results.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Максимальная просадка: {backtest_results.get('max_drawdown', 0)*100:.2f}%")
    logger.info(f"  Win Rate: {backtest_results.get('win_rate', 0)*100:.2f}%")
    logger.info(f"  Всего сделок: {backtest_results.get('total_trades', 0)}")
    
    logger.end_stage("backtesting", 
                    total_return=backtest_results.get('total_return', 0)*100,
                    sharpe_ratio=backtest_results.get('sharpe_ratio', 0))
    
    return backtest_results

def analyze_results(config: dict, results: dict, logger):
    """Анализ и визуализация результатов"""
    logger.start_stage("results_analysis")
    
    logger.info("📊 Анализ результатов...")
    
    min_sharpe = config['validation']['min_sharpe_ratio']
    min_win_rate = config['validation']['min_win_rate']
    max_dd = config['validation']['max_drawdown']
    
    passed_validation = True
    
    if results['sharpe_ratio'] < min_sharpe:
        logger.warning(f"⚠️ Sharpe Ratio ({results['sharpe_ratio']:.2f}) ниже минимального ({min_sharpe})")
        passed_validation = False
    
    if results['win_rate'] < min_win_rate:
        logger.warning(f"⚠️ Win Rate ({results['win_rate']:.2%}) ниже минимального ({min_win_rate:.2%})")
        passed_validation = False
    
    if abs(results['max_drawdown']) > max_dd:
        logger.warning(f"⚠️ Max Drawdown ({results['max_drawdown']:.2%}) превышает лимит ({max_dd:.2%})")
        passed_validation = False
    
    if passed_validation:
        logger.info("✅ Все валидационные тесты пройдены!")
    else:
        logger.warning("❌ Некоторые валидационные тесты не пройдены")
    
    logger.end_stage("results_analysis", validation_passed=passed_validation)
    
    return passed_validation

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Crypto AI Trading System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['data', 'train', 'backtest', 'full', 'demo', 'interactive', 'production', 'inference', 'validate', 'monitor'],
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
    parser.add_argument('--production', action='store_true',
                       help='Использовать production конфигурацию (config_production.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Путь к checkpoint для fine-tuning (например: models_saved/best_model_20250710_150018.pth)')
    
    args = parser.parse_args()
    
    # Определяем production режим и загружаем конфигурацию
    is_production_mode = args.production or args.mode in ['production', 'inference', 'validate']
    
    if is_production_mode:
        # Используем ProductionConfig для production режимов
        config_manager = ProductionConfig(args.config, production_mode=True)
        config = config_manager.config
        logger_name = "CryptoAI-Production"
    else:
        # Обычная загрузка конфигурации
        config = load_config(args.config)
        logger_name = "CryptoAI"
    
    # Создаем logger сразу
    logger = get_logger(logger_name)
    
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
    if args.production or args.mode == 'production':
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
        
        if args.mode in ['train', 'full', 'production']:
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
                        model_path = fine_tuner.save_checkpoint(epoch, val_metrics, is_best=True)
                    
                    logger.info(f"Epoch {epoch+1}/{fine_tuning_epochs} - "
                              f"Train Loss: {train_metrics['loss']:.4f}, "
                              f"Val Loss: {val_metrics['loss']:.4f}, "
                              f"Direction Acc: {val_metrics.get('direction_accuracy', 0):.3f}")
                
                model = fine_tuner.model
                
            else:
                # Обычное обучение модели с унифицированной конфигурацией
                model, model_path, train_loader = train_model(config_updated, train_loader, val_loader, logger)
        
        if args.mode in ['backtest', 'full']:
            if args.mode == 'backtest':
                if not args.model_path:
                    logger.error("Необходимо указать --model-path для режима backtest")
                    return
                
                logger.info(f"📥 Загрузка модели: {args.model_path}")
                
                # Загрузка модели
                checkpoint = torch.load(args.model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
                
                # Создание модели с конфигурацией
                from models.patchtst_unified import UnifiedPatchTSTForTrading
                model = UnifiedPatchTSTForTrading(config_updated)
                
                # Загрузка весов
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
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
            from models.patchtst_unified import create_unified_model
            model = create_unified_model(config)
            
            checkpoint = torch.load(args.model_path)
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