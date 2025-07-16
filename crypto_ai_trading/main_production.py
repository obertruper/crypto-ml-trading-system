#!/usr/bin/env python3
"""
Crypto AI Trading System - Production Ready версия
Включает мониторинг, валидацию и защиту от ошибок
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
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Версия системы
__version__ = "3.0.0-production"

class ProductionConfig:
    """Управление production конфигурацией"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
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


def main():
    parser = argparse.ArgumentParser(description='Crypto AI Trading System - Production')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Путь к конфигурации')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'validate', 'monitor'], 
                       default='train', help='Режим работы')
    parser.add_argument('--model-path', type=str, help='Путь к модели для inference/validate')
    parser.add_argument('--data-path', type=str, help='Путь к данным для inference')
    args = parser.parse_args()
    
    # Инициализация
    logger = get_logger('main_production')
    logger.info("="*80)
    logger.info(f"🚀 Crypto AI Trading System v{__version__}")
    logger.info(f"📅 Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🔧 Режим: {args.mode}")
    logger.info("="*80)
    
    try:
        # Загрузка и валидация конфигурации
        config_manager = ProductionConfig(args.config)
        config = config_manager.config
        
        if args.mode == 'train':
            # Полный цикл обучения с валидацией
            logger.info("🎓 Запуск обучения в production режиме...")
            
            # Импорт необходимых модулей
            from prepare_trading_data import main as prepare_data_main
            from training.unified_trainer import UnifiedTrainingPipeline
            
            # 1. Подготовка данных
            logger.info("📊 Этап 1: Подготовка данных...")
            prepare_data_main()
            
            # 2. Обучение
            logger.info("🧠 Этап 2: Обучение модели...")
            pipeline = UnifiedTrainingPipeline(config)
            model, model_path, metrics = pipeline.train()
            
            # 3. Валидация
            logger.info("✅ Этап 3: Production валидация...")
            validator = ModelValidator(config, logger)
            
            # Загружаем val_loader для валидации
            from data.precomputed_dataset import create_precomputed_loaders
            _, val_loader, _ = create_precomputed_loaders(config, logger)
            
            if validator.validate_model(model, val_loader):
                logger.info("🎉 Модель прошла production валидацию!")
                logger.info(f"📦 Модель сохранена: {model_path}")
            else:
                logger.error("❌ Модель не прошла production валидацию!")
                logger.error("Необходимо дополнительное обучение или изменение параметров")
        
        elif args.mode == 'inference':
            # Production inference
            if not args.model_path:
                logger.error("❌ Необходимо указать --model-path для inference")
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
        
        elif args.mode == 'validate':
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
            from data.precomputed_dataset import create_precomputed_loaders
            _, val_loader, _ = create_precomputed_loaders(config, logger)
            
            # Валидация
            validator = ModelValidator(config, logger)
            if validator.validate_model(model, val_loader):
                logger.info("✅ Модель прошла валидацию!")
            else:
                logger.error("❌ Модель не прошла валидацию!")
        
        elif args.mode == 'monitor':
            # Мониторинг обучения
            logger.info("📊 Запуск мониторинга...")
            
            import subprocess
            subprocess.run(['python', 'monitor_training.py'])
        
        logger.info("="*80)
        logger.info("✅ Выполнение завершено успешно!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        logger.exception("Подробности ошибки:")
        sys.exit(1)


if __name__ == "__main__":
    main()