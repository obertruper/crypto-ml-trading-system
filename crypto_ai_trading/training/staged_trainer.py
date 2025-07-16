"""
Поэтапный трейнер для выхода из локального минимума
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import copy
from pathlib import Path

from training.optimized_trainer import OptimizedTrainer
from utils.logger import get_logger


class StagedTrainer:
    """
    Трейнер с поэтапным обучением для предотвращения схлопывания модели
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = get_logger("StagedTrainer")
        
        # Проверяем конфигурацию поэтапного обучения
        self.staged_config = config.get('production', {}).get('staged_training', {})
        self.enabled = self.staged_config.get('enabled', False)
        
        if not self.enabled:
            self.logger.warning("⚠️ Поэтапное обучение отключено в конфигурации")
            return
            
        self.stages = self.staged_config.get('stages', [])
        if not self.stages:
            self.logger.error("❌ Не найдены этапы обучения в конфигурации")
            self.enabled = False
            return
            
        self.logger.info(f"✅ Поэтапное обучение включено с {len(self.stages)} этапами")
        
        # Сохраняем оригинальную конфигурацию
        self.original_config = copy.deepcopy(config)
        
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        """
        Запуск поэтапного обучения
        """
        if not self.enabled:
            # Fallback к обычному обучению
            trainer = OptimizedTrainer(self.model, self.config, self.device)
            return trainer.train(train_loader, val_loader)
            
        self.logger.info("🚀 Начало поэтапного обучения")
        self.logger.info("="*80)
        
        all_history = {}
        total_epochs = 0
        
        for stage_idx, stage in enumerate(self.stages):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"📊 ЭТАП {stage_idx + 1}/{len(self.stages)}: {stage['name']}")
            self.logger.info(f"📝 {stage.get('description', 'Без описания')}")
            self.logger.info(f"⏱️ Эпох: {stage['epochs']}")
            self.logger.info("="*80)
            
            # Создаем конфигурацию для текущего этапа
            stage_config = self._create_stage_config(stage)
            
            # Создаем трейнер для этапа
            trainer = OptimizedTrainer(self.model, stage_config, self.device)
            
            # Настраиваем активные losses
            self._configure_losses(trainer, stage)
            
            # Обучение на этапе
            stage_history = trainer.train(train_loader, val_loader)
            
            # Сохраняем историю
            stage_name = f"stage_{stage_idx}_{stage['name']}"
            all_history[stage_name] = stage_history
            total_epochs += stage['epochs']
            
            # Анализ результатов этапа
            self._analyze_stage_results(stage_name, stage_history, val_loader)
            
            # Проверка на схлопывание после критических этапов
            if stage_idx < 2:  # Первые два этапа критичны
                if self._check_collapse(trainer, val_loader):
                    self.logger.error("⚠️ Обнаружено схлопывание модели! Применяем коррекцию...")
                    self._apply_correction(stage_idx)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ Поэтапное обучение завершено! Всего эпох: {total_epochs}")
        self.logger.info("="*80)
        
        return all_history
        
    def _create_stage_config(self, stage: Dict) -> Dict:
        """
        Создает конфигурацию для конкретного этапа
        """
        # Копируем оригинальную конфигурацию
        stage_config = copy.deepcopy(self.original_config)
        
        # Обновляем параметры из этапа
        if 'learning_rate' in stage:
            stage_config['model']['learning_rate'] = stage['learning_rate']
            self.logger.info(f"📈 Learning rate: {stage['learning_rate']}")
            
        if 'dropout' in stage:
            stage_config['model']['dropout'] = stage['dropout']
            stage_config['model']['attention_dropout'] = stage['dropout'] * 0.5
            self.logger.info(f"💧 Dropout: {stage['dropout']}")
            
        if 'label_smoothing' in stage:
            stage_config['model']['label_smoothing'] = stage['label_smoothing']
            self.logger.info(f"🔄 Label smoothing: {stage['label_smoothing']}")
            
        if 'class_weights' in stage:
            stage_config['loss']['class_weights'] = stage['class_weights']
            self.logger.info(f"⚖️ Class weights: {stage['class_weights']}")
            
        if 'gradient_clip' in stage:
            stage_config['model']['gradient_clip'] = stage['gradient_clip']
            self.logger.info(f"✂️ Gradient clipping: {stage['gradient_clip']}")
            
        # Устанавливаем количество эпох
        stage_config['model']['epochs'] = stage['epochs']
        
        return stage_config
        
    def _configure_losses(self, trainer: OptimizedTrainer, stage: Dict):
        """
        Настраивает активные loss функции для этапа
        """
        active_losses = stage.get('active_losses', ['all'])
        
        if hasattr(trainer.criterion, 'set_active_losses'):
            trainer.criterion.set_active_losses(active_losses)
            self.logger.info(f"🎯 Активные losses: {active_losses}")
            
    def _check_collapse(self, trainer: OptimizedTrainer, val_loader: DataLoader) -> bool:
        """
        Проверяет, не схлопнулась ли модель в один класс
        """
        if not val_loader:
            return False
            
        # Получаем предсказания на валидации
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(val_loader):
                if batch_idx > 10:  # Проверяем только первые батчи
                    break
                    
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                # Извлекаем direction предсказания
                if hasattr(outputs, '_direction_logits'):
                    direction_logits = outputs._direction_logits[:, 0, :]  # 15m
                    preds = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
                    predictions.extend(preds.cpu().numpy())
                    
        if not predictions:
            return False
            
        # Проверяем распределение
        import numpy as np
        unique, counts = np.unique(predictions, return_counts=True)
        max_ratio = max(counts) / sum(counts)
        
        collapse_threshold = self.config.get('loss', {}).get('collapse_threshold', 0.8)
        
        if max_ratio > collapse_threshold:
            self.logger.warning(f"⚠️ Схлопывание обнаружено! Один класс составляет {max_ratio*100:.1f}%")
            return True
            
        return False
        
    def _apply_correction(self, stage_idx: int):
        """
        Применяет коррекцию при обнаружении схлопывания
        """
        # Консервативная переинициализация direction head
        if hasattr(self.model, 'direction_head'):
            self.logger.info("🔧 Переинициализация direction head...")
            
            for module in self.model.direction_head.modules():
                if isinstance(module, nn.Linear) and module.out_features == 12:
                    # Консервативная инициализация для предотвращения NaN
                    nn.init.xavier_uniform_(module.weight, gain=0.3)
                    if module.bias is not None:
                        with torch.no_grad():
                            bias = module.bias.view(4, 3)
                            bias[:, 0] = 0.3    # LONG bias (умеренно положительный)
                            bias[:, 1] = 0.3    # SHORT bias (умеренно положительный)  
                            bias[:, 2] = -1.2   # FLAT bias (сильно отрицательный для подавления)
                            
        # Не увеличиваем learning rate чтобы избежать NaN
        self.logger.info("🔧 Коррекция применена без изменения learning rate")
                
    def _analyze_stage_results(self, stage_name: str, history: Dict, val_loader: DataLoader):
        """
        Анализирует результаты этапа
        """
        if 'val_loss' in history and history['val_loss']:
            final_val_loss = history['val_loss'][-1]
            best_val_loss = min(history['val_loss'])
            self.logger.info(f"📊 Результаты этапа {stage_name}:")
            self.logger.info(f"   Final val loss: {final_val_loss:.4f}")
            self.logger.info(f"   Best val loss: {best_val_loss:.4f}")