"""
Fine-tuning модуль для дообучения существующей модели
Использует предобученные веса и оптимизирует только определенные слои
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import yaml
from tqdm import tqdm
import time

from models.patchtst_unified import UnifiedPatchTST as UnifiedPatchTST
from training.optimized_trainer import OptimizedTrainer
from utils.logger import get_logger


class FineTuner(OptimizedTrainer):
    """Специализированный trainer для fine-tuning существующих моделей"""
    
    def __init__(self, 
                 model: nn.Module, 
                 config: Dict, 
                 checkpoint_path: str,
                 device: Optional[torch.device] = None):
        super().__init__(model, config, device)
        
        self.logger = get_logger("FineTuner")
        self.checkpoint_path = checkpoint_path
        
        # Fine-tuning специфичные параметры
        self.freeze_backbone = config.get('fine_tuning', {}).get('freeze_backbone', True)
        self.unfreeze_layers = config.get('fine_tuning', {}).get('unfreeze_layers', ['direction', 'confidence'])
        self.curriculum_learning = config.get('fine_tuning', {}).get('curriculum_learning', True)
        self.noise_injection_std = config.get('fine_tuning', {}).get('noise_injection_std', 0.01)
        
        # Загружаем предобученные веса
        self._load_pretrained_weights()
        
        # Настраиваем какие слои обучать
        self._setup_trainable_layers()
        
        # Инициализируем temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Mixup параметры (усилены для fine-tuning)
        self.use_mixup = True
        self.mixup_alpha = config.get('fine_tuning', {}).get('mixup_alpha', 0.3)
        
        self.logger.info(f"✅ Fine-tuning инициализирован с checkpoint: {checkpoint_path}")
        self.logger.info(f"   - Заморожен backbone: {self.freeze_backbone}")
        self.logger.info(f"   - Обучаемые слои: {self.unfreeze_layers}")
        self.logger.info(f"   - Curriculum learning: {self.curriculum_learning}")
        self.logger.info(f"   - Mixup alpha: {self.mixup_alpha}")
    
    def _load_pretrained_weights(self):
        """Загрузка предобученных весов"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Загружаем веса модели
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.logger.info(f"✅ Загружены веса из {self.checkpoint_path}")
        
        # Загружаем метрики если есть
        if 'metrics' in checkpoint:
            self.logger.info(f"📊 Предыдущие метрики:")
            for key, value in checkpoint['metrics'].items():
                self.logger.info(f"   - {key}: {value:.4f}")
    
    def _setup_trainable_layers(self):
        """Настройка обучаемых слоев"""
        # Сначала замораживаем все параметры
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Размораживаем только указанные слои
        trainable_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # Проверяем, нужно ли разморозить этот слой
            should_unfreeze = any(layer_name in name for layer_name in self.unfreeze_layers)
            
            if should_unfreeze:
                param.requires_grad = True
                trainable_params += param.numel()
                if param.dim() > 1:  # Только для матриц весов
                    self.logger.debug(f"   ✅ Разморожен: {name} ({param.shape})")
        
        # Всегда размораживаем temperature для calibration
        self.temperature.requires_grad = True
        
        self.logger.info(f"📊 Обучаемые параметры: {trainable_params:,} / {total_params:,} "
                        f"({100 * trainable_params / total_params:.1f}%)")
    
    def inject_noise(self, inputs: torch.Tensor) -> torch.Tensor:
        """Добавление шума к входным данным для регуляризации"""
        if self.training and self.noise_injection_std > 0:
            noise = torch.randn_like(inputs) * self.noise_injection_std
            return inputs + noise
        return inputs
    
    def apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Применение temperature scaling для калибровки уверенности"""
        return logits / self.temperature
    
    def get_curriculum_weights(self, targets: Dict[str, torch.Tensor], epoch: int) -> torch.Tensor:
        """Получение весов для curriculum learning"""
        if not self.curriculum_learning:
            return torch.ones(targets['direction_15m'].shape[0], device=self.device)
        
        # Начинаем с простых примеров (высокая волатильность)
        future_returns = targets.get('future_return_15m', torch.zeros_like(targets['direction_15m']))
        volatility = torch.abs(future_returns)
        
        # Прогресс обучения (0 -> 1)
        progress = min(epoch / 20, 1.0)  # Полный curriculum за 20 эпох
        
        # В начале фокусируемся на высокой волатильности
        # Позже включаем все примеры
        threshold = torch.quantile(volatility, 1.0 - progress * 0.8)  # От 20% до 100% данных
        
        weights = torch.ones_like(volatility)
        weights[volatility < threshold] = 0.1 + 0.9 * progress  # Плавное увеличение веса
        
        return weights
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Модифицированная эпоха обучения с fine-tuning техниками"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc="Fine-tuning", leave=False)
        
        for batch_idx, (inputs, targets, info) in enumerate(progress_bar):
            # Перенос на GPU
            inputs = inputs.to(self.device, non_blocking=True)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device, non_blocking=True)
            else:
                targets = {k: v.to(self.device, non_blocking=True) 
                          for k, v in targets.items()}
            
            # Добавляем шум к входным данным
            inputs = self.inject_noise(inputs)
            
            # Применяем mixup с увеличенным alpha
            if self.use_mixup and np.random.rand() < 0.5:  # 50% вероятность mixup
                inputs, targets_a, targets_b, lam = self.mixup_data(inputs, targets, self.mixup_alpha)
                
                # Forward pass с mixup
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    
                    # Temperature scaling для direction логитов
                    if hasattr(outputs, '_direction_logits'):
                        outputs._direction_logits = self.apply_temperature_scaling(outputs._direction_logits)
                    
                    # Mixup loss
                    loss_a = self.loss_fn(outputs, targets_a)
                    loss_b = self.loss_fn(outputs, targets_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
            else:
                # Обычный forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    
                    # Temperature scaling
                    if hasattr(outputs, '_direction_logits'):
                        outputs._direction_logits = self.apply_temperature_scaling(outputs._direction_logits)
                    
                    loss = self.loss_fn(outputs, targets)
            
            # Curriculum learning веса
            if self.curriculum_learning:
                curriculum_weights = self.get_curriculum_weights(targets, self.current_epoch)
                loss = (loss * curriculum_weights).mean()
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Обновляем EMA
                    self._update_ema()
            else:
                loss.backward()
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Обновляем EMA
                    self._update_ema()
            
            # Накапливаем метрики
            batch_loss = loss.detach() * self.gradient_accumulation_steps
            epoch_loss += batch_loss
            
            # Обновляем прогресс бар
            progress_bar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Средние метрики за эпоху
        num_batches = len(train_loader)
        epoch_metrics['loss'] = (epoch_loss / num_batches).item()
        
        return epoch_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Сохранение checkpoint с дополнительной информацией о fine-tuning"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'temperature': self.temperature.item(),
            'fine_tuning_config': {
                'original_checkpoint': self.checkpoint_path,
                'freeze_backbone': self.freeze_backbone,
                'unfreeze_layers': self.unfreeze_layers,
                'mixup_alpha': self.mixup_alpha,
                'noise_injection_std': self.noise_injection_std
            }
        }
        
        if self.use_ema and self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        # Сохраняем
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if is_best:
            path = Path(f'models_saved/best_finetuned_model_{timestamp}.pth')
        else:
            path = Path(f'models_saved/checkpoint_finetuned_epoch_{epoch}_{timestamp}.pth')
            
        torch.save(checkpoint, path)
        self.logger.info(f"💾 Checkpoint сохранен: {path}")
        
        return path


def create_fine_tuner(config: Dict, checkpoint_path: str, device: Optional[torch.device] = None) -> FineTuner:
    """Фабричная функция для создания FineTuner"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создаем модель
    model = UnifiedPatchTST(config).to(device)
    
    # Создаем FineTuner
    fine_tuner = FineTuner(
        model=model,
        config=config,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    return fine_tuner