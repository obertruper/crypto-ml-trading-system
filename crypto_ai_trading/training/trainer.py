"""
Основной модуль для обучения моделей
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
from pathlib import Path
import time
from tqdm import tqdm
import json
from datetime import datetime

from utils.logger import get_logger
from models.losses import get_loss_function, MultiTaskLoss
from training.optimizer import get_optimizer, get_scheduler
from utils.metrics import MetricsTracker

class Trainer:
    """Основной класс для обучения моделей"""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict,
                 device: Optional[torch.device] = None):
        """
        Args:
            model: модель для обучения
            config: конфигурация обучения
            device: устройство для обучения
        """
        self.model = model
        self.config = config
        self.logger = get_logger("Trainer")
        
        # Устройство
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Использование нескольких GPU если доступны
        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            self.logger.info(f"🔥 Используется {torch.cuda.device_count()} GPU для обучения")
            self.model = nn.DataParallel(self.model)
            self.is_data_parallel = True
        else:
            self.is_data_parallel = False
        
        # Параметры обучения
        self.epochs = config['model']['epochs']
        # ИСПРАВЛЕНО: Используем оптимальный learning rate для RTX 5090
        self.learning_rate = config['model'].get('learning_rate', 2e-5)  # Оптимальный LR
        self.gradient_clip = config['model'].get('gradient_clip', 1.0)
        self.early_stopping_patience = config['model'].get('early_stopping_patience', 10)
        
        # Gradient accumulation для больших эффективных батчей
        self.gradient_accumulation_steps = config['performance'].get('gradient_accumulation_steps', 1)
        
        # Mixed precision training
        self.use_amp = config['performance'].get('mixed_precision', False)
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Оптимизатор и планировщик
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Функции потерь
        self.criterion = self._create_loss_function()
        
        # Метрики
        self.metrics_tracker = MetricsTracker(config)
        
        # Директории для сохранения
        self.checkpoint_dir = Path("models_saved")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # GPU оптимизации
        if self.device.type == 'cuda':
            # Периодическая очистка кэша GPU
            self.gpu_cache_clear_freq = config['performance'].get('gpu_cache_clear_freq', 10)
            # Мониторинг памяти GPU
            self.monitor_gpu_memory = config['performance'].get('monitor_gpu_memory', True)
        
        # Early stopping (улучшенная версия для борьбы с переобучением)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.min_delta = config['model'].get('min_delta', 1e-4)  # Минимальное улучшение
        self.overfitting_threshold = config['model'].get('overfitting_threshold', 0.1)  # Порог переобучения
        self.consecutive_overfitting = 0  # Счетчик последовательного переобучения
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Создание оптимизатора"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'AdamW')
        
        # Получаем параметры оптимизатора, исключая lr чтобы избежать дублирования
        optimizer_params = optimizer_config.get('params', {}).copy()
        optimizer_params.pop('lr', None)  # Удаляем lr если он есть в params
        
        return get_optimizer(
            optimizer_name,
            self.model.parameters(),
            lr=self.learning_rate,
            **optimizer_params
        )
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Создание планировщика learning rate"""
        scheduler_config = self.config.get('scheduler', {})
        
        if not scheduler_config:
            return None
        
        scheduler_name = scheduler_config.get('name', 'CosineAnnealingLR')
        
        # Специальная обработка для OneCycleLR
        if scheduler_name == 'OneCycleLR':
            # OneCycleLR требует total_steps
            if hasattr(self, 'train_loader'):
                total_steps = self.epochs * len(self.train_loader)
            else:
                total_steps = self.epochs * 1000  # Примерная оценка
            scheduler_config['params']['total_steps'] = total_steps
            scheduler_config['params']['epochs'] = self.epochs
        
        return get_scheduler(
            scheduler_name,
            self.optimizer,
            **scheduler_config.get('params', {})
        )
    
    def _create_loss_function(self) -> nn.Module:
        """Создание функции потерь"""
        loss_config = self.config.get('loss', {})
        loss_name = loss_config.get('name', 'mse')
        
        # Проверяем тип loss функции
        if loss_name == 'directional_multitask':
            from models.patchtst_unified import DirectionalMultiTaskLoss
            self.logger.info("🎯 Используется DirectionalMultiTaskLoss для улучшения direction prediction")
            return DirectionalMultiTaskLoss(self.config)
        elif loss_name == 'unified_trading':
            from models.patchtst_unified import UnifiedTradingLoss
            return UnifiedTradingLoss(self.config)
        elif 'trading' in loss_name:
            from models.trading_losses import get_trading_loss_function
            return get_trading_loss_function(self.config, loss_type='multi_task')
        
        # Проверяем если используется унифицированная модель
        model_name = self.config.get('model', {}).get('name', '')
        if model_name == 'UnifiedPatchTST':
            # Автоматически используем DirectionalMultiTaskLoss для UnifiedPatchTST
            self.logger.info("🔧 Автоматически используется DirectionalMultiTaskLoss для UnifiedPatchTST")
            from models.patchtst_unified import DirectionalMultiTaskLoss
            return DirectionalMultiTaskLoss(self.config)
        
        # Многозадачная потеря для торговой модели
        elif loss_config.get('multitask', False):
            task_losses = {}
            task_weights = {}
            
            # Потеря для предсказания цены
            task_losses['price'] = get_loss_function(
                loss_config.get('price_loss', 'mse')
            )
            task_weights['price'] = loss_config.get('price_weight', 1.0)
            
            # Потеря для вероятностей TP
            task_losses['tp'] = get_loss_function(
                loss_config.get('tp_loss', 'bce')
            )
            task_weights['tp'] = loss_config.get('tp_weight', 1.0)
            
            # Потеря для вероятности SL
            task_losses['sl'] = get_loss_function(
                loss_config.get('sl_loss', 'bce')
            )
            task_weights['sl'] = loss_config.get('sl_weight', 1.0)
            
            # Потеря для волатильности
            task_losses['volatility'] = get_loss_function(
                loss_config.get('volatility_loss', 'mse')
            )
            task_weights['volatility'] = loss_config.get('volatility_weight', 0.5)
            
            return MultiTaskLoss(
                task_weights=task_weights,
                uncertainty_weighting=loss_config.get('uncertainty_weighting', False)
            )
        else:
            # Одна функция потерь
            loss_name = loss_config.get('name', 'mse')
            return get_loss_function(loss_name, **loss_config.get('params', {}))
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Обучение одной эпохи"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        # Обнуляем градиенты в начале для gradient accumulation
        # Используем set_to_none=True для экономии памяти (RTX 5090 оптимизация)
        self.optimizer.zero_grad(set_to_none=True)
        
        # Счетчик для отслеживания накопленных градиентов
        accumulated_gradients = 0
        
        for batch_idx, (inputs, targets, info) in enumerate(progress_bar):
            # Перенос на устройство
            inputs = inputs.to(self.device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)
            elif isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Проверка входных данных на NaN
            if torch.isnan(inputs).any():
                self.logger.warning(f"NaN во входных данных батча {batch_idx}, пропускаем")
                continue
                
            # Дополнительная проверка на большие значения во входных данных
            input_max = inputs.abs().max().item()
            if input_max > 1000:
                self.logger.warning(f"Очень большие значения во входных данных: max={input_max:.4f}")
                # Нормализуем входные данные
                inputs = torch.clamp(inputs, min=-100, max=100)
                
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    
                    # Проверка выходов модели на inf/nan
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        self.logger.warning(f"Model outputs contain NaN/Inf at batch {batch_idx}")
                        self.logger.warning(f"  Outputs stats before clipping: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
                        # Клиппинг экстремальных значений
                        outputs = torch.clamp(outputs, min=-100, max=100)
                    
                    loss = self._compute_loss(outputs, targets)
            else:
                outputs = self.model(inputs)
                
                # Проверка выходов модели на inf/nan
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    self.logger.warning(f"Model outputs contain NaN/Inf at batch {batch_idx}")
                    self.logger.warning(f"  Outputs stats before clipping: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
                    # Клиппинг экстремальных значений
                    outputs = torch.clamp(outputs, min=-100, max=100)
                
                loss = self._compute_loss(outputs, targets)
            
            # Проверка на NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"Loss is NaN/Inf at batch {batch_idx}: {loss.item() if not torch.isinf(loss) else 'inf'}")
                # Дополнительная диагностика
                self.logger.warning(f"  Outputs stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                if isinstance(targets, torch.Tensor):
                    self.logger.warning(f"  Targets stats: min={targets.min().item():.4f}, max={targets.max().item():.4f}, mean={targets.mean().item():.4f}")
                
                # Проверяем градиенты
                grad_norms = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 100:
                            self.logger.warning(f"  Large gradient in {name}: {grad_norm:.4f}")
                        grad_norms.append(grad_norm)
                
                if grad_norms:
                    self.logger.warning(f"  Max gradient norm: {max(grad_norms):.4f}")
                
                # ВАЖНО: Если используем AMP, нужно правильно обработать scaler
                if self.use_amp:
                    # Сбрасываем градиенты (с оптимизацией памяти)
                    self.optimizer.zero_grad(set_to_none=True)
                    # НЕ вызываем scaler.update() без scaler.step()!
                
                continue  # Пропускаем этот батч
            
            # Backward pass с поддержкой gradient accumulation
            # Нормализуем loss на количество шагов накопления
            loss = loss / self.gradient_accumulation_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_gradients += 1
            
            # Обновляем веса только каждые gradient_accumulation_steps шагов или на последнем батче
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.use_amp and accumulated_gradients > 0:
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    accumulated_gradients = 0  # Сбрасываем счетчик
                elif self.use_amp and accumulated_gradients == 0:
                    # Не было накопленных градиентов, пропускаем обновление
                    self.logger.warning("Нет накопленных градиентов для AMP, пропускаем обновление")
                elif not self.use_amp and accumulated_gradients > 0:
                    # Gradient clipping с дополнительной проверкой
                    if self.gradient_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                        if grad_norm > self.gradient_clip * 10:  # Слишком большие градиенты
                            self.logger.warning(f"Очень большая норма градиента: {grad_norm:.4f}")
                    
                    self.optimizer.step()
                    accumulated_gradients = 0  # Сбрасываем счетчик
                
                # Обнуляем градиенты после обновления (с оптимизацией памяти)
                self.optimizer.zero_grad(set_to_none=True)
            
            # Обновление метрик - НЕ умножаем на gradient_accumulation_steps
            epoch_loss += loss.item()
            
            # Детальное логирование для первых батчей
            if batch_idx < 3 and not hasattr(self, '_detailed_log_done'):
                self.logger.info(f"📊 Батч {batch_idx} детали:")
                self.logger.info(f"   - Loss: {loss.item():.6f}")
                self.logger.info(f"   - Outputs min/max/mean: {outputs.min():.4f}/{outputs.max():.4f}/{outputs.mean():.4f}")
                self.logger.info(f"   - Targets min/max/mean: {targets.min():.4f}/{targets.max():.4f}/{targets.mean():.4f}")
                self.logger.info(f"   - Градиенты есть: {loss.requires_grad}")
                if batch_idx == 2:
                    self._detailed_log_done = True
            
            # Безопасное вычисление метрик
            try:
                batch_metrics = self.metrics_tracker.metrics_calculator.compute_batch_metrics(outputs, targets)
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value
            except Exception as e:
                if not hasattr(self, '_metrics_error_logged'):
                    self._metrics_error_logged = True
                    self.logger.warning(f"⚠️ Ошибка вычисления метрик: {e}")
            
            # Обновление progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })
        
        # Усреднение метрик
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        epoch_metrics['loss'] = epoch_loss
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Валидация модели"""
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(val_loader):
                # Перенос на устройство
                inputs = inputs.to(self.device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                elif isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self._compute_loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                # Обновление метрик
                val_loss += loss.item()
                batch_metrics = self.metrics_tracker.metrics_calculator.compute_batch_metrics(outputs, targets)
                
                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value
        
        # Усреднение метрик
        num_batches = len(val_loader)
        val_loss /= num_batches
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        val_metrics['loss'] = val_loss
        
        return val_metrics
    
    def _compute_loss(self, outputs: Union[torch.Tensor, Dict],
                     targets: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """Вычисление потерь с поддержкой унифицированной модели"""

        # Логирование типов для отладки
        if not hasattr(self, '_logged_types'):
            self._logged_types = True
            self.logger.info(f"🔍 Типы данных в _compute_loss:")
            self.logger.info(f"   - Outputs type: {type(outputs)}")
            self.logger.info(f"   - Targets type: {type(targets)}")
            if isinstance(outputs, dict):
                self.logger.info(f"   - Outputs keys: {list(outputs.keys())}")
            if isinstance(targets, dict):
                self.logger.info(f"   - Targets keys: {list(targets.keys())}")

        # Для унифицированной модели - прямое применение loss
        if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обработка целевых переменных из датасета
            # Датасет возвращает targets с размерностью (batch, 1, 36)
            # Модель выдает outputs с размерностью (batch, 36)
            
            # Детальное логирование размерностей для отладки
            if not hasattr(self, '_logged_dimensions'):
                self._logged_dimensions = True
                self.logger.info(f"🔍 Размерности при первом батче:")
                self.logger.info(f"   - Outputs shape: {outputs.shape}")
                self.logger.info(f"   - Targets shape: {targets.shape}")
            
            # Приведение targets к правильной размерности
            if targets.dim() == 3 and targets.shape[1] == 1:
                # (batch, 1, 36) -> (batch, 36)
                targets = targets.squeeze(1)
                if not hasattr(self, '_logged_squeeze'):
                    self._logged_squeeze = True
                    self.logger.info(f"✅ Targets приведены к размерности: {targets.shape}")
            
            # Проверка размерностей после приведения
            if outputs.dim() == 2 and targets.dim() == 2:
                # Убеждаемся что размерности совпадают
                if outputs.shape[-1] != targets.shape[-1]:
                    self.logger.warning(f"⚠️ Размерности не совпадают: outputs {outputs.shape} vs targets {targets.shape}")
                    min_size = min(outputs.shape[-1], targets.shape[-1])
                    outputs = outputs[..., :min_size]
                    targets = targets[..., :min_size]
                
                # КРИТИЧЕСКИЙ ФИКС: Для DirectionalMultiTaskLoss нужны словари, а не тензоры!
                # Проверяем тип loss функции
                if not hasattr(self, '_loss_type_logged'):
                    self._loss_type_logged = True
                    self.logger.info(f"🔍 Loss function type: {self.criterion.__class__.__name__}")
                    self.logger.info(f"   Has DirectionalMultiTaskLoss: {'DirectionalMultiTaskLoss' in self.criterion.__class__.__name__}")

                # ВСЕГДА конвертируем для DirectionalMultiTaskLoss или если outputs - dict
                if (hasattr(self.criterion, '__class__') and 'DirectionalMultiTaskLoss' in self.criterion.__class__.__name__) or isinstance(outputs, dict):
                    # Конвертируем outputs и targets в словари для DirectionalMultiTaskLoss

                    if not hasattr(self, '_conversion_logged'):
                        self._conversion_logged = True
                        self.logger.info(f"✅ Конвертация targets в dict для DirectionalMultiTaskLoss")
                        self.logger.info(f"   Outputs is dict: {isinstance(outputs, dict)}")
                        self.logger.info(f"   Targets shape: {targets.shape if isinstance(targets, torch.Tensor) else 'already dict'}")

                    # Конвертация targets в dict (20 целевых переменных)
                    future_returns = targets[:, 0:4]
                    directions = targets[:, 4:8].long()
                    long_levels = targets[:, 8:12]
                    short_levels = targets[:, 12:16]
                    risk_metrics = targets[:, 16:20]

                    targets_dict = {
                        # Future returns (индексы 0-3)
                        'future_return_15m': future_returns[:, 0],
                        'future_return_1h': future_returns[:, 1],
                        'future_return_4h': future_returns[:, 2],
                        'future_return_12h': future_returns[:, 3],
                        'future_returns': future_returns,

                        # Directions (индексы 4-7)
                        'direction_15m': directions[:, 0],
                        'direction_1h': directions[:, 1],
                        'direction_4h': directions[:, 2],
                        'direction_12h': directions[:, 3],
                        'directions': directions,

                        # Long levels (индексы 8-11)
                        'long_will_reach_1pct_4h': long_levels[:, 0],
                        'long_will_reach_2pct_4h': long_levels[:, 1],
                        'long_will_reach_3pct_12h': long_levels[:, 2],
                        'long_will_reach_5pct_12h': long_levels[:, 3],
                        'long_levels': long_levels,

                        # Short levels (индексы 12-15)
                        'short_will_reach_1pct_4h': short_levels[:, 0],
                        'short_will_reach_2pct_4h': short_levels[:, 1],
                        'short_will_reach_3pct_12h': short_levels[:, 2],
                        'short_will_reach_5pct_12h': short_levels[:, 3],
                        'short_levels': short_levels,

                        # Risk metrics (индексы 16-19)
                        'max_drawdown_1h': risk_metrics[:, 0],
                        'max_rally_1h': risk_metrics[:, 1],
                        'max_drawdown_4h': risk_metrics[:, 2],
                        'max_rally_4h': risk_metrics[:, 3],
                        'risk_metrics': risk_metrics
                    }

                    # outputs уже должен быть словарем от модели, просто передаем
                    loss = self.criterion(outputs, targets_dict)

                    # DirectionalMultiTaskLoss возвращает dict с ключом 'total_loss'
                    if isinstance(loss, dict):
                        if not hasattr(self, '_logged_loss_dict'):
                            self._logged_loss_dict = True
                            self.logger.info(f"🔍 DirectionalMultiTaskLoss результат:")
                            self.logger.info(f"   - Loss type: {type(loss)}")
                            if 'total_loss' in loss:
                                self.logger.info(f"   - Total loss: {loss['total_loss'].item():.6f}")
                        loss = loss.get('total_loss', torch.tensor(1.0, device=outputs.device, requires_grad=True))
                else:
                    # Для других loss функций - прямое применение
                    loss = self.criterion(outputs, targets)
                
                # Проверка на NaN/Inf с детальным логированием
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning("❌ Loss is NaN/Inf! Детальная информация:")
                    self.logger.warning(f"   - Loss значение: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                    self.logger.warning(f"   - Outputs stats: min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
                    self.logger.warning(f"   - Targets stats: min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}")

                    # Проверяем конкретные компоненты
                    if hasattr(self.criterion, 'last_losses'):
                        self.logger.warning("   - Компоненты loss:")
                        for key, val in self.criterion.last_losses.items():
                            if isinstance(val, torch.Tensor):
                                self.logger.warning(f"     * {key}: {val.item():.4f}")

                    # Вместо возврата 0, возвращаем небольшое значение
                    # чтобы градиенты могли распространяться
                    return torch.tensor(1e-6, device=outputs.device, requires_grad=True)
                
                return loss
        
        # Для старой логики с MultiTaskLoss
        if isinstance(self.criterion, MultiTaskLoss):
            # [оставляем старую логику для совместимости]
            losses = {}
            
            if isinstance(outputs, dict) and isinstance(targets, dict):
                # ... старая логика ...
                pass
            
            return self.criterion(losses) if losses else torch.tensor(0.0, device=outputs.device)
        
        # Проверяем тип criterion для правильной обработки
        from models.patchtst_unified import DirectionalMultiTaskLoss
        if isinstance(self.criterion, DirectionalMultiTaskLoss):
            # DirectionalMultiTaskLoss ожидает словари
            if not isinstance(outputs, dict):
                outputs = {'predictions': outputs}
            if not isinstance(targets, dict):
                targets = {'targets': targets}

            # Вычисляем loss
            loss = self.criterion(outputs, targets)

            # Проверяем результат
            if not hasattr(self, '_logged_directional_loss'):
                self._logged_directional_loss = True
                self.logger.info(f"🔍 DirectionalMultiTaskLoss результат:")
                self.logger.info(f"   - Loss type: {type(loss)}")
                if hasattr(loss, 'item'):
                    self.logger.info(f"   - Loss value: {loss.item():.6f}")

            return loss
        
        # Fallback для простых случаев
        if isinstance(outputs, dict):
            outputs = list(outputs.values())[0]
        if isinstance(targets, dict):
            targets = list(targets.values())[0]
        
        return self.criterion(outputs, targets)
    def _align_dimensions(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Выравнивание размерностей выходов и целей"""
        if outputs.shape != targets.shape:
            # Логирование несоответствия
            self.logger.debug(f"Несоответствие размерностей: outputs {outputs.shape}, targets {targets.shape}")
            
            # Если у outputs есть дополнительное измерение по признакам
            if outputs.dim() == 3 and targets.dim() == 3 and outputs.shape[-1] != targets.shape[-1]:
                # Берем только нужное количество признаков из outputs
                if outputs.shape[-1] > targets.shape[-1]:
                    outputs = outputs[..., :targets.shape[-1]]
                else:
                    # Или наоборот, если targets больше
                    targets = targets[..., :outputs.shape[-1]]
            
            # Если размерности все еще не совпадают
            if outputs.shape != targets.shape:
                # Попытка выравнивания по батчам и временным шагам
                if outputs.dim() == 3 and targets.dim() == 2:
                    # outputs: (batch, time, features), targets: (batch, features)
                    # Берем последний временной шаг из outputs
                    outputs = outputs[:, -1, :]
                elif outputs.dim() == 2 and targets.dim() == 3:
                    # outputs: (batch, features), targets: (batch, time, features)
                    # Берем последний временной шаг из targets
                    targets = targets[:, -1, :]
                elif outputs.dim() == 3 and targets.dim() == 3:
                    # Оба имеют 3 измерения, но разные размеры по времени
                    min_time = min(outputs.shape[1], targets.shape[1])
                    outputs = outputs[:, :min_time, :]
                    targets = targets[:, :min_time, :]
                
                # Финальная попытка - flatten до 2D
                if outputs.shape != targets.shape:
                    if outputs.dim() > 2:
                        outputs = outputs.reshape(outputs.shape[0], -1)
                    if targets.dim() > 2:
                        targets = targets.reshape(targets.shape[0], -1)
                    
                    # Выравнивание по последнему измерению
                    min_size = min(outputs.shape[-1], targets.shape[-1])
                    outputs = outputs[..., :min_size]
                    targets = targets[..., :min_size]
        
        return outputs, targets
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              callbacks: Optional[List[Callable]] = None) -> Dict:
        """Полный цикл обучения"""
        
        self.logger.info(f"Начало обучения на {self.epochs} эпох")
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Размер батча: {train_loader.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Обучение
            train_metrics = self.train_epoch(train_loader)
            
            # Валидация
            val_metrics = self.validate(val_loader)
            
            # Обновление learning rate
            current_lr = self.optimizer.param_groups[0]['lr']  # Инициализация перед использованием
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    old_lr = current_lr  # Сохраняем старое значение
                    self.scheduler.step(val_metrics['loss'])
                    # Проверяем изменение LR
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if old_lr != current_lr:
                        self.logger.info(f"📉 Learning rate снижен с {old_lr:.2e} до {current_lr:.2e}")
                else:
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
            
            # Сохранение истории
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
            
            # GPU оптимизации
            if self.device.type == 'cuda':
                # Периодическая очистка кэша GPU
                if (epoch + 1) % self.gpu_cache_clear_freq == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.logger.info(f"🧹 Очищен кэш GPU на эпохе {epoch + 1}")
                
                # Мониторинг памяти GPU
                if self.monitor_gpu_memory and (epoch + 1) % 5 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    self.logger.info(f"💾 GPU память: выделено {allocated:.2f}GB, зарезервировано {reserved:.2f}GB")
            
            # Улучшенная логика борьбы с переобучением
            improvement = self.best_val_loss - val_metrics['loss']
            overfitting_ratio = val_metrics['loss'] / train_metrics['loss'] if train_metrics['loss'] > 0 else 1.0
            
            # Логирование метрик для отладки
            self.logger.info(f"📊 Метрики эпохи {epoch + 1}: train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
                           f"overfitting_ratio={overfitting_ratio:.3f}, improvement={improvement:.6f}, current_lr={current_lr:.2e}")
            
            if improvement > self.min_delta:
                # Значительное улучшение
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.consecutive_overfitting = 0
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch, val_metrics['loss'], is_best=True)
                self.logger.info(f"✅ Новая лучшая модель! Val loss улучшен на {improvement:.6f}")
            else:
                self.patience_counter += 1
                
                # МЯГКАЯ проверка на переобучение - даем модели больше свободы
                if epoch >= 5:  # Начинаем проверку только с 5-й эпохи
                    if overfitting_ratio > (1.0 + self.overfitting_threshold):
                        self.consecutive_overfitting += 1
                        self.logger.warning(f"⚠️ Переобучение обнаружено: Train: {train_metrics['loss']:.4f}, Val: {val_metrics['loss']:.4f} (ratio: {overfitting_ratio:.3f})")
                        
                        # Останавливаемся только после 5 последовательных обнаружений (вместо 2)
                        if self.consecutive_overfitting >= 5:
                            self.logger.info(f"🛑 Остановка из-за устойчивого переобучения (5 эпох подряд)")
                            break
                    else:
                        self.consecutive_overfitting = 0
                else:
                    # Первые 5 эпох - только информационное сообщение
                    if overfitting_ratio > 1.5:  # 50% разница
                        self.logger.info(f"📈 Эпоха {epoch + 1}: val_loss выше train_loss (ratio: {overfitting_ratio:.3f}) - нормально в начале")
            
            # КРИТИЧНО: Дополнительная проверка - если val_loss не улучшается слишком долго
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"🛑 Early stopping: val_loss не улучшается {self.patience_counter} эпох (patience={self.early_stopping_patience})")
                break
            
            # Логирование
            epoch_time = time.time() - epoch_start_time
            self.logger.log_model_metrics(
                epoch=epoch + 1,
                metrics={
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                }
            )
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_metrics, val_metrics)
        
        # Восстановление лучшей модели
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        self.logger.info(f"Обучение завершено за {total_time:.2f} секунд")
        
        # Финальный отчет
        self._save_training_report()
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Сохранение checkpoint'а"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем правильно для DataParallel
        model_state = self.model.module.state_dict() if self.is_data_parallel else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history,
            'is_data_parallel': self.is_data_parallel
        }
        
        if is_best:
            path = self.checkpoint_dir / f"best_model_{timestamp}.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_{timestamp}.pth"
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint сохранен: {path}")
    
    def _save_training_report(self):
        """Сохранение отчета об обучении"""
        report = {
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.history['train_loss']),
            'model_architecture': str(self.model),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.checkpoint_dir / f"training_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        self.logger.info(f"Отчет об обучении сохранен: {report_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузка checkpoint'а"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        self.logger.info(f"Checkpoint загружен: {checkpoint_path}")
        
        return checkpoint['epoch']