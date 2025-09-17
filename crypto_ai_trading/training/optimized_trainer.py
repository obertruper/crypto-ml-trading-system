"""
Оптимизированный Trainer для максимальной утилизации GPU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import time
from tqdm import tqdm
from collections import deque

from utils.logger import get_logger
from training.trainer import Trainer
import torch.nn.functional as F

class AntiCollapseMonitor:
    """
    🚨 Система мониторинга и предотвращения схлопывания FLAT класса
    
    Особенности:
    - Мониторинг распределения классов каждые N батчей
    - Автоматическая коррекция параметров
    - Адаптивное управление learning rate
    """
    
    def __init__(self, config):
        self.logger = get_logger("AntiCollapseMonitor")
        self.config = config
        
        # Пороги для обнаружения схлопывания (смягченные для поэтапного обучения)
        self.min_flat_ratio = 0.05  # Минимальная доля FLAT (5%) - очень мягко
        self.max_single_class_ratio = 0.85  # Максимальная доля одного класса (85%)
        self.min_entropy = 0.3  # Минимальная энтропия (снижено)
        
        # Счетчики статистики
        self.reset_epoch_stats()
        
        # Параметры коррекции
        self.correction_applied = False
        self.correction_count = 0
        self.max_corrections_per_epoch = 3  # Максимум коррекций за эпоху
    
    def reset_epoch_stats(self):
        """Сброс статистики на новую эпоху"""
        self.epoch_class_counts = {'long': 0, 'short': 0, 'flat': 0}
        self.epoch_total_samples = 0
        self.collapse_alerts = 0
        self.correction_applied = False
        self.correction_count = 0
        self.entropy_history = deque(maxlen=10)
    
    def check_batch_predictions(self, outputs, targets, batch_idx):
        """
        Проверка распределения классов в текущем батче
        
        Returns:
            bool: True если обнаружено схлопывание
        """
        # Извлекаем direction предсказания для 15m (4-я колонка)
        if hasattr(outputs, '_direction_logits'):
            direction_logits = outputs._direction_logits[:, 0, :]  # 15m timeframe
            probs = torch.softmax(direction_logits, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)
        else:
            # Fallback: используем пороги из основных outputs
            # outputs может быть dict или tensor
            if isinstance(outputs, dict):
                # Если dict, берем direction_15m логиты
                if 'direction_15m' in outputs:
                    direction_logits = outputs['direction_15m']  # [batch_size, 3]
                    pred_classes = torch.argmax(direction_logits, dim=-1)  # [batch_size]
                else:
                    # Fallback на старый формат
                    pred_classes = torch.zeros(outputs[list(outputs.keys())[0]].shape[0], dtype=torch.long)
            elif hasattr(outputs, 'prediction'):
                pred_tensor = outputs.prediction
                # Проверяем что pred_tensor действительно тензор
                if isinstance(pred_tensor, torch.Tensor):
                    if pred_tensor.shape[-1] == 20:
                        # Старый формат с 20 outputs
                        pred_values = pred_tensor[:, 4]  # direction_15m
                        pred_classes = torch.zeros_like(pred_values, dtype=torch.long)
                        pred_classes[pred_values < 0.67] = 0  # LONG
                        pred_classes[(pred_values >= 0.67) & (pred_values < 1.33)] = 1  # SHORT
                        pred_classes[pred_values >= 1.33] = 2  # FLAT
                    else:
                        pred_classes = torch.zeros(pred_tensor.shape[0], dtype=torch.long)
                else:
                    # Если не тензор, создаем заглушку
                    batch_size = targets['direction_15m'].shape[0] if isinstance(targets, dict) else targets.shape[0]
                    pred_classes = torch.zeros(batch_size, dtype=torch.long)
            else:
                # Если outputs - обычный тензор
                if isinstance(outputs, torch.Tensor):
                    pred_tensor = outputs
                    if len(pred_tensor.shape) > 1 and pred_tensor.shape[-1] >= 5:
                        pred_values = pred_tensor[:, 4]  # direction_15m
                        pred_classes = torch.zeros_like(pred_values, dtype=torch.long)
                        pred_classes[pred_values < 0.67] = 0  # LONG
                        pred_classes[(pred_values >= 0.67) & (pred_values < 1.33)] = 1  # SHORT
                        pred_classes[pred_values >= 1.33] = 2  # FLAT
                    else:
                        pred_classes = torch.zeros(pred_tensor.shape[0], dtype=torch.long)
                else:
                    # Если не тензор, создаем заглушку
                    batch_size = targets['direction_15m'].shape[0] if isinstance(targets, dict) else targets.shape[0]
                    pred_classes = torch.zeros(batch_size, dtype=torch.long)
        
        # Подсчитываем распределение
        batch_size = pred_classes.shape[0]
        long_count = (pred_classes == 0).sum().item()
        short_count = (pred_classes == 1).sum().item()
        flat_count = (pred_classes == 2).sum().item()
        
        # Обновляем общую статистику
        self.epoch_class_counts['long'] += long_count
        self.epoch_class_counts['short'] += short_count
        self.epoch_class_counts['flat'] += flat_count
        self.epoch_total_samples += batch_size
        
        # Рассчитываем текущие соотношения
        if self.epoch_total_samples > 0:
            long_ratio = self.epoch_class_counts['long'] / self.epoch_total_samples
            short_ratio = self.epoch_class_counts['short'] / self.epoch_total_samples
            flat_ratio = self.epoch_class_counts['flat'] / self.epoch_total_samples
        else:
            return False
        
        # Рассчитываем энтропию
        ratios = torch.tensor([long_ratio, short_ratio, flat_ratio])
        ratios = ratios + 1e-8  # Избегаем log(0)
        entropy = -(ratios * torch.log(ratios)).sum().item()
        normalized_entropy = entropy / np.log(3)  # Нормализуем
        self.entropy_history.append(normalized_entropy)
        
        # Логирование каждые 50 батчей
        if batch_idx % 50 == 0:
            self.logger.info(
                f"📈 Батч {batch_idx}: LONG={long_ratio:.1%} SHORT={short_ratio:.1%} "
                f"FLAT={flat_ratio:.1%} | Энтропия={normalized_entropy:.3f}"
            )
        
        # Проверка на схлопывание
        collapse_detected = False
        
        # 1. Проверка FLAT класса
        if flat_ratio < self.min_flat_ratio:
            self.logger.warning(
                f"⚠️ FLAT схлопывание: {flat_ratio:.1%} < {self.min_flat_ratio:.1%}"
            )
            collapse_detected = True
            
        # 2. Проверка доминирования одного класса
        max_ratio = max(long_ratio, short_ratio, flat_ratio)
        if max_ratio > self.max_single_class_ratio:
            dominant_class = ['LONG', 'SHORT', 'FLAT'][np.argmax([long_ratio, short_ratio, flat_ratio])]
            self.logger.warning(
                f"⚠️ Доминирование {dominant_class}: {max_ratio:.1%} > {self.max_single_class_ratio:.1%}"
            )
            collapse_detected = True
            
        # 3. Проверка энтропии
        if normalized_entropy < self.min_entropy:
            self.logger.warning(
                f"⚠️ Низкая энтропия: {normalized_entropy:.3f} < {self.min_entropy:.3f}"
            )
            collapse_detected = True
        
        if collapse_detected:
            self.collapse_alerts += 1
            
        return collapse_detected
    
    def apply_correction(self, optimizer, scheduler, model):
        """
        Применение автоматической коррекции
        
        Returns:
            bool: True если коррекция применена
        """
        if self.correction_count >= self.max_corrections_per_epoch:
            self.logger.info("⚠️ Максимум коррекций за эпоху достигнут")
            return False
            
        self.correction_count += 1
        
        # 1. Снижаем learning rate на 20%
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            # Ограничиваем минимальный LR
            min_lr = 1e-7
            new_lr = max(param_group['lr'] * 0.8, min_lr)
            param_group['lr'] = new_lr
            self.logger.info(f"🔧 LR снижен: {old_lr:.6f} → {param_group['lr']:.6f}")
        
        # 2. Применяем gradient clipping маленький
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        self.logger.info("🔧 Применен мягкий gradient clipping (0.5)")
        
        # 3. Добавляем шум к direction head для FLAT класса
        if hasattr(model, 'direction_heads'):
            with torch.no_grad():
                for head in model.direction_heads:
                    # Добавляем маленький шум в bias для FLAT класса (2-й индекс)
                    if hasattr(head, 'bias') and head.bias is not None:
                        noise = torch.randn_like(head.bias[2:3]) * 0.01  # Маленький шум
                        head.bias[2:3] += noise  # Только для FLAT
            self.logger.info("🔧 Добавлен шум в direction heads для FLAT")
        
        self.correction_applied = True
        return True


class OptimizedTrainer(Trainer):
    """Оптимизированная версия Trainer с максимальной производительностью GPU"""
    
    def __init__(self, model: nn.Module, config: Dict, device: Optional[torch.device] = None):
        super().__init__(model, config, device)
        
        self.logger = get_logger("OptimizedTrainer")
        
        # Применяем заморозку/разморозку модулей согласно конфигу и пересоздаем оптимизатор/шедулер
        self._apply_freeze_from_config()
        # Переинициализируем оптимизатор/шедулер, чтобы учитывать requires_grad
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # ВАЖНО: Всегда пересоздаем scaler после создания нового оптимизатора
        # Scaler должен быть связан с текущим оптимизатором
        if self.use_amp:
            # Удаляем старый scaler если он есть
            if hasattr(self, 'scaler'):
                del self.scaler
            # Создаем новый scaler
            self.scaler = GradScaler()
            # Устанавливаем флаг для отслеживания первого шага
            self._scaler_initialized = False
            self.logger.debug("✅ GradScaler пересоздан для нового оптимизатора")

        # Оптимизации для GPU
        self.async_metrics = True  # Асинхронный расчет метрик
        self.log_interval = 10  # Логировать каждые N батчей
        self.metrics_buffer = deque(maxlen=100)  # Буфер для метрик
        
        # 🚨 НОВАЯ СИСТЕМА: Anti-Collapse мониторинг и коррекция
        self.anti_collapse_monitor = AntiCollapseMonitor(config)
        self.class_monitoring_freq = 25  # Проверять каждые 25 батчей
        self.last_class_check_batch = 0
        
        # EMA (Exponential Moving Average) для весов модели - ОТКЛЮЧЕНО
        self.use_ema = False  # Принудительно отключено для предотвращения маскирования переобучения
        self.ema_decay = config.get('model', {}).get('ema_decay', 0.99)  # Снижен decay
        self.ema_model = None
        if self.use_ema:
            self._init_ema()
            self.logger.info(f"✅ EMA включен с decay={self.ema_decay}")
        
        # Dropout schedule параметры - ОТКЛЮЧЕНО
        self.use_dropout_schedule = False  # Принудительно отключено
        if self.use_dropout_schedule:
            self.initial_dropout = config.get('model', {}).get('dropout', 0.3)
            self.final_dropout = 0.1  # Минимальный dropout
            self.dropout_warmup_epochs = 20  # Эпохи для снижения dropout
            self.logger.info(f"✅ Dropout Schedule включен: {self.initial_dropout} → {self.final_dropout}")
        
        # Mixup augmentation для direction задачи - ОТКЛЮЧЕНО
        self.use_mixup = False  # Принудительно отключено для многозадачного обучения
        if self.use_mixup:
            self.mixup_alpha = config.get('model', {}).get('mixup_alpha', 0.2)
            self.logger.info(f"✅ Mixup augmentation включен: alpha={self.mixup_alpha}")
        
        # Компиляция модели для ускорения (PyTorch 2.0+)
        if hasattr(torch, 'compile') and config.get('model', {}).get('compile_model', False):
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                if 'RTX 5090' in gpu_name:
                    self.logger.warning("⚠️ torch.compile отключен для RTX 5090 (sm_120) - не поддерживается текущей версией PyTorch")
                else:
                    self.logger.info("🚀 Компиляция модели с torch.compile...")
                    self.model = torch.compile(self.model, mode='reduce-overhead')
            else:
                self.logger.info("🚀 Компиляция модели с torch.compile...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
        
        # CUDA оптимизации
        if self.device.type == 'cuda':
            # Включаем TensorFloat-32 для ускорения на Ampere+ (RTX 30xx и выше)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Benchmark mode для cudnn
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            self.logger.info("✅ GPU оптимизации включены:")
            self.logger.info(f"   - TF32: {torch.backends.cuda.matmul.allow_tf32}")
            self.logger.info(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            self.logger.info(f"   - Mixed Precision: {self.use_amp}")

    def _apply_freeze_from_config(self):
        """Заморозка/разморозка параметров модели согласно конфигу этапа.
        
        training.train_only_modules: список имён модулей модели, которые обучаем; остальные — freeze.
        training.freeze_modules: явный список модулей для заморозки.
        """
        training_cfg = self.config.get('training', {})
        train_only = training_cfg.get('train_only_modules')
        freeze_modules = training_cfg.get('freeze_modules', [])

        if train_only is None and not freeze_modules:
            return

        # Утилита: установить requires_grad для всех параметров модуля
        def set_requires_grad(module: nn.Module, flag: bool):
            for p in module.parameters():
                p.requires_grad = flag

        # По умолчанию размораживаем всё
        set_requires_grad(self.model, True)

        if isinstance(train_only, list):
            # Замораживаем всё кроме указанных модулей
            set_requires_grad(self.model, False)
            
            # ВАЖНО: всегда оставляем backbone активным для градиентов
            # чтобы избежать ошибки "does not require grad"
            if hasattr(self.model, 'backbone'):
                set_requires_grad(self.model.backbone, True)
            
            for name in train_only:
                if hasattr(self.model, name):
                    set_requires_grad(getattr(self.model, name), True)
                else:
                    self.logger.warning(f"⚠️ Модуль '{name}' не найден в модели — пропускаем")
            self.logger.info(f"🔒 Обучаем только модули: {train_only} + backbone")

        # Дополнительно замораживаем явно указанные модули
        if isinstance(freeze_modules, list) and freeze_modules:
            for name in freeze_modules:
                if hasattr(self.model, name):
                    set_requires_grad(getattr(self.model, name), False)
                else:
                    self.logger.warning(f"⚠️ Модуль для заморозки '{name}' не найден в модели")
            self.logger.info(f"🧊 Заморожены модули: {freeze_modules}")
    
    def update_dropout(self, epoch: int):
        """Обновляет dropout rate согласно расписанию"""
        if not self.use_dropout_schedule:
            return
            
        if epoch < self.dropout_warmup_epochs:
            # Линейное снижение dropout
            progress = epoch / self.dropout_warmup_epochs
            current_dropout = self.initial_dropout - (self.initial_dropout - self.final_dropout) * progress
            
            # Обновляем dropout во всех слоях модели
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = current_dropout
                    
            if epoch % 5 == 0:
                self.logger.info(f"📊 Dropout обновлен: {current_dropout:.3f}")
    
    def _init_ema(self):
        """Инициализация EMA модели"""
        import copy
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def _update_ema(self):
        """Обновление весов EMA модели"""
        if not self.use_ema or self.ema_model is None:
            return
            
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
        """Применяет mixup augmentation к данным"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Оптимизированное обучение одной эпохи с Anti-Collapse системой"""
        self.model.train()
        
        # Критическое логирование для отладки
        self.logger.info(f"🔍 train_epoch начат. use_amp={self.use_amp}, gradient_accumulation_steps={self.gradient_accumulation_steps}")
        self.logger.info(f"🔍 DataLoader: {len(train_loader)} батчей, batch_size={train_loader.batch_size}")
        
        epoch_loss = torch.tensor(0.0, device=self.device)  # Инициализируем как тензор на устройстве
        epoch_metrics = {}
        batch_times = deque(maxlen=50)
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        # Инициализация мониторинга классов в начале эпохи
        self.anti_collapse_monitor.reset_epoch_stats()
        
        # Обнуление градиентов с оптимизацией памяти
        self.optimizer.zero_grad(set_to_none=True)
        
        # НЕ пересоздаем scaler здесь - он уже создан в __init__
        # Это критически важно для корректной работы с оптимизатором
        
        # Время начала эпохи
        epoch_start = time.time()
        last_log_time = epoch_start
        
        # Адаптивный learning rate scheduler
        adaptive_lr_applied = False
        
        # Проверяем, что dataloader не пустой
        if len(train_loader) == 0:
            self.logger.warning("⚠️ DataLoader пустой, пропускаем эпоху")
            return {'loss': 0.0, 'collapse_alerts': 0, 'corrections_applied': 0}
        
        batches_processed = 0
        accumulated_steps = 0  # Счетчик накопленных градиентов с последнего optimizer.step()
        
        try:
            for batch_idx, (inputs, targets, info) in enumerate(progress_bar):
                batch_start = time.time()
                
                # Отладочное логирование для первых батчей
                if batch_idx < 3:
                    self.logger.info(f"🔍 Батч {batch_idx}: inputs shape={inputs.shape if hasattr(inputs, 'shape') else 'unknown'}")
                    self.logger.info(f"🔍 gradient_accumulation_steps: {self.gradient_accumulation_steps}, total batches: {len(train_loader)}")
                
                # Асинхронный перенос на GPU с non_blocking
                inputs = inputs.to(self.device, non_blocking=True)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device, non_blocking=True)
                elif isinstance(targets, dict):
                    targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
                
                # Обработка info dict (содержит idx как тензор)
                if isinstance(info, dict) and 'idx' in info and isinstance(info['idx'], torch.Tensor):
                    info['idx'] = info['idx'].to(self.device, non_blocking=True)
                
                # Forward pass с AMP
                if self.use_amp:
                    with autocast():
                        try:
                            outputs = self.model(inputs, return_dict=True)
                        except TypeError:
                            outputs = self.model(inputs)

                        # КРИТИЧЕСКИЙ FIX: Обработка tuple output (tensor, attention_weights)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # Берем только тензор

                        loss = self._compute_loss(outputs, targets)
                else:
                    try:
                        outputs = self.model(inputs, return_dict=True)
                    except TypeError:
                        outputs = self.model(inputs)

                    # КРИТИЧЕСКИЙ FIX: Обработка tuple output (tensor, attention_weights)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Берем только тензор

                    loss = self._compute_loss(outputs, targets)

                # Проверка на NaN/Inf
                loss_to_check = loss['total_loss'] if isinstance(loss, dict) else loss
                if torch.isnan(loss_to_check) or torch.isinf(loss_to_check):
                    self.logger.warning(f"⚠️ NaN/Inf обнаружен в loss на батче {batch_idx}")
                    self.logger.warning(f"   Loss value: {loss_to_check.item()}")
                    # Пропускаем этот батч
                    self.optimizer.zero_grad()
                    continue
                    
                # Проверка outputs на NaN
                outputs_to_check = outputs if isinstance(outputs, torch.Tensor) else None
                if outputs_to_check is not None and torch.isnan(outputs_to_check).any():
                    self.logger.warning(f"⚠️ NaN обнаружен в outputs на батче {batch_idx}")
                    self.logger.warning(f"   NaN count: {torch.isnan(outputs_to_check).sum().item()}")
                    # Пропускаем этот батч
                    self.optimizer.zero_grad()
                    continue
                
                # Нормализация loss для gradient accumulation
                if isinstance(loss, dict):
                    loss_for_backward = loss['total_loss'] / self.gradient_accumulation_steps
                else:
                    loss_for_backward = loss / self.gradient_accumulation_steps
                
                # Backward pass
                # Для первого батча первой эпохи не используем AMP чтобы избежать проблем с инициализацией scaler
                use_amp_for_backward = self.use_amp and not (hasattr(self, '_scaler_initialized') and not self._scaler_initialized and batch_idx == 0)
                
                if use_amp_for_backward:
                    if batch_idx < 3:
                        self.logger.debug(f"🔍 Backward pass с AMP на батче {batch_idx}, loss={loss_for_backward.item():.4f}")
                    self.scaler.scale(loss_for_backward).backward()
                    if batch_idx < 3:
                        self.logger.debug(f"🔍 Backward pass завершен, scaler._scale={self.scaler._scale}")
                else:
                    if batch_idx == 0 and hasattr(self, '_scaler_initialized') and not self._scaler_initialized:
                        self.logger.debug(f"🔧 Первый батч - backward без AMP для инициализации")
                    loss_for_backward.backward()
                
                batches_processed += 1
                accumulated_steps += 1
                
                # Обновление весов каждые N шагов
                should_update = (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader)
                if should_update:
                    if batch_idx < 10:  # Логируем первые несколько обновлений
                        self.logger.info(f"🔍 Optimizer step на батче {batch_idx+1}, accumulated_steps={accumulated_steps}, use_amp={self.use_amp}")
                    
                    # Проверяем, первый ли это шаг с новым оптимизатором
                    is_first_step = hasattr(self, '_scaler_initialized') and not self._scaler_initialized
                    
                    if is_first_step and accumulated_steps > 0:
                        # Первый шаг - не используем scaler
                        self.logger.debug("🔧 Первый шаг оптимизатора - без AMP для инициализации")
                        if self.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                        self.optimizer.step()
                        # Помечаем что scaler инициализирован
                        self._scaler_initialized = True
                    elif self.use_amp and accumulated_steps > 0:
                        try:
                            # Обычный путь с AMP
                            # Unscale перед clipping
                            self.scaler.unscale_(self.optimizer)
                            # 🚨 УЛУЧШЕННЫЙ Gradient clipping с мониторингом
                            if self.gradient_clip > 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                                # Логируем аномальные градиенты
                                if grad_norm > self.gradient_clip * 2:
                                    self.logger.warning(f"⚠️ Высокая норма градиентов: {grad_norm:.4f} (обрезана до {self.gradient_clip})")
                            # Optimizer step
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        except RuntimeError as e:
                            if "No inf checks were recorded" in str(e):
                                # Если не было scale операций, делаем обычный step
                                self.logger.warning(f"⚠️ Scaler ошибка после {accumulated_steps} шагов накопления. Выполняем обычный optimizer step")
                                self.logger.warning(f"   Детали ошибки: {e}")
                                try:
                                    # Пытаемся очистить состояние scaler
                                    if hasattr(self.scaler, '_per_optimizer_states'):
                                        self.scaler._per_optimizer_states.clear()
                                except:
                                    pass  # Игнорируем ошибки очистки
                                
                                try:
                                    if self.gradient_clip > 0:
                                        # Градиенты могли быть unscaled, пробуем клиппировать
                                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                                except:
                                    pass  # Игнорируем ошибки клиппирования
                                
                                # Делаем обычный optimizer step
                                self.optimizer.step()
                                
                                # Пересоздаем scaler после ошибки
                                try:
                                    self.scaler = GradScaler()
                                    self.logger.warning("   Scaler пересоздан после ошибки")
                                except:
                                    self.use_amp = False
                                    self.logger.warning("   Отключаем AMP из-за проблем со scaler")
                            else:
                                raise e
                        # ВСЕГДА сбрасываем счетчик после любого update (успешного или fallback)
                        accumulated_steps = 0
                    elif self.use_amp and accumulated_steps == 0:
                        # Если AMP включен, но не было накопленных градиентов
                        self.logger.warning("⚠️ AMP включен, но нет накопленных градиентов. Пропускаем optimizer step.")
                        # Не вызываем scaler.update() без scaler.step()!
                    else:
                        if accumulated_steps > 0:
                            # 🚨 УЛУЧШЕННЫЙ Gradient clipping с мониторингом
                            if self.gradient_clip > 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                                # Логируем аномальные градиенты
                                if grad_norm > self.gradient_clip * 2:
                                    self.logger.warning(f"⚠️ Высокая норма градиентов: {grad_norm:.4f} (обрезана до {self.gradient_clip})")
                            # Optimizer step
                            self.optimizer.step()
                            accumulated_steps = 0
                            
                            # Обновление EMA модели
                            self._update_ema()
                    
                    # Обнуление градиентов
                    self.optimizer.zero_grad(set_to_none=True)
                
                # Асинхронное накопление метрик (без .item() каждый батч)
                if isinstance(loss, dict):
                    batch_loss = loss['total_loss'].detach()
                else:
                    batch_loss = loss.detach()
                
                # Накапливаем loss БЕЗ умножения на gradient_accumulation_steps
                epoch_loss += batch_loss
                
                # 🚨 НОВОЕ: Мониторинг классового распределения
                if (batch_idx - self.last_class_check_batch) >= self.class_monitoring_freq:
                    collapse_detected = self.anti_collapse_monitor.check_batch_predictions(
                        outputs, targets, batch_idx
                    )
                    
                    if collapse_detected:
                        self.logger.warning(f"⚠️ СХЛОПЫВАНИЕ обнаружено на батче {batch_idx}!")
                        # Применяем коррекцию
                        corrected = self.anti_collapse_monitor.apply_correction(
                            self.optimizer, self.scheduler, self.model
                        )
                        if corrected:
                            self.logger.info("✅ Применена автоматическая коррекция")
                            adaptive_lr_applied = True
                    
                    self.last_class_check_batch = batch_idx
                
                # Время батча
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Периодическое логирование
                if batch_idx % self.log_interval == 0:
                    # Синхронизация для получения актуальных значений
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # Правильное вычисление текущего loss для отображения
                    # batch_loss уже detached, нужно только взять значение
                    if isinstance(batch_loss, torch.Tensor):
                        # НЕ умножаем на gradient_accumulation_steps для отображения!
                        current_loss = batch_loss.item()
                    else:
                        current_loss = float(batch_loss)

                    # Проверка на очень малые значения
                    if current_loss < 1e-8:
                        self.logger.warning(f"⚠️ Loss очень мал: {current_loss:.8f} на батче {batch_idx}")
                    avg_batch_time = np.mean(batch_times)
                    samples_per_sec = train_loader.batch_size / avg_batch_time
                    
                    # Расширенная статистика GPU для RTX 5090
                    if self.device.type == 'cuda':
                        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                        gpu_memory_cached = torch.cuda.memory_cached() / 1024**3
                        
                        # Утилизация памяти (выделенная/зарезервированная)
                        gpu_utilization = (gpu_memory_allocated / gpu_memory_reserved * 100) if gpu_memory_reserved > 0 else 0
                        
                        # Дополнительные метрики для RTX 5090
                        if hasattr(torch.cuda, 'memory_stats'):
                            memory_stats = torch.cuda.memory_stats()
                            allocated_bytes = memory_stats.get("allocated_bytes.all.current", 0)
                            reserved_bytes = memory_stats.get("reserved_bytes.all.current", 0)
                            max_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1024**3
                            max_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1024**3
                            
                            # Эффективность использования памяти
                            memory_efficiency = (allocated_bytes / reserved_bytes * 100) if reserved_bytes > 0 else 0
                        else:
                            max_allocated = gpu_memory_allocated
                            max_reserved = gpu_memory_reserved
                            memory_efficiency = gpu_utilization
                    else:
                        gpu_memory_allocated = 0
                        gpu_memory_reserved = 0
                        gpu_memory_cached = 0
                        gpu_utilization = 0
                        max_allocated = 0
                        max_reserved = 0
                        memory_efficiency = 0
                    
                    progress_bar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'samples/s': f'{samples_per_sec:.0f}',
                        'gpu_mem': f'{gpu_memory_allocated:.1f}/{gpu_memory_reserved:.1f}GB',
                        'gpu_use': f'{gpu_utilization:.0f}%',
                        'mem_eff': f'{memory_efficiency:.0f}%',
                        'batch_ms': f'{avg_batch_time*1000:.0f}'
                    })
                    
                    # Детальное логирование каждые 100 батчей
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        elapsed = time.time() - last_log_time
                        self.logger.info(f"Batch {batch_idx}: "
                                       f"loss={current_loss:.4f}, "
                                       f"speed={samples_per_sec:.0f} samples/s, "
                                       f"time={elapsed:.1f}s")
                        
                        # Расширенное логирование GPU для RTX 5090
                        if self.device.type == 'cuda':
                            self.logger.info(f"GPU Memory: {gpu_memory_allocated:.1f}GB / {gpu_memory_reserved:.1f}GB "
                                           f"(util: {gpu_utilization:.1f}%, eff: {memory_efficiency:.1f}%)")
                            self.logger.info(f"Peak Memory: {max_allocated:.1f}GB allocated, {max_reserved:.1f}GB reserved")
                            
                            # Целевая утилизация для RTX 5090
                            if gpu_utilization < 85:
                                self.logger.warning(f"⚠️ GPU утилизация ниже оптимальной: {gpu_utilization:.1f}% < 85%")
                            elif gpu_utilization > 95:
                                self.logger.info(f"🔥 Отличная GPU утилизация: {gpu_utilization:.1f}%")
                        
                        last_log_time = time.time()
                
                # Периодическая очистка кэша GPU
                if self.device.type == 'cuda' and batch_idx % self.gpu_cache_clear_freq == 0:
                    torch.cuda.empty_cache()
        
        except RuntimeError as e:
            self.logger.error(f"❌ RuntimeError в train_epoch: {e}")
            self.logger.error(f"   Batch idx: {batch_idx if 'batch_idx' in locals() else 'unknown'}")
            self.logger.error(f"   Accumulated steps: {accumulated_steps}")
            self.logger.error(f"   Batches processed: {batches_processed}")
            raise
        
        # Финальная синхронизация
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Вычисление средних метрик
        # Исправлено: правильное вычисление среднего loss
        if isinstance(epoch_loss, torch.Tensor):
            avg_loss = (epoch_loss / max(len(train_loader), 1)).item()
        else:
            avg_loss = epoch_loss / max(len(train_loader), 1)
            
        epoch_time = time.time() - epoch_start
        
        self.logger.info(f"📊 Эпоха завершена за {epoch_time:.1f}с")
        self.logger.info(f"   Средняя скорость: {len(train_loader.dataset)/epoch_time:.0f} samples/s")
        self.logger.info(f"   Средний loss: {avg_loss:.6f}")
        
        # Проверяем, что были обработаны батчи
        if batches_processed == 0:
            self.logger.warning("⚠️ Не было обработано ни одного батча!")
            return {'loss': 0.0, 'collapse_alerts': 0, 'corrections_applied': 0}
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'samples_per_second': len(train_loader.dataset) / epoch_time,
            'collapse_alerts': self.anti_collapse_monitor.collapse_alerts,
            'corrections_applied': self.anti_collapse_monitor.correction_count,
            'adaptive_lr_applied': adaptive_lr_applied
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Оптимизированная валидация"""
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                # Асинхронный перенос на GPU
                inputs = inputs.to(self.device, non_blocking=True)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device, non_blocking=True)
                elif isinstance(targets, dict):
                    targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
                
                # Обработка info dict (содержит idx как тензор)
                if isinstance(info, dict) and 'idx' in info and isinstance(info['idx'], torch.Tensor):
                    info['idx'] = info['idx'].to(self.device, non_blocking=True)
                
                # Forward pass с AMP
                if self.use_amp:
                    with autocast():
                        try:
                            outputs = self.model(inputs, return_dict=True)
                        except TypeError:
                            outputs = self.model(inputs)

                        # КРИТИЧЕСКИЙ FIX: Обработка tuple output (tensor, attention_weights)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # Берем только тензор

                        loss = self._compute_loss(outputs, targets)
                else:
                    try:
                        outputs = self.model(inputs, return_dict=True)
                    except TypeError:
                        outputs = self.model(inputs)

                    # КРИТИЧЕСКИЙ FIX: Обработка tuple output (tensor, attention_weights)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Берем только тензор

                    loss = self._compute_loss(outputs, targets)
                
                # Накопление loss (поддержка dict)
                loss_to_add = loss['total_loss'] if isinstance(loss, dict) else loss
                val_loss += loss_to_add.detach()
                
                # Периодическая очистка кэша
                if self.device.type == 'cuda' and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Финальная синхронизация
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        avg_val_loss = (val_loss / len(val_loader)).item()
        
        return {
            'val_loss': avg_val_loss
        }
    
    # Алиас для совместимости
    validate_epoch = validate
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        """Оптимизированный процесс обучения"""
        self.logger.info("🚀 Начало оптимизированного обучения")
        self.logger.info(f"   Эпох: {self.epochs}")
        self.logger.info(f"   Batch size: {train_loader.batch_size}")
        self.logger.info(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
        self.logger.info(f"   Effective batch size: {train_loader.batch_size * self.gradient_accumulation_steps}")
        
        # Информация о GPU
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"🔥 GPU: {gpu_name}")
            self.logger.info(f"   Память GPU: {gpu_memory_total:.1f} GB")
            self.logger.info(f"   Mixed Precision: {'Включено' if self.use_amp else 'Выключено'}")
            if 'RTX 5090' in gpu_name:
                self.logger.info("   ⚡ Обнаружена RTX 5090 - используются оптимизации для sm_120")
        
        best_val_loss = float('inf')
        best_macro_f1 = 0.0
        patience_counter = 0
        
        # 🚨 НОВОЕ: Early stopping по схлопыванию (смягчено для поэтапного обучения)
        collapse_patience = 10  # Максимум 10 эпох с схлопыванием (увеличено)
        collapse_counter = 0
        min_flat_ratio_epochs = 0.05  # Минимальная доля FLAT для early stopping (снижено)
        
        for epoch in range(self.epochs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Эпоха {epoch + 1}/{self.epochs}")
            
            # Обновляем dropout согласно расписанию
            self.update_dropout(epoch)
            
            # Обновляем эпоху в loss функции для динамических весов
            if hasattr(self.criterion, 'set_epoch'):
                self.criterion.set_epoch(epoch)
                if epoch < 10:  # Логируем warmup прогресс
                    current_weight = self.criterion.get_dynamic_direction_weight()
                    self.logger.info(f"📈 Direction loss weight (warmup): {current_weight:.2f}")
            
            # Обучение
            self.logger.debug(f"Вызов train_epoch для эпохи {epoch + 1}")
            try:
                train_metrics = self.train_epoch(train_loader)
            except RuntimeError as e:
                if "No inf checks were recorded" in str(e):
                    self.logger.error(f"❌ Ошибка GradScaler на эпохе {epoch + 1}: {e}")
                    self.logger.info("Пытаемся продолжить без AMP...")
                    # Отключаем AMP и пробуем снова
                    self.use_amp = False
                    train_metrics = self.train_epoch(train_loader)
                else:
                    raise
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Логирование информации о anti-collapse
            if train_metrics.get('collapse_alerts', 0) > 0:
                self.logger.warning(f"⚠️ Предупреждения о схлопывании: {train_metrics['collapse_alerts']}")
            if train_metrics.get('corrections_applied', 0) > 0:
                self.logger.info(f"✅ Применено коррекций: {train_metrics['corrections_applied']}")
            
            # Анализ важности признаков на первой эпохе
            if epoch == 0 and hasattr(train_loader.dataset, 'feature_cols'):
                try:
                    from utils.feature_importance import FeatureImportanceAnalyzer
                    analyzer = FeatureImportanceAnalyzer()
                    
                    # Получаем имена признаков
                    feature_names = train_loader.dataset.feature_cols
                    
                    # Анализируем важность через градиенты
                    # Используем модель в eval режиме но с градиентами
                    self.model.eval()
                    importance = analyzer.analyze_gradient_importance(
                        self.model, train_loader, feature_names, 
                        device=self.device, num_batches=3
                    )
                    self.model.train()  # Возвращаем в train режим
                    
                    # Логируем топ-25 важных признаков
                    if importance:
                        analyzer.log_top_features(importance, top_n=25, 
                                                 title="Топ-25 важных признаков (анализ градиентов)")
                    else:
                        self.logger.warning("Анализ важности вернул пустой результат")
                        
                except ImportError as e:
                    self.logger.warning(f"Не удалось импортировать FeatureImportanceAnalyzer: {e}")
                except AttributeError as e:
                    self.logger.warning(f"Отсутствует атрибут feature_cols в dataset: {e}")
                except Exception as e:
                    self.logger.warning(f"Не удалось проанализировать важность признаков: {e}")
                    import traceback
                    self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Валидация с расширенными метриками
            if val_loader is not None:
                val_metrics = self.validate_with_enhanced_metrics(val_loader)
                self.history['val_loss'].append(val_metrics['val_loss'])
                
                # Early stopping по macro F1 вместо val loss
                current_macro_f1 = val_metrics.get('macro_f1_overall', 0.0)
                current_flat_ratio = val_metrics.get('class_distribution', {}).get('flat', 0.0)
                
                # Проверка на схлопывание FLAT класса
                if current_flat_ratio < min_flat_ratio_epochs:
                    collapse_counter += 1
                    self.logger.warning(
                        f"⚠️ FLAT схлопывание ({current_flat_ratio:.1%}): {collapse_counter}/{collapse_patience}"
                    )
                else:
                    collapse_counter = 0  # Сбрасываем счетчик если FLAT восстановился
                
                if current_macro_f1 > best_macro_f1 + self.config['model'].get('min_delta', 0.001):
                    best_macro_f1 = current_macro_f1
                    best_val_loss = val_metrics['val_loss']  # Сохраняем для логирования
                    patience_counter = 0
                    # Сохранение лучшей модели
                    self._save_checkpoint(epoch, val_metrics['val_loss'], is_best=True)
                    self.logger.info(f"✅ Новая лучшая модель! Macro F1: {best_macro_f1:.4f}")
                else:
                    patience_counter += 1
                
                self.logger.info(f"📈 Train Loss: {train_metrics['loss']:.4f}, "
                               f"Val Loss: {val_metrics['val_loss']:.4f}, "
                               f"Macro F1: {val_metrics.get('macro_f1_overall', 0):.4f} "
                               f"(best F1: {best_macro_f1:.4f}, patience: {patience_counter})")
                
                # Детальное логирование распределения предсказаний направлений
                if 'predicted_distribution' in val_metrics:
                    pred_dist = val_metrics['predicted_distribution']
                    actual_dist = val_metrics.get('actual_distribution', {})
                    
                    self.logger.info("📊 Распределение предсказанных направлений:")
                    for timeframe in ['15m', '1h', '4h', '12h']:
                        if f'direction_{timeframe}' in pred_dist:
                            pred = pred_dist[f'direction_{timeframe}']
                            actual = actual_dist.get(f'direction_{timeframe}', {})
                            self.logger.info(f"   {timeframe}: LONG: {pred.get('LONG', 0):.1%} "
                                           f"(факт: {actual.get('LONG', 0):.1%}), "
                                           f"SHORT: {pred.get('SHORT', 0):.1%} "
                                           f"(факт: {actual.get('SHORT', 0):.1%}), "
                                           f"FLAT: {pred.get('FLAT', 0):.1%} "
                                           f"(факт: {actual.get('FLAT', 0):.1%})")
                
                # Логирование энтропии предсказаний
                if 'prediction_entropy' in val_metrics and val_metrics['prediction_entropy']:
                    avg_entropy = val_metrics['prediction_entropy'].get('average', 0)
                    # Проверяем, что энтропия рассчитана корректно
                    if avg_entropy > 0:
                        self.logger.info(f"🎲 Средняя энтропия предсказаний: {avg_entropy:.3f} "
                                       f"(макс: {np.log(3):.3f})")
                        
                        # Предупреждение о схлопывании
                        if avg_entropy < 0.5:
                            self.logger.warning("⚠️ НИЗКАЯ ЭНТРОПИЯ! Модель может схлопываться в один класс!")
                    else:
                        # Рассчитываем энтропию из распределения предсказаний
                        if 'predicted_distribution' in val_metrics:
                            total_entropy = 0
                            count = 0
                            for tf in ['15m', '1h', '4h', '12h']:
                                dist = val_metrics['predicted_distribution'].get(f'direction_{tf}', {})
                                if dist:
                                    # Вычисляем энтропию из распределения
                                    probs = np.array([dist.get('LONG', 0), dist.get('SHORT', 0), dist.get('FLAT', 0)])
                                    probs = probs / probs.sum() if probs.sum() > 0 else probs
                                    # Избегаем log(0)
                                    probs = np.clip(probs, 1e-8, 1.0)
                                    entropy = -np.sum(probs * np.log(probs))
                                    normalized_entropy = entropy / np.log(3)
                                    total_entropy += normalized_entropy
                                    count += 1
                            
                            if count > 0:
                                avg_entropy = total_entropy / count
                                self.logger.info(f"🎲 Средняя энтропия предсказаний (пересчитана): {avg_entropy:.3f} "
                                               f"(макс: 1.099)")
                            else:
                                self.logger.info("🎲 Не удалось рассчитать энтропию предсказаний")
                
                # Проверка early stopping
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info("⚠️ Early stopping triggered (по F1 score)!")
                    break
                    
                if collapse_counter >= collapse_patience:
                    self.logger.error(f"⚠️ Early stopping из-за схлопывания FLAT! ({collapse_counter} эпох подряд)")
                    self.logger.error("📉 Модель неспособна поддерживать баланс классов")
                    break
            else:
                self.logger.info(f"📈 Train Loss: {train_metrics['loss']:.4f}")
                # Проверка на схлопывание и без валидации
                if train_metrics.get('collapse_alerts', 0) > 10:
                    self.logger.warning("⚠️ Много предупреждений о схлопывании за эпоху!")
            
            # Scheduler step
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    # ReduceLROnPlateau требует метрику
                    if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                        if val_loader is not None:
                            # Используем macro F1 вместо val_loss если доступно
                            # ИСПРАВЛЕНО: monitor теперь на верхнем уровне scheduler, а не в params
                            metric_name = self.config.get('scheduler', {}).get('monitor', 'val_loss')
                            if metric_name == 'val_macro_f1':
                                metric_value = val_metrics.get('macro_f1_overall', 0.0)
                                self.logger.info(f"📊 Scheduler: используем macro_f1_overall = {metric_value:.4f}")
                            else:
                                metric_value = val_metrics.get(metric_name, val_metrics['val_loss'])
                            self.scheduler.step(metric_value)
                        else:
                            self.scheduler.step(train_metrics['loss'])
                    else:
                        self.scheduler.step()
            
            # Периодическое логирование confusion matrix (каждые 5 эпох)
            if (epoch + 1) % 5 == 0 and val_loader is not None:
                self.log_confusion_matrix(val_loader, epoch + 1)
            
            # Периодическое сохранение
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, train_metrics['loss'], is_best=False)
        
        # Финальная статистика по anti-collapse
        total_collapse_alerts = 0
        total_corrections = 0
        for i, train_loss in enumerate(self.history.get('train_loss', [])):
            # Подсчитываем из train metrics если доступны
            pass  # Статистика уже логируется во время обучения
        
        self.logger.info("✅ Обучение завершено!")
        if hasattr(self, 'anti_collapse_monitor'):
            total_alerts = sum(getattr(self.anti_collapse_monitor, 'total_alerts', 0) for _ in range(1))
            if total_alerts > 0:
                self.logger.info(f"📈 Общая статистика anti-collapse система работала активно")
        
        return self.history
    
    def compute_direction_metrics(self, outputs, targets) -> Dict[str, float]:
        """
        Расчет метрик для direction предсказаний
        
        Args:
            outputs: torch.Tensor (batch_size, 20) или dict с ключами direction_*
            targets: torch.Tensor (batch_size, 20) или dict с ключами direction_*
            
        Returns:
            Dict с метриками directional accuracy и win rate
        """
        metrics = {}
        
        # КРИТИЧЕСКИЙ FIX: Обработка tuple (tensor, attention_weights)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Берем только тензор

        # Извлекаем direction переменные в зависимости от типа
        if isinstance(outputs, dict):
            # Если outputs - словарь, извлекаем отдельные ключи
            direction_outputs = {
                '15m': outputs.get('direction_15m'),
                '1h': outputs.get('direction_1h'),
                '4h': outputs.get('direction_4h'),
                '12h': outputs.get('direction_12h')
            }
        else:
            # Если outputs не словарь, проверяем что это тензор
            if isinstance(outputs, torch.Tensor):
                # Если outputs - тензор, извлекаем индексы 4-7
                direction_outputs = outputs[:, 4:8]  # direction_15m, 1h, 4h, 12h
            else:
                # Fallback: создаем пустой словарь
                direction_outputs = {
                    '15m': None,
                    '1h': None,
                    '4h': None,
                    '12h': None
                }
        
        if isinstance(targets, dict):
            # Если targets - словарь, извлекаем отдельные ключи
            direction_targets = {
                '15m': targets.get('direction_15m'),
                '1h': targets.get('direction_1h'),
                '4h': targets.get('direction_4h'),
                '12h': targets.get('direction_12h')
            }
        else:
            # Если targets не словарь, проверяем что это тензор
            if isinstance(targets, torch.Tensor):
                # Если targets - тензор, извлекаем индексы 4-7
                direction_targets = targets[:, 4:8]
            else:
                # Fallback: создаем пустой словарь
                direction_targets = {
                    '15m': None,
                    '1h': None,
                    '4h': None,
                    '12h': None
                }
        
        # Расчет directional accuracy для каждого таймфрейма
        timeframes = ['15m', '1h', '4h', '12h']
        
        for i, tf in enumerate(timeframes):
            # Пропускаем если данных нет (для словарей)
            if isinstance(direction_outputs, dict):
                if direction_outputs[tf] is None or (isinstance(direction_targets, dict) and direction_targets[tf] is None):
                    continue
                pred_outputs = direction_outputs[tf]
                true_targets = direction_targets[tf] if isinstance(direction_targets, dict) else direction_targets[:, i]
            else:
                pred_outputs = direction_outputs[:, i]
                true_targets = direction_targets[:, i] if isinstance(direction_targets, torch.Tensor) else direction_targets[tf]
            
            # ИСПРАВЛЕНИЕ: Правильная интерпретация выходов модели
            # Если outputs имеет _direction_logits, используем их
            if hasattr(outputs, '_direction_logits'):
                # Используем логиты для получения классов через softmax + argmax
                direction_logits = outputs._direction_logits[:, i, :]  # (batch_size, 3)
                probs = torch.softmax(direction_logits, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1)
                # Логирование уверенности
                with torch.no_grad():
                    max_probs = probs.max(dim=-1)[0].detach().cpu()
                    if i == 0:
                        import numpy as _np
                        mp = max_probs.numpy()
                        q25, q50, q75 = _np.quantile(mp, [0.25, 0.5, 0.75])
                        self.logger.info(f"📊 {tf} probs: max_p mean={mp.mean():.3f}, q25={q25:.3f}, q50={q50:.3f}, q75={q75:.3f}")
                self.logger.debug(f"Using _direction_logits for {tf}")
            elif isinstance(pred_outputs, torch.Tensor) and pred_outputs.dim() > 1 and pred_outputs.shape[-1] == 3:
                # Если это логиты (batch_size, 3), применяем softmax
                probs = torch.softmax(pred_outputs, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1)
            else:
                # УЛУЧШЕННЫЙ Fallback: используем пороги для интерпретации
                # Вместо простого округления, используем пороги для классификации
                pred_values = pred_outputs
                
                # Логируем диапазон значений для отладки
                if i == 0:
                    self.logger.info(f"📊 {tf} raw outputs: min={pred_values.min():.3f}, max={pred_values.max():.3f}, mean={pred_values.mean():.3f}")
                
                # Используем пороги для классификации
                # < 0.67 = LONG (0)
                # 0.67 - 1.33 = SHORT (1)  
                # > 1.33 = FLAT (2)
                pred_classes = torch.zeros_like(pred_values, dtype=torch.long)
                pred_classes[pred_values < 0.67] = 0  # LONG
                pred_classes[(pred_values >= 0.67) & (pred_values < 1.33)] = 1  # SHORT
                pred_classes[pred_values >= 1.33] = 2  # FLAT
            
            true_classes = true_targets.long()
            
            # Точность предсказания направления
            correct = (pred_classes == true_classes).float()
            accuracy = correct.mean().item()
            metrics[f'direction_accuracy_{tf}'] = accuracy
            
            # Мониторинг разнообразия предсказаний
            unique_preds, pred_counts = torch.unique(pred_classes, return_counts=True)
            pred_distribution = {}
            for class_idx in range(3):  # LONG=0, SHORT=1, FLAT=2
                count = pred_counts[unique_preds == class_idx].sum().item() if (unique_preds == class_idx).any() else 0
                pred_distribution[class_idx] = count
            
            # Вычисляем энтропию предсказаний для оценки разнообразия
            total_preds = pred_classes.shape[0]
            pred_probs = torch.tensor([pred_distribution.get(i, 0) / total_preds for i in range(3)])
            pred_probs = pred_probs + 1e-8  # Избегаем log(0)
            entropy = -(pred_probs * torch.log(pred_probs)).sum().item()
            normalized_entropy = entropy / np.log(3)  # Нормализуем к [0, 1]
            
            metrics[f'pred_entropy_{tf}'] = normalized_entropy
            metrics[f'pred_long_ratio_{tf}'] = pred_distribution.get(0, 0) / total_preds
            metrics[f'pred_short_ratio_{tf}'] = pred_distribution.get(1, 0) / total_preds
            metrics[f'pred_flat_ratio_{tf}'] = pred_distribution.get(2, 0) / total_preds
            
            # Точность для UP/DOWN (исключаем FLAT)
            non_flat_mask = (true_classes != 2)
            if non_flat_mask.sum() > 0:
                up_down_correct = correct[non_flat_mask].mean().item()
                metrics[f'up_down_accuracy_{tf}'] = up_down_correct
            
            # Процент правильных UP предсказаний
            up_mask = (true_classes == 0)
            if up_mask.sum() > 0:
                up_correct = correct[up_mask].mean().item()
                metrics[f'up_accuracy_{tf}'] = up_correct
            
            # Процент правильных DOWN предсказаний  
            down_mask = (true_classes == 1)
            if down_mask.sum() > 0:
                down_correct = correct[down_mask].mean().item()
                metrics[f'down_accuracy_{tf}'] = down_correct
        
        # Общая directional accuracy (среднее по всем таймфреймам)
        overall_accuracy = np.mean([metrics[f'direction_accuracy_{tf}'] for tf in timeframes])
        metrics['direction_accuracy_overall'] = overall_accuracy
        
        # Вычисляем macro F1 для каждого таймфрейма
        macro_f1_scores = []
        for i, tf in enumerate(timeframes):
            # Извлекаем предсказания и цели
            if hasattr(outputs, '_direction_logits'):
                direction_logits = outputs._direction_logits[:, i, :]
                pred_classes = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
            else:
                # Используем пороги вместо округления
                if isinstance(outputs, dict):
                    # Если outputs это словарь, берем соответствующие direction логиты
                    direction_key = f'direction_{tf}'
                    if direction_key in outputs:
                        direction_logits = outputs[direction_key]
                        # Если это логиты (3 класса), используем argmax
                        if direction_logits.shape[-1] == 3:
                            pred_classes = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
                        else:
                            # Если это одно значение, применяем пороги
                            pred_values = direction_logits.squeeze(-1) if direction_logits.dim() > 1 else direction_logits
                            pred_classes = torch.zeros_like(pred_values, dtype=torch.long)
                            pred_classes[pred_values < 0.67] = 0  # LONG
                            pred_classes[(pred_values >= 0.67) & (pred_values < 1.33)] = 1  # SHORT
                            pred_classes[pred_values >= 1.33] = 2  # FLAT
                    else:
                        # Если ключа нет, пропускаем
                        continue
                else:
                    # Если outputs это тензор
                    pred_values = outputs[:, 4+i]
                    pred_classes = torch.zeros_like(pred_values, dtype=torch.long)
                    pred_classes[pred_values < 0.67] = 0  # LONG
                    pred_classes[(pred_values >= 0.67) & (pred_values < 1.33)] = 1  # SHORT
                    pred_classes[pred_values >= 1.33] = 2  # FLAT
            
            # Обработка targets
            if isinstance(targets, dict):
                direction_key = f'direction_{tf}'
                if direction_key in targets:
                    true_classes = targets[direction_key].long()
                else:
                    continue
            else:
                true_classes = targets[:, 4+i].long()
            
            # Вычисляем F1 для каждого класса
            f1_scores = []
            for class_idx in range(3):
                tp = ((pred_classes == class_idx) & (true_classes == class_idx)).sum().item()
                fp = ((pred_classes == class_idx) & (true_classes != class_idx)).sum().item()
                fn = ((pred_classes != class_idx) & (true_classes == class_idx)).sum().item()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
            
            macro_f1 = np.mean(f1_scores)
            metrics[f'macro_f1_{tf}'] = macro_f1
            macro_f1_scores.append(macro_f1)
        
        # Общий macro F1
        metrics['macro_f1_overall'] = np.mean(macro_f1_scores)
        
        # Общая метрика разнообразия предсказаний
        overall_entropy = np.mean([metrics[f'pred_entropy_{tf}'] for tf in timeframes])
        metrics['pred_entropy_overall'] = overall_entropy
        
        # Средние соотношения классов
        metrics['pred_long_ratio_overall'] = np.mean([metrics[f'pred_long_ratio_{tf}'] for tf in timeframes])
        metrics['pred_short_ratio_overall'] = np.mean([metrics[f'pred_short_ratio_{tf}'] for tf in timeframes])
        metrics['pred_flat_ratio_overall'] = np.mean([metrics[f'pred_flat_ratio_{tf}'] for tf in timeframes])
        
        # Предупреждение если модель предсказывает слишком однообразно
        if overall_entropy < 0.3:
            self.logger.warning(f"⚠️ Низкое разнообразие предсказаний! Энтропия: {overall_entropy:.3f}")
            self.logger.warning(f"   LONG: {metrics['pred_long_ratio_overall']:.1%}, "
                              f"SHORT: {metrics['pred_short_ratio_overall']:.1%}, "
                              f"FLAT: {metrics['pred_flat_ratio_overall']:.1%}")
        
        return metrics
    
    def compute_trading_metrics(self, outputs, targets) -> Dict[str, float]:
        """
        Расчет торговых метрик: win rate, profit factor, etc.

        Args:
            outputs: torch.Tensor (batch_size, 20) или dict с ключами direction_* и future_return_*
            targets: torch.Tensor (batch_size, 20) или dict с ключами direction_* и future_return_*

        Returns:
            Dict с торговыми метриками
        """
        metrics = {}

        # КРИТИЧЕСКИЙ FIX: Обработка tuple (tensor, attention_weights)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Берем только тензор

        # Извлекаем нужные переменные в зависимости от типа
        if isinstance(targets, dict):
            future_returns = {
                '15m': targets.get('future_return_15m'),
                '1h': targets.get('future_return_1h'),
                '4h': targets.get('future_return_4h'),
                '12h': targets.get('future_return_12h')
            }
        else:
            if isinstance(targets, torch.Tensor):
                future_returns = targets[:, 0:4]  # future_return_15m, 1h, 4h, 12h
            else:
                future_returns = {'15m': None, '1h': None, '4h': None, '12h': None}
        
        if isinstance(outputs, dict):
            direction_outputs = {
                '15m': outputs.get('direction_15m'),
                '1h': outputs.get('direction_1h'),
                '4h': outputs.get('direction_4h'),
                '12h': outputs.get('direction_12h')
            }
        else:
            if isinstance(outputs, torch.Tensor):
                direction_outputs = outputs[:, 4:8]
            else:
                direction_outputs = {'15m': None, '1h': None, '4h': None, '12h': None}
        
        if isinstance(targets, dict):
            direction_targets = {
                '15m': targets.get('direction_15m'),
                '1h': targets.get('direction_1h'),
                '4h': targets.get('direction_4h'),
                '12h': targets.get('direction_12h')
            }
        else:
            if isinstance(targets, torch.Tensor):
                direction_targets = targets[:, 4:8]
            else:
                direction_targets = {'15m': None, '1h': None, '4h': None, '12h': None}
        
        timeframes = ['15m', '1h', '4h', '12h']
        
        for i, tf in enumerate(timeframes):
            # Извлекаем данные в зависимости от типа
            if isinstance(future_returns, dict):
                if future_returns[tf] is None:
                    continue
                future_return = future_returns[tf]
            else:
                future_return = future_returns[:, i]
            
            if isinstance(direction_outputs, dict):
                if direction_outputs[tf] is None:
                    continue
                dir_output = direction_outputs[tf]
            else:
                dir_output = direction_outputs[:, i]
            
            if isinstance(direction_targets, dict):
                if direction_targets[tf] is None:
                    continue
                dir_target = direction_targets[tf]
            else:
                dir_target = direction_targets[:, i]
            
            # ИСПРАВЛЕНИЕ: Правильная интерпретация предсказанных направлений
            if hasattr(outputs, '_direction_logits'):
                # Используем логиты для получения классов через softmax + argmax
                direction_logits = outputs._direction_logits[:, i, :]  # (batch_size, 3)
                probs = torch.softmax(direction_logits, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1)
                # Логирование уверенности для каждого tf
                with torch.no_grad():
                    max_probs = probs.max(dim=-1)[0].detach().cpu()
                    import numpy as _np
                    mp = max_probs.numpy()
                    q25, q50, q75 = _np.quantile(mp, [0.25, 0.5, 0.75])
                    self.logger.info(f"📊 {tf} probs: max_p mean={mp.mean():.3f}, q25={q25:.3f}, q50={q50:.3f}, q75={q75:.3f}")
                self.logger.debug(f"Using _direction_logits for {tf}")
            elif isinstance(dir_output, torch.Tensor) and dir_output.dim() > 1 and dir_output.shape[-1] == 3:
                # Если это логиты (batch_size, 3), применяем softmax
                probs = torch.softmax(dir_output, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1)
            else:
                # УЛУЧШЕННЫЙ Fallback: используем пороги для интерпретации
                # Вместо простого округления, используем пороги для классификации
                pred_values = dir_output
                
                # Логируем диапазон значений для отладки
                if i == 0:
                    self.logger.info(f"📊 {tf} raw outputs: min={pred_values.min():.3f}, max={pred_values.max():.3f}, mean={pred_values.mean():.3f}")
                
                # Используем пороги для классификации
                # < 0.67 = LONG (0)
                # 0.67 - 1.33 = SHORT (1)  
                # > 1.33 = FLAT (2)
                pred_classes = torch.zeros_like(pred_values, dtype=torch.long)
                pred_classes[pred_values < 0.67] = 0  # LONG
                pred_classes[(pred_values >= 0.67) & (pred_values < 1.33)] = 1  # SHORT
                pred_classes[pred_values >= 1.33] = 2  # FLAT
            
            true_returns = future_return
            
            # Имитируем торговые сигналы
            # UP предсказание (0) = LONG позиция
            long_mask = (pred_classes == 0)
            # DOWN предсказание (1) = SHORT позиция  
            short_mask = (pred_classes == 1)
            # FLAT предсказание (2) = нет позиции
            
            # Всегда пытаемся рассчитать метрики, даже если нет сигналов
            # Это поможет отследить проблему
            if True:  # Убираем условие для отладки
                # Расчет P&L
                pnl = torch.zeros_like(true_returns)
                
                # УЛУЧШЕНИЕ 1: Фильтр по уверенности модели
                # Получаем вероятности из логитов
                if hasattr(outputs, '_direction_logits'):
                    direction_logits = outputs._direction_logits[:, i, :]
                    probs = torch.softmax(direction_logits, dim=-1)
                    # Берем максимальную вероятность для каждого предсказания
                    max_probs = probs.max(dim=-1)[0]
                elif isinstance(dir_output, torch.Tensor) and dir_output.dim() > 1 and dir_output.shape[-1] == 3:
                    # Если это логиты, используем softmax вероятности
                    probs = torch.softmax(dir_output, dim=-1)
                    max_probs = probs.max(dim=-1)[0]
                else:
                    # Для fallback используем простую эвристику
                    # Чем дальше от центра (1.0 = FLAT), тем увереннее
                    # Используем dir_output как скалярное значение
                    distance_from_center = torch.abs(dir_output - 1.0)
                    max_probs = torch.sigmoid(distance_from_center * 2)  # Преобразуем в [0, 1]
                
                # Пороги уверенности для торговых сигналов
                # 45-50% - умеренные пороги для баланса качества и количества
                confidence_thresholds = {'15m': 0.45, '1h': 0.45, '4h': 0.50, '12h': 0.50}
                confidence_threshold = confidence_thresholds[tf]
                high_confidence = max_probs > confidence_threshold
                
                # Логирование для отладки для КАЖДОГО таймфрейма
                self.logger.info(f"📊 {tf} Распределение предсказаний:")
                self.logger.info(f"   LONG: {long_mask.sum().item()}/{len(long_mask)} ({long_mask.float().mean()*100:.1f}%)")
                self.logger.info(f"   SHORT: {short_mask.sum().item()}/{len(short_mask)} ({short_mask.float().mean()*100:.1f}%)")
                self.logger.info(f"   FLAT: {(pred_classes == 2).sum().item()}/{len(pred_classes)} ({(pred_classes == 2).float().mean()*100:.1f}%)")
                self.logger.info(f"   Уверенность > {confidence_threshold:.0%}: {high_confidence.sum().item()} сигналов")
                
                # СНИЖЕННЫЕ требования к движению - учитываем реальную волатильность крипто
                # Даже 0.1% движение может быть прибыльным с правильным risk management
                min_move_thresholds = {'15m': 0.0005, '1h': 0.001, '4h': 0.0015, '12h': 0.002}
                min_move_threshold = min_move_thresholds[tf]
                significant_move = torch.abs(true_returns) > min_move_threshold
                
                # Применяем фильтры
                valid_long = long_mask & high_confidence & significant_move
                valid_short = short_mask & high_confidence & significant_move
                
                # Логирование после фильтрации для каждого таймфрейма
                self.logger.info(f"   {tf} После фильтров: LONG={valid_long.sum().item()}, SHORT={valid_short.sum().item()}")
                
                # LONG позиции: прибыль = изменение цены
                if valid_long.sum() > 0:
                    pnl[valid_long] = true_returns[valid_long]
                
                # SHORT позиции: прибыль = -изменение цены
                if valid_short.sum() > 0:
                    pnl[valid_short] = -true_returns[valid_short]
                
                # Убираем комиссию
                commission = 0.001  # 0.1%
                trading_mask = valid_long | valid_short
                pnl[trading_mask] -= commission
                
                # Win Rate
                profitable_trades = pnl[trading_mask] > 0
                if trading_mask.sum() > 0:
                    win_rate = profitable_trades.float().mean().item()
                    metrics[f'win_rate_{tf}'] = win_rate
                    
                    # Логирование win rate для каждого таймфрейма
                    total_trades = trading_mask.sum().item()
                    profitable = profitable_trades.sum().item()
                    self.logger.info(f"   Win Rate {tf}: {win_rate:.1%} ({profitable}/{total_trades} сделок)")
                    
                    # Profit Factor
                    profits = pnl[trading_mask & (pnl > 0)]
                    losses = pnl[trading_mask & (pnl < 0)]
                    
                    if len(profits) > 0 and len(losses) > 0:
                        profit_factor = profits.sum().item() / abs(losses.sum().item())
                        metrics[f'profit_factor_{tf}'] = profit_factor
                    
                    # Средний P&L
                    avg_pnl = pnl[trading_mask].mean().item()
                    metrics[f'avg_pnl_{tf}'] = avg_pnl
                    
                    # Максимальная просадка (упрощенная)
                    cumulative_pnl = torch.cumsum(pnl[trading_mask], dim=0)
                    running_max = torch.cummax(cumulative_pnl, dim=0)[0]
                    drawdown = running_max - cumulative_pnl
                    max_drawdown = drawdown.max().item()
                    metrics[f'max_drawdown_{tf}'] = max_drawdown
        
        # УЛУЧШЕНИЕ 3: Стратегия комбинирования таймфреймов для повышения win rate
        # Торгуем только когда несколько таймфреймов подтверждают сигнал
        
        # Извлекаем предсказания всех таймфреймов
        all_predictions = []
        for i, tf in enumerate(timeframes):
            if hasattr(outputs, '_direction_logits'):
                direction_logits = outputs._direction_logits[:, i, :]
                pred_classes = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
            elif isinstance(direction_outputs, dict):
                # Если direction_outputs это словарь, берем соответствующие direction логиты
                if direction_outputs[tf] is not None:
                    dir_output_tf = direction_outputs[tf]
                    if dir_output_tf.shape[-1] == 3:
                        # Если это логиты (3 класса), используем argmax
                        pred_classes = torch.argmax(torch.softmax(dir_output_tf, dim=-1), dim=-1)
                    else:
                        # Если это одно значение, округляем
                        pred_classes = torch.round(dir_output_tf.squeeze(-1) if dir_output_tf.dim() > 1 else dir_output_tf).clamp(0, 2).long()
                else:
                    # Если нет данных для этого tf, пропускаем
                    continue
            else:
                # Если direction_outputs это тензор
                pred_classes = torch.round(direction_outputs[:, i]).clamp(0, 2).long()
            all_predictions.append(pred_classes)
        
        # Стратегия: голосование большинством
        # Торгуем только если минимум 3 из 4 таймфреймов согласны
        all_predictions = torch.stack(all_predictions, dim=1)  # (batch_size, 4)
        
        # Подсчитываем голоса для каждого класса
        long_votes = (all_predictions == 0).sum(dim=1)  # Количество LONG голосов
        short_votes = (all_predictions == 1).sum(dim=1)  # Количество SHORT голосов
        
        # Требуем минимум 3 согласованных сигнала для торговли
        consensus_threshold = 3
        strong_long_signal = long_votes >= consensus_threshold
        strong_short_signal = short_votes >= consensus_threshold
        
        if strong_long_signal.sum() > 0 or strong_short_signal.sum() > 0:
            # Используем максимальный таймфрейм (12h) для оценки PnL
            if isinstance(future_returns, dict):
                true_returns = future_returns['12h']  # 12h returns
            else:
                true_returns = future_returns[:, 3]  # 12h returns
            consensus_pnl = torch.zeros_like(true_returns)
            
            # LONG с консенсусом
            if strong_long_signal.sum() > 0:
                consensus_pnl[strong_long_signal] = true_returns[strong_long_signal]
            
            # SHORT с консенсусом  
            if strong_short_signal.sum() > 0:
                consensus_pnl[strong_short_signal] = -true_returns[strong_short_signal]
            
            # Комиссия
            consensus_trading_mask = strong_long_signal | strong_short_signal
            consensus_pnl[consensus_trading_mask] -= 0.001
            
            # Win rate с консенсусом
            if consensus_trading_mask.sum() > 0:
                consensus_win_rate = (consensus_pnl[consensus_trading_mask] > 0).float().mean().item()
                metrics['win_rate_consensus'] = consensus_win_rate
                
                # Количество сделок с консенсусом
                metrics['consensus_trades_ratio'] = consensus_trading_mask.float().mean().item()
        
        # Общие метрики
        if any(f'win_rate_{tf}' in metrics for tf in timeframes):
            # Средний win rate
            win_rates = [metrics[f'win_rate_{tf}'] for tf in timeframes if f'win_rate_{tf}' in metrics]
            if win_rates:
                metrics['win_rate_overall'] = np.mean(win_rates)
        
        # Если есть консенсус метрика, используем её как основную
        if 'win_rate_consensus' in metrics:
            metrics['win_rate_primary'] = metrics['win_rate_consensus']
        elif 'win_rate_overall' in metrics:
            metrics['win_rate_primary'] = metrics['win_rate_overall']
        else:
            # Если нет win rate, значит нет торговых сигналов
            metrics['win_rate_primary'] = 0.0
            self.logger.warning("⚠️ Нет торговых сигналов! Модель предсказывает только FLAT")
        
        return metrics
    
    def compute_class_distribution_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                          timeframe: str) -> Dict[str, float]:
        """
        Расчет детальных метрик распределения классов
        
        Args:
            predictions: предсказанные классы
            targets: истинные классы
            timeframe: таймфрейм ('15m', '1h', '4h', '12h')
            
        Returns:
            Dict с метриками по классам
        """
        metrics = {}
        
        # Подсчет классов
        unique_pred, pred_counts = torch.unique(predictions, return_counts=True)
        unique_true, true_counts = torch.unique(targets, return_counts=True)
        
        # Распределение предсказаний
        total_samples = len(predictions)
        for class_id in [0, 1, 2]:  # LONG, SHORT, FLAT
            class_name = ['LONG', 'SHORT', 'FLAT'][class_id]
            
            # Количество предсказаний данного класса
            pred_count = pred_counts[unique_pred == class_id].sum().item() if (unique_pred == class_id).any() else 0
            true_count = true_counts[unique_true == class_id].sum().item() if (unique_true == class_id).any() else 0
            
            # Процентное распределение
            pred_pct = (pred_count / total_samples * 100) if total_samples > 0 else 0
            true_pct = (true_count / total_samples * 100) if total_samples > 0 else 0
            
            metrics[f'{timeframe}_{class_name}_pred_pct'] = pred_pct
            metrics[f'{timeframe}_{class_name}_true_pct'] = true_pct
            metrics[f'{timeframe}_{class_name}_diff_pct'] = abs(pred_pct - true_pct)
            
            # Точность для конкретного класса (precision)
            if pred_count > 0:
                class_correct = ((predictions == class_id) & (targets == class_id)).sum().item()
                precision = class_correct / pred_count
                metrics[f'{timeframe}_{class_name}_precision'] = precision
            else:
                metrics[f'{timeframe}_{class_name}_precision'] = 0.0
            
            # Полнота для конкретного класса (recall)
            if true_count > 0:
                class_correct = ((predictions == class_id) & (targets == class_id)).sum().item()
                recall = class_correct / true_count
                metrics[f'{timeframe}_{class_name}_recall'] = recall
            else:
                metrics[f'{timeframe}_{class_name}_recall'] = 0.0
        
        # Общий дисбаланс предсказаний
        pred_entropy = -sum([
            (c/total_samples) * torch.log(torch.tensor(c/total_samples + 1e-8)) 
            for c in pred_counts.cpu().numpy()
        ]).item() if total_samples > 0 else 0
        
        max_entropy = torch.log(torch.tensor(3.0)).item()  # log(3) для 3 классов
        normalized_entropy = pred_entropy / max_entropy if max_entropy > 0 else 0
        
        metrics[f'{timeframe}_prediction_entropy'] = normalized_entropy
        metrics[f'{timeframe}_prediction_diversity'] = len(unique_pred)
        
        return metrics
    
    def log_confusion_matrix(self, val_loader: DataLoader, epoch: int):
        """
        Логирование confusion matrix для direction предсказаний
        
        Args:
            val_loader: DataLoader для валидации
            epoch: номер эпохи
        """
        self.model.eval()
        
        # Словарь для хранения confusion matrices по таймфреймам
        confusion_matrices = {tf: torch.zeros(3, 3, dtype=torch.long) for tf in ['15m', '1h', '4h', '12h']}
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc="Computing Confusion Matrix", leave=False):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Приводим targets к правильной размерности если нужно
                if targets.dim() == 3 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                
                try:
                    outputs = self.model(inputs, return_dict=True)
                except TypeError:
                    outputs = self.model(inputs)
                
                # КРИТИЧЕСКИЙ FIX: Обработка разных типов output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Берем только тензор

                # Обрабатываем каждый таймфрейм
                for i, tf in enumerate(['15m', '1h', '4h', '12h']):
                    if isinstance(outputs, dict):
                        # Если модель вернула словарь (return_dict=True)
                        direction_key = f'direction_{tf}'
                        if direction_key in outputs:
                            direction_logits = outputs[direction_key]
                            pred_classes = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
                        else:
                            continue
                    elif hasattr(outputs, '_direction_logits'):
                        direction_logits = outputs._direction_logits[:, i, :]
                        pred_classes = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
                    elif isinstance(outputs, torch.Tensor):
                        # Проверяем размерность outputs
                        if outputs.dim() == 1:
                            # Если outputs одномерный, пропускаем эту итерацию
                            continue
                        elif outputs.dim() == 2 and outputs.shape[1] > 4+i:
                            direction_outputs = outputs[:, 4+i]
                            pred_classes = torch.round(direction_outputs).clamp(0, 2).long()
                        else:
                            # Если структура outputs неожиданная, пропускаем
                            continue
                    else:
                        continue
                    
                    true_classes = targets[:, 4+i].long()
                    
                    # Обновляем confusion matrix
                    for t, p in zip(true_classes, pred_classes):
                        confusion_matrices[tf][t.item(), p.item()] += 1
        
        # Логируем confusion matrices
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Confusion Matrices - Epoch {epoch}")
        self.logger.info(f"{'='*60}")
        
        for tf, cm in confusion_matrices.items():
            self.logger.info(f"\n🕐 Таймфрейм {tf}:")
            self.logger.info("   Pred→  LONG  SHORT  FLAT")
            self.logger.info("True↓")
            
            class_names = ['LONG', 'SHORT', 'FLAT']
            for i, class_name in enumerate(class_names):
                row = cm[i]
                total = row.sum().item()
                if total > 0:
                    percentages = (row.float() / total * 100).numpy()
                    self.logger.info(f"{class_name:5s}  {row[0]:5d} {row[1]:5d} {row[2]:5d}  "
                                   f"({percentages[0]:4.1f}% {percentages[1]:4.1f}% {percentages[2]:4.1f}%)")
                else:
                    self.logger.info(f"{class_name:5s}     0     0     0")
            
            # Общая точность
            correct = cm.diag().sum().item()
            total = cm.sum().item()
            accuracy = correct / total if total > 0 else 0
            self.logger.info(f"\nОбщая точность: {accuracy:.3f} ({correct}/{total})")
            
            # F1 scores по классам
            f1_scores = []
            for i in range(3):
                tp = cm[i, i].item()
                fp = cm[:, i].sum().item() - tp
                fn = cm[i, :].sum().item() - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
                
                self.logger.info(f"F1 {class_names[i]}: {f1:.3f} (P={precision:.3f}, R={recall:.3f})")
            
            # Macro F1
            macro_f1 = np.mean(f1_scores)
            self.logger.info(f"Macro F1: {macro_f1:.3f}")
        
        self.logger.info(f"{'='*60}\n")
    
    def validate_with_enhanced_metrics(self, val_loader: DataLoader) -> Dict[str, float]:
        """Валидация с расширенными метриками для direction и trading"""
        self.model.eval()
        
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        all_direction_logits = []  # сохраняем логиты направлений для корректных метрик
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(tqdm(val_loader, desc="Enhanced Validation", leave=False)):
                # Асинхронный перенос на GPU
                inputs = inputs.to(self.device, non_blocking=True)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device, non_blocking=True)
                elif isinstance(targets, dict):
                    targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

                # Обработка info dict (содержит idx как тензор)
                if isinstance(info, dict) and 'idx' in info and isinstance(info['idx'], torch.Tensor):
                    info['idx'] = info['idx'].to(self.device, non_blocking=True)

                # Forward pass с AMP
                if self.use_amp:
                    with autocast():
                        try:
                            outputs = self.model(inputs, return_dict=True)
                        except TypeError:
                            outputs = self.model(inputs)

                        # КРИТИЧЕСКИЙ FIX: Обработка tuple output (tensor, attention_weights)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # Берем только тензор, игнорируем attention_weights

                        loss = self._compute_loss(outputs, targets)
                else:
                    try:
                        outputs = self.model(inputs, return_dict=True)
                    except TypeError:
                        outputs = self.model(inputs)

                    # КРИТИЧЕСКИЙ FIX: Обработка tuple output (tensor, attention_weights)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Берем только тензор, игнорируем attention_weights

                    loss = self._compute_loss(outputs, targets)
                
                # Накопление loss (поддержка dict)
                loss_to_add = loss['total_loss'] if isinstance(loss, dict) else loss
                val_loss += loss_to_add.detach()
                
                # Сохраняем outputs и targets для расчета метрик
                if isinstance(outputs, torch.Tensor):
                    all_outputs.append(outputs.detach().cpu())
                elif isinstance(outputs, dict):
                    # Конвертируем dict предсказаний в тензор [batch, 20]
                    try:
                        parts = []
                        # Future returns
                        parts.append(outputs['future_return_15m'])
                        parts.append(outputs['future_return_1h'])
                        parts.append(outputs['future_return_4h'])
                        parts.append(outputs['future_return_12h'])
                        # Direction: берём классы через argmax по логитам
                        for tf in ['15m', '1h', '4h', '12h']:
                            logits = outputs[f'direction_{tf}']  # [B,3]
                            classes = torch.argmax(torch.softmax(logits, dim=-1), dim=-1, keepdim=True).float()
                            parts.append(classes)
                        # Long/Short levels
                        parts.append(outputs['long_will_reach_1pct_4h'])
                        parts.append(outputs['long_will_reach_2pct_4h'])
                        parts.append(outputs['long_will_reach_3pct_12h'])
                        parts.append(outputs['long_will_reach_5pct_12h'])
                        parts.append(outputs['short_will_reach_1pct_4h'])
                        parts.append(outputs['short_will_reach_2pct_4h'])
                        parts.append(outputs['short_will_reach_3pct_12h'])
                        parts.append(outputs['short_will_reach_5pct_12h'])
                        # Risk metrics
                        parts.append(outputs['max_drawdown_1h'])
                        parts.append(outputs['max_rally_1h'])
                        parts.append(outputs['max_drawdown_4h'])
                        parts.append(outputs['max_rally_4h'])
                        tensor_out = torch.cat(parts, dim=1).detach().cpu()
                        all_outputs.append(tensor_out)
                    except Exception as e:
                        self.logger.warning(
                            f"⚠️ Не удалось конвертировать outputs dict в тензор: {e}. "
                            f"Пропускаем сохранение all_outputs для этого батча."
                        )
                # Обрабатываем targets в зависимости от типа
                if isinstance(targets, torch.Tensor):
                    all_targets.append(targets.detach().cpu())
                elif isinstance(targets, dict):
                    # Если targets - dict, берем только тензоры из него
                    targets_cpu = {}
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets_cpu[k] = v.detach().cpu()
                    all_targets.append(targets_cpu)
                else:
                    # Если targets - это что-то другое (tuple и т.д.)
                    self.logger.warning(f"⚠️ Неожиданный тип targets: {type(targets)}")
                    all_targets.append(targets)
                # Логиты направлений могут теряться при конкатенации тензоров
                # Сохраняем логиты направлений, если доступны
                if hasattr(outputs, '_direction_logits'):
                    all_direction_logits.append(outputs._direction_logits.detach().cpu())
                elif isinstance(outputs, dict) and 'direction_15m' in outputs:
                    try:
                        dir_logits = torch.stack([
                            outputs['direction_15m'],
                            outputs['direction_1h'],
                            outputs['direction_4h'],
                            outputs['direction_12h']
                        ], dim=1)  # (B,4,3)
                        all_direction_logits.append(dir_logits.detach().cpu())
                    except Exception as e:
                        self.logger.warning(f"⚠️ Не удалось извлечь direction logits из outputs dict: {e}")
                
                # Периодическая очистка кэша
                if self.device.type == 'cuda' and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Финальная синхронизация
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Проверяем что есть данные для объединения
        if not all_outputs:
            self.logger.warning("⚠️ Val loader пустой - невозможно рассчитать метрики")
            return {
                'val_loss': float('inf'),
                'macro_f1_overall': 0.0,
                'win_rate_overall': 0.0
            }
        
        # Объединяем все батчи
        all_outputs = torch.cat(all_outputs, dim=0)
        
        # Обрабатываем targets - могут быть как тензоры, так и dict
        if all_targets and isinstance(all_targets[0], torch.Tensor):
            all_targets = torch.cat(all_targets, dim=0)
        elif all_targets and isinstance(all_targets[0], dict):
            # Если targets - словари, конкатенируем каждый ключ отдельно
            combined_targets = {}
            for key in all_targets[0].keys():
                combined_targets[key] = torch.cat([t[key] for t in all_targets], dim=0)
            all_targets = combined_targets
        else:
            self.logger.warning(f"⚠️ Неожиданный формат targets: {type(all_targets[0]) if all_targets else 'empty'}")
            all_targets = torch.zeros_like(all_outputs) if all_outputs.numel() > 0 else torch.zeros(1)
        direction_logits = None
        if len(all_direction_logits) > 0:
            direction_logits = torch.cat(all_direction_logits, dim=0)  # (N, 4, 3)
        
        # Приводим targets к правильной размерности если нужно
        if isinstance(all_targets, torch.Tensor):
            if all_targets.dim() == 3 and all_targets.shape[1] == 1:
                all_targets = all_targets.squeeze(1)
        
        # Расчет базовых метрик
        avg_val_loss = (val_loss / len(val_loader)).item()
        metrics = {'val_loss': avg_val_loss}
        
        # Расчет enhanced метрик
        try:
            # НЕ пытаемся установить атрибут на тензор - это вызывает ошибку
            # Вместо этого передаем логиты отдельно если нужно
            direction_metrics = self.compute_direction_metrics(all_outputs, all_targets)
            trading_metrics = self.compute_trading_metrics(all_outputs, all_targets)
            
            metrics.update(direction_metrics)
            metrics.update(trading_metrics)
            
            # Логирование ключевых метрик
            if 'direction_accuracy_overall' in metrics:
                self.logger.info(f"📊 Direction Accuracy: {metrics['direction_accuracy_overall']:.3f}")
            
            if 'macro_f1_overall' in metrics:
                self.logger.info(f"📊 Macro F1 Score: {metrics['macro_f1_overall']:.3f}")
            
            # Логирование разнообразия предсказаний
            if 'pred_entropy_overall' in metrics:
                self.logger.info(f"🎲 Prediction Diversity: Entropy={metrics['pred_entropy_overall']:.3f} "
                               f"(LONG: {metrics.get('pred_long_ratio_overall', 0):.1%}, "
                               f"SHORT: {metrics.get('pred_short_ratio_overall', 0):.1%}, "
                               f"FLAT: {metrics.get('pred_flat_ratio_overall', 0):.1%})")
            
            # Добавляем детальные метрики по классам
            timeframes = ['15m', '1h', '4h', '12h']
            for i, tf in enumerate(timeframes):
                # Извлекаем предсказания и цели для текущего таймфрейма
                if hasattr(all_outputs, '_direction_logits'):
                    direction_logits = all_outputs._direction_logits[:, i, :]
                    pred_classes = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
                else:
                    # Проверяем тип all_outputs
                    if isinstance(all_outputs, dict):
                        # Если outputs - словарь, используем соответствующий ключ
                        direction_key = f'direction_{tf}'
                        if direction_key in all_outputs:
                            direction_outputs = all_outputs[direction_key]
                            pred_classes = torch.argmax(torch.softmax(direction_outputs, dim=-1), dim=-1)
                        else:
                            continue  # Пропускаем если ключа нет
                    else:
                        # Если outputs - тензор, используем индексацию
                        direction_outputs = all_outputs[:, 4+i]
                        pred_classes = torch.round(direction_outputs).clamp(0, 2).long()
                
                # Извлекаем истинные классы
                if isinstance(all_targets, dict):
                    # Если targets - словарь, используем соответствующий ключ
                    direction_key = f'direction_{tf}'
                    if direction_key in all_targets:
                        true_classes = all_targets[direction_key].long()
                    else:
                        continue  # Пропускаем если ключа нет
                else:
                    # Если targets - тензор, используем индексацию
                    true_classes = all_targets[:, 4+i].long()
                
                # Расчет метрик распределения классов
                class_metrics = self.compute_class_distribution_metrics(pred_classes, true_classes, tf)
                metrics.update(class_metrics)
                
                # Логирование ключевых метрик по классам
                if i == 0:  # Детально логируем только для первого таймфрейма
                    self.logger.info(f"\n📊 Детальные метрики классов для {tf}:")
                    for class_name in ['LONG', 'SHORT', 'FLAT']:
                        pred_pct = class_metrics.get(f'{tf}_{class_name}_pred_pct', 0)
                        true_pct = class_metrics.get(f'{tf}_{class_name}_true_pct', 0)
                        precision = class_metrics.get(f'{tf}_{class_name}_precision', 0)
                        recall = class_metrics.get(f'{tf}_{class_name}_recall', 0)
                        
                        self.logger.info(f"   {class_name}: pred={pred_pct:.1f}% true={true_pct:.1f}% | "
                                       f"precision={precision:.3f} recall={recall:.3f}")
            
            # Логирование уверенности если доступна
            if hasattr(all_outputs, '_confidence_scores'):
                confidence_scores = all_outputs._confidence_scores.cpu()
                # Преобразуем из [-1, 1] в [0, 1] (так как используем Tanh в модели)
                confidence_probs = (confidence_scores + 1) / 2
                
                avg_confidence = confidence_probs.mean().item()
                min_confidence = confidence_probs.min().item()
                max_confidence = confidence_probs.max().item()
                
                self.logger.info(f"💪 Confidence Scores: avg={avg_confidence:.3f}, "
                               f"min={min_confidence:.3f}, max={max_confidence:.3f}")
                
                # Процент высокоуверенных предсказаний
                high_conf_threshold = 0.6
                high_conf_ratio = (confidence_probs > high_conf_threshold).float().mean().item()
                self.logger.info(f"   Высокоуверенных (>{high_conf_threshold}): {high_conf_ratio:.1%}")
            
            if 'win_rate_overall' in metrics:
                self.logger.info(f"💰 Win Rate: {metrics['win_rate_overall']:.3f}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка расчета enhanced метрик: {e}")
        
        # Добавляем сводные метрики распределения для логирования
        predicted_distribution = {}
        actual_distribution = {}
        prediction_entropy = {}
        
        # Добавляем в метрики для использования в StagedTrainingManager
        overall_entropy = 0.0
        overall_long_pred = 0.0
        overall_short_pred = 0.0
        overall_flat_pred = 0.0
        
        for i, tf in enumerate(['15m', '1h', '4h', '12h']):
            # Извлекаем предсказания направлений
            if hasattr(all_outputs, '_direction_logits'):
                direction_logits = all_outputs._direction_logits[:, i, :]
                probs = torch.softmax(direction_logits, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1)
                
                # Энтропия предсказаний
                log_probs = torch.log(probs + 1e-8)
                entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
                normalized_entropy = entropy / np.log(3)  # Нормализуем на макс энтропию
                prediction_entropy[f'direction_{tf}'] = normalized_entropy
                
                # Отладка: проверяем значения
                if i == 0:  # Только для первого таймфрейма
                    self.logger.debug(f"🔍 Direction entropy debug for {tf}:")
                    self.logger.debug(f"   Raw entropy: {entropy:.4f}")
                    self.logger.debug(f"   Normalized: {normalized_entropy:.4f}")
                    self.logger.debug(f"   Probs shape: {probs.shape}")
                    self.logger.debug(f"   Sample probs: {probs[0].cpu().numpy()}")
            else:
                pred_classes = torch.round(all_outputs[:, 4+i]).clamp(0, 2).long()
                # Для обратной совместимости считаем энтропию из распределения классов
                unique, counts = torch.unique(pred_classes, return_counts=True)
                probs = counts.float() / len(pred_classes)
                # Создаем полное распределение вероятностей для 3 классов
                full_probs = torch.zeros(3)
                for cls, prob in zip(unique, probs):
                    if cls < 3:
                        full_probs[cls] = prob
                # Считаем энтропию
                full_probs = full_probs + 1e-8  # Избегаем log(0)
                entropy = -torch.sum(full_probs * torch.log(full_probs)).item()
                normalized_entropy = entropy / np.log(3)
                prediction_entropy[f'direction_{tf}'] = normalized_entropy
            
            true_classes = all_targets[:, 4+i].long()
            
            # Распределение предсказаний
            pred_dist = {}
            actual_dist = {}
            for cls, name in enumerate(['LONG', 'SHORT', 'FLAT']):
                pred_dist[name] = (pred_classes == cls).float().mean().item()
                actual_dist[name] = (true_classes == cls).float().mean().item()
            
            predicted_distribution[f'direction_{tf}'] = pred_dist
            actual_distribution[f'direction_{tf}'] = actual_dist
            
            # Накапливаем для общих метрик (только для первого таймфрейма 15m)
            if i == 0:
                overall_entropy = prediction_entropy.get(f'direction_{tf}', 0.0)
                overall_long_pred = pred_dist.get('LONG', 0.0)
                overall_short_pred = pred_dist.get('SHORT', 0.0)
                overall_flat_pred = pred_dist.get('FLAT', 0.0)
        
        # Средняя энтропия
        if prediction_entropy:
            avg_entropy = np.mean(list(prediction_entropy.values()))
            prediction_entropy['average'] = avg_entropy
        
        # Добавляем метрики для StagedTrainingManager
        metrics['entropy'] = overall_entropy
        metrics['class_distribution'] = {
            'long': overall_long_pred,
            'short': overall_short_pred,
            'flat': overall_flat_pred
        }
        
        metrics['predicted_distribution'] = predicted_distribution
        metrics['actual_distribution'] = actual_distribution
        metrics['prediction_entropy'] = prediction_entropy
        
        return metrics
