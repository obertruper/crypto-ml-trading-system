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

class OptimizedTrainer(Trainer):
    """Оптимизированная версия Trainer с максимальной производительностью GPU"""
    
    def __init__(self, model: nn.Module, config: Dict, device: Optional[torch.device] = None):
        super().__init__(model, config, device)
        
        self.logger = get_logger("OptimizedTrainer")
        
        # Оптимизации для GPU
        self.async_metrics = True  # Асинхронный расчет метрик
        self.log_interval = 10  # Логировать каждые N батчей
        self.metrics_buffer = deque(maxlen=100)  # Буфер для метрик
        
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
        """Оптимизированное обучение одной эпохи"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {}
        batch_times = deque(maxlen=50)
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        # Обнуление градиентов с оптимизацией памяти
        self.optimizer.zero_grad(set_to_none=True)
        
        # Время начала эпохи
        epoch_start = time.time()
        last_log_time = epoch_start
        
        for batch_idx, (inputs, targets, info) in enumerate(progress_bar):
            batch_start = time.time()
            
            # Асинхронный перенос на GPU с non_blocking
            inputs = inputs.to(self.device, non_blocking=True)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device, non_blocking=True)
            elif isinstance(targets, dict):
                targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
            
            # Forward pass с AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
            
            # Проверка на NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"⚠️ NaN/Inf обнаружен в loss на батче {batch_idx}")
                self.logger.warning(f"   Loss value: {loss.item()}")
                # Пропускаем этот батч
                self.optimizer.zero_grad()
                continue
                
            # Проверка outputs на NaN
            if torch.isnan(outputs).any():
                self.logger.warning(f"⚠️ NaN обнаружен в outputs на батче {batch_idx}")
                self.logger.warning(f"   NaN count: {torch.isnan(outputs).sum().item()}")
                # Пропускаем этот батч
                self.optimizer.zero_grad()
                continue
            
            # Нормализация loss для gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Обновление весов каждые N шагов
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Unscale перед clipping
                    self.scaler.unscale_(self.optimizer)
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Обновление EMA модели
                    self._update_ema()
                
                # Обнуление градиентов
                self.optimizer.zero_grad(set_to_none=True)
            
            # Асинхронное накопление метрик (без .item() каждый батч)
            batch_loss = loss.detach() * self.gradient_accumulation_steps
            epoch_loss += batch_loss
            
            # Время батча
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Периодическое логирование
            if batch_idx % self.log_interval == 0:
                # Синхронизация для получения актуальных значений
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                current_loss = batch_loss.item()
                avg_batch_time = np.mean(batch_times)
                samples_per_sec = train_loader.batch_size / avg_batch_time
                
                # Расширенная статистика GPU для RTX 5090
                if self.device.type == 'cuda':
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    gpu_utilization = (gpu_memory_allocated / gpu_memory_reserved * 100) if gpu_memory_reserved > 0 else 0
                else:
                    gpu_memory_allocated = 0
                    gpu_memory_reserved = 0
                    gpu_utilization = 0
                
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'samples/s': f'{samples_per_sec:.0f}',
                    'gpu_mem': f'{gpu_memory_allocated:.1f}/{gpu_memory_reserved:.1f}GB',
                    'gpu_use': f'{gpu_utilization:.0f}%',
                    'batch_ms': f'{avg_batch_time*1000:.0f}'
                })
                
                # Детальное логирование каждые 100 батчей
                if batch_idx % 100 == 0 and batch_idx > 0:
                    elapsed = time.time() - last_log_time
                    self.logger.info(f"Batch {batch_idx}: "
                                   f"loss={current_loss:.4f}, "
                                   f"speed={samples_per_sec:.0f} samples/s, "
                                   f"time={elapsed:.1f}s")
                    last_log_time = time.time()
            
            # Периодическая очистка кэша GPU
            if self.device.type == 'cuda' and batch_idx % self.gpu_cache_clear_freq == 0:
                torch.cuda.empty_cache()
        
        # Финальная синхронизация
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Вычисление средних метрик
        avg_loss = (epoch_loss / len(train_loader)).item()
        epoch_time = time.time() - epoch_start
        
        self.logger.info(f"📊 Эпоха завершена за {epoch_time:.1f}с")
        self.logger.info(f"   Средняя скорость: {len(train_loader.dataset)/epoch_time:.0f} samples/s")
        self.logger.info(f"   Средний loss: {avg_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'samples_per_second': len(train_loader.dataset) / epoch_time
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
                
                # Forward pass с AMP
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self._compute_loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                # Накопление loss
                val_loss += loss.detach()
                
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
        patience_counter = 0
        
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
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Валидация с расширенными метриками
            if val_loader is not None:
                val_metrics = self.validate_with_enhanced_metrics(val_loader)
                self.history['val_loss'].append(val_metrics['val_loss'])
                
                # Early stopping
                if val_metrics['val_loss'] < best_val_loss - self.config['model'].get('min_delta', 0.0001):
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    # Сохранение лучшей модели
                    self._save_checkpoint(epoch, val_metrics['val_loss'], is_best=True)
                else:
                    patience_counter += 1
                
                self.logger.info(f"📈 Train Loss: {train_metrics['loss']:.4f}, "
                               f"Val Loss: {val_metrics['val_loss']:.4f} "
                               f"(best: {best_val_loss:.4f}, patience: {patience_counter})")
                
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info("⚠️ Early stopping triggered!")
                    break
            else:
                self.logger.info(f"📈 Train Loss: {train_metrics['loss']:.4f}")
            
            # Scheduler step
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    # ReduceLROnPlateau требует метрику
                    if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                        if val_loader is not None:
                            self.scheduler.step(val_metrics['val_loss'])
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
        
        self.logger.info("✅ Обучение завершено!")
        return self.history
    
    def compute_direction_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Расчет метрик для direction предсказаний
        
        Args:
            outputs: (batch_size, 20) - выходы модели
            targets: (batch_size, 20) - целевые значения
            
        Returns:
            Dict с метриками directional accuracy и win rate
        """
        metrics = {}
        
        # Извлекаем direction переменные (индексы 4-7)
        direction_outputs = outputs[:, 4:8]  # direction_15m, 1h, 4h, 12h
        direction_targets = targets[:, 4:8]
        
        # Расчет directional accuracy для каждого таймфрейма
        timeframes = ['15m', '1h', '4h', '12h']
        
        for i, tf in enumerate(timeframes):
            # ИСПРАВЛЕНИЕ: Правильная интерпретация выходов модели
            # Если outputs имеет _direction_logits, используем их
            if hasattr(outputs, '_direction_logits'):
                # Используем логиты для получения классов через softmax + argmax
                direction_logits = outputs._direction_logits[:, i, :]  # (batch_size, 3)
                pred_classes = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
            else:
                # Fallback: интерпретируем как непрерывные значения [0, 2]
                pred_classes = torch.round(direction_outputs[:, i]).clamp(0, 2).long()
            
            true_classes = direction_targets[:, i].long()
            
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
    
    def compute_trading_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Расчет торговых метрик: win rate, profit factor, etc.
        
        Args:
            outputs: (batch_size, 20) - выходы модели
            targets: (batch_size, 20) - целевые значения
            
        Returns:
            Dict с торговыми метриками
        """
        metrics = {}
        
        # Извлекаем нужные переменные
        future_returns = targets[:, 0:4]  # future_return_15m, 1h, 4h, 12h
        direction_outputs = outputs[:, 4:8]
        direction_targets = targets[:, 4:8]
        
        timeframes = ['15m', '1h', '4h', '12h']
        
        for i, tf in enumerate(timeframes):
            # ИСПРАВЛЕНИЕ: Правильная интерпретация предсказанных направлений
            if hasattr(outputs, '_direction_logits'):
                # Используем логиты для получения классов через softmax + argmax
                direction_logits = outputs._direction_logits[:, i, :]  # (batch_size, 3)
                pred_classes = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
            else:
                # Fallback: интерпретируем как непрерывные значения [0, 2]
                pred_classes = torch.round(direction_outputs[:, i]).clamp(0, 2).long()
            
            true_returns = future_returns[:, i]
            
            # Имитируем торговые сигналы
            # UP предсказание (0) = LONG позиция
            long_mask = (pred_classes == 0)
            # DOWN предсказание (1) = SHORT позиция  
            short_mask = (pred_classes == 1)
            # FLAT предсказание (2) = нет позиции
            
            if long_mask.sum() > 0 or short_mask.sum() > 0:
                # Расчет P&L
                pnl = torch.zeros_like(true_returns)
                
                # LONG позиции: прибыль = изменение цены
                if long_mask.sum() > 0:
                    pnl[long_mask] = true_returns[long_mask]
                
                # SHORT позиции: прибыль = -изменение цены
                if short_mask.sum() > 0:
                    pnl[short_mask] = -true_returns[short_mask]
                
                # Убираем комиссию
                commission = 0.001  # 0.1%
                trading_mask = long_mask | short_mask
                pnl[trading_mask] -= commission
                
                # Win Rate
                profitable_trades = pnl[trading_mask] > 0
                if trading_mask.sum() > 0:
                    win_rate = profitable_trades.float().mean().item()
                    metrics[f'win_rate_{tf}'] = win_rate
                    
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
        
        # Общие метрики
        if any(f'win_rate_{tf}' in metrics for tf in timeframes):
            # Средний win rate
            win_rates = [metrics[f'win_rate_{tf}'] for tf in timeframes if f'win_rate_{tf}' in metrics]
            if win_rates:
                metrics['win_rate_overall'] = np.mean(win_rates)
        
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
                
                outputs = self.model(inputs)
                
                # Обрабатываем каждый таймфрейм
                for i, tf in enumerate(['15m', '1h', '4h', '12h']):
                    if hasattr(outputs, '_direction_logits'):
                        direction_logits = outputs._direction_logits[:, i, :]
                        pred_classes = torch.argmax(torch.softmax(direction_logits, dim=-1), dim=-1)
                    else:
                        direction_outputs = outputs[:, 4+i]
                        pred_classes = torch.round(direction_outputs).clamp(0, 2).long()
                    
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
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(tqdm(val_loader, desc="Enhanced Validation", leave=False)):
                # Асинхронный перенос на GPU
                inputs = inputs.to(self.device, non_blocking=True)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device, non_blocking=True)
                elif isinstance(targets, dict):
                    targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
                
                # Forward pass с AMP
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self._compute_loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                # Накопление loss
                val_loss += loss.detach()
                
                # Сохраняем outputs и targets для расчета метрик
                all_outputs.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
                
                # Периодическая очистка кэша
                if self.device.type == 'cuda' and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Финальная синхронизация
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Объединяем все батчи
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Приводим targets к правильной размерности если нужно
        if all_targets.dim() == 3 and all_targets.shape[1] == 1:
            all_targets = all_targets.squeeze(1)
        
        # Расчет базовых метрик
        avg_val_loss = (val_loss / len(val_loader)).item()
        metrics = {'val_loss': avg_val_loss}
        
        # Расчет enhanced метрик
        try:
            direction_metrics = self.compute_direction_metrics(all_outputs, all_targets)
            trading_metrics = self.compute_trading_metrics(all_outputs, all_targets)
            
            metrics.update(direction_metrics)
            metrics.update(trading_metrics)
            
            # Логирование ключевых метрик
            if 'direction_accuracy_overall' in metrics:
                self.logger.info(f"📊 Direction Accuracy: {metrics['direction_accuracy_overall']:.3f}")
            
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
                    direction_outputs = all_outputs[:, 4+i]
                    pred_classes = torch.round(direction_outputs).clamp(0, 2).long()
                
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
        
        return metrics