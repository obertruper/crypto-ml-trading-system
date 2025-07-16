"""
Утилиты для диагностики и предотвращения NaN в процессе обучения
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

class NaNDiagnostics:
    """Класс для отслеживания и диагностики NaN значений"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.nan_history = {
            'inputs': [],
            'outputs': [],
            'gradients': [],
            'losses': []
        }
        self.batch_counter = 0
        
    def check_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Проверка тензора на NaN/Inf с подробной диагностикой"""
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan or has_inf:
            self.logger.warning(f"⚠️ Обнаружены проблемы в {name}:")
            if has_nan:
                nan_count = torch.isnan(tensor).sum().item()
                nan_pct = nan_count / tensor.numel() * 100
                self.logger.warning(f"  - NaN: {nan_count} значений ({nan_pct:.2f}%)")
            
            if has_inf:
                inf_count = torch.isinf(tensor).sum().item()
                inf_pct = inf_count / tensor.numel() * 100
                self.logger.warning(f"  - Inf: {inf_count} значений ({inf_pct:.2f}%)")
            
            # Статистика по валидным значениям
            valid_mask = ~(torch.isnan(tensor) | torch.isinf(tensor))
            if valid_mask.any():
                valid_values = tensor[valid_mask]
                self.logger.warning(f"  - Валидные значения: min={valid_values.min().item():.4f}, "
                                  f"max={valid_values.max().item():.4f}, "
                                  f"mean={valid_values.mean().item():.4f}, "
                                  f"std={valid_values.std().item():.4f}")
            
            # Детальная информация о форме и типе
            self.logger.warning(f"  - Shape: {tensor.shape}, dtype: {tensor.dtype}")
            
            return True
        
        return False
    
    def check_model_parameters(self, model: torch.nn.Module) -> Dict[str, bool]:
        """Проверка всех параметров модели на NaN/Inf"""
        problematic_params = {}
        
        for name, param in model.named_parameters():
            if param is not None:
                has_issues = self.check_tensor(param.data, f"parameter {name}")
                if has_issues:
                    problematic_params[name] = True
                    
                # Проверка градиентов если они есть
                if param.grad is not None:
                    grad_issues = self.check_tensor(param.grad, f"gradient {name}")
                    if grad_issues:
                        problematic_params[f"{name}_grad"] = True
        
        return problematic_params
    
    def sanitize_tensor(self, tensor: torch.Tensor, 
                       replace_nan: float = 0.0,
                       replace_inf: float = 1e6) -> torch.Tensor:
        """Замена NaN и Inf значений в тензоре"""
        # Сохраняем оригинальное устройство
        device = tensor.device
        
        # Замена NaN
        if torch.isnan(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=replace_nan)
        
        # Замена Inf
        if torch.isinf(tensor).any():
            tensor = torch.where(torch.isinf(tensor), 
                               torch.full_like(tensor, replace_inf).to(device), 
                               tensor)
        
        return tensor
    
    def check_dataframe(self, df: pd.DataFrame, name: str = "dataframe") -> Dict[str, int]:
        """Проверка DataFrame на NaN значения"""
        nan_counts = df.isna().sum()
        problematic_cols = nan_counts[nan_counts > 0]
        
        if len(problematic_cols) > 0:
            self.logger.warning(f"⚠️ NaN значения в {name}:")
            for col, count in problematic_cols.items():
                pct = count / len(df) * 100
                self.logger.warning(f"  - {col}: {count} значений ({pct:.2f}%)")
        
        return problematic_cols.to_dict()
    
    def log_batch_stats(self, 
                       inputs: torch.Tensor,
                       outputs: torch.Tensor,
                       loss: torch.Tensor,
                       batch_idx: int):
        """Логирование статистики батча для отладки"""
        self.batch_counter += 1
        
        # Проверка на проблемы
        input_issues = self.check_tensor(inputs, f"batch {batch_idx} inputs")
        output_issues = self.check_tensor(outputs, f"batch {batch_idx} outputs")
        loss_issues = self.check_tensor(loss, f"batch {batch_idx} loss")
        
        # Сохранение истории
        if input_issues:
            self.nan_history['inputs'].append(batch_idx)
        if output_issues:
            self.nan_history['outputs'].append(batch_idx)
        if loss_issues:
            self.nan_history['losses'].append(batch_idx)
        
        # Периодический отчет
        if self.batch_counter % 100 == 0:
            self.print_summary()
    
    def print_summary(self):
        """Вывод сводки по NaN проблемам"""
        total_batches = self.batch_counter
        
        self.logger.info("📊 Сводка по NaN/Inf проблемам:")
        for category, batch_list in self.nan_history.items():
            if batch_list:
                count = len(batch_list)
                pct = count / total_batches * 100
                self.logger.info(f"  - {category}: {count} батчей ({pct:.2f}%)")
                if count <= 5:
                    self.logger.info(f"    Батчи: {batch_list}")
                else:
                    self.logger.info(f"    Первые 5: {batch_list[:5]}")
    
    def create_nan_safe_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Обертка оптимизатора для предотвращения NaN в градиентах"""
        original_step = optimizer.step
        
        def safe_step(closure=None):
            # Проверка градиентов перед шагом оптимизатора
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        # Замена NaN/Inf в градиентах
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            self.logger.warning("Обнаружены NaN/Inf в градиентах, заменяем на 0")
                            param.grad = self.sanitize_tensor(param.grad, replace_nan=0.0, replace_inf=0.0)
            
            # Вызов оригинального step
            return original_step(closure)
        
        optimizer.step = safe_step
        return optimizer


def add_gradient_hooks(model: torch.nn.Module, logger: Optional[logging.Logger] = None):
    """Добавление хуков для отслеживания градиентов"""
    logger = logger or logging.getLogger(__name__)
    
    def gradient_hook(module, grad_input, grad_output):
        """Хук для проверки градиентов"""
        module_name = module.__class__.__name__
        
        # Проверка grad_output
        for i, grad in enumerate(grad_output):
            if grad is not None:
                if torch.isnan(grad).any():
                    logger.warning(f"NaN в grad_output[{i}] модуля {module_name}")
                if torch.isinf(grad).any():
                    logger.warning(f"Inf в grad_output[{i}] модуля {module_name}")
                    
                # Статистика градиентов
                grad_norm = grad.norm().item()
                if grad_norm > 100:
                    logger.warning(f"Большая норма градиента в {module_name}: {grad_norm:.4f}")
    
    # Регистрация хуков для всех модулей
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Только для листовых модулей
            module.register_backward_hook(gradient_hook)
    
    logger.info(f"Зарегистрированы хуки для отслеживания градиентов")


def stabilize_model_initialization(model: torch.nn.Module, method: str = 'xavier'):
    """Стабильная инициализация весов модели"""
    
    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            if method == 'xavier':
                torch.nn.init.xavier_uniform_(module.weight, gain=0.5)  # Меньший gain для стабильности
            elif method == 'kaiming':
                torch.nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
                
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            if module.weight is not None:
                torch.nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    model.apply(init_weights)
    
    # Особая инициализация для позиционных эмбеддингов
    for name, param in model.named_parameters():
        if 'pos_embedding' in name or 'positional' in name:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
    
    return model