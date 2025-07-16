"""
Патч для trainer.py для работы с унифицированной моделью
"""

import torch
import torch.nn as nn
from typing import Dict, Union


def create_unified_loss_function(config: Dict) -> nn.Module:
    """Создание правильной loss функции для унифицированной модели"""
    from models.patchtst_unified import UnifiedTradingLoss
    return UnifiedTradingLoss(config)


def compute_loss_unified(outputs: torch.Tensor, 
                        targets: torch.Tensor,
                        criterion: nn.Module) -> torch.Tensor:
    """
    Упрощенная функция compute_loss для унифицированной архитектуры
    
    Args:
        outputs: (batch_size, 36) - предсказания модели
        targets: (batch_size, 36) - целевые значения
        criterion: loss функция
        
    Returns:
        loss: скаляр
    """
    # Проверка размерностей
    if outputs.shape[-1] != targets.shape[-1]:
        # Если размерности не совпадают, обрезаем или паддим
        if outputs.shape[-1] < targets.shape[-1]:
            # Обрезаем targets
            targets = targets[..., :outputs.shape[-1]]
        else:
            # Обрезаем outputs
            outputs = outputs[..., :targets.shape[-1]]
    
    # Применяем loss функцию
    loss = criterion(outputs, targets)
    
    # Проверка на NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        # Возвращаем безопасное значение
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    
    return loss


def patch_trainer_for_unified_model():
    """
    Применяет патч к trainer.py для работы с унифицированной моделью
    """
    import sys
    import os
    
    # Читаем оригинальный файл
    trainer_path = os.path.join(os.path.dirname(__file__), 'trainer.py')
    
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Патч 1: Исправляем создание loss функции
    old_loss_creation = """def _create_loss_function(self) -> nn.Module:
        \"\"\"Создание функции потерь\"\"\"
        loss_config = self.config.get('loss', {})
        loss_name = loss_config.get('name', 'mse')"""
    
    new_loss_creation = """def _create_loss_function(self) -> nn.Module:
        \"\"\"Создание функции потерь\"\"\"
        loss_config = self.config.get('loss', {})
        loss_name = loss_config.get('name', 'mse')
        
        # Проверяем если используется унифицированная модель
        model_name = self.config.get('model', {}).get('name', '')
        if model_name == 'UnifiedPatchTST':
            from models.patchtst_unified import UnifiedTradingLoss
            return UnifiedTradingLoss(self.config)"""
    
    content = content.replace(old_loss_creation, new_loss_creation)
    
    # Патч 2: Исправляем compute_loss
    old_compute = """def _compute_loss(self, outputs: Union[torch.Tensor, Dict], 
                     targets: Union[torch.Tensor, Dict]) -> torch.Tensor:
        \"\"\"Вычисление потерь\"\"\""""
    
    new_compute = """def _compute_loss(self, outputs: Union[torch.Tensor, Dict], 
                     targets: Union[torch.Tensor, Dict]) -> torch.Tensor:
        \"\"\"Вычисление потерь\"\"\"
        
        # Для унифицированной модели
        if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            if outputs.dim() == 2 and targets.dim() == 2:  # (batch, features)
                # Убеждаемся что размерности совпадают
                if outputs.shape[-1] != targets.shape[-1]:
                    min_size = min(outputs.shape[-1], targets.shape[-1])
                    outputs = outputs[..., :min_size]
                    targets = targets[..., :min_size]
                
                # Применяем loss напрямую
                return self.criterion(outputs, targets)"""
    
    content = content.replace(old_compute, new_compute)
    
    # Патч 3: Добавляем scheduler поддержку
    scheduler_patch = """
        # Специальная обработка для OneCycleLR
        if scheduler_name == 'OneCycleLR':
            # OneCycleLR требует total_steps
            if not hasattr(self, 'total_steps'):
                self.total_steps = self.epochs * 1000  # Примерная оценка
            scheduler_config['params']['total_steps'] = self.total_steps
            scheduler_config['params']['epochs'] = self.epochs"""
    
    if "scheduler_name = scheduler_config.get('name'" in content and "OneCycleLR" not in content:
        insert_pos = content.find("return get_scheduler(")
        content = content[:insert_pos] + scheduler_patch + "\n        \n        " + content[insert_pos:]
    
    # Сохраняем патченный файл
    patched_path = trainer_path.replace('.py', '_patched.py')
    with open(patched_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Патч создан: {patched_path}")
    print("Для применения выполните:")
    print(f"cp {trainer_path} {trainer_path}.backup")
    print(f"cp {patched_path} {trainer_path}")
    
    return patched_path


# Дополнительные улучшения для trainer
class ImprovedMetricsTracker:
    """Улучшенный трекер метрик с поддержкой многозадачности"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = {}
        self.output_names = self._get_output_names()
        
    def _get_output_names(self):
        """Получает имена выходов из конфига или модели"""
        # Берем из UnifiedPatchTSTForTrading
        from models.patchtst_unified import UnifiedPatchTSTForTrading
        dummy_model = UnifiedPatchTSTForTrading(self.config)
        return dummy_model.get_output_names()
    
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Вычисляет метрики для всех выходов"""
        metrics = {}
        
        # Общие метрики
        with torch.no_grad():
            # MAE для future returns
            mae_returns = torch.abs(outputs[:, :4] - targets[:, :4]).mean()
            metrics['mae_returns'] = mae_returns.item()
            
            # Accuracy для бинарных предсказаний
            binary_preds = (outputs[:, 4:] > 0.5).float()
            binary_targets = targets[:, 4:]
            accuracy = (binary_preds == binary_targets).float().mean()
            metrics['accuracy'] = accuracy.item()
            
            # Специфичные метрики для TP/SL
            tp_indices = list(range(4, 7)) + list(range(8, 11)) + \
                        list(range(20, 23)) + list(range(24, 27))
            sl_indices = [7, 11, 23, 27]
            
            if tp_indices:
                tp_accuracy = (binary_preds[:, [i-4 for i in tp_indices if i >= 4]] == 
                              binary_targets[:, [i-4 for i in tp_indices if i >= 4]]).float().mean()
                metrics['tp_accuracy'] = tp_accuracy.item()
            
            if sl_indices:
                sl_accuracy = (binary_preds[:, [i-4 for i in sl_indices if i >= 4]] == 
                              binary_targets[:, [i-4 for i in sl_indices if i >= 4]]).float().mean()
                metrics['sl_accuracy'] = sl_accuracy.item()
        
        return metrics


if __name__ == "__main__":
    # Применяем патч
    patch_trainer_for_unified_model()