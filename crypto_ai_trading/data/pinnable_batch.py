"""
Класс для батчей с поддержкой pin_memory для RTX 5090
"""
import torch
from typing import NamedTuple, Optional


class PinnableBatch(NamedTuple):
    """Батч данных с поддержкой pin_memory"""
    inputs: torch.Tensor
    targets: torch.Tensor
    indices: torch.Tensor
    
    def pin_memory(self):
        """Перемещает все тензоры в pinned memory"""
        return PinnableBatch(
            inputs=self.inputs.pin_memory(),
            targets=self.targets.pin_memory(),
            indices=self.indices.pin_memory()
        )
    
    def to(self, device, non_blocking=True):
        """Перемещает все тензоры на устройство"""
        return PinnableBatch(
            inputs=self.inputs.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
            indices=self.indices.to(device, non_blocking=non_blocking)
        )


def robust_collate_fn(batch):
    """Улучшенная функция collate для RTX 5090 с гарантированной поддержкой pin_memory
    
    Возвращает PinnableBatch который корректно обрабатывается PyTorch DataLoader
    """
    # Извлекаем компоненты батча
    inputs_list = []
    targets_list = []
    indices_list = []
    
    for item in batch:
        inputs, targets, info = item
        inputs_list.append(inputs)
        targets_list.append(targets)
        
        # Обрабатываем индексы
        if isinstance(info, dict) and 'idx' in info:
            indices_list.append(info['idx'])
        elif isinstance(info, (int, torch.Tensor)):
            indices_list.append(info)
        else:
            indices_list.append(0)  # Значение по умолчанию
    
    # Создаем батчи
    inputs_batch = torch.stack(inputs_list)
    targets_batch = torch.stack(targets_list)
    indices_batch = torch.tensor(indices_list, dtype=torch.long)
    
    # Возвращаем PinnableBatch
    return PinnableBatch(
        inputs=inputs_batch,
        targets=targets_batch,
        indices=indices_batch
    )