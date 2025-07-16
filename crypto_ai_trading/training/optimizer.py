"""
Конфигурации оптимизаторов и планировщиков learning rate
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, List
import math


def get_optimizer(optimizer_name: str, 
                 parameters, 
                 lr: float = 1e-3,
                 **kwargs) -> optim.Optimizer:
    """Создание оптимизатора по имени"""
    
    optimizers = {
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
        'Adamax': optim.Adamax,
        'Adadelta': optim.Adadelta,
        'Adagrad': optim.Adagrad,
        'LBFGS': optim.LBFGS,
        'RAdam': RAdam,
        'Lookahead': Lookahead,
        'Ranger': Ranger
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Специальная обработка для некоторых оптимизаторов
    if optimizer_name == 'SGD':
        kwargs.setdefault('momentum', 0.9)
        kwargs.setdefault('nesterov', True)
    elif optimizer_name == 'AdamW':
        kwargs.setdefault('weight_decay', 0.01)
        kwargs.setdefault('betas', (0.9, 0.999))
    elif optimizer_name == 'Lookahead':
        # Lookahead требует базовый оптимизатор
        base_optimizer = get_optimizer('AdamW', parameters, lr, **kwargs)
        return Lookahead(base_optimizer, k=5, alpha=0.5)
    elif optimizer_name == 'Ranger':
        # Ranger = RAdam + Lookahead
        return Ranger(parameters, lr=lr, **kwargs)
    
    return optimizers[optimizer_name](parameters, lr=lr, **kwargs)


def get_scheduler(scheduler_name: str,
                 optimizer: optim.Optimizer,
                 **kwargs) -> _LRScheduler:
    """Создание планировщика learning rate"""
    
    schedulers = {
        'StepLR': optim.lr_scheduler.StepLR,
        'MultiStepLR': optim.lr_scheduler.MultiStepLR,
        'ExponentialLR': optim.lr_scheduler.ExponentialLR,
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
        'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
        'CyclicLR': optim.lr_scheduler.CyclicLR,
        'OneCycleLR': optim.lr_scheduler.OneCycleLR,
        'LinearWarmup': LinearWarmupScheduler,
        'CosineWarmup': CosineWarmupScheduler
    }
    
    if scheduler_name not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    # Специальная обработка параметров для разных планировщиков
    if scheduler_name == 'StepLR':
        kwargs.setdefault('step_size', 10)
        kwargs.setdefault('gamma', 0.1)
    elif scheduler_name == 'MultiStepLR':
        kwargs.setdefault('milestones', [30, 60, 90])
        kwargs.setdefault('gamma', 0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        kwargs.setdefault('T_max', 100)
        kwargs.setdefault('eta_min', 1e-6)
    elif scheduler_name == 'OneCycleLR':
        kwargs.setdefault('max_lr', 0.1)
        kwargs.setdefault('total_steps', 1000)
    
    return schedulers[scheduler_name](optimizer, **kwargs)


class RAdam(optim.Optimizer):
    """Rectified Adam оптимизатор"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state['step'] += 1
                buffered = [None, None, None]

                N_sma_max = 2 / (1 - beta2) - 1
                beta2_t = beta2 ** state['step']
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # More conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                    ) / (1 - beta1 ** state['step'])
                elif self.degenerated_to_sgd:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                else:
                    step_size = -1

                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                elif step_size > 0:
                    p_data_fp32.add_(exp_avg, alpha=-step_size)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])

                p.data.copy_(p_data_fp32)

        return loss


class Lookahead(optim.Optimizer):
    """Lookahead оптимизатор"""
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.defaults = base_optimizer.defaults
        self.slow_state = [[p.clone().detach() for p in group['params']]
                          for group in self.param_groups]
        
        for group in self.param_groups:
            group['counter'] = 0

    def update(self, group, slow_weights):
        """Update the slow weights"""
        for p, q in zip(group['params'], slow_weights):
            if p.grad is None:
                continue
            q.data.add_(p.data - q.data, alpha=self.alpha)
            p.data.copy_(q.data)

    def update_lookahead(self):
        """Lookahead update"""
        for group, slow_weights in zip(self.param_groups, self.slow_state):
            self.update(group, slow_weights)

    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group['counter'] == 0:
                self.update_lookahead()
            group['counter'] += 1
            if group['counter'] >= self.k:
                group['counter'] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'fast_state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        }
        fast_state_dict = {
            'state': state_dict['fast_state'],
            'param_groups': state_dict['param_groups'],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.slow_state = [[p.clone().detach() for p in group['params']]
                          for group in self.param_groups]


class Ranger(optim.Optimizer):
    """Ranger оптимизатор = RAdam + Lookahead"""
    
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, betas=(0.95, 0.999),
                 eps=1e-8, weight_decay=0):
        
        # Создаем RAdam
        radam = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # Оборачиваем в Lookahead
        self.optimizer = Lookahead(radam, k=k, alpha=alpha)
        
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = self.optimizer.defaults
    
    def step(self, closure=None):
        return self.optimizer.step(closure)
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class LinearWarmupScheduler(_LRScheduler):
    """Планировщик с линейным разогревом"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Линейный разогрев
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Косинусное затухание после разогрева
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor 
                   for base_lr in self.base_lrs]


class CosineWarmupScheduler(_LRScheduler):
    """Планировщик с косинусным разогревом и затуханием"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Косинусный разогрев
            warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - self.last_epoch / self.warmup_epochs)))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Косинусное затухание
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor 
                   for base_lr in self.base_lrs]


def create_optimizer_groups(model: torch.nn.Module,
                          lr: float,
                          weight_decay: float = 0.01) -> List[Dict[str, Any]]:
    """
    Создание групп параметров с разными настройками для оптимизатора
    
    Args:
        model: модель
        lr: базовый learning rate
        weight_decay: коэффициент регуляризации
    
    Returns:
        Список групп параметров
    """
    # Параметры без weight decay (biases и нормализация)
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    
    # Группа 1: параметры с weight decay
    decay_params = []
    # Группа 2: параметры без weight decay
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_groups = [
        {
            'params': decay_params,
            'lr': lr,
            'weight_decay': weight_decay
        },
        {
            'params': no_decay_params,
            'lr': lr,
            'weight_decay': 0.0
        }
    ]
    
    return optimizer_groups


def get_linear_schedule_with_warmup(optimizer, 
                                   num_warmup_steps,
                                   num_training_steps,
                                   last_epoch=-1):
    """
    Создание планировщика с линейным разогревом и линейным затуханием
    
    Args:
        optimizer: оптимизатор
        num_warmup_steps: количество шагов разогрева
        num_training_steps: общее количество шагов обучения
        last_epoch: последняя эпоха
        
    Returns:
        Планировщик LR
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)