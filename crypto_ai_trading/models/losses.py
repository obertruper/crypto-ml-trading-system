"""
Кастомные функции потерь для торговых моделей
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class TradingLoss(nn.Module):
    """Базовый класс для торговых функций потерь"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Применение метода редукции"""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class DirectionalLoss(TradingLoss):
    """Потери с учетом направления движения цены"""
    
    def __init__(self, 
                 alpha: float = 0.7,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: вес для правильного направления (1-alpha для magnitude)
            reduction: метод редукции
        """
        super().__init__(reduction)
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: предсказанные значения
            target: истинные значения
        """
        # MSE компонент
        mse_loss = self.mse(pred, target)
        
        # Направленный компонент (правильное предсказание знака)
        pred_direction = torch.sign(pred)
        target_direction = torch.sign(target)
        direction_correct = (pred_direction == target_direction).float()
        
        # Штраф за неправильное направление
        direction_loss = 1.0 - direction_correct
        
        # Комбинированная потеря
        loss = self.alpha * direction_loss + (1 - self.alpha) * mse_loss
        
        return self._reduce(loss)


class ProfitLoss(TradingLoss):
    """Потери на основе потенциальной прибыли/убытка"""
    
    def __init__(self,
                 transaction_cost: float = 0.001,
                 risk_penalty: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            transaction_cost: комиссия за транзакцию
            risk_penalty: множитель штрафа за убытки
            reduction: метод редукции
        """
        super().__init__(reduction)
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                entry_prices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: предсказанные цены/доходности
            target: истинные цены/доходности
            entry_prices: цены входа (опционально)
        """
        # Расчет прибыли/убытка
        if entry_prices is not None:
            # Если есть цены входа, считаем реальный P&L
            pred_pnl = (pred - entry_prices) / entry_prices - self.transaction_cost
            target_pnl = (target - entry_prices) / entry_prices - self.transaction_cost
        else:
            # Иначе работаем с доходностями
            pred_pnl = pred - self.transaction_cost
            target_pnl = target - self.transaction_cost
        
        # Ошибка предсказания P&L
        pnl_error = target_pnl - pred_pnl
        
        # Асимметричный штраф: больше штрафуем за упущенную прибыль и реализованные убытки
        loss = torch.where(
            target_pnl > 0,  # Если должна была быть прибыль
            torch.where(
                pred_pnl > 0,  # И мы предсказали прибыль
                torch.abs(pnl_error),  # Обычная ошибка
                self.risk_penalty * torch.abs(pnl_error)  # Штраф за упущенную прибыль
            ),
            torch.where(
                pred_pnl < 0,  # Если предсказали убыток (правильно)
                torch.abs(pnl_error),  # Обычная ошибка
                self.risk_penalty * torch.abs(pnl_error)  # Штраф за неувиденный риск
            )
        )
        
        return self._reduce(loss)


class SharpeRatioLoss(TradingLoss):
    """Потери на основе коэффициента Шарпа"""
    
    def __init__(self,
                 risk_free_rate: float = 0.0,
                 epsilon: float = 1e-8,
                 reduction: str = 'mean'):
        """
        Args:
            risk_free_rate: безрисковая ставка
            epsilon: малое значение для стабильности
            reduction: метод редукции
        """
        super().__init__(reduction)
        self.risk_free_rate = risk_free_rate
        self.epsilon = epsilon
    
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            returns: доходности портфеля [batch_size, time_steps]
        """
        # Средняя доходность
        mean_return = returns.mean(dim=1)
        
        # Стандартное отклонение
        std_return = returns.std(dim=1) + self.epsilon
        
        # Коэффициент Шарпа (отрицательный для минимизации)
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
        
        # Потеря = -Sharpe Ratio (минимизируем негативный Sharpe)
        loss = -sharpe_ratio
        
        return self._reduce(loss)


class MaxDrawdownLoss(TradingLoss):
    """Потери на основе максимальной просадки"""
    
    def __init__(self,
                 lambda_dd: float = 0.5,
                 reduction: str = 'mean'):
        """
        Args:
            lambda_dd: вес для просадки в общей потере
            reduction: метод редукции
        """
        super().__init__(reduction)
        self.lambda_dd = lambda_dd
    
    def forward(self, 
                returns: torch.Tensor,
                target_returns: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            returns: предсказанные доходности [batch_size, time_steps]
            target_returns: целевые доходности (опционально)
        """
        # Кумулятивные доходности
        cum_returns = torch.cumprod(1 + returns, dim=1)
        
        # Максимальная просадка
        running_max = torch.cummax(cum_returns, dim=1)[0]
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min(dim=1)[0]
        
        # Базовая потеря - штраф за просадку
        loss = -self.lambda_dd * max_drawdown
        
        # Если есть целевые доходности, добавляем MSE
        if target_returns is not None:
            mse_loss = F.mse_loss(returns, target_returns, reduction='none').mean(dim=1)
            loss = loss + (1 - self.lambda_dd) * mse_loss
        
        return self._reduce(loss)


class RiskAdjustedLoss(TradingLoss):
    """Комбинированная потеря с учетом риска"""
    
    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 transaction_cost: float = 0.001,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: вес для ошибки предсказания
            beta: вес для риска (волатильности)
            gamma: вес для Sharpe ratio
            transaction_cost: комиссия
            reduction: метод редукции
        """
        super().__init__(reduction)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.transaction_cost = transaction_cost
        
        # Компоненты потерь
        self.mse_loss = nn.MSELoss(reduction='none')
        self.sharpe_loss = SharpeRatioLoss(reduction='none')
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                volatility: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: предсказанные значения
            target: истинные значения
            volatility: предсказанная/истинная волатильность
        """
        batch_size = pred.size(0)
        
        # 1. Ошибка предсказания
        prediction_loss = self.mse_loss(pred, target).mean(dim=1)
        
        # 2. Риск (волатильность)
        if volatility is not None:
            risk_loss = volatility.mean(dim=1) if volatility.dim() > 1 else volatility
        else:
            # Используем стандартное отклонение ошибки как прокси для риска
            risk_loss = (pred - target).std(dim=1)
        
        # 3. Sharpe ratio компонент
        returns = pred - self.transaction_cost
        sharpe_loss = self.sharpe_loss(returns)
        
        # Комбинированная потеря
        loss = (self.alpha * prediction_loss + 
                self.beta * risk_loss + 
                self.gamma * sharpe_loss)
        
        return self._reduce(loss)


class FocalLoss(TradingLoss):
    """Focal Loss для работы с несбалансированными классами"""
    
    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: веса классов
            gamma: фокусирующий параметр
            reduction: метод редукции
        """
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: предсказанные логиты [batch_size, num_classes]
            target: целевые классы [batch_size]
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Вероятности
        p = torch.exp(-ce_loss)
        
        # Focal term
        focal_term = (1 - p) ** self.gamma
        
        # Применяем веса классов если заданы
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        return self._reduce(focal_loss)


class TripletLoss(TradingLoss):
    """Triplet Loss для обучения представлений"""
    
    def __init__(self,
                 margin: float = 1.0,
                 p: int = 2,
                 reduction: str = 'mean'):
        """
        Args:
            margin: отступ между positive и negative
            p: норма для расстояния
            reduction: метод редукции
        """
        super().__init__(reduction)
        self.margin = margin
        self.p = p
    
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: якорные примеры
            positive: положительные примеры (схожие)
            negative: отрицательные примеры (различные)
        """
        # Расстояния
        pos_dist = F.pairwise_distance(anchor, positive, p=self.p)
        neg_dist = F.pairwise_distance(anchor, negative, p=self.p)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return self._reduce(loss)


class MultiTaskLoss(nn.Module):
    """Многозадачная функция потерь"""
    
    def __init__(self,
                 task_weights: Optional[Dict[str, float]] = None,
                 uncertainty_weighting: bool = False):
        """
        Args:
            task_weights: веса для каждой задачи
            uncertainty_weighting: использовать автоматическое взвешивание
        """
        super().__init__()
        self.task_weights = task_weights or {}
        self.uncertainty_weighting = uncertainty_weighting
        
        if uncertainty_weighting:
            # Learnable parameters для автоматического взвешивания
            self.log_vars = nn.ParameterDict()
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: словарь с потерями для каждой задачи
        """
        if self.uncertainty_weighting:
            # Автоматическое взвешивание на основе неопределенности
            total_loss = 0
            for task_name, loss in losses.items():
                if task_name not in self.log_vars:
                    self.log_vars[task_name] = nn.Parameter(torch.zeros(1))
                
                precision = torch.exp(-self.log_vars[task_name])
                total_loss += precision * loss + self.log_vars[task_name]
        else:
            # Фиксированные веса
            total_loss = 0
            for task_name, loss in losses.items():
                weight = self.task_weights.get(task_name, 1.0)
                total_loss += weight * loss
        
        return total_loss


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """Фабрика для создания функций потерь"""
    
    # Импортируем DirectionalMultiTaskLoss локально для избежания циклических импортов
    if loss_name == 'directional_multitask':
        from models.patchtst_unified import DirectionalMultiTaskLoss
        return DirectionalMultiTaskLoss(**kwargs)
    
    loss_functions = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': nn.HuberLoss,
        'directional': DirectionalLoss,
        'profit': ProfitLoss,
        'sharpe': SharpeRatioLoss,
        'drawdown': MaxDrawdownLoss,
        'risk_adjusted': RiskAdjustedLoss,
        'focal': FocalLoss,
        'triplet': TripletLoss,
        'multitask': MultiTaskLoss
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)