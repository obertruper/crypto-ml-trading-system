"""
Специализированные loss функции для торговых моделей
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TradingMultiTaskLoss(nn.Module):
    """Многозадачная loss функция для торговой модели"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Веса для различных компонентов loss
        self.direction_weight = 1.0
        self.tp_weight = 0.8
        self.sl_weight = 1.2  # Больший вес для stop loss
        self.entry_time_weight = 0.5
        
        # Базовые loss функции
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        Вычисление loss для всех компонентов
        
        Args:
            predictions: Тензор с предсказаниями модели
            targets: Тензор с целевыми значениями
            
        Returns:
            Скаляр loss
        """
        # ИСПРАВЛЕНИЕ: Проверяем типы входных данных
        if isinstance(predictions, dict) and isinstance(targets, dict):
            # Если действительно словари
            losses = {}
            
            # Loss для направления (LONG/SHORT/NEUTRAL)
            if 'direction_probs' in predictions and 'direction' in targets:
                direction_loss = self.ce_loss(
                    predictions['direction_probs'], 
                    targets['direction']
                )
                losses['direction_loss'] = direction_loss * self.direction_weight
            
            # Loss для вероятностей TP (LONG)
            if 'long_tp_probs' in predictions and 'long_tp_targets' in targets:
                long_tp_loss = self.bce_loss(
                    predictions['long_tp_probs'],
                    targets['long_tp_targets']
                )
                losses['long_tp_loss'] = long_tp_loss * self.tp_weight
            
            # Loss для вероятности SL (LONG)
            if 'long_sl_prob' in predictions and 'long_sl_target' in targets:
                long_sl_loss = self.bce_loss(
                    predictions['long_sl_prob'],
                    targets['long_sl_target']
                )
                losses['long_sl_loss'] = long_sl_loss * self.sl_weight
            
            # Loss для оптимального времени входа (LONG)
            if 'long_entry_time' in predictions and 'long_entry_time_target' in targets:
                long_entry_loss = self.mse_loss(
                    predictions['long_entry_time'],
                    targets['long_entry_time_target']
                )
                losses['long_entry_loss'] = long_entry_loss * self.entry_time_weight
            
            # Аналогично для SHORT
            if 'short_tp_probs' in predictions and 'short_tp_targets' in targets:
                short_tp_loss = self.bce_loss(
                    predictions['short_tp_probs'],
                    targets['short_tp_targets']
                )
                losses['short_tp_loss'] = short_tp_loss * self.tp_weight
            
            if 'short_sl_prob' in predictions and 'short_sl_target' in targets:
                short_sl_loss = self.bce_loss(
                    predictions['short_sl_prob'],
                    targets['short_sl_target']
                )
                losses['short_sl_loss'] = short_sl_loss * self.sl_weight
            
            if 'short_entry_time' in predictions and 'short_entry_time_target' in targets:
                short_entry_loss = self.mse_loss(
                    predictions['short_entry_time'],
                    targets['short_entry_time_target']
                )
                losses['short_entry_loss'] = short_entry_loss * self.entry_time_weight
            
            # Общий loss
            if losses:
                total_loss = sum(losses.values())
                losses['total_loss'] = total_loss
                return total_loss
            else:
                # Если нет подходящих loss компонентов, используем MSE
                return self.mse_loss(predictions.get('output', list(predictions.values())[0]), 
                                   targets.get('output', list(targets.values())[0]))
        else:
            # Простой случай - тензоры напрямую
            # Для торговой модели используем MSE loss по умолчанию
            return self.mse_loss(predictions, targets)


class RiskAwareLoss(nn.Module):
    """Loss функция с учетом риск-менеджмента"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.risk_config = config.get('risk_management', {})
        
        # Параметры риск-менеджмента
        self.max_drawdown_penalty = 2.0
        self.sharpe_bonus = 0.5
        
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor,
                prices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Loss с учетом риск-метрик
        
        Args:
            predictions: Предсказания модели
            targets: Целевые значения
            prices: Цены для расчета риск-метрик
            
        Returns:
            Loss значение с учетом риска
        """
        # Базовый MSE loss
        base_loss = F.mse_loss(predictions, targets)
        
        if prices is None:
            return base_loss
        
        # Расчет потенциальной просадки
        potential_returns = predictions - prices.unsqueeze(-1)
        potential_drawdown = torch.min(potential_returns, dim=1)[0]
        max_drawdown = torch.abs(torch.min(potential_drawdown))
        
        # Штраф за большую просадку
        drawdown_penalty = self.max_drawdown_penalty * torch.relu(
            max_drawdown - self.risk_config.get('max_drawdown', 0.2)
        )
        
        # Бонус за хороший Sharpe ratio (упрощенная версия)
        returns_std = torch.std(potential_returns)
        returns_mean = torch.mean(potential_returns)
        sharpe_approx = returns_mean / (returns_std + 1e-6)
        sharpe_bonus = self.sharpe_bonus * torch.relu(sharpe_approx - 1.5)
        
        # Итоговый loss
        total_loss = base_loss + drawdown_penalty - sharpe_bonus
        
        return total_loss


class ProbabilisticTradingLoss(nn.Module):
    """Loss для вероятностной модели торговли"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Фокальный параметр для сложных примеров
        self.focal_gamma = 2.0
        
    def focal_bce_loss(self, predictions: torch.Tensor, 
                       targets: torch.Tensor, 
                       gamma: float = 2.0) -> torch.Tensor:
        """
        Focal Binary Cross Entropy Loss
        Больший вес для сложных примеров
        """
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Focal weight
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = (1 - pt) ** gamma
        
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()
    
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Вычисление loss с фокусом на сложные примеры
        """
        total_loss = 0.0
        
        # Loss для вероятностей TP с focal loss
        if 'tp_probs' in predictions and 'tp_targets' in targets:
            tp_loss = self.focal_bce_loss(
                predictions['tp_probs'],
                targets['tp_targets'],
                gamma=self.focal_gamma
            )
            total_loss += tp_loss
        
        # Loss для вероятности SL с увеличенным focal weight
        if 'sl_prob' in predictions and 'sl_target' in targets:
            sl_loss = self.focal_bce_loss(
                predictions['sl_prob'],
                targets['sl_target'],
                gamma=self.focal_gamma * 1.5  # Больший фокус на SL
            )
            total_loss += sl_loss * 1.5  # Больший вес для SL
        
        return total_loss


def get_trading_loss_function(config: Dict, loss_type: str = 'multi_task') -> nn.Module:
    """
    Фабрика для создания loss функций
    
    Args:
        config: Конфигурация
        loss_type: Тип loss функции
        
    Returns:
        Loss функция
    """
    if loss_type == 'multi_task':
        return TradingMultiTaskLoss(config)
    elif loss_type == 'risk_aware':
        return RiskAwareLoss(config)
    elif loss_type == 'probabilistic':
        return ProbabilisticTradingLoss(config)
    else:
        raise ValueError(f"Неизвестный тип loss функции: {loss_type}")