"""
Специализированные метрики для оценки качества предсказания направления движения
и торговой эффективности модели
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass 
class TradingMetrics:
    """Структура для хранения торговых метрик"""
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_return: float
    total_trades: int
    profitable_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float


class DirectionMetricsCalculator:
    """Калькулятор метрик для направленных предсказаний"""
    
    def __init__(self, commission: float = 0.001, risk_free_rate: float = 0.0):
        """
        Args:
            commission: Комиссия за сделку (в долях)
            risk_free_rate: Безрисковая ставка для расчета Sharpe
        """
        self.commission = commission
        self.risk_free_rate = risk_free_rate
    
    def directional_accuracy(self, 
                           predictions: torch.Tensor, 
                           targets: torch.Tensor,
                           exclude_flat: bool = False) -> Dict[str, float]:
        """
        Расчет точности предсказания направления
        
        Args:
            predictions: (N,) тензор с предсказанными классами [0, 1, 2]
            targets: (N,) тензор с истинными классами [0, 1, 2]
            exclude_flat: Исключить FLAT класс из расчета
            
        Returns:
            Dict с метриками точности
        """
        assert predictions.shape == targets.shape
        
        # Преобразуем к long для индексации
        predictions = predictions.long()
        targets = targets.long()
        
        metrics = {}
        
        if exclude_flat:
            # Исключаем FLAT (класс 2)
            mask = (targets != 2)
            if mask.sum() == 0:
                return {'accuracy': 0.0, 'up_accuracy': 0.0, 'down_accuracy': 0.0}
            
            pred_filtered = predictions[mask]
            target_filtered = targets[mask]
            
            # Общая точность UP/DOWN
            correct = (pred_filtered == target_filtered).float()
            metrics['up_down_accuracy'] = correct.mean().item()
            
        else:
            # Общая точность всех классов
            correct = (predictions == targets).float()
            metrics['overall_accuracy'] = correct.mean().item()
        
        # Точность по классам
        for class_idx, class_name in enumerate(['UP', 'DOWN', 'FLAT']):
            class_mask = (targets == class_idx)
            if class_mask.sum() > 0:
                class_correct = correct[class_mask].mean().item()
                metrics[f'{class_name.lower()}_accuracy'] = class_correct
        
        # Матрица ошибок в виде метрик
        confusion_metrics = self._confusion_matrix_metrics(predictions, targets)
        metrics.update(confusion_metrics)
        
        return metrics
    
    def _confusion_matrix_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Метрики из матрицы ошибок"""
        metrics = {}
        
        # Для UP класса (0)
        up_true_positives = ((predictions == 0) & (targets == 0)).sum().item()
        up_false_positives = ((predictions == 0) & (targets != 0)).sum().item()
        up_false_negatives = ((predictions != 0) & (targets == 0)).sum().item()
        
        if up_true_positives + up_false_positives > 0:
            up_precision = up_true_positives / (up_true_positives + up_false_positives)
            metrics['up_precision'] = up_precision
        
        if up_true_positives + up_false_negatives > 0:
            up_recall = up_true_positives / (up_true_positives + up_false_negatives)
            metrics['up_recall'] = up_recall
        
        # Для DOWN класса (1)
        down_true_positives = ((predictions == 1) & (targets == 1)).sum().item()
        down_false_positives = ((predictions == 1) & (targets != 1)).sum().item()
        down_false_negatives = ((predictions != 1) & (targets == 1)).sum().item()
        
        if down_true_positives + down_false_positives > 0:
            down_precision = down_true_positives / (down_true_positives + down_false_positives)
            metrics['down_precision'] = down_precision
        
        if down_true_positives + down_false_negatives > 0:
            down_recall = down_true_positives / (down_true_positives + down_false_negatives)
            metrics['down_recall'] = down_recall
        
        return metrics
    
    def trading_performance(self, 
                          direction_predictions: torch.Tensor,
                          actual_returns: torch.Tensor,
                          exclude_flat: bool = True) -> TradingMetrics:
        """
        Расчет торговых метрик на основе направленных сигналов
        
        Args:
            direction_predictions: (N,) предсказанные направления [0=UP, 1=DOWN, 2=FLAT]
            actual_returns: (N,) фактические доходности в процентах
            exclude_flat: Исключить FLAT сигналы из торговли
            
        Returns:
            TradingMetrics со всеми торговыми показателями
        """
        assert direction_predictions.shape == actual_returns.shape
        
        # Создаем торговые сигналы
        if exclude_flat:
            trading_mask = direction_predictions != 2  # Исключаем FLAT
        else:
            trading_mask = torch.ones_like(direction_predictions, dtype=torch.bool)
        
        if trading_mask.sum() == 0:
            # Нет торговых сигналов
            return TradingMetrics(
                win_rate=0.0, profit_factor=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, avg_return=0.0, total_trades=0,
                profitable_trades=0, avg_win=0.0, avg_loss=0.0,
                largest_win=0.0, largest_loss=0.0
            )
        
        # Фильтруем данные по торговым сигналам
        trade_signals = direction_predictions[trading_mask]
        trade_returns = actual_returns[trading_mask]
        
        # Расчет P&L
        pnl = torch.zeros_like(trade_returns)
        
        # LONG позиции (UP предсказание)
        long_mask = (trade_signals == 0)
        pnl[long_mask] = trade_returns[long_mask] / 100.0  # Из процентов в доли
        
        # SHORT позиции (DOWN предсказание)  
        short_mask = (trade_signals == 1)
        pnl[short_mask] = -trade_returns[short_mask] / 100.0  # Инвертируем для шорта
        
        # FLAT позиции (если не исключены)
        flat_mask = (trade_signals == 2)
        pnl[flat_mask] = 0.0  # Нет позиции = нет P&L
        
        # Вычитаем комиссию от всех торгуемых позиций
        active_trading_mask = (trade_signals != 2)
        pnl[active_trading_mask] -= self.commission
        
        # Базовые статистики
        total_trades = active_trading_mask.sum().item()
        profitable_mask = pnl > 0
        profitable_trades = profitable_mask.sum().item()
        
        if total_trades == 0:
            win_rate = 0.0
        else:
            win_rate = profitable_trades / total_trades
        
        # Прибыли и убытки
        profits = pnl[profitable_mask]
        losses = pnl[~profitable_mask & active_trading_mask]
        
        avg_win = profits.mean().item() if len(profits) > 0 else 0.0
        avg_loss = losses.mean().item() if len(losses) > 0 else 0.0
        
        largest_win = profits.max().item() if len(profits) > 0 else 0.0
        largest_loss = losses.min().item() if len(losses) > 0 else 0.0
        
        # Profit Factor
        total_profit = profits.sum().item() if len(profits) > 0 else 0.0
        total_loss = abs(losses.sum().item()) if len(losses) > 0 else 0.0
        
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        else:
            profit_factor = float('inf') if total_profit > 0 else 0.0
        
        # Средняя доходность
        avg_return = pnl[active_trading_mask].mean().item() if total_trades > 0 else 0.0
        
        # Sharpe Ratio
        if total_trades > 1:
            returns_std = pnl[active_trading_mask].std().item()
            if returns_std > 0:
                sharpe_ratio = (avg_return - self.risk_free_rate) / returns_std
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Максимальная просадка
        if total_trades > 0:
            cumulative_pnl = torch.cumsum(pnl[active_trading_mask], dim=0)
            running_max = torch.cummax(cumulative_pnl, dim=0)[0]
            drawdown = running_max - cumulative_pnl
            max_drawdown = drawdown.max().item()
        else:
            max_drawdown = 0.0
        
        return TradingMetrics(
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_return=avg_return,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss
        )
    
    def batch_evaluate(self, 
                      predictions: torch.Tensor, 
                      targets: torch.Tensor,
                      returns: torch.Tensor,
                      timeframe_weights: Optional[Dict[str, float]] = None) -> Dict[str, Union[float, TradingMetrics]]:
        """
        Оценка батча предсказаний по всем таймфреймам
        
        Args:
            predictions: (batch_size, 4) direction предсказания для [15m, 1h, 4h, 12h]
            targets: (batch_size, 4) истинные direction метки
            returns: (batch_size, 4) фактические доходности
            timeframe_weights: Веса важности для разных таймфреймов
            
        Returns:
            Dict с метриками по таймфреймам и агрегированными
        """
        if timeframe_weights is None:
            timeframe_weights = {'15m': 0.2, '1h': 0.3, '4h': 0.35, '12h': 0.15}
        
        timeframes = ['15m', '1h', '4h', '12h']
        results = {}
        
        # Метрики по каждому таймфрейму
        for i, tf in enumerate(timeframes):
            # Direction accuracy
            dir_metrics = self.directional_accuracy(
                predictions[:, i], 
                targets[:, i], 
                exclude_flat=True
            )
            
            # Trading performance
            trading_metrics = self.trading_performance(
                predictions[:, i],
                returns[:, i],
                exclude_flat=True
            )
            
            results[f'{tf}_direction'] = dir_metrics
            results[f'{tf}_trading'] = trading_metrics
        
        # Агрегированные метрики
        weighted_accuracy = sum(
            results[f'{tf}_direction']['up_down_accuracy'] * timeframe_weights[tf]
            for tf in timeframes
            if 'up_down_accuracy' in results[f'{tf}_direction']
        )
        
        weighted_win_rate = sum(
            results[f'{tf}_trading'].win_rate * timeframe_weights[tf]
            for tf in timeframes
        )
        
        results['overall_direction_accuracy'] = weighted_accuracy
        results['overall_win_rate'] = weighted_win_rate
        
        return results
    
    def create_performance_report(self, metrics: Dict) -> str:
        """Создание текстового отчета по метрикам"""
        report = []
        report.append("=" * 60)
        report.append("ОТЧЕТ ПО НАПРАВЛЕННЫМ ПРЕДСКАЗАНИЯМ")
        report.append("=" * 60)
        
        # Общие метрики
        if 'overall_direction_accuracy' in metrics:
            acc = metrics['overall_direction_accuracy']
            report.append(f"📊 Общая точность направления: {acc:.3f} ({acc*100:.1f}%)")
        
        if 'overall_win_rate' in metrics:
            wr = metrics['overall_win_rate'] 
            report.append(f"💰 Общий Win Rate: {wr:.3f} ({wr*100:.1f}%)")
        
        report.append("")
        
        # По таймфреймам
        timeframes = ['15m', '1h', '4h', '12h']
        for tf in timeframes:
            trading_key = f'{tf}_trading'
            direction_key = f'{tf}_direction'
            
            if trading_key in metrics and direction_key in metrics:
                trading = metrics[trading_key]
                direction = metrics[direction_key]
                
                report.append(f"📈 {tf.upper()} таймфрейм:")
                
                if 'up_down_accuracy' in direction:
                    acc = direction['up_down_accuracy']
                    report.append(f"   Direction Accuracy: {acc:.3f} ({acc*100:.1f}%)")
                
                report.append(f"   Win Rate: {trading.win_rate:.3f} ({trading.win_rate*100:.1f}%)")
                report.append(f"   Profit Factor: {trading.profit_factor:.2f}")
                report.append(f"   Trades: {trading.total_trades} (profitable: {trading.profitable_trades})")
                
                if trading.sharpe_ratio != 0:
                    report.append(f"   Sharpe Ratio: {trading.sharpe_ratio:.2f}")
                
                report.append("")
        
        # Рекомендации
        report.append("🎯 РЕКОМЕНДАЦИИ:")
        
        overall_acc = metrics.get('overall_direction_accuracy', 0)
        overall_wr = metrics.get('overall_win_rate', 0)
        
        if overall_acc < 0.52:
            report.append("   ⚠️  Direction Accuracy ниже 52% - модель предсказывает хуже случайного")
            report.append("   💡 Рекомендация: пересмотреть архитектуру или признаки")
        elif overall_acc > 0.55:
            report.append("   ✅ Отличная Direction Accuracy - модель хорошо предсказывает направление")
        
        if overall_wr < 0.50:
            report.append("   ⚠️  Win Rate ниже 50% - стратегия убыточна")
            report.append("   💡 Рекомендация: улучшить risk management или пороги входа")
        elif overall_wr > 0.55:
            report.append("   ✅ Отличный Win Rate - стратегия потенциально прибыльна")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def quick_direction_evaluation(predictions: torch.Tensor, 
                             targets: torch.Tensor, 
                             returns: torch.Tensor) -> Dict[str, float]:
    """
    Быстрая оценка качества направленных предсказаний
    
    Args:
        predictions: (N, 4) direction предсказания  
        targets: (N, 4) истинные direction метки
        returns: (N, 4) фактические доходности
        
    Returns:
        Dict с ключевыми метриками
    """
    calculator = DirectionMetricsCalculator()
    
    # Усредняем по всем таймфреймам
    all_preds = predictions.flatten()
    all_targets = targets.flatten() 
    all_returns = returns.flatten()
    
    # Direction accuracy
    dir_metrics = calculator.directional_accuracy(all_preds, all_targets, exclude_flat=True)
    
    # Trading performance
    trading_metrics = calculator.trading_performance(all_preds, all_returns, exclude_flat=True)
    
    return {
        'direction_accuracy': dir_metrics.get('up_down_accuracy', 0.0),
        'win_rate': trading_metrics.win_rate,
        'profit_factor': trading_metrics.profit_factor,
        'total_trades': trading_metrics.total_trades,
        'sharpe_ratio': trading_metrics.sharpe_ratio
    }