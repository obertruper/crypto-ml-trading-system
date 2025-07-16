"""
Калькулятор метрик для оценки моделей
Адаптировано из xgboost_v3
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve
)
from typing import Dict, Tuple, Any
import logging

from config import Config

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Класс для вычисления метрик"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         task_type: str = None) -> Dict[str, float]:
        """
        Универсальный метод для расчета метрик
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказания
            task_type: Тип задачи (если None - берется из конфига)
            
        Returns:
            Словарь с метриками
        """
        if task_type is None:
            task_type = self.config.training.task_type
            
        if task_type == "regression":
            return self.calculate_regression_metrics(y_true, y_pred)
        else:
            return self.calculate_classification_metrics(y_true, y_pred)
    
    def calculate_regression_metrics(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик для регрессии
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказания
            
        Returns:
            Словарь с метриками
        """
        # Базовые метрики
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Направленная точность (важно для трейдинга)
        direction_accuracy = np.mean((y_pred > 0) == (y_true > 0))
        
        # Дополнительные метрики
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Процентили ошибок
        percentiles = np.percentile(np.abs(errors), [25, 50, 75, 90, 95])
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'mean_error': mean_error,
            'std_error': std_error,
            'error_p25': percentiles[0],
            'error_p50': percentiles[1],
            'error_p75': percentiles[2],
            'error_p90': percentiles[3],
            'error_p95': percentiles[4]
        }
        
        return metrics
    
    def calculate_classification_metrics(self, y_true: np.ndarray, 
                                       y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик для бинарной классификации
        
        Args:
            y_true: Истинные метки
            y_pred_proba: Вероятности положительного класса
            
        Returns:
            Словарь с метриками
        """
        # Оптимальный порог
        threshold = self._find_optimal_threshold(y_true, y_pred_proba)
        
        # Бинарные предсказания
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Базовые метрики
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.5
            
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Дополнительные метрики
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        
        # G-mean (среднее геометрическое)
        gmean = np.sqrt(recall * specificity)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'gmean': gmean,
            'optimal_threshold': threshold,
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
        
        return metrics
    
    def _find_optimal_threshold(self, y_true: np.ndarray, 
                               y_pred_proba: np.ndarray) -> float:
        """
        Поиск оптимального порога для классификации
        
        Args:
            y_true: Истинные метки
            y_pred_proba: Вероятности
            
        Returns:
            Оптимальный порог
        """
        if not self.config.training.optimize_threshold:
            return self.config.training.classification_threshold / 100
        
        # Получаем ROC кривую
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # В зависимости от метрики оптимизации
        if self.config.training.threshold_metric == 'gmean':
            # G-mean
            gmeans = np.sqrt(tpr * (1 - fpr))
            ix = np.argmax(gmeans)
        elif self.config.training.threshold_metric == 'f1':
            # F1-score для каждого порога
            f1_scores = []
            for thresh in thresholds:
                y_pred = (y_pred_proba >= thresh).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                f1_scores.append(f1)
            ix = np.argmax(f1_scores)
        else:  # Youden's J statistic
            J = tpr - fpr
            ix = np.argmax(J)
        
        optimal_threshold = thresholds[ix]
        
        logger.info(f"🎯 Оптимальный порог: {optimal_threshold:.3f}")
        
        return optimal_threshold
    
    def calculate_trading_metrics(self, returns: np.ndarray, 
                                predictions: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик специфичных для трейдинга
        
        Args:
            returns: Фактические доходности
            predictions: Предсказанные доходности или сигналы
            
        Returns:
            Словарь с трейдинг метриками
        """
        # Sharpe Ratio (упрощенный)
        if predictions.std() > 0:
            sharpe = predictions.mean() / predictions.std() * np.sqrt(252 * 96)  # 96 = 24h/15min
        else:
            sharpe = 0
            
        # Hit Rate (процент правильных направлений)
        hit_rate = np.mean((predictions > 0) == (returns > 0))
        
        # Profit Factor
        gains = returns[predictions > 0]
        losses = returns[predictions < 0]
        
        total_gains = gains[gains > 0].sum() if len(gains[gains > 0]) > 0 else 0
        total_losses = abs(losses[losses < 0].sum()) if len(losses[losses < 0]) > 0 else 1
        
        profit_factor = total_gains / total_losses if total_losses > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'sharpe_ratio': sharpe,
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_return': cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
        }
        
        return metrics