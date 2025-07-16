"""
Расчет метрик для оценки моделей
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, average_precision_score, matthews_corrcoef
)
import logging
from typing import Dict, Tuple, Optional, Union

from config import Config

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Класс для расчета метрик"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Расчет метрик для регрессии"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mean_error': np.mean(y_pred - y_true),
            'std_error': np.std(y_pred - y_true)
        }
        
        # Направленная точность (правильно предсказанное направление)
        direction_accuracy = ((y_true > 0) == (y_pred > 0)).mean()
        metrics['direction_accuracy'] = direction_accuracy
        
        # Процентные метрики
        metrics['mae_percent'] = metrics['mae']
        metrics['rmse_percent'] = metrics['rmse']
        
        return metrics
        
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                       threshold: Optional[float] = None) -> Dict[str, float]:
        """Расчет метрик для классификации"""
        if threshold is None and self.config.training.optimize_threshold:
            # Находим оптимальный порог
            threshold = self.find_optimal_threshold(y_true, y_pred_proba, 
                                                   method=self.config.training.threshold_metric)
        elif threshold is None:
            threshold = 0.5
            
        y_pred = (y_pred_proba > threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'threshold': threshold
        }
        
        # ROC-AUC и AP
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.5
            metrics['avg_precision'] = 0.0
            
        # Matthews correlation coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Дополнительные метрики
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
        
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             method: str = "gmean") -> float:
        """
        Поиск оптимального порога для классификации
        
        Args:
            y_true: Истинные метки
            y_pred_proba: Предсказанные вероятности
            method: Метод поиска ("gmean", "f1", "profit")
            
        Returns:
            Оптимальный порог
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        if method == "gmean":
            # G-mean = sqrt(sensitivity * specificity)
            gmean = np.sqrt(tpr * (1 - fpr))
            optimal_idx = np.argmax(gmean)
        elif method == "f1":
            # Максимизация F1-score
            f1_scores = []
            for threshold in thresholds:
                y_pred = (y_pred_proba > threshold).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                f1_scores.append(f1)
            optimal_idx = np.argmax(f1_scores)
        elif method == "profit":
            # Максимизация прибыли (требует дополнительных параметров)
            # Простая эвристика: минимизация суммы ошибок
            total_errors = fpr + (1 - tpr)
            optimal_idx = np.argmin(total_errors)
        else:
            # По умолчанию используем точку ближайшую к (0,1)
            distances = np.sqrt(fpr**2 + (1-tpr)**2)
            optimal_idx = np.argmin(distances)
            
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"   Оптимальный порог ({method}): {optimal_threshold:.4f}")
        logger.info(f"   TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f}")
        
        return optimal_threshold
        
    def calculate_multiclass_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Расчет метрик для мультиклассовой классификации"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
        
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Расчет метрик специфичных для трейдинга
        
        Args:
            y_true: Истинные метки (0/1 для классификации или returns для регрессии)
            y_pred: Предсказания модели
            returns: Фактические доходности (если доступны)
            
        Returns:
            Словарь с трейдинговыми метриками
        """
        trading_metrics = {}
        
        if self.config.training.task_type == "classification_binary":
            # Для бинарной классификации
            n_trades = y_pred.sum()
            n_profitable = ((y_pred == 1) & (y_true == 1)).sum()
            
            trading_metrics['total_trades'] = int(n_trades)
            trading_metrics['win_rate'] = n_profitable / n_trades if n_trades > 0 else 0
            
            if returns is not None:
                # Расчет доходности стратегии
                strategy_returns = returns * y_pred
                trading_metrics['total_return'] = strategy_returns.sum()
                trading_metrics['avg_return_per_trade'] = strategy_returns.sum() / n_trades if n_trades > 0 else 0
                trading_metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(strategy_returns)
                trading_metrics['max_drawdown'] = self._calculate_max_drawdown(strategy_returns.cumsum())
                
        elif self.config.training.task_type == "regression":
            # Для регрессии
            # Считаем сколько раз модель предсказала положительную доходность
            n_positive_pred = (y_pred > 0).sum()
            n_correct_positive = ((y_pred > 0) & (y_true > 0)).sum()
            
            trading_metrics['positive_predictions'] = int(n_positive_pred)
            trading_metrics['positive_accuracy'] = n_correct_positive / n_positive_pred if n_positive_pred > 0 else 0
            
        return trading_metrics
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Расчет коэффициента Шарпа"""
        excess_returns = returns - risk_free_rate
        
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
            
        # Годовой Sharpe (предполагаем 15-минутные свечи)
        periods_per_year = 4 * 24 * 365  # 4 свечи в час * 24 часа * 365 дней
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
        
        return sharpe
        
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Расчет максимальной просадки"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
        
        return abs(drawdown.min())
        
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  target_names: Optional[list] = None):
        """Вывод подробного отчета по классификации"""
        if target_names is None:
            if self.config.training.task_type == "classification_binary":
                target_names = ['Не входить', 'Входить']
            else:
                target_names = [f'Класс {i}' for i in range(len(np.unique(y_true)))]
                
        report = classification_report(y_true, y_pred, target_names=target_names)
        logger.info(f"\n📊 Подробный отчет классификации:\n{report}")
        
    def create_metrics_summary(self, metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Создание сводной таблицы метрик"""
        summary_data = []
        
        for model_name, metrics in metrics_dict.items():
            row = {'Model': model_name}
            row.update(metrics)
            summary_data.append(row)
            
        df = pd.DataFrame(summary_data)
        
        # Форматирование
        float_cols = df.select_dtypes(include=[np.float64]).columns
        for col in float_cols:
            df[col] = df[col].round(4)
            
        return df