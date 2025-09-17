"""
Анализатор качества предсказаний для всех 20 целевых переменных UnifiedPatchTST
Предоставляет детальную оценку производительности модели по каждой задаче
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from pathlib import Path
import json
import logging
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, log_loss,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import get_logger


@dataclass
class VariableMetrics:
    """Метрики для одной целевой переменной"""
    variable_name: str
    variable_type: str  # 'regression', 'classification', 'binary_classification'
    n_samples: int
    
    # Метрики для регрессии
    mse: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    
    # Метрики для классификации
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    
    # Метрики для бинарной классификации
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    log_loss_score: Optional[float] = None
    
    # Статистики распределений
    pred_mean: Optional[float] = None
    pred_std: Optional[float] = None
    true_mean: Optional[float] = None
    true_std: Optional[float] = None
    
    # Дополнительные метрики
    correlation: Optional[float] = None
    calibration_error: Optional[float] = None
    reliability: Optional[float] = None
    
    # Информация о классах (для классификации)
    class_distribution: Optional[Dict[str, int]] = None
    confusion_matrix_values: Optional[List[List[int]]] = None


class PredictionQualityAnalyzer:
    """
    Анализатор качества предсказаний для модели с 20 целевыми переменными
    
    Анализирует:
    - 4 переменные возвратов (regression)
    - 4 переменные направлений (multi-class classification) 
    - 8 переменных достижения уровней (binary classification)
    - 4 риск-метрики (regression)
    """
    
    def __init__(self, target_config: Optional[Dict] = None):
        self.logger = get_logger("PredictionQualityAnalyzer")
        
        # Конфигурация целевых переменных (соответствует config.yaml)
        if target_config is None:
            self.target_config = self._get_default_target_config()
        else:
            self.target_config = target_config
            
        self.variable_metrics: List[VariableMetrics] = []
        
        self.logger.info("📊 Инициализация анализатора качества предсказаний")
        self.logger.info(f"   🎯 Целевых переменных: {len(self.target_config)}")
        
    def _get_default_target_config(self) -> List[Dict]:
        """Возвращает стандартную конфигурацию 20 целевых переменных"""
        
        return [
            # Возвраты (4) - регрессия
            {'name': 'future_return_15m', 'type': 'regression', 'index': 0},
            {'name': 'future_return_1h', 'type': 'regression', 'index': 1},
            {'name': 'future_return_4h', 'type': 'regression', 'index': 2}, 
            {'name': 'future_return_12h', 'type': 'regression', 'index': 3},
            
            # Направления (4) - классификация 3 класса (LONG=0, SHORT=1, FLAT=2)
            {'name': 'direction_15m', 'type': 'classification', 'index': 4, 'n_classes': 3},
            {'name': 'direction_1h', 'type': 'classification', 'index': 5, 'n_classes': 3},
            {'name': 'direction_4h', 'type': 'classification', 'index': 6, 'n_classes': 3},
            {'name': 'direction_12h', 'type': 'classification', 'index': 7, 'n_classes': 3},
            
            # LONG уровни (4) - бинарная классификация
            {'name': 'long_will_reach_1pct_4h', 'type': 'binary_classification', 'index': 8},
            {'name': 'long_will_reach_2pct_4h', 'type': 'binary_classification', 'index': 9},
            {'name': 'long_will_reach_3pct_12h', 'type': 'binary_classification', 'index': 10},
            {'name': 'long_will_reach_5pct_12h', 'type': 'binary_classification', 'index': 11},
            
            # SHORT уровни (4) - бинарная классификация
            {'name': 'short_will_reach_1pct_4h', 'type': 'binary_classification', 'index': 12},
            {'name': 'short_will_reach_2pct_4h', 'type': 'binary_classification', 'index': 13},
            {'name': 'short_will_reach_3pct_12h', 'type': 'binary_classification', 'index': 14},
            {'name': 'short_will_reach_5pct_12h', 'type': 'binary_classification', 'index': 15},
            
            # Риск-метрики (4) - регрессия
            {'name': 'max_drawdown_1h', 'type': 'regression', 'index': 16},
            {'name': 'max_rally_1h', 'type': 'regression', 'index': 17},
            {'name': 'max_drawdown_4h', 'type': 'regression', 'index': 18},
            {'name': 'max_rally_4h', 'type': 'regression', 'index': 19}
        ]
    
    def analyze_single_variable(self, 
                               y_true: np.ndarray,
                               y_pred: np.ndarray, 
                               variable_config: Dict,
                               y_pred_proba: Optional[np.ndarray] = None) -> VariableMetrics:
        """
        Анализирует качество предсказаний для одной переменной
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            variable_config: Конфигурация переменной
            y_pred_proba: Вероятности (для классификации)
            
        Returns:
            Метрики для переменной
        """
        
        var_name = variable_config['name']
        var_type = variable_config['type']
        
        # Убираем NaN значения
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if y_pred_proba is not None:
            valid_mask = valid_mask & ~np.isnan(y_pred_proba).any(axis=1)
            
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if y_pred_proba is not None:
            y_pred_proba_clean = y_pred_proba[valid_mask]
        else:
            y_pred_proba_clean = None
            
        n_samples = len(y_true_clean)
        
        if n_samples < 10:
            self.logger.warning(f"⚠️ Недостаточно данных для {var_name}: {n_samples}")
            return VariableMetrics(
                variable_name=var_name,
                variable_type=var_type,
                n_samples=n_samples
            )
        
        # Базовые статистики
        pred_mean = float(np.mean(y_pred_clean))
        pred_std = float(np.std(y_pred_clean))
        true_mean = float(np.mean(y_true_clean))
        true_std = float(np.std(y_true_clean))
        
        correlation = float(np.corrcoef(y_true_clean, y_pred_clean)[0, 1]) if n_samples > 1 else 0.0
        
        metrics = VariableMetrics(
            variable_name=var_name,
            variable_type=var_type,
            n_samples=n_samples,
            pred_mean=pred_mean,
            pred_std=pred_std,
            true_mean=true_mean,
            true_std=true_std,
            correlation=correlation
        )
        
        # Рассчитываем метрики в зависимости от типа задачи
        if var_type == 'regression':
            metrics = self._calculate_regression_metrics(metrics, y_true_clean, y_pred_clean)
            
        elif var_type == 'classification':
            n_classes = variable_config.get('n_classes', 3)
            metrics = self._calculate_classification_metrics(
                metrics, y_true_clean, y_pred_clean, y_pred_proba_clean, n_classes
            )
            
        elif var_type == 'binary_classification':
            metrics = self._calculate_binary_classification_metrics(
                metrics, y_true_clean, y_pred_clean, y_pred_proba_clean
            )
        
        return metrics
    
    def _calculate_regression_metrics(self, 
                                    metrics: VariableMetrics,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray) -> VariableMetrics:
        """Рассчитывает метрики для регрессии"""
        
        try:
            metrics.mse = float(mean_squared_error(y_true, y_pred))
            metrics.mae = float(mean_absolute_error(y_true, y_pred))
            metrics.rmse = float(np.sqrt(metrics.mse))
            metrics.r2 = float(r2_score(y_true, y_pred))
            
            # MAPE (Mean Absolute Percentage Error)
            # Избегаем деление на ноль
            non_zero_mask = np.abs(y_true) > 1e-8
            if np.sum(non_zero_mask) > 0:
                mape_values = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
                metrics.mape = float(np.mean(mape_values)) * 100
            else:
                metrics.mape = None
                
        except Exception as e:
            self.logger.error(f"Ошибка в расчете метрик регрессии для {metrics.variable_name}: {str(e)}")
            
        return metrics
    
    def _calculate_classification_metrics(self,
                                        metrics: VariableMetrics,
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray],
                                        n_classes: int) -> VariableMetrics:
        """Рассчитывает метрики для многоклассовой классификации"""
        
        try:
            # Конвертируем в целые числа
            y_true_int = y_true.astype(int)
            y_pred_int = y_pred.astype(int)
            
            # Базовые метрики
            metrics.accuracy = float(accuracy_score(y_true_int, y_pred_int))
            
            # Метрики с разными стратегиями усреднения
            metrics.precision = float(precision_score(y_true_int, y_pred_int, average='macro', zero_division=0))
            metrics.recall = float(recall_score(y_true_int, y_pred_int, average='macro', zero_division=0))
            metrics.f1 = float(f1_score(y_true_int, y_pred_int, average='macro', zero_division=0))
            metrics.macro_f1 = metrics.f1
            metrics.weighted_f1 = float(f1_score(y_true_int, y_pred_int, average='weighted', zero_division=0))
            
            # Распределение классов
            unique_true, counts_true = np.unique(y_true_int, return_counts=True)
            metrics.class_distribution = {f"class_{cls}": int(count) for cls, count in zip(unique_true, counts_true)}
            
            # Confusion matrix
            cm = confusion_matrix(y_true_int, y_pred_int)
            metrics.confusion_matrix_values = cm.tolist()
            
            # Log loss (если есть вероятности)
            if y_pred_proba is not None and y_pred_proba.shape[1] == n_classes:
                try:
                    metrics.log_loss_score = float(log_loss(y_true_int, y_pred_proba))
                    
                    # Калибровка (reliability)
                    metrics.reliability = self._calculate_reliability(y_true_int, y_pred_proba)
                    
                except Exception as e:
                    self.logger.debug(f"Не удалось рассчитать log_loss для {metrics.variable_name}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Ошибка в расчете метрик классификации для {metrics.variable_name}: {str(e)}")
            
        return metrics
    
    def _calculate_binary_classification_metrics(self,
                                               metrics: VariableMetrics,
                                               y_true: np.ndarray,
                                               y_pred: np.ndarray,
                                               y_pred_proba: Optional[np.ndarray]) -> VariableMetrics:
        """Рассчитывает метрики для бинарной классификации"""
        
        try:
            # Конвертируем в бинарные значения
            y_true_bin = (y_true > 0.5).astype(int)
            y_pred_bin = (y_pred > 0.5).astype(int)
            
            # Базовые метрики
            metrics.accuracy = float(accuracy_score(y_true_bin, y_pred_bin))
            metrics.precision = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
            metrics.recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
            metrics.f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
            
            # Распределение классов
            unique_true, counts_true = np.unique(y_true_bin, return_counts=True)
            metrics.class_distribution = {f"class_{cls}": int(count) for cls, count in zip(unique_true, counts_true)}
            
            # Confusion matrix
            cm = confusion_matrix(y_true_bin, y_pred_bin)
            metrics.confusion_matrix_values = cm.tolist()
            
            # ROC-AUC и PR-AUC
            if len(np.unique(y_true_bin)) > 1:  # Должно быть больше одного класса
                try:
                    # Для ROC-AUC используем непрерывные предсказания
                    metrics.auc_roc = float(roc_auc_score(y_true_bin, y_pred))
                    
                    # PR-AUC
                    precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin, y_pred)
                    metrics.auc_pr = float(np.trapz(precision_vals, recall_vals))
                    
                except Exception as e:
                    self.logger.debug(f"Не удалось рассчитать AUC для {metrics.variable_name}: {str(e)}")
            
            # Log loss (если есть вероятности)
            if y_pred_proba is not None:
                try:
                    # Для бинарной классификации берем вероятность положительного класса
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                        pos_proba = y_pred_proba[:, 1]
                    else:
                        pos_proba = y_pred_proba.flatten()
                        
                    metrics.log_loss_score = float(log_loss(y_true_bin, pos_proba))
                    
                    # Калибровка
                    metrics.calibration_error = self._calculate_calibration_error(y_true_bin, pos_proba)
                    
                except Exception as e:
                    self.logger.debug(f"Не удалось рассчитать log_loss для {metrics.variable_name}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Ошибка в расчете метрик бинарной классификации для {metrics.variable_name}: {str(e)}")
            
        return metrics
    
    def _calculate_reliability(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """Рассчитывает reliability (калибровку) для многоклассовой классификации"""
        
        try:
            # Предсказанный класс
            y_pred_class = np.argmax(y_pred_proba, axis=1)
            
            # Максимальная вероятность
            max_proba = np.max(y_pred_proba, axis=1)
            
            # Корректность предсказания
            correct = (y_pred_class == y_true).astype(int)
            
            # Разбиваем на бины по уверенности
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            reliability = 0
            total_samples = 0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (max_proba > bin_lower) & (max_proba <= bin_upper)
                prop_in_bin = in_bin.sum()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = correct[in_bin].mean()
                    avg_confidence_in_bin = max_proba[in_bin].mean()
                    reliability += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    total_samples += prop_in_bin
            
            return reliability / total_samples if total_samples > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """Рассчитывает Expected Calibration Error (ECE) для бинарной классификации"""
        
        try:
            # Разбиваем на бины по предсказанной вероятности
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            total_samples = len(y_pred_proba)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.sum() / total_samples
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
            
        except Exception:
            return 0.0
    
    def analyze_all_variables(self,
                            y_true: Union[np.ndarray, torch.Tensor],
                            y_pred: Union[np.ndarray, torch.Tensor],
                            y_pred_proba: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict:
        """
        Анализирует качество предсказаний для всех 20 переменных
        
        Args:
            y_true: Истинные значения (N, 20) или (N, 1, 20)
            y_pred: Предсказанные значения (N, 20)
            y_pred_proba: Вероятности для классификации (опционально)
            
        Returns:
            Словарь с результатами анализа
        """
        
        self.logger.info("🔍 Начало анализа качества предсказаний для всех переменных")
        
        # Конвертация в numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().detach().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().detach().numpy()
        if y_pred_proba is not None and isinstance(y_pred_proba, torch.Tensor):
            y_pred_proba = y_pred_proba.cpu().detach().numpy()
        
        # Обработка размерности
        if y_true.ndim == 3 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)  # (N, 1, 20) -> (N, 20)
        if y_pred.ndim == 3 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)
        
        n_samples, n_variables = y_true.shape
        expected_variables = len(self.target_config)
        
        if n_variables != expected_variables:
            self.logger.warning(f"⚠️ Ожидалось {expected_variables} переменных, получено {n_variables}")
        
        self.logger.info(f"📊 Анализ данных: {n_samples} наблюдений, {n_variables} переменных")
        
        # Анализируем каждую переменную
        all_metrics = []
        
        for i, var_config in enumerate(self.target_config):
            if i >= n_variables:
                self.logger.warning(f"⚠️ Переменная {var_config['name']} отсутствует в данных")
                continue
                
            self.logger.debug(f"   Анализируем {var_config['name']}")
            
            # Извлекаем данные для переменной
            y_true_var = y_true[:, i]
            y_pred_var = y_pred[:, i]
            
            # Извлекаем вероятности если доступны
            y_pred_proba_var = None
            if y_pred_proba is not None:
                if var_config['type'] == 'classification' and var_config.get('n_classes', 3) == 3:
                    # Для классификации направлений: 3 класса  
                    start_idx = i * 3
                    end_idx = start_idx + 3
                    if end_idx <= y_pred_proba.shape[1]:
                        y_pred_proba_var = y_pred_proba[:, start_idx:end_idx]
                elif var_config['type'] == 'binary_classification':
                    # Для бинарной классификации: 1 вероятность
                    if i < y_pred_proba.shape[1]:
                        y_pred_proba_var = y_pred_proba[:, i:i+1]
            
            # Анализируем переменную
            try:
                var_metrics = self.analyze_single_variable(
                    y_true_var, y_pred_var, var_config, y_pred_proba_var
                )
                all_metrics.append(var_metrics)
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка анализа {var_config['name']}: {str(e)}")
                # Создаем пустые метрики
                empty_metrics = VariableMetrics(
                    variable_name=var_config['name'],
                    variable_type=var_config['type'],
                    n_samples=0
                )
                all_metrics.append(empty_metrics)
        
        self.variable_metrics = all_metrics
        
        # Создаем сводный отчет
        summary_report = self._generate_summary_report(all_metrics)
        
        # Группировка по типам задач
        grouped_analysis = self._group_analysis_by_task_type(all_metrics)
        
        # Общие выводы и рекомендации
        insights = self._generate_insights(all_metrics, grouped_analysis)
        
        comprehensive_analysis = {
            'summary': summary_report,
            'grouped_analysis': grouped_analysis,
            'detailed_metrics': [self._metrics_to_dict(m) for m in all_metrics],
            'insights_and_recommendations': insights,
            'metadata': {
                'n_samples': n_samples,
                'n_variables_analyzed': len(all_metrics),
                'analysis_timestamp': datetime.now().isoformat(),
                'target_config': self.target_config
            }
        }
        
        self.logger.info("✅ Анализ качества предсказаний завершен")
        
        return comprehensive_analysis
    
    def _generate_summary_report(self, metrics_list: List[VariableMetrics]) -> Dict:
        """Генерирует сводный отчет по всем переменным"""
        
        # Группируем по типам
        regression_metrics = [m for m in metrics_list if m.variable_type == 'regression']
        classification_metrics = [m for m in metrics_list if m.variable_type == 'classification']
        binary_metrics = [m for m in metrics_list if m.variable_type == 'binary_classification']
        
        summary = {
            'overall': {
                'total_variables': len(metrics_list),
                'regression_variables': len(regression_metrics),
                'classification_variables': len(classification_metrics),
                'binary_classification_variables': len(binary_metrics)
            }
        }
        
        # Суммарные метрики по типам задач
        if regression_metrics:
            r2_scores = [m.r2 for m in regression_metrics if m.r2 is not None]
            correlations = [m.correlation for m in regression_metrics if m.correlation is not None]
            
            summary['regression'] = {
                'mean_r2': np.mean(r2_scores) if r2_scores else None,
                'mean_correlation': np.mean(correlations) if correlations else None,
                'variables_with_good_r2': len([r2 for r2 in r2_scores if r2 > 0.1]),
                'best_performing_variable': max(regression_metrics, key=lambda x: x.r2 or 0).variable_name if r2_scores else None
            }
        
        if classification_metrics:
            f1_scores = [m.f1 for m in classification_metrics if m.f1 is not None]
            accuracies = [m.accuracy for m in classification_metrics if m.accuracy is not None]
            
            summary['classification'] = {
                'mean_f1': np.mean(f1_scores) if f1_scores else None,
                'mean_accuracy': np.mean(accuracies) if accuracies else None,
                'variables_with_good_f1': len([f1 for f1 in f1_scores if f1 > 0.4]),
                'best_performing_variable': max(classification_metrics, key=lambda x: x.f1 or 0).variable_name if f1_scores else None
            }
        
        if binary_metrics:
            f1_scores = [m.f1 for m in binary_metrics if m.f1 is not None]
            auc_scores = [m.auc_roc for m in binary_metrics if m.auc_roc is not None]
            
            summary['binary_classification'] = {
                'mean_f1': np.mean(f1_scores) if f1_scores else None,
                'mean_auc': np.mean(auc_scores) if auc_scores else None,
                'variables_with_good_f1': len([f1 for f1 in f1_scores if f1 > 0.3]),
                'variables_with_good_auc': len([auc for auc in auc_scores if auc > 0.6]),
                'best_performing_variable': max(binary_metrics, key=lambda x: x.auc_roc or 0).variable_name if auc_scores else None
            }
        
        return summary
    
    def _group_analysis_by_task_type(self, metrics_list: List[VariableMetrics]) -> Dict:
        """Группирует анализ по типам задач"""
        
        grouped = {
            'returns_prediction': {
                'variables': ['future_return_15m', 'future_return_1h', 'future_return_4h', 'future_return_12h'],
                'metrics': []
            },
            'direction_prediction': {
                'variables': ['direction_15m', 'direction_1h', 'direction_4h', 'direction_12h'],
                'metrics': []
            },
            'long_levels_prediction': {
                'variables': ['long_will_reach_1pct_4h', 'long_will_reach_2pct_4h', 
                             'long_will_reach_3pct_12h', 'long_will_reach_5pct_12h'],
                'metrics': []
            },
            'short_levels_prediction': {
                'variables': ['short_will_reach_1pct_4h', 'short_will_reach_2pct_4h',
                             'short_will_reach_3pct_12h', 'short_will_reach_5pct_12h'],
                'metrics': []
            },
            'risk_metrics_prediction': {
                'variables': ['max_drawdown_1h', 'max_rally_1h', 'max_drawdown_4h', 'max_rally_4h'],
                'metrics': []
            }
        }
        
        # Распределяем метрики по группам
        for metrics in metrics_list:
            for group_name, group_info in grouped.items():
                if metrics.variable_name in group_info['variables']:
                    group_info['metrics'].append(self._metrics_to_dict(metrics))
                    break
        
        # Рассчитываем групповые статистики
        for group_name, group_info in grouped.items():
            group_metrics = group_info['metrics']
            
            if group_metrics:
                if group_name in ['returns_prediction', 'risk_metrics_prediction']:
                    # Метрики регрессии
                    r2_values = [m.get('r2') for m in group_metrics if m.get('r2') is not None]
                    correlations = [m.get('correlation') for m in group_metrics if m.get('correlation') is not None]
                    
                    group_info['group_stats'] = {
                        'mean_r2': np.mean(r2_values) if r2_values else None,
                        'mean_correlation': np.mean(correlations) if correlations else None,
                        'best_r2': max(r2_values) if r2_values else None,
                        'worst_r2': min(r2_values) if r2_values else None
                    }
                    
                elif group_name == 'direction_prediction':
                    # Метрики многоклассовой классификации
                    f1_values = [m.get('f1') for m in group_metrics if m.get('f1') is not None]
                    acc_values = [m.get('accuracy') for m in group_metrics if m.get('accuracy') is not None]
                    
                    group_info['group_stats'] = {
                        'mean_f1': np.mean(f1_values) if f1_values else None,
                        'mean_accuracy': np.mean(acc_values) if acc_values else None,
                        'best_f1': max(f1_values) if f1_values else None,
                        'worst_f1': min(f1_values) if f1_values else None
                    }
                    
                else:  # long_levels_prediction, short_levels_prediction
                    # Метрики бинарной классификации
                    f1_values = [m.get('f1') for m in group_metrics if m.get('f1') is not None]
                    auc_values = [m.get('auc_roc') for m in group_metrics if m.get('auc_roc') is not None]
                    
                    group_info['group_stats'] = {
                        'mean_f1': np.mean(f1_values) if f1_values else None,
                        'mean_auc': np.mean(auc_values) if auc_values else None,
                        'best_auc': max(auc_values) if auc_values else None,
                        'worst_auc': min(auc_values) if auc_values else None
                    }
        
        return grouped
    
    def _generate_insights(self, metrics_list: List[VariableMetrics], grouped_analysis: Dict) -> Dict:
        """Генерирует выводы и рекомендации"""
        
        insights = {
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'priority_improvements': []
        }
        
        # Анализируем сильные стороны
        strong_variables = []
        
        for metrics in metrics_list:
            if metrics.variable_type == 'regression' and metrics.r2 is not None and metrics.r2 > 0.2:
                strong_variables.append(f"{metrics.variable_name} (R²={metrics.r2:.3f})")
            elif metrics.variable_type in ['classification', 'binary_classification'] and metrics.f1 is not None and metrics.f1 > 0.4:
                strong_variables.append(f"{metrics.variable_name} (F1={metrics.f1:.3f})")
        
        if strong_variables:
            insights['strengths'].append(f"Хорошее качество предсказаний: {', '.join(strong_variables)}")
        
        # Анализируем слабые стороны  
        weak_variables = []
        
        for metrics in metrics_list:
            if metrics.variable_type == 'regression' and metrics.r2 is not None and metrics.r2 < 0.05:
                weak_variables.append(f"{metrics.variable_name} (R²={metrics.r2:.3f})")
            elif metrics.variable_type in ['classification', 'binary_classification'] and metrics.f1 is not None and metrics.f1 < 0.2:
                weak_variables.append(f"{metrics.variable_name} (F1={metrics.f1:.3f})")
        
        if weak_variables:
            insights['weaknesses'].append(f"Слабое качество предсказаний: {', '.join(weak_variables)}")
        
        # Анализируем каждую группу задач
        for group_name, group_info in grouped_analysis.items():
            group_stats = group_info.get('group_stats', {})
            
            if group_name == 'direction_prediction':
                mean_f1 = group_stats.get('mean_f1')
                if mean_f1 is not None:
                    if mean_f1 > 0.4:
                        insights['strengths'].append(f"Направления торговли предсказываются хорошо (средний F1={mean_f1:.3f})")
                    elif mean_f1 < 0.3:
                        insights['weaknesses'].append(f"Слабое предсказание направлений торговли (средний F1={mean_f1:.3f})")
                        insights['priority_improvements'].append("Улучшить модель предсказания направлений торговли")
            
            elif group_name == 'returns_prediction':
                mean_r2 = group_stats.get('mean_r2')
                if mean_r2 is not None:
                    if mean_r2 > 0.15:
                        insights['strengths'].append(f"Доходности предсказываются удовлетворительно (средний R²={mean_r2:.3f})")
                    elif mean_r2 < 0.05:
                        insights['weaknesses'].append(f"Слабое предсказание доходностей (средний R²={mean_r2:.3f})")
                        insights['priority_improvements'].append("Улучшить модель предсказания доходностей")
        
        # Общие рекомендации
        insights['recommendations'].extend([
            "Сосредоточиться на улучшении переменных с низкими метриками",
            "Рассмотреть увеличение сложности модели для слабо предсказуемых переменных",
            "Проанализировать качество данных для переменных с очень низкими показателями",
            "Возможно, использовать разные архитектуры/головы для разных типов задач"
        ])
        
        if len(insights['priority_improvements']) == 0:
            insights['priority_improvements'].append("Основные метрики в приемлемом диапазоне, продолжить мониторинг")
        
        return insights
    
    def _metrics_to_dict(self, metrics: VariableMetrics) -> Dict:
        """Конвертирует метрики в словарь для JSON сериализации"""
        
        return {
            'variable_name': metrics.variable_name,
            'variable_type': metrics.variable_type,
            'n_samples': metrics.n_samples,
            'mse': metrics.mse,
            'mae': metrics.mae,
            'rmse': metrics.rmse,
            'r2': metrics.r2,
            'mape': metrics.mape,
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1': metrics.f1,
            'macro_f1': metrics.macro_f1,
            'weighted_f1': metrics.weighted_f1,
            'auc_roc': metrics.auc_roc,
            'auc_pr': metrics.auc_pr,
            'log_loss_score': metrics.log_loss_score,
            'pred_mean': metrics.pred_mean,
            'pred_std': metrics.pred_std,
            'true_mean': metrics.true_mean,
            'true_std': metrics.true_std,
            'correlation': metrics.correlation,
            'calibration_error': metrics.calibration_error,
            'reliability': metrics.reliability,
            'class_distribution': metrics.class_distribution,
            'confusion_matrix_values': metrics.confusion_matrix_values
        }
    
    def save_analysis_results(self, results: Dict, output_dir: str = "validation/prediction_quality_results"):
        """Сохраняет результаты анализа"""
        
        # Создаем директорию
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем JSON
        json_file = output_path / f"prediction_quality_analysis_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Создаем читаемый отчет
        report_file = output_path / f"PREDICTION_QUALITY_REPORT_{timestamp}.md"
        report_content = self._create_readable_report(results)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"💾 Результаты анализа сохранены:")
        self.logger.info(f"   📄 JSON: {json_file}")
        self.logger.info(f"   📖 Отчет: {report_file}")
        
        return json_file, report_file
    
    def _create_readable_report(self, results: Dict) -> str:
        """Создает читаемый отчет анализа качества предсказаний"""
        
        timestamp = results['metadata']['analysis_timestamp']
        n_samples = results['metadata']['n_samples']
        summary = results['summary']
        insights = results['insights_and_recommendations']
        grouped = results['grouped_analysis']
        
        report = f"""# Отчет по качеству предсказаний модели

**Дата анализа:** {timestamp}  
**Количество образцов:** {n_samples}  
**Переменных проанализировано:** {summary['overall']['total_variables']}

## 📊 Общая сводка

- **Регрессионные задачи:** {summary['overall']['regression_variables']} 
- **Классификация (направления):** {summary['overall']['classification_variables']}
- **Бинарная классификация (уровни):** {summary['overall']['binary_classification_variables']}

"""
        
        # Метрики по типам задач
        if 'regression' in summary:
            reg_info = summary['regression']
            report += f"""### 📈 Регрессия (доходности и риски)

- **Средний R²:** {reg_info.get('mean_r2', 0):.3f}
- **Средняя корреляция:** {reg_info.get('mean_correlation', 0):.3f}
- **Переменных с хорошим R² (>0.1):** {reg_info.get('variables_with_good_r2', 0)}
- **Лучшая переменная:** {reg_info.get('best_performing_variable', 'N/A')}

"""
        
        if 'classification' in summary:
            class_info = summary['classification']
            report += f"""### 🎯 Классификация (направления)

- **Средний F1:** {class_info.get('mean_f1', 0):.3f}
- **Средняя точность:** {class_info.get('mean_accuracy', 0):.3f}
- **Переменных с хорошим F1 (>0.4):** {class_info.get('variables_with_good_f1', 0)}
- **Лучшая переменная:** {class_info.get('best_performing_variable', 'N/A')}

"""
        
        if 'binary_classification' in summary:
            binary_info = summary['binary_classification']
            report += f"""### 🎲 Бинарная классификация (уровни TP)

- **Средний F1:** {binary_info.get('mean_f1', 0):.3f}
- **Средний AUC:** {binary_info.get('mean_auc', 0):.3f}
- **Переменных с хорошим F1 (>0.3):** {binary_info.get('variables_with_good_f1', 0)}
- **Переменных с хорошим AUC (>0.6):** {binary_info.get('variables_with_good_auc', 0)}
- **Лучшая переменная:** {binary_info.get('best_performing_variable', 'N/A')}

"""
        
        # Детальный анализ по группам
        report += "## 🔍 Детальный анализ по группам задач\n\n"
        
        group_names = {
            'returns_prediction': '📈 Предсказание доходностей',
            'direction_prediction': '🎯 Предсказание направлений',
            'long_levels_prediction': '🟢 Предсказание LONG уровней',
            'short_levels_prediction': '🔴 Предсказание SHORT уровней',
            'risk_metrics_prediction': '⚠️ Предсказание рисков'
        }
        
        for group_key, group_name in group_names.items():
            if group_key in grouped:
                group_data = grouped[group_key]
                group_stats = group_data.get('group_stats', {})
                
                report += f"### {group_name}\n\n"
                
                if group_key in ['returns_prediction', 'risk_metrics_prediction']:
                    mean_r2 = group_stats.get('mean_r2')
                    mean_corr = group_stats.get('mean_correlation')
                    if mean_r2 is not None:
                        report += f"- **Средний R²:** {mean_r2:.3f}\n"
                        report += f"- **Средняя корреляция:** {mean_corr:.3f}\n"
                        report += f"- **Диапазон R²:** {group_stats.get('worst_r2', 0):.3f} - {group_stats.get('best_r2', 0):.3f}\n"
                else:
                    mean_f1 = group_stats.get('mean_f1')
                    if mean_f1 is not None:
                        report += f"- **Средний F1:** {mean_f1:.3f}\n"
                        if 'mean_auc' in group_stats:
                            report += f"- **Средний AUC:** {group_stats.get('mean_auc'):.3f}\n"
                        if 'mean_accuracy' in group_stats:
                            report += f"- **Средняя точность:** {group_stats.get('mean_accuracy'):.3f}\n"
                
                # Список переменных в группе
                variables = group_data.get('variables', [])
                report += f"- **Переменные:** {', '.join(variables)}\n\n"
        
        # Выводы и рекомендации
        report += "## 💡 Выводы и рекомендации\n\n"
        
        if insights.get('strengths'):
            report += "### ✅ Сильные стороны\n\n"
            for strength in insights['strengths']:
                report += f"- {strength}\n"
            report += "\n"
        
        if insights.get('weaknesses'):
            report += "### ⚠️ Слабые стороны\n\n"
            for weakness in insights['weaknesses']:
                report += f"- {weakness}\n"
            report += "\n"
        
        if insights.get('priority_improvements'):
            report += "### 🎯 Приоритетные улучшения\n\n"
            for improvement in insights['priority_improvements']:
                report += f"- {improvement}\n"
            report += "\n"
        
        if insights.get('recommendations'):
            report += "### 📋 Общие рекомендации\n\n"
            for recommendation in insights['recommendations']:
                report += f"- {recommendation}\n"
            report += "\n"
        
        report += "---\n*Отчет сгенерирован анализатором качества предсказаний*"
        
        return report


def main():
    """Пример использования анализатора качества предсказаний"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Анализ качества предсказаний модели")
    parser.add_argument('--true-file', required=True, help='Файл с истинными значениями (npy)')
    parser.add_argument('--pred-file', required=True, help='Файл с предсказаниями (npy)')
    parser.add_argument('--proba-file', help='Файл с вероятностями (npy, опционально)')
    parser.add_argument('--output-dir', default='validation/prediction_quality_results',
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    print("📊 Загрузка данных...")
    
    # Загружаем данные
    y_true = np.load(args.true_file)
    y_pred = np.load(args.pred_file)
    
    y_pred_proba = None
    if args.proba_file:
        y_pred_proba = np.load(args.proba_file)
        print(f"📈 Вероятности загружены: {y_pred_proba.shape}")
    
    print(f"🎯 Истинные значения: {y_true.shape}")
    print(f"🔮 Предсказания: {y_pred.shape}")
    
    # Создаем анализатор
    analyzer = PredictionQualityAnalyzer()
    
    # Запускаем анализ
    print("🔍 Запуск анализа качества предсказаний...")
    results = analyzer.analyze_all_variables(y_true, y_pred, y_pred_proba)
    
    # Сохраняем результаты
    json_file, report_file = analyzer.save_analysis_results(results, args.output_dir)
    
    print("✅ Анализ завершен!")
    
    # Выводим краткие результаты
    summary = results['summary']
    print(f"\n📊 Краткие результаты:")
    
    if 'regression' in summary:
        reg_info = summary['regression']
        print(f"   📈 Регрессия - средний R²: {reg_info.get('mean_r2', 0):.3f}")
    
    if 'classification' in summary:
        class_info = summary['classification']
        print(f"   🎯 Классификация - средний F1: {class_info.get('mean_f1', 0):.3f}")
    
    if 'binary_classification' in summary:
        binary_info = summary['binary_classification']
        print(f"   🎲 Бинарная классификация - средний AUC: {binary_info.get('mean_auc', 0):.3f}")
    
    print(f"\n📄 Детальные результаты сохранены в {json_file}")
    print(f"📖 Читаемый отчет сохранен в {report_file}")


if __name__ == "__main__":
    main()