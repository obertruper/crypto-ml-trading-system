"""
Метрики для оценки моделей
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
import pandas as pd


class MetricsCalculator:
    """Класс для расчета различных метрик"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: конфигурация с параметрами метрик
        """
        self.config = config
        self.reset()
        
    def reset(self):
        """Сброс накопленных значений"""
        self.predictions = []
        self.targets = []
        self.losses = []
        self.batch_count = 0
        
    def update(self, 
               predictions: Union[torch.Tensor, Dict],
               targets: Union[torch.Tensor, Dict],
               loss: Optional[float] = None):
        """
        Обновление накопленных значений
        
        Args:
            predictions: предсказания модели
            targets: истинные значения
            loss: значение функции потерь
        """
        # Конвертация в numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        elif isinstance(predictions, dict):
            predictions = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                          for k, v in predictions.items()}
            
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        elif isinstance(targets, dict):
            targets = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                      for k, v in targets.items()}
        
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if loss is not None:
            self.losses.append(loss)
            
        self.batch_count += 1
        
    def compute(self) -> Dict[str, float]:
        """
        Вычисление всех метрик
        
        Returns:
            Словарь с метриками
        """
        if self.batch_count == 0:
            return {}
            
        # Объединение батчей
        if isinstance(self.predictions[0], dict):
            all_predictions = self._aggregate_dict_batches(self.predictions)
            all_targets = self._aggregate_dict_batches(self.targets)
            
            # Расчет метрик для многозадачной модели
            metrics = self._compute_multitask_metrics(all_predictions, all_targets)
        else:
            all_predictions = np.concatenate(self.predictions, axis=0)
            all_targets = np.concatenate(self.targets, axis=0)
            
            # Расчет метрик для простой модели
            metrics = self._compute_single_task_metrics(all_predictions, all_targets)
        
        # Добавление средней потери
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
            
        return metrics
    
    def compute_batch_metrics(self,
                            predictions: Union[torch.Tensor, Dict],
                            targets: Union[torch.Tensor, Dict]) -> Dict[str, float]:
        """
        Вычисление метрик для одного батча
        
        Args:
            predictions: предсказания
            targets: истинные значения
            
        Returns:
            Словарь с метриками
        """
        # Конвертация в numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        elif isinstance(predictions, dict):
            predictions = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                          for k, v in predictions.items()}
            
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        elif isinstance(targets, dict):
            targets = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                      for k, v in targets.items()}
        
        if isinstance(predictions, dict):
            return self._compute_multitask_metrics(predictions, targets)
        else:
            return self._compute_single_task_metrics(predictions, targets)
    
    def compute_for_subset(self, predictions, targets) -> Dict[str, float]:
        """Вычисление метрик для подмножества данных"""
        if isinstance(predictions, dict):
            return self._compute_multitask_metrics(predictions, targets)
        else:
            return self._compute_single_task_metrics(predictions, targets)
    
    def _aggregate_dict_batches(self, batches: List[Dict]) -> Dict:
        """Объединение батчей словарей"""
        aggregated = {}
        
        for key in batches[0].keys():
            values = [batch[key] for batch in batches]
            if isinstance(values[0], np.ndarray):
                aggregated[key] = np.concatenate(values, axis=0)
            else:
                aggregated[key] = values
                
        return aggregated
    
    def _compute_single_task_metrics(self, 
                                   predictions: np.ndarray,
                                   targets: np.ndarray) -> Dict[str, float]:
        """Расчет метрик для одной задачи (включая многомерные выходы)"""
        metrics = {}
        
        # Обработка различных размерностей для торговой модели с 36 выходами
        if predictions.ndim == 2 and targets.ndim == 3:
            # predictions: (batch, 36), targets: (batch, 1, 36) или (batch, time, 36)
            if targets.shape[1] == 1:
                # Убираем лишнее измерение
                targets = targets.squeeze(1)
            else:
                # Берем последний временной шаг если time > 1
                targets = targets[:, -1, :]
        elif predictions.ndim == 3 and targets.ndim == 2:
            # predictions: (batch, time, features), targets: (batch, features)
            # Берем последний временной шаг из predictions
            predictions = predictions[:, -1, :]
        elif predictions.ndim == 3:
            # Reshape to 2D для вычисления метрик
            batch_size, time_steps, n_features = predictions.shape
            predictions = predictions.reshape(-1, n_features)
            
            # ИСПРАВЛЕНИЕ: Проверяем размерности targets
            if targets.ndim == 3:
                targets = targets.reshape(-1, n_features)
            elif targets.ndim == 2:
                # Если targets 2D, то возможно нужно расширить или обрезать
                if targets.shape[0] != predictions.shape[0]:
                    # Приводим к одинаковому размеру
                    min_samples = min(predictions.shape[0], targets.shape[0])
                    predictions = predictions[:min_samples]
                    targets = targets[:min_samples]
            
        # ИСПРАВЛЕНИЕ: Дополнительная проверка размерностей
        if predictions.shape != targets.shape:
            # Приводим к минимальному общему размеру
            min_samples = min(predictions.shape[0], targets.shape[0])
            min_features = min(predictions.shape[1] if predictions.ndim > 1 else 1, 
                              targets.shape[1] if targets.ndim > 1 else 1)
            
            predictions = predictions[:min_samples]
            targets = targets[:min_samples]
            
            if predictions.ndim > 1 and targets.ndim > 1:
                predictions = predictions[:, :min_features]
                targets = targets[:, :min_features]
        
        # Проверка на NaN
        if np.isnan(predictions).any() or np.isnan(targets).any():
            # Заменяем NaN на 0 для вычисления метрик
            predictions = np.nan_to_num(predictions, nan=0.0)
            targets = np.nan_to_num(targets, nan=0.0)
        
        # Получаем актуальное количество признаков после всех обработок
        n_features = predictions.shape[1] if predictions.ndim > 1 else 1
        
        # Специальная обработка для торговой модели с 36 выходами
        if predictions.ndim == 2 and n_features == 36:
            # Названия для 36 целевых переменных
            target_names = [
                'future_return_1', 'future_return_2', 'future_return_3', 'future_return_4',
                'long_tp1_hit', 'long_tp2_hit', 'long_tp3_hit', 'long_sl_hit',
                'short_tp1_hit', 'short_tp2_hit', 'short_tp3_hit', 'short_sl_hit',
                'long_tp1_time', 'long_tp2_time', 'long_tp3_time', 'long_sl_time',
                'short_tp1_time', 'short_tp2_time', 'short_tp3_time', 'short_sl_time',
                'long_tp1_reached', 'long_tp2_reached', 'long_tp3_reached', 'long_sl_reached',
                'short_tp1_reached', 'short_tp2_reached', 'short_tp3_reached', 'short_sl_reached',
                'long_optimal_entry', 'long_optimal_exit', 'short_optimal_entry', 'short_optimal_exit',
                'long_max_profit', 'short_max_profit', 'long_max_loss', 'short_max_loss'
            ]
            
            # Вычисляем метрики для групп переменных
            # Future returns (регрессия)
            future_returns_idx = list(range(4))
            fr_pred = predictions[:, future_returns_idx]
            fr_target = targets[:, future_returns_idx]
            metrics['future_returns_mse'] = mean_squared_error(fr_target.flatten(), fr_pred.flatten())
            metrics['future_returns_mae'] = mean_absolute_error(fr_target.flatten(), fr_pred.flatten())
            
            # Hit вероятности (бинарная классификация)
            hit_idx = list(range(4, 12))
            hit_pred = predictions[:, hit_idx]
            hit_target = targets[:, hit_idx]
            hit_pred_binary = (hit_pred > 0.5).astype(int)
            hit_target_binary = hit_target.astype(int)
            metrics['hit_accuracy'] = accuracy_score(hit_target_binary.flatten(), hit_pred_binary.flatten())
            
            # Time predictions (регрессия)
            time_idx = list(range(12, 20))
            time_pred = predictions[:, time_idx]
            time_target = targets[:, time_idx]
            # Только для ненулевых значений
            mask = time_target > 0
            if mask.any():
                metrics['time_mae'] = mean_absolute_error(time_target[mask], time_pred[mask])
            
            # Общие метрики
            metrics['overall_mse'] = mean_squared_error(targets.flatten(), predictions.flatten())
            metrics['overall_mae'] = mean_absolute_error(targets.flatten(), predictions.flatten())
            
            return metrics
            
        # Вычисляем метрики для каждого признака (старая логика)
        elif predictions.ndim > 1 and n_features > 1:
            for i in range(min(n_features, 5)):  # Ограничиваем для производительности
                pred_i = predictions[:, i]
                target_i = targets[:, i]
                
                # Пропускаем если все значения NaN
                if not np.isnan(pred_i).all() and not np.isnan(target_i).all():
                    metrics[f'feature_{i}_mse'] = mean_squared_error(target_i, pred_i)
                    metrics[f'feature_{i}_mae'] = mean_absolute_error(target_i, pred_i)
                
            # Общие метрики
            try:
                metrics['mse'] = mean_squared_error(targets.flatten(), predictions.flatten())
                metrics['mae'] = mean_absolute_error(targets.flatten(), predictions.flatten())
            except:
                metrics['mse'] = np.nan
                metrics['mae'] = np.nan
            return metrics
        
        # Проверка размерности для 2D
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        if targets.ndim > 1 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        # Регрессионные метрики
        if len(np.unique(targets)) > 10:  # Предполагаем регрессию
            metrics['mse'] = mean_squared_error(targets, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(targets, predictions)
            metrics['r2'] = r2_score(targets, predictions)
            
            # Дополнительные метрики
            metrics['mape'] = self._mean_absolute_percentage_error(targets, predictions)
            metrics['directional_accuracy'] = self._directional_accuracy(targets, predictions)
            
        else:  # Классификация
            # Преобразование вероятностей в классы
            if predictions.ndim > 1:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = (predictions > 0.5).astype(int)
                
            metrics['accuracy'] = accuracy_score(targets, pred_classes)
            
            # Дополнительные метрики для бинарной классификации
            if len(np.unique(targets)) == 2:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets, pred_classes, average='binary'
                )
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1'] = f1
                
                # AUC если есть вероятности
                if predictions.ndim == 1 or predictions.shape[1] == 1:
                    try:
                        metrics['auc'] = roc_auc_score(targets, predictions)
                    except:
                        pass
                        
        return metrics
    
    def _compute_multitask_metrics(self,
                                 predictions: Dict,
                                 targets: Dict) -> Dict[str, float]:
        """Расчет метрик для многозадачной модели"""
        metrics = {}
        
        # Метрики для предсказания цены
        if 'price_pred' in predictions and 'future_returns' in targets:
            price_metrics = self._compute_price_metrics(
                predictions['price_pred'],
                targets['future_returns']
            )
            for k, v in price_metrics.items():
                metrics[f'price_{k}'] = v
        
        # Метрики для вероятностей TP
        if 'tp_probs' in predictions and 'tp_targets' in targets:
            tp_metrics = self._compute_tp_metrics(
                predictions['tp_probs'],
                targets['tp_targets']
            )
            for k, v in tp_metrics.items():
                metrics[f'tp_{k}'] = v
        
        # Метрики для SL
        if 'sl_prob' in predictions and 'sl_target' in targets:
            sl_metrics = self._compute_sl_metrics(
                predictions['sl_prob'],
                targets['sl_target']
            )
            for k, v in sl_metrics.items():
                metrics[f'sl_{k}'] = v
        
        # Метрики для волатильности
        if 'volatility' in predictions and 'volatility_target' in targets:
            vol_metrics = self._compute_volatility_metrics(
                predictions['volatility'],
                targets['volatility_target']
            )
            for k, v in vol_metrics.items():
                metrics[f'volatility_{k}'] = v
        
        # Общие торговые метрики
        trading_metrics = self._compute_trading_metrics(predictions, targets)
        metrics.update(trading_metrics)
        
        return metrics
    
    def _compute_price_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Метрики для предсказания цены"""
        return {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'directional_accuracy': self._directional_accuracy(targets, predictions),
            'return_correlation': np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
        }
    
    def _compute_tp_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Метрики для вероятностей Take Profit"""
        metrics = {}
        
        # Для каждого уровня TP
        for i in range(predictions.shape[1]):
            if i < targets.shape[1]:
                pred = predictions[:, i]
                target = targets[:, i]
                
                # Бинарные метрики
                pred_binary = (pred > 0.5).astype(int)
                
                metrics[f'tp{i+1}_accuracy'] = accuracy_score(target, pred_binary)
                
                # Precision, Recall, F1
                if len(np.unique(target)) > 1:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        target, pred_binary, average='binary', zero_division=0
                    )
                    metrics[f'tp{i+1}_precision'] = precision
                    metrics[f'tp{i+1}_recall'] = recall
                    metrics[f'tp{i+1}_f1'] = f1
                    
                    # AUC
                    try:
                        metrics[f'tp{i+1}_auc'] = roc_auc_score(target, pred)
                    except:
                        pass
        
        return metrics
    
    def _compute_sl_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Метрики для Stop Loss"""
        pred_binary = (predictions > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(targets, pred_binary)
        }
        
        if len(np.unique(targets)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, pred_binary, average='binary', zero_division=0
            )
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            try:
                metrics['auc'] = roc_auc_score(targets, predictions)
            except:
                pass
                
        return metrics
    
    def _compute_volatility_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Метрики для волатильности"""
        return {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'correlation': np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
        }
    
    def _compute_trading_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """Общие торговые метрики"""
        metrics = {}
        
        # Потенциальная прибыль
        if 'tp_probs' in predictions and 'tp_targets' in targets:
            metrics['potential_profit'] = self._calculate_potential_profit(
                predictions['tp_probs'],
                targets['tp_targets']
            )
        
        # Риск-метрики
        if 'sl_prob' in predictions and 'sl_target' in targets:
            metrics['risk_ratio'] = self._calculate_risk_ratio(
                predictions['sl_prob'],
                targets['sl_target']
            )
        
        return metrics
    
    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAPE метрика"""
        epsilon = 1e-10
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Точность предсказания направления"""
        return np.mean(np.sign(y_true) == np.sign(y_pred))
    
    def _calculate_potential_profit(self, tp_probs: np.ndarray, tp_targets: np.ndarray) -> float:
        """Расчет потенциальной прибыли"""
        # Упрощенный расчет: средняя вероятность достижения TP
        tp_levels = [1.2, 2.4, 3.5, 5.8]  # % прибыли
        
        expected_profit = 0
        for i, level in enumerate(tp_levels):
            if i < tp_probs.shape[1] and i < tp_targets.shape[1]:
                # Вероятность * уровень прибыли * точность предсказания
                prob = tp_probs[:, i].mean()
                actual = tp_targets[:, i].mean()
                expected_profit += prob * level * (1 - abs(prob - actual))
                
        return expected_profit
    
    def _calculate_risk_ratio(self, sl_prob: np.ndarray, sl_target: np.ndarray) -> float:
        """Расчет соотношения риска"""
        # Средняя вероятность SL vs реальная частота
        pred_risk = sl_prob.mean()
        actual_risk = sl_target.mean()
        
        return pred_risk / (actual_risk + 1e-10)


class MetricsTracker:
    """Класс для отслеживания метрик во время обучения"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_calculator = MetricsCalculator(config)
        self.history = {
            'train': {},
            'val': {}
        }
        
    def update(self, mode: str, epoch: int, metrics: Dict):
        """
        Обновление истории метрик
        
        Args:
            mode: 'train' или 'val'
            epoch: номер эпохи
            metrics: словарь с метриками
        """
        for key, value in metrics.items():
            if key not in self.history[mode]:
                self.history[mode][key] = []
            self.history[mode][key].append(value)
            
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """
        Получение эпохи с лучшей метрикой
        
        Args:
            metric: название метрики
            mode: 'min' или 'max'
            
        Returns:
            Номер эпохи
        """
        if metric not in self.history['val']:
            return -1
            
        values = self.history['val'][metric]
        if mode == 'min':
            return np.argmin(values)
        else:
            return np.argmax(values)
            
    def get_current_metrics(self, mode: str = 'val') -> Dict:
        """Получение последних метрик"""
        current_metrics = {}
        
        for key, values in self.history[mode].items():
            if values:
                current_metrics[key] = values[-1]
                
        return current_metrics
    
    def save_to_csv(self, filepath: str):
        """Сохранение истории метрик в CSV"""
        # Преобразование в DataFrame
        train_df = pd.DataFrame(self.history['train'])
        val_df = pd.DataFrame(self.history['val'])
        
        # Добавление префиксов
        train_df = train_df.add_prefix('train_')
        val_df = val_df.add_prefix('val_')
        
        # Объединение
        df = pd.concat([train_df, val_df], axis=1)
        df.index.name = 'epoch'
        
        # Сохранение
        df.to_csv(filepath)
        
    def plot_metrics(self, metrics: List[str], save_path: Optional[str] = None):
        """Визуализация метрик"""
        import matplotlib.pyplot as plt
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots((n_metrics + 1) // 2, 2, figsize=(12, 4 * ((n_metrics + 1) // 2)))
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Данные для графика
            train_data = self.history['train'].get(metric, [])
            val_data = self.history['val'].get(metric, [])
            
            if train_data:
                ax.plot(train_data, label='Train', linewidth=2)
            if val_data:
                ax.plot(val_data, label='Val', linewidth=2)
                
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Удаление лишних subplot'ов
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def calculate_trading_metrics(predictions: np.ndarray,
                            actual_prices: np.ndarray,
                            positions: np.ndarray,
                            transaction_cost: float = 0.001) -> Dict[str, float]:
    """
    Расчет торговых метрик для бэктеста
    
    Args:
        predictions: предсказанные цены/доходности
        actual_prices: реальные цены
        positions: размеры позиций
        transaction_cost: комиссия
        
    Returns:
        Словарь с торговыми метриками
    """
    # Расчет доходностей
    returns = np.diff(actual_prices) / actual_prices[:-1]
    
    # Доходности стратегии
    strategy_returns = positions[:-1] * returns - np.abs(np.diff(positions)) * transaction_cost
    
    # Метрики
    metrics = {
        'total_return': np.prod(1 + strategy_returns) - 1,
        'annual_return': (1 + metrics['total_return']) ** (252 / len(returns)) - 1,
        'sharpe_ratio': np.sqrt(252) * strategy_returns.mean() / strategy_returns.std(),
        'max_drawdown': calculate_max_drawdown(strategy_returns),
        'win_rate': (strategy_returns > 0).sum() / len(strategy_returns),
        'profit_factor': calculate_profit_factor(strategy_returns),
        'calmar_ratio': metrics['annual_return'] / abs(metrics['max_drawdown'])
    }
    
    return metrics


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Расчет максимальной просадки"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_profit_factor(returns: np.ndarray) -> float:
    """Расчет profit factor"""
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return profits / losses if losses > 0 else np.inf