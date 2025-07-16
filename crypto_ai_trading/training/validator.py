"""
Модуль валидации моделей
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from utils.logger import get_logger
from utils.metrics import MetricsCalculator
from utils.visualization import TradingVisualizer


class ModelValidator:
    """Класс для валидации и оценки моделей"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 device: Optional[torch.device] = None):
        """
        Args:
            model: модель для валидации
            config: конфигурация
            device: устройство для вычислений
        """
        self.model = model
        self.config = config
        self.logger = get_logger("ModelValidator")
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        
        # Калькулятор метрик
        self.metrics_calculator = MetricsCalculator(config)
        
        # Директория для результатов
        self.results_dir = Path("results") / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_model(self, 
                      val_loader: DataLoader,
                      return_predictions: bool = False) -> Dict:
        """
        Валидация модели на валидационном наборе
        
        Args:
            val_loader: DataLoader для валидации
            return_predictions: возвращать ли предсказания
            
        Returns:
            Словарь с метриками и опционально предсказаниями
        """
        self.logger.info("Начало валидации модели...")
        
        all_predictions = []
        all_targets = []
        all_info = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(tqdm(val_loader, desc="Validation")):
                # Перенос на устройство
                inputs = inputs.to(self.device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                elif isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Предсказание
                outputs = self.model(inputs)
                
                # Сохранение для анализа
                if return_predictions:
                    all_predictions.append(self._to_numpy(outputs))
                    all_targets.append(self._to_numpy(targets))
                    all_info.extend(info)
                
                # Обновление метрик
                self.metrics_calculator.update(outputs, targets, info)
        
        # Вычисление финальных метрик
        metrics = self.metrics_calculator.compute()
        
        # Подготовка результатов
        results = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if return_predictions:
            results['predictions'] = self._aggregate_predictions(all_predictions)
            results['targets'] = self._aggregate_predictions(all_targets)
            results['info'] = all_info
        
        # Сохранение результатов
        self._save_validation_results(results)
        
        self.logger.info(f"Валидация завершена. Основные метрики:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.4f}")
        
        return results
    
    def evaluate_trading_performance(self,
                                   test_loader: DataLoader,
                                   initial_capital: float = 100000) -> Dict:
        """
        Оценка торговой производительности модели
        
        Args:
            test_loader: DataLoader для тестирования
            initial_capital: начальный капитал
            
        Returns:
            Словарь с торговыми метриками
        """
        self.logger.info("Оценка торговой производительности...")
        
        # Получаем предсказания
        results = self.validate_model(test_loader, return_predictions=True)
        predictions = results['predictions']
        info = results['info']
        
        # Симуляция торговли
        trading_results = self._simulate_trading(predictions, info, initial_capital)
        
        # Расчет торговых метрик
        trading_metrics = self._calculate_trading_metrics(trading_results)
        
        # Визуализация
        self._plot_trading_results(trading_results)
        
        return trading_metrics
    
    def cross_validate(self,
                      data_loader: DataLoader,
                      n_folds: int = 5) -> Dict:
        """
        Кросс-валидация модели
        
        Args:
            data_loader: полный DataLoader
            n_folds: количество фолдов
            
        Returns:
            Словарь с результатами кросс-валидации
        """
        self.logger.info(f"Начало {n_folds}-fold кросс-валидации...")
        
        # Разделение данных на фолды
        dataset_size = len(data_loader.dataset)
        fold_size = dataset_size // n_folds
        
        fold_results = []
        
        for fold in range(n_folds):
            self.logger.info(f"Fold {fold + 1}/{n_folds}")
            
            # Создание индексов для валидации
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else dataset_size
            
            val_indices = list(range(val_start, val_end))
            train_indices = list(range(0, val_start)) + list(range(val_end, dataset_size))
            
            # Создание подмножеств
            val_subset = torch.utils.data.Subset(data_loader.dataset, val_indices)
            
            # Создание DataLoader для валидации
            val_loader = DataLoader(
                val_subset,
                batch_size=data_loader.batch_size,
                shuffle=False,
                num_workers=data_loader.num_workers
            )
            
            # Валидация на фолде
            fold_metrics = self.validate_model(val_loader)['metrics']
            fold_results.append(fold_metrics)
        
        # Агрегация результатов
        cv_results = self._aggregate_cv_results(fold_results)
        
        # Сохранение результатов
        self._save_cv_results(cv_results)
        
        return cv_results
    
    def analyze_predictions(self,
                          predictions: Dict,
                          targets: Dict,
                          info: List[Dict]) -> Dict:
        """
        Детальный анализ предсказаний
        
        Args:
            predictions: предсказания модели
            targets: истинные значения
            info: дополнительная информация
            
        Returns:
            Словарь с результатами анализа
        """
        analysis = {}
        
        # Анализ по символам
        symbol_performance = self._analyze_by_symbol(predictions, targets, info)
        analysis['symbol_performance'] = symbol_performance
        
        # Анализ по времени
        temporal_analysis = self._analyze_temporal_patterns(predictions, targets, info)
        analysis['temporal_patterns'] = temporal_analysis
        
        # Анализ ошибок
        error_analysis = self._analyze_errors(predictions, targets, info)
        analysis['error_analysis'] = error_analysis
        
        # Анализ уверенности
        if 'tp_probs' in predictions:
            confidence_analysis = self._analyze_confidence(predictions, targets)
            analysis['confidence_analysis'] = confidence_analysis
        
        return analysis
    
    def _to_numpy(self, tensor_or_dict: Union[torch.Tensor, Dict]) -> Union[np.ndarray, Dict]:
        """Конвертация тензоров в numpy arrays"""
        if isinstance(tensor_or_dict, torch.Tensor):
            return tensor_or_dict.cpu().numpy()
        elif isinstance(tensor_or_dict, dict):
            return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                   for k, v in tensor_or_dict.items()}
        else:
            return tensor_or_dict
    
    def _aggregate_predictions(self, predictions_list: List) -> Union[np.ndarray, Dict]:
        """Агрегация батчей предсказаний"""
        if not predictions_list:
            return None
            
        if isinstance(predictions_list[0], dict):
            # Словарь предсказаний
            aggregated = {}
            for key in predictions_list[0].keys():
                aggregated[key] = np.concatenate([p[key] for p in predictions_list], axis=0)
            return aggregated
        else:
            # Простой массив
            return np.concatenate(predictions_list, axis=0)
    
    def _simulate_trading(self, 
                         predictions: Dict,
                         info: List[Dict],
                         initial_capital: float) -> pd.DataFrame:
        """Симуляция торговли на основе предсказаний"""
        
        # Создание DataFrame с результатами
        results = []
        capital = initial_capital
        position = 0
        
        for i in range(len(info)):
            symbol = info[i]['symbol']
            timestamp = info[i].get('target_start_time', i)
            last_close = info[i]['last_close']
            
            # Торговые сигналы
            if 'tp_probs' in predictions:
                tp_probs = predictions['tp_probs'][i]
                sl_prob = predictions['sl_prob'][i]
                
                # Простая стратегия: покупаем если вероятность TP > порога
                if tp_probs[0] > 0.6 and position == 0:  # TP 1.2%
                    position = capital * 0.1 / last_close  # 10% капитала
                    entry_price = last_close
                elif sl_prob > 0.4 and position > 0:  # Стоп-лосс
                    capital += position * last_close - position * entry_price
                    position = 0
            
            results.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'capital': capital,
                'position': position,
                'price': last_close
            })
        
        return pd.DataFrame(results)
    
    def _calculate_trading_metrics(self, trading_results: pd.DataFrame) -> Dict:
        """Расчет торговых метрик"""
        
        # Доходность
        initial_capital = trading_results['capital'].iloc[0]
        final_capital = trading_results['capital'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Расчет дневных доходностей
        daily_returns = trading_results['capital'].pct_change().dropna()
        
        # Метрики
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(trading_results['capital']),
            'win_rate': self._calculate_win_rate(daily_returns),
            'profit_factor': self._calculate_profit_factor(daily_returns),
            'number_of_trades': len(trading_results[trading_results['position'] != 0])
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, capital_series: pd.Series) -> float:
        """Расчет максимальной просадки"""
        cummax = capital_series.cummax()
        drawdown = (capital_series - cummax) / cummax
        return drawdown.min()
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Расчет процента выигрышных сделок"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Расчет profit factor"""
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return profits / losses if losses > 0 else np.inf
    
    def _analyze_by_symbol(self, predictions, targets, info) -> Dict:
        """Анализ производительности по символам"""
        symbol_metrics = {}
        
        # Группировка по символам
        symbols = [i['symbol'] for i in info]
        unique_symbols = list(set(symbols))
        
        for symbol in unique_symbols:
            symbol_indices = [i for i, s in enumerate(symbols) if s == symbol]
            
            # Фильтрация предсказаний для символа
            if isinstance(predictions, dict):
                symbol_preds = {k: v[symbol_indices] for k, v in predictions.items()}
                symbol_targets = {k: v[symbol_indices] for k, v in targets.items()}
            else:
                symbol_preds = predictions[symbol_indices]
                symbol_targets = targets[symbol_indices]
            
            # Расчет метрик для символа
            symbol_metrics[symbol] = self.metrics_calculator.compute_for_subset(
                symbol_preds, symbol_targets
            )
        
        return symbol_metrics
    
    def _analyze_temporal_patterns(self, predictions, targets, info) -> Dict:
        """Анализ временных паттернов"""
        # Извлечение временных меток
        timestamps = [i.get('target_start_time', idx) for idx, i in enumerate(info)]
        
        # Группировка по часам дня, дням недели и т.д.
        temporal_analysis = {
            'hourly_performance': {},
            'daily_performance': {},
            'monthly_performance': {}
        }
        
        # Здесь можно добавить более детальный анализ
        
        return temporal_analysis
    
    def _analyze_errors(self, predictions, targets, info) -> Dict:
        """Анализ ошибок предсказаний"""
        error_analysis = {}
        
        if 'price_pred' in predictions and 'future_returns' in targets:
            errors = predictions['price_pred'] - targets['future_returns']
            
            error_analysis['mean_error'] = np.mean(errors)
            error_analysis['std_error'] = np.std(errors)
            error_analysis['mae'] = np.mean(np.abs(errors))
            error_analysis['rmse'] = np.sqrt(np.mean(errors ** 2))
            
            # Квантили ошибок
            error_analysis['error_quantiles'] = {
                '5%': np.percentile(errors, 5),
                '25%': np.percentile(errors, 25),
                '50%': np.percentile(errors, 50),
                '75%': np.percentile(errors, 75),
                '95%': np.percentile(errors, 95)
            }
        
        return error_analysis
    
    def _analyze_confidence(self, predictions, targets) -> Dict:
        """Анализ уверенности модели"""
        confidence_analysis = {}
        
        if 'tp_probs' in predictions:
            tp_probs = predictions['tp_probs']
            
            # Калибровка вероятностей
            confidence_analysis['calibration'] = self._calculate_calibration(
                tp_probs, targets.get('tp_targets')
            )
            
            # Распределение уверенности
            confidence_analysis['confidence_distribution'] = {
                'mean': np.mean(tp_probs),
                'std': np.std(tp_probs),
                'min': np.min(tp_probs),
                'max': np.max(tp_probs)
            }
        
        return confidence_analysis
    
    def _calculate_calibration(self, predicted_probs, true_labels) -> Dict:
        """Расчет калибровки вероятностей"""
        if true_labels is None:
            return {}
            
        # Разбиение на бины
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        calibration = {
            'bin_accuracy': [],
            'bin_confidence': [],
            'bin_count': []
        }
        
        for i in range(n_bins):
            bin_mask = (predicted_probs >= bin_boundaries[i]) & (predicted_probs < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(true_labels[bin_mask])
                bin_confidence = np.mean(predicted_probs[bin_mask])
                calibration['bin_accuracy'].append(bin_accuracy)
                calibration['bin_confidence'].append(bin_confidence)
                calibration['bin_count'].append(np.sum(bin_mask))
        
        # Expected Calibration Error
        ece = np.sum([
            count * np.abs(acc - conf) 
            for acc, conf, count in zip(
                calibration['bin_accuracy'],
                calibration['bin_confidence'],
                calibration['bin_count']
            )
        ]) / np.sum(calibration['bin_count'])
        
        calibration['ece'] = ece
        
        return calibration
    
    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Агрегация результатов кросс-валидации"""
        cv_results = {
            'mean_metrics': {},
            'std_metrics': {},
            'fold_metrics': fold_results
        }
        
        # Вычисление средних и стандартных отклонений
        all_keys = fold_results[0].keys()
        
        for key in all_keys:
            values = [fold[key] for fold in fold_results if isinstance(fold[key], (int, float))]
            if values:
                cv_results['mean_metrics'][key] = np.mean(values)
                cv_results['std_metrics'][key] = np.std(values)
        
        return cv_results
    
    def _plot_trading_results(self, trading_results: pd.DataFrame):
        """Визуализация результатов торговли"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # График капитала
        axes[0].plot(trading_results.index, trading_results['capital'])
        axes[0].set_title('Динамика капитала')
        axes[0].set_ylabel('Капитал')
        axes[0].grid(True)
        
        # График позиций
        axes[1].plot(trading_results.index, trading_results['position'])
        axes[1].set_title('Размер позиции')
        axes[1].set_ylabel('Позиция')
        axes[1].set_xlabel('Время')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'trading_results.png')
        plt.close()
    
    def _save_validation_results(self, results: Dict):
        """Сохранение результатов валидации"""
        # Сохранение метрик
        with open(self.results_dir / 'validation_metrics.json', 'w') as f:
            json.dump(results['metrics'], f, indent=4, default=str)
        
        # Сохранение полных результатов
        if 'predictions' in results:
            np.save(self.results_dir / 'predictions.npy', results['predictions'])
            np.save(self.results_dir / 'targets.npy', results['targets'])
        
        self.logger.info(f"Результаты сохранены в {self.results_dir}")
    
    def _save_cv_results(self, cv_results: Dict):
        """Сохранение результатов кросс-валидации"""
        with open(self.results_dir / 'cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=4, default=str)
        
        # Создание отчета
        report = "Результаты кросс-валидации:\n\n"
        report += "Средние метрики:\n"
        for key, value in cv_results['mean_metrics'].items():
            std = cv_results['std_metrics'].get(key, 0)
            report += f"  {key}: {value:.4f} ± {std:.4f}\n"
        
        with open(self.results_dir / 'cv_report.txt', 'w') as f:
            f.write(report)


def validate_checkpoint(checkpoint_path: str,
                       test_loader: DataLoader,
                       config: Dict) -> Dict:
    """
    Быстрая функция для валидации сохраненной модели
    
    Args:
        checkpoint_path: путь к checkpoint'у
        test_loader: DataLoader для тестирования
        config: конфигурация
        
    Returns:
        Результаты валидации
    """
    from models.patchtst import PatchTSTForPrediction
    
    # Загрузка модели
    model = PatchTSTForPrediction(**config['model'])
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Создание валидатора
    validator = ModelValidator(model, config)
    
    # Валидация
    results = validator.validate_model(test_loader, return_predictions=True)
    
    # Торговая производительность
    trading_metrics = validator.evaluate_trading_performance(test_loader)
    results['trading_metrics'] = trading_metrics
    
    return results