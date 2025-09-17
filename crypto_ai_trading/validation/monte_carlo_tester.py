"""
Monte Carlo тестирование для валидации статистической значимости торговых стратегий
Включает permutation tests, bootstrap анализ и stress testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import warnings
from pathlib import Path
import json
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
from scipy import stats
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import get_logger


@dataclass
class MonteCarloConfig:
    """Конфигурация Monte Carlo тестов"""
    n_permutations: int = 1000         # Количество пермутаций
    n_bootstrap: int = 500             # Количество bootstrap выборок
    confidence_level: float = 0.95     # Доверительный интервал
    min_observations: int = 100        # Минимум наблюдений для теста
    parallel_jobs: int = -1            # Параллельные процессы (-1 = все CPU)
    random_seed: int = 42              # Seed для воспроизводимости
    
    # Параметры stress testing
    stress_scenarios: int = 100        # Количество сценариев стресс-теста
    volatility_multipliers: List[float] = None  # Множители волатильности
    correlation_shifts: List[float] = None      # Сдвиги корреляции
    drawdown_limits: List[float] = None         # Лимиты просадки


@dataclass
class PermutationResult:
    """Результат permutation теста"""
    observed_metric: float
    permuted_distribution: List[float]
    p_value: float
    z_score: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    null_hypothesis_rejected: bool


class MonteCarloTester:
    """
    Система Monte Carlo тестирования для торговых стратегий
    
    Включает:
    - Permutation тесты для статистической значимости
    - Bootstrap анализ для доверительных интервалов
    - Stress тестирование при различных рыночных условиях
    - Анализ robustness стратегии
    """
    
    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()
        self.logger = get_logger("MonteCarloTester")
        
        # Устанавливаем defaults если не заданы
        if self.config.volatility_multipliers is None:
            self.config.volatility_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
            
        if self.config.correlation_shifts is None:
            self.config.correlation_shifts = [-0.3, -0.1, 0.0, 0.1, 0.3]
            
        if self.config.drawdown_limits is None:
            self.config.drawdown_limits = [0.05, 0.10, 0.15, 0.20, 0.25]
        
        # Фиксируем seed для воспроизводимости
        np.random.seed(self.config.random_seed)
        
        self.logger.info("🎲 Инициализация Monte Carlo тестера")
        self.logger.info(f"   🔄 Пермутаций: {self.config.n_permutations}")
        self.logger.info(f"   📊 Bootstrap: {self.config.n_bootstrap}")
        self.logger.info(f"   🌪️ Стресс-сценариев: {self.config.stress_scenarios}")
        
    def permutation_test(self, 
                        strategy_returns: np.ndarray,
                        benchmark_returns: Optional[np.ndarray] = None,
                        metric_func: Callable = None,
                        alternative: str = 'two-sided') -> PermutationResult:
        """
        Permutation тест для статистической значимости метрики
        
        Args:
            strategy_returns: Доходности стратегии
            benchmark_returns: Доходности бенчмарка (опционально)
            metric_func: Функция расчета метрики (по умолчанию среднее)
            alternative: Тип гипотезы ('two-sided', 'greater', 'less')
            
        Returns:
            Результат permutation теста
        """
        
        if len(strategy_returns) < self.config.min_observations:
            raise ValueError(f"Недостаточно данных: {len(strategy_returns)} < {self.config.min_observations}")
        
        # Функция метрики по умолчанию
        if metric_func is None:
            metric_func = lambda x: np.mean(x) if len(x) > 0 else 0
        
        # Рассчитываем наблюдаемую метрику
        if benchmark_returns is not None:
            # Тест относительно бенчмарка
            excess_returns = strategy_returns - benchmark_returns
            observed_metric = metric_func(excess_returns)
            all_returns = np.concatenate([strategy_returns, benchmark_returns])
        else:
            # Тест относительно нуля (случайная стратегия)
            observed_metric = metric_func(strategy_returns)
            all_returns = strategy_returns.copy()
        
        self.logger.info(f"🎯 Наблюдаемая метрика: {observed_metric:.4f}")
        self.logger.info(f"🔄 Запуск {self.config.n_permutations} пермутаций...")
        
        # Генерируем пермутированное распределение
        permuted_metrics = []
        n_obs = len(strategy_returns)
        
        for i in range(self.config.n_permutations):
            if benchmark_returns is not None:
                # Случайное перемешивание между стратегией и бенчмарком
                shuffled_returns = np.random.permutation(all_returns)
                fake_strategy = shuffled_returns[:n_obs]
                fake_benchmark = shuffled_returns[n_obs:]
                fake_excess = fake_strategy - fake_benchmark
                permuted_metric = metric_func(fake_excess)
            else:
                # Случайное перемешивание знаков доходностей
                signs = np.random.choice([-1, 1], size=len(all_returns))
                shuffled_returns = all_returns * signs
                permuted_metric = metric_func(shuffled_returns)
            
            permuted_metrics.append(permuted_metric)
            
            if (i + 1) % 100 == 0:
                self.logger.debug(f"   Выполнено пермутаций: {i + 1}")
        
        permuted_metrics = np.array(permuted_metrics)
        
        # Рассчитываем p-value
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(permuted_metrics) >= np.abs(observed_metric))
        elif alternative == 'greater':
            p_value = np.mean(permuted_metrics >= observed_metric)
        elif alternative == 'less':
            p_value = np.mean(permuted_metrics <= observed_metric)
        else:
            raise ValueError(f"Неизвестный тип гипотезы: {alternative}")
        
        # Z-score
        perm_mean = np.mean(permuted_metrics)
        perm_std = np.std(permuted_metrics)
        z_score = (observed_metric - perm_mean) / (perm_std + 1e-8)
        
        # Доверительный интервал
        alpha = 1 - self.config.confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        confidence_interval = (
            np.percentile(permuted_metrics, lower_percentile),
            np.percentile(permuted_metrics, upper_percentile)
        )
        
        # Статистическая значимость
        is_significant = p_value < (1 - self.config.confidence_level)
        
        result = PermutationResult(
            observed_metric=observed_metric,
            permuted_distribution=permuted_metrics.tolist(),
            p_value=p_value,
            z_score=z_score,
            confidence_interval=confidence_interval,
            is_significant=is_significant,
            null_hypothesis_rejected=is_significant
        )
        
        self.logger.info(f"📊 Результат permutation теста:")
        self.logger.info(f"   p-value: {p_value:.4f}")
        self.logger.info(f"   Z-score: {z_score:.2f}")
        self.logger.info(f"   Значимость: {'ДА' if is_significant else 'НЕТ'}")
        
        return result
    
    def bootstrap_analysis(self, 
                          returns: np.ndarray,
                          metric_funcs: Dict[str, Callable] = None) -> Dict:
        """
        Bootstrap анализ для доверительных интервалов метрик
        
        Args:
            returns: Массив доходностей
            metric_funcs: Словарь функций метрик {name: function}
            
        Returns:
            Результаты bootstrap анализа
        """
        
        if len(returns) < self.config.min_observations:
            raise ValueError(f"Недостаточно данных: {len(returns)} < {self.config.min_observations}")
        
        # Функции метрик по умолчанию
        if metric_funcs is None:
            metric_funcs = {
                'mean_return': np.mean,
                'std_return': np.std,
                'sharpe_ratio': lambda x: np.mean(x) / (np.std(x) + 1e-8),
                'max_drawdown': self._calculate_max_drawdown,
                'hit_rate': lambda x: np.mean(x > 0),
                'skewness': lambda x: stats.skew(x) if len(x) > 2 else 0,
                'kurtosis': lambda x: stats.kurtosis(x) if len(x) > 3 else 0
            }
        
        self.logger.info(f"🔄 Bootstrap анализ: {self.config.n_bootstrap} выборок")
        
        bootstrap_results = {}
        
        for metric_name, metric_func in metric_funcs.items():
            self.logger.debug(f"   Анализ метрики: {metric_name}")
            
            # Наблюдаемое значение
            observed_value = metric_func(returns)
            
            # Bootstrap выборки
            bootstrap_values = []
            
            for i in range(self.config.n_bootstrap):
                # Создаем bootstrap выборку с возвращением
                bootstrap_sample = resample(returns, n_samples=len(returns), random_state=i)
                bootstrap_value = metric_func(bootstrap_sample)
                bootstrap_values.append(bootstrap_value)
            
            bootstrap_values = np.array(bootstrap_values)
            
            # Доверительный интервал
            alpha = 1 - self.config.confidence_level
            lower_percentile = 100 * (alpha / 2)
            upper_percentile = 100 * (1 - alpha / 2)
            
            confidence_interval = (
                np.percentile(bootstrap_values, lower_percentile),
                np.percentile(bootstrap_values, upper_percentile)
            )
            
            # Bias и standard error
            bootstrap_mean = np.mean(bootstrap_values)
            bias = bootstrap_mean - observed_value
            standard_error = np.std(bootstrap_values)
            
            bootstrap_results[metric_name] = {
                'observed': observed_value,
                'bootstrap_mean': bootstrap_mean,
                'bias': bias,
                'standard_error': standard_error,
                'confidence_interval': confidence_interval,
                'bootstrap_distribution': bootstrap_values.tolist()
            }
        
        return bootstrap_results
    
    def stress_test_strategy(self, 
                           returns: np.ndarray,
                           prices: Optional[np.ndarray] = None) -> Dict:
        """
        Stress тестирование стратегии при различных рыночных условиях
        
        Args:
            returns: Исторические доходности
            prices: Исторические цены (опционально)
            
        Returns:
            Результаты stress тестирования
        """
        
        self.logger.info("🌪️ Запуск stress тестирования")
        
        stress_results = {
            'volatility_stress': self._volatility_stress_test(returns),
            'correlation_stress': self._correlation_stress_test(returns),
            'regime_change_stress': self._regime_change_stress_test(returns),
            'black_swan_stress': self._black_swan_stress_test(returns),
            'liquidity_stress': self._liquidity_stress_test(returns, prices)
        }
        
        # Общая оценка устойчивости
        stress_results['overall_robustness'] = self._calculate_overall_robustness(stress_results)
        
        return stress_results
    
    def _volatility_stress_test(self, returns: np.ndarray) -> Dict:
        """Stress тест при различных уровнях волатильности"""
        
        original_vol = np.std(returns)
        original_sharpe = np.mean(returns) / (original_vol + 1e-8)
        
        vol_results = {}
        
        for vol_mult in self.config.volatility_multipliers:
            # Масштабируем доходности
            scaled_returns = returns * vol_mult
            
            # Рассчитываем метрики
            new_vol = np.std(scaled_returns)
            new_sharpe = np.mean(scaled_returns) / (new_vol + 1e-8)
            max_dd = self._calculate_max_drawdown(scaled_returns)
            
            vol_results[f"vol_{vol_mult}x"] = {
                'volatility': new_vol,
                'sharpe_ratio': new_sharpe,
                'max_drawdown': max_dd,
                'returns_std': new_vol,
                'hit_rate': np.mean(scaled_returns > 0)
            }
        
        return vol_results
    
    def _correlation_stress_test(self, returns: np.ndarray) -> Dict:
        """Stress тест при изменении корреляционной структуры"""
        
        # Для демонстрации создаем искусственные корреляции
        n = len(returns)
        corr_results = {}
        
        for corr_shift in self.config.correlation_shifts:
            # Добавляем коррелированный шум
            corr_noise = np.random.randn(n) * np.std(returns) * 0.5
            
            if corr_shift != 0:
                # Создаем корреляцию
                correlation = corr_shift
                adjusted_returns = returns + correlation * corr_noise
            else:
                adjusted_returns = returns.copy()
            
            # Рассчитываем метрики
            sharpe = np.mean(adjusted_returns) / (np.std(adjusted_returns) + 1e-8)
            max_dd = self._calculate_max_drawdown(adjusted_returns)
            
            corr_results[f"corr_{corr_shift:+.1f}"] = {
                'mean_return': np.mean(adjusted_returns),
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'volatility': np.std(adjusted_returns)
            }
        
        return corr_results
    
    def _regime_change_stress_test(self, returns: np.ndarray) -> Dict:
        """Stress тест при смене рыночных режимов"""
        
        n = len(returns)
        regime_results = {}
        
        # Различные рыночные режимы
        regimes = {
            'bull_market': {'trend': 0.001, 'vol_mult': 0.8},  # Растущий рынок, низкая волатильность
            'bear_market': {'trend': -0.002, 'vol_mult': 1.5}, # Падающий рынок, высокая волатильность
            'sideways': {'trend': 0.0, 'vol_mult': 1.2},       # Боковой тренд
            'crisis': {'trend': -0.005, 'vol_mult': 3.0}       # Кризис
        }
        
        for regime_name, regime_params in regimes.items():
            # Моделируем режим
            trend_component = np.full(n, regime_params['trend'])
            vol_adjusted_returns = returns * regime_params['vol_mult']
            regime_returns = vol_adjusted_returns + trend_component
            
            # Рассчитываем метрики
            sharpe = np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)
            max_dd = self._calculate_max_drawdown(regime_returns)
            
            regime_results[regime_name] = {
                'mean_return': np.mean(regime_returns),
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'volatility': np.std(regime_returns),
                'hit_rate': np.mean(regime_returns > 0)
            }
        
        return regime_results
    
    def _black_swan_stress_test(self, returns: np.ndarray) -> Dict:
        """Stress тест при экстремальных событиях (black swans)"""
        
        swan_results = {}
        
        # Различные типы черных лебедей
        swan_scenarios = [
            {'name': 'flash_crash', 'shock': -0.20, 'duration': 1},      # Мгновенный обвал -20%
            {'name': 'sustained_crash', 'shock': -0.10, 'duration': 5},  # Продолжительное падение
            {'name': 'volatility_spike', 'shock': 0.0, 'vol_mult': 5.0}, # Резкий всплеск волатильности
            {'name': 'liquidity_crisis', 'shock': -0.05, 'gap': 0.02}    # Кризис ликвидности с гэпами
        ]
        
        for scenario in swan_scenarios:
            shocked_returns = returns.copy()
            
            if scenario['name'] == 'flash_crash':
                # Однократный шок в случайной точке
                shock_idx = np.random.randint(len(returns))
                shocked_returns[shock_idx] = scenario['shock']
                
            elif scenario['name'] == 'sustained_crash':
                # Продолжительное падение
                start_idx = np.random.randint(len(returns) - scenario['duration'])
                for i in range(scenario['duration']):
                    shocked_returns[start_idx + i] += scenario['shock'] / scenario['duration']
                    
            elif scenario['name'] == 'volatility_spike':
                # Резкое увеличение волатильности
                spike_length = min(10, len(returns) // 4)
                start_idx = np.random.randint(len(returns) - spike_length)
                shocked_returns[start_idx:start_idx + spike_length] *= scenario['vol_mult']
                
            elif scenario['name'] == 'liquidity_crisis':
                # Добавляем случайные гэпы
                n_gaps = max(1, len(returns) // 50)
                gap_indices = np.random.choice(len(returns), n_gaps, replace=False)
                for idx in gap_indices:
                    shocked_returns[idx] += np.random.choice([-1, 1]) * scenario['gap']
            
            # Рассчитываем метрики после шока
            sharpe = np.mean(shocked_returns) / (np.std(shocked_returns) + 1e-8)
            max_dd = self._calculate_max_drawdown(shocked_returns)
            
            swan_results[scenario['name']] = {
                'mean_return': np.mean(shocked_returns),
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'volatility': np.std(shocked_returns),
                'recovery_time': self._calculate_recovery_time(shocked_returns),
                'tail_ratio': self._calculate_tail_ratio(shocked_returns)
            }
        
        return swan_results
    
    def _liquidity_stress_test(self, returns: np.ndarray, prices: Optional[np.ndarray] = None) -> Dict:
        """Stress тест ликвидности (требует данные о ценах)"""
        
        if prices is None:
            # Восстанавливаем цены из доходностей
            prices = np.cumprod(1 + returns) * 100  # Предполагаем начальную цену 100
        
        liquidity_results = {}
        
        # Различные сценарии ликвидности
        scenarios = [
            {'name': 'normal_liquidity', 'bid_ask_spread': 0.001, 'slippage': 0.0005},
            {'name': 'reduced_liquidity', 'bid_ask_spread': 0.005, 'slippage': 0.002},
            {'name': 'crisis_liquidity', 'bid_ask_spread': 0.02, 'slippage': 0.01},
            {'name': 'extreme_illiquidity', 'bid_ask_spread': 0.05, 'slippage': 0.03}
        ]
        
        for scenario in scenarios:
            # Применяем торговые издержки
            trading_costs = scenario['bid_ask_spread'] + scenario['slippage']
            
            # Предполагаем что половина доходности теряется на торговых издержках
            # при активной торговле
            liquidity_adjusted_returns = returns - np.abs(returns) * trading_costs
            
            # Рассчитываем метрики
            sharpe = np.mean(liquidity_adjusted_returns) / (np.std(liquidity_adjusted_returns) + 1e-8)
            max_dd = self._calculate_max_drawdown(liquidity_adjusted_returns)
            
            liquidity_results[scenario['name']] = {
                'mean_return': np.mean(liquidity_adjusted_returns),
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'trading_costs': trading_costs,
                'net_profit_margin': np.mean(liquidity_adjusted_returns) / (np.mean(np.abs(returns)) + 1e-8)
            }
        
        return liquidity_results
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Рассчитывает максимальную просадку"""
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_recovery_time(self, returns: np.ndarray) -> int:
        """Рассчитывает время восстановления после просадки"""
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        
        # Находим максимальную просадку
        drawdown = (cumulative - running_max) / running_max
        max_dd_idx = np.argmin(drawdown)
        
        # Находим восстановление после максимальной просадки
        recovery_idx = max_dd_idx
        max_before_dd = running_max[max_dd_idx]
        
        for i in range(max_dd_idx + 1, len(cumulative)):
            if cumulative[i] >= max_before_dd:
                recovery_idx = i
                break
        
        return recovery_idx - max_dd_idx
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Рассчитывает отношение хвостов распределения"""
        
        # Отношение 95-го процентиля к 5-му процентилю
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        return abs(p95 / p5) if p5 != 0 else float('inf')
    
    def _calculate_overall_robustness(self, stress_results: Dict) -> Dict:
        """Рассчитывает общую оценку устойчивости стратегии"""
        
        robustness_scores = []
        
        # Анализируем каждую категорию стресс-тестов
        for category, results in stress_results.items():
            if category in ['volatility_stress', 'correlation_stress', 'regime_change_stress', 
                          'black_swan_stress', 'liquidity_stress']:
                
                # Собираем Sharpe ratios из всех сценариев
                sharpe_ratios = []
                for scenario_name, metrics in results.items():
                    if 'sharpe_ratio' in metrics:
                        sharpe_ratios.append(metrics['sharpe_ratio'])
                
                if sharpe_ratios:
                    # Консистентность = обратная к коэффициенту вариации
                    mean_sharpe = np.mean(sharpe_ratios)
                    std_sharpe = np.std(sharpe_ratios)
                    consistency = 1 / (std_sharpe / (abs(mean_sharpe) + 1e-8) + 1e-8)
                    
                    # Нормализуем и ограничиваем
                    category_score = max(0, min(1, consistency / 10))
                    robustness_scores.append(category_score)
        
        overall_robustness = {
            'overall_score': np.mean(robustness_scores) if robustness_scores else 0,
            'category_scores': dict(zip(
                ['volatility', 'correlation', 'regime_change', 'black_swan', 'liquidity'],
                robustness_scores
            )),
            'robustness_rating': self._get_robustness_rating(np.mean(robustness_scores) if robustness_scores else 0)
        }
        
        return overall_robustness
    
    def _get_robustness_rating(self, score: float) -> str:
        """Преобразует числовую оценку в текстовый рейтинг"""
        
        if score >= 0.8:
            return "ВЫСОКАЯ"
        elif score >= 0.6:
            return "УМЕРЕННАЯ"
        elif score >= 0.4:
            return "НИЗКАЯ"
        else:
            return "ОЧЕНЬ НИЗКАЯ"
    
    def comprehensive_validation(self, 
                               strategy_returns: np.ndarray,
                               benchmark_returns: Optional[np.ndarray] = None,
                               prices: Optional[np.ndarray] = None,
                               save_results: bool = True) -> Dict:
        """
        Комплексная валидация стратегии с использованием всех методов
        
        Args:
            strategy_returns: Доходности стратегии
            benchmark_returns: Доходности бенчмарка (опционально)
            prices: Исторические цены (опционально)
            save_results: Сохранить результаты
            
        Returns:
            Комплексные результаты валидации
        """
        
        self.logger.info("🔬 Запуск комплексной Monte Carlo валидации")
        self.logger.info(f"   📊 Данных: {len(strategy_returns)} наблюдений")
        
        validation_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_observations': len(strategy_returns),
                'period_start': None,  # Можно добавить если есть временные метки
                'period_end': None,
                'config': self.config.__dict__
            }
        }
        
        # 1. Permutation тесты для ключевых метрик
        self.logger.info("🎲 Permutation тесты...")
        
        metrics_to_test = {
            'mean_return': np.mean,
            'sharpe_ratio': lambda x: np.mean(x) / (np.std(x) + 1e-8),
            'hit_rate': lambda x: np.mean(x > 0),
            'max_drawdown': self._calculate_max_drawdown
        }
        
        permutation_results = {}
        for metric_name, metric_func in metrics_to_test.items():
            try:
                perm_result = self.permutation_test(
                    strategy_returns, benchmark_returns, metric_func
                )
                permutation_results[metric_name] = perm_result.__dict__
            except Exception as e:
                self.logger.error(f"Ошибка в permutation тесте для {metric_name}: {str(e)}")
                permutation_results[metric_name] = {'error': str(e)}
        
        validation_results['permutation_tests'] = permutation_results
        
        # 2. Bootstrap анализ
        self.logger.info("🔄 Bootstrap анализ...")
        
        try:
            bootstrap_results = self.bootstrap_analysis(strategy_returns)
            validation_results['bootstrap_analysis'] = bootstrap_results
        except Exception as e:
            self.logger.error(f"Ошибка в bootstrap анализе: {str(e)}")
            validation_results['bootstrap_analysis'] = {'error': str(e)}
        
        # 3. Stress тестирование
        self.logger.info("🌪️ Stress тестирование...")
        
        try:
            stress_results = self.stress_test_strategy(strategy_returns, prices)
            validation_results['stress_tests'] = stress_results
        except Exception as e:
            self.logger.error(f"Ошибка в stress тестировании: {str(e)}")
            validation_results['stress_tests'] = {'error': str(e)}
        
        # 4. Общая оценка и рекомендации
        overall_assessment = self._generate_overall_assessment(validation_results)
        validation_results['overall_assessment'] = overall_assessment
        
        # Сохраняем результаты
        if save_results:
            self._save_validation_results(validation_results)
        
        self.logger.info("✅ Комплексная валидация завершена")
        self._log_summary(validation_results)
        
        return validation_results
    
    def _generate_overall_assessment(self, results: Dict) -> Dict:
        """Генерирует общую оценку и рекомендации"""
        
        assessment = {
            'statistical_significance': {},
            'robustness': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Анализ статистической значимости
        perm_results = results.get('permutation_tests', {})
        significant_metrics = []
        
        for metric, result in perm_results.items():
            if isinstance(result, dict) and result.get('is_significant', False):
                significant_metrics.append(metric)
        
        assessment['statistical_significance'] = {
            'significant_metrics': significant_metrics,
            'significance_score': len(significant_metrics) / max(1, len(perm_results)),
            'overall_significant': len(significant_metrics) >= len(perm_results) * 0.5
        }
        
        # Анализ устойчивости
        stress_results = results.get('stress_tests', {})
        robustness_info = stress_results.get('overall_robustness', {})
        
        assessment['robustness'] = {
            'robustness_score': robustness_info.get('overall_score', 0),
            'robustness_rating': robustness_info.get('robustness_rating', 'НЕИЗВЕСТНО'),
            'weak_areas': self._identify_weak_areas(stress_results)
        }
        
        # Риск-анализ
        bootstrap_results = results.get('bootstrap_analysis', {})
        
        assessment['risk_assessment'] = {
            'max_drawdown_risk': self._assess_drawdown_risk(bootstrap_results),
            'return_stability': self._assess_return_stability(bootstrap_results),
            'tail_risk': self._assess_tail_risk(stress_results)
        }
        
        # Генерируем рекомендации
        recommendations = []
        
        if not assessment['statistical_significance']['overall_significant']:
            recommendations.append("🚨 КРИТИЧНО: Стратегия не демонстрирует статистически значимые результаты")
        
        if assessment['robustness']['robustness_score'] < 0.5:
            recommendations.append("⚠️ Низкая устойчивость к стресс-сценариям. Требуется улучшение стратегии")
        
        if assessment['risk_assessment']['max_drawdown_risk'] == 'HIGH':
            recommendations.append("📉 Высокий риск просадки. Рекомендуется улучшить риск-менеджмент")
        
        if assessment['risk_assessment']['tail_risk'] == 'HIGH':
            recommendations.append("🦅 Высокий tail risk. Стратегия уязвима к экстремальным событиям")
        
        if not recommendations:
            recommendations.append("✅ Стратегия демонстрирует хорошие показатели во всех тестах")
        
        assessment['recommendations'] = recommendations
        
        return assessment
    
    def _identify_weak_areas(self, stress_results: Dict) -> List[str]:
        """Определяет слабые места в stress тестах"""
        
        weak_areas = []
        
        # Анализируем каждую категорию стресс-тестов
        categories = ['volatility_stress', 'correlation_stress', 'regime_change_stress', 
                     'black_swan_stress', 'liquidity_stress']
        
        for category in categories:
            if category in stress_results:
                results = stress_results[category]
                
                # Анализируем Sharpe ratios в сценариях
                sharpe_ratios = []
                for scenario, metrics in results.items():
                    if isinstance(metrics, dict) and 'sharpe_ratio' in metrics:
                        sharpe_ratios.append(metrics['sharpe_ratio'])
                
                if sharpe_ratios:
                    min_sharpe = min(sharpe_ratios)
                    if min_sharpe < 0.5:  # Низкий Sharpe в каком-то сценарии
                        weak_areas.append(category.replace('_stress', ''))
        
        return weak_areas
    
    def _assess_drawdown_risk(self, bootstrap_results: Dict) -> str:
        """Оценивает риск просадки на основе bootstrap анализа"""
        
        if 'max_drawdown' in bootstrap_results:
            dd_info = bootstrap_results['max_drawdown']
            ci_upper = dd_info.get('confidence_interval', [0, 0])[1]
            
            if abs(ci_upper) > 0.25:  # Более 25% просадки в худшем случае
                return 'HIGH'
            elif abs(ci_upper) > 0.15:
                return 'MODERATE' 
            else:
                return 'LOW'
        
        return 'UNKNOWN'
    
    def _assess_return_stability(self, bootstrap_results: Dict) -> str:
        """Оценивает стабильность доходности"""
        
        if 'mean_return' in bootstrap_results:
            return_info = bootstrap_results['mean_return']
            standard_error = return_info.get('standard_error', 0)
            observed = return_info.get('observed', 0)
            
            # Коэффициент вариации
            cv = standard_error / (abs(observed) + 1e-8)
            
            if cv > 2.0:
                return 'LOW'
            elif cv > 1.0:
                return 'MODERATE'
            else:
                return 'HIGH'
        
        return 'UNKNOWN'
    
    def _assess_tail_risk(self, stress_results: Dict) -> str:
        """Оценивает tail risk на основе stress тестов"""
        
        if 'black_swan_stress' in stress_results:
            swan_results = stress_results['black_swan_stress']
            
            # Ищем худший сценарий
            worst_sharpe = float('inf')
            for scenario, metrics in swan_results.items():
                if isinstance(metrics, dict) and 'sharpe_ratio' in metrics:
                    worst_sharpe = min(worst_sharpe, metrics['sharpe_ratio'])
            
            if worst_sharpe < -1.0:  # Очень плохой Sharpe в экстремальных сценариях
                return 'HIGH'
            elif worst_sharpe < 0:
                return 'MODERATE'
            else:
                return 'LOW'
        
        return 'UNKNOWN'
    
    def _save_validation_results(self, results: Dict):
        """Сохраняет результаты валидации"""
        
        # Создаем директорию
        results_dir = Path("validation/monte_carlo_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем JSON
        json_file = results_dir / f"monte_carlo_validation_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Создаем отчет
        report_file = results_dir / f"MONTE_CARLO_REPORT_{timestamp}.md"
        report_content = self._create_validation_report(results)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"💾 Результаты сохранены:")
        self.logger.info(f"   📄 JSON: {json_file}")
        self.logger.info(f"   📖 Отчет: {report_file}")
    
    def _create_validation_report(self, results: Dict) -> str:
        """Создает отчет валидации в формате Markdown"""
        
        timestamp = results['metadata']['timestamp']
        n_obs = results['metadata']['n_observations']
        assessment = results.get('overall_assessment', {})
        
        report = f"""# Monte Carlo Validation Report

**Дата анализа:** {timestamp}  
**Количество наблюдений:** {n_obs}  

## 📊 Общая оценка

"""
        
        # Статистическая значимость
        sig_info = assessment.get('statistical_significance', {})
        if sig_info.get('overall_significant', False):
            report += "✅ **Статистическая значимость:** ПОДТВЕРЖДЕНА\n"
        else:
            report += "❌ **Статистическая значимость:** НЕ ПОДТВЕРЖДЕНА\n"
        
        report += f"- Значимых метрик: {len(sig_info.get('significant_metrics', []))}\n"
        report += f"- Балл значимости: {sig_info.get('significance_score', 0):.2f}\n\n"
        
        # Устойчивость
        robust_info = assessment.get('robustness', {})
        report += f"🛡️ **Устойчивость:** {robust_info.get('robustness_rating', 'НЕИЗВЕСТНО')}\n"
        report += f"- Балл устойчивости: {robust_info.get('robustness_score', 0):.2f}\n"
        
        weak_areas = robust_info.get('weak_areas', [])
        if weak_areas:
            report += f"- Слабые области: {', '.join(weak_areas)}\n"
        report += "\n"
        
        # Риск-анализ
        risk_info = assessment.get('risk_assessment', {})
        report += "⚠️ **Риск-анализ:**\n"
        report += f"- Риск просадки: {risk_info.get('max_drawdown_risk', 'НЕИЗВЕСТНО')}\n"
        report += f"- Стабильность доходности: {risk_info.get('return_stability', 'НЕИЗВЕСТНО')}\n"
        report += f"- Tail risk: {risk_info.get('tail_risk', 'НЕИЗВЕСТНО')}\n\n"
        
        # Рекомендации
        recommendations = assessment.get('recommendations', [])
        report += "## 🎯 Рекомендации\n\n"
        
        for rec in recommendations:
            report += f"{rec}\n\n"
        
        # Детальные результаты
        report += "## 📈 Детальные результаты\n\n"
        
        # Permutation тесты
        perm_results = results.get('permutation_tests', {})
        if perm_results:
            report += "### Permutation тесты\n\n"
            report += "| Метрика | Наблюдаемое | p-value | Z-score | Значимость |\n"
            report += "|---------|-------------|---------|---------|------------|\n"
            
            for metric, result in perm_results.items():
                if isinstance(result, dict) and 'observed_metric' in result:
                    obs = result.get('observed_metric', 0)
                    p_val = result.get('p_value', 1)
                    z_score = result.get('z_score', 0)
                    sig = '✅' if result.get('is_significant', False) else '❌'
                    
                    report += f"| {metric} | {obs:.4f} | {p_val:.4f} | {z_score:.2f} | {sig} |\n"
            
            report += "\n"
        
        # Stress тесты (сводка)
        stress_results = results.get('stress_tests', {})
        if 'overall_robustness' in stress_results:
            robustness = stress_results['overall_robustness']
            report += "### Stress тестирование\n\n"
            report += f"**Общий балл устойчивости:** {robustness.get('overall_score', 0):.2f}\n\n"
            
            category_scores = robustness.get('category_scores', {})
            if category_scores:
                report += "**Баллы по категориям:**\n"
                for category, score in category_scores.items():
                    report += f"- {category}: {score:.2f}\n"
                report += "\n"
        
        report += "---\n*Отчет сгенерирован системой Monte Carlo валидации*"
        
        return report
    
    def _log_summary(self, results: Dict):
        """Выводит краткую сводку результатов"""
        
        assessment = results.get('overall_assessment', {})
        
        self.logger.info("📋 Сводка результатов:")
        
        sig_info = assessment.get('statistical_significance', {})
        self.logger.info(f"   📊 Статистическая значимость: {sig_info.get('significance_score', 0):.2f}")
        
        robust_info = assessment.get('robustness', {})
        self.logger.info(f"   🛡️ Устойчивость: {robust_info.get('robustness_rating', 'НЕИЗВЕСТНО')}")
        
        risk_info = assessment.get('risk_assessment', {})
        self.logger.info(f"   ⚠️ Риск просадки: {risk_info.get('max_drawdown_risk', 'НЕИЗВЕСТНО')}")
        
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            self.logger.info("   💡 Ключевые рекомендации:")
            for rec in recommendations[:2]:  # Показываем только первые 2
                self.logger.info(f"      {rec}")


def main():
    """Пример использования Monte Carlo тестера"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Monte Carlo валидация торговой стратегии")
    parser.add_argument('--returns-file', required=True, help='Файл с доходностями (CSV)')
    parser.add_argument('--benchmark-file', help='Файл с доходностями бенчмарка (CSV)')
    parser.add_argument('--n-permutations', type=int, default=1000, help='Количество пермутаций')
    parser.add_argument('--n-bootstrap', type=int, default=500, help='Количество bootstrap выборок')
    parser.add_argument('--confidence', type=float, default=0.95, help='Уровень доверия')
    
    args = parser.parse_args()
    
    # Создаем конфигурацию
    config = MonteCarloConfig(
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence
    )
    
    # Создаем тестер
    tester = MonteCarloTester(config)
    
    # Загружаем данные
    returns_df = pd.read_csv(args.returns_file)
    strategy_returns = returns_df['returns'].values
    
    benchmark_returns = None
    if args.benchmark_file:
        benchmark_df = pd.read_csv(args.benchmark_file)
        benchmark_returns = benchmark_df['returns'].values
    
    print("🧪 Запуск Monte Carlo валидации...")
    print(f"📊 Доходностей стратегии: {len(strategy_returns)}")
    
    if benchmark_returns is not None:
        print(f"📈 Доходностей бенчмарка: {len(benchmark_returns)}")
    
    # Запускаем комплексную валидацию
    results = tester.comprehensive_validation(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        save_results=True
    )
    
    print("✅ Валидация завершена!")
    
    # Выводим основные результаты
    assessment = results.get('overall_assessment', {})
    
    sig_info = assessment.get('statistical_significance', {})
    print(f"📊 Статистическая значимость: {sig_info.get('significance_score', 0):.1%}")
    
    robust_info = assessment.get('robustness', {})
    print(f"🛡️ Устойчивость: {robust_info.get('robustness_rating', 'НЕИЗВЕСТНО')}")
    
    print("\n💡 Рекомендации:")
    for rec in assessment.get('recommendations', []):
        print(f"   {rec}")


if __name__ == "__main__":
    main()