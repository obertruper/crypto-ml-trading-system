"""
Модуль для расчета размера позиций
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger


@dataclass
class PositionInfo:
    """Информация о позиции"""
    size: float  # Размер позиции в базовой валюте
    units: float  # Количество единиц актива
    risk_amount: float  # Сумма риска в валюте
    leverage: float  # Используемое плечо
    margin_required: float  # Требуемая маржа
    position_value: float  # Полная стоимость позиции


class PositionSizer:
    """Базовый класс для расчета размера позиций"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: конфигурация с параметрами
        """
        self.config = config
        self.logger = get_logger("PositionSizer")
        
        # Основные параметры
        self.max_risk_per_trade = config['risk_management']['max_risk_per_trade']
        self.max_portfolio_risk = config['risk_management']['max_portfolio_risk']
        self.max_position_size = config['risk_management']['max_position_size']
        self.max_leverage = config['risk_management'].get('max_leverage', 10)
        self.min_position_size = config['risk_management'].get('min_position_size', 0.001)
        
        # Комиссии
        self.maker_fee = config['risk_management'].get('maker_fee', 0.0002)
        self.taker_fee = config['risk_management'].get('taker_fee', 0.0005)
        
    def calculate_position_size(self,
                              capital: float,
                              entry_price: float,
                              stop_loss: float,
                              confidence: float = 1.0,
                              existing_positions: Optional[Dict] = None) -> PositionInfo:
        """
        Расчет размера позиции
        
        Args:
            capital: доступный капитал
            entry_price: цена входа
            stop_loss: уровень стоп-лосса
            confidence: уверенность в сигнале (0-1)
            existing_positions: текущие открытые позиции
            
        Returns:
            Информация о позиции
        """
        # Расчет риска на сделку
        risk_per_unit = abs(entry_price - stop_loss)
        risk_percentage = risk_per_unit / entry_price
        
        # Базовый размер позиции по риску
        risk_amount = capital * self.max_risk_per_trade * confidence
        position_units = risk_amount / risk_per_unit
        position_value = position_units * entry_price
        
        # Применение ограничений
        position_value = self._apply_constraints(
            position_value, capital, existing_positions
        )
        
        # Пересчет единиц после ограничений
        position_units = position_value / entry_price
        actual_risk = position_units * risk_per_unit
        
        # Расчет плеча и маржи
        leverage = position_value / capital
        margin_required = position_value / self.max_leverage
        
        return PositionInfo(
            size=position_value,
            units=position_units,
            risk_amount=actual_risk,
            leverage=leverage,
            margin_required=margin_required,
            position_value=position_value
        )
    
    def _apply_constraints(self,
                         position_value: float,
                         capital: float,
                         existing_positions: Optional[Dict] = None) -> float:
        """Применение ограничений к размеру позиции"""
        
        # Ограничение по максимальному размеру позиции
        max_position = capital * self.max_position_size
        position_value = min(position_value, max_position)
        
        # Ограничение по общему риску портфеля
        if existing_positions:
            total_risk = sum(pos.get('risk', 0) for pos in existing_positions.values())
            available_risk = capital * self.max_portfolio_risk - total_risk
            if available_risk > 0:
                position_value = min(position_value, available_risk / self.max_risk_per_trade)
        
        # Ограничение по плечу
        max_with_leverage = capital * self.max_leverage
        position_value = min(position_value, max_with_leverage)
        
        # Минимальный размер позиции
        min_position = capital * self.min_position_size
        if position_value < min_position:
            position_value = 0  # Не открываем слишком маленькие позиции
        
        return position_value


class KellyPositionSizer(PositionSizer):
    """Расчет размера позиции по критерию Келли"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.kelly_fraction = config['risk_management'].get('kelly_fraction', 0.25)
        self.lookback_periods = config['risk_management'].get('kelly_lookback', 100)
        
    def calculate_kelly_size(self,
                           win_probability: float,
                           avg_win: float,
                           avg_loss: float,
                           capital: float) -> float:
        """
        Расчет оптимального размера по формуле Келли
        
        Args:
            win_probability: вероятность выигрыша
            avg_win: средний выигрыш
            avg_loss: средний проигрыш
            capital: капитал
            
        Returns:
            Оптимальный размер позиции
        """
        if avg_loss == 0:
            return 0
        
        # Формула Келли: f = (p * b - q) / b
        # где p - вероятность выигрыша, q - вероятность проигрыша, b - отношение выигрыша к проигрышу
        loss_probability = 1 - win_probability
        win_loss_ratio = avg_win / abs(avg_loss)
        
        kelly_percentage = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
        
        # Применение дробного Келли для снижения риска
        kelly_percentage = kelly_percentage * self.kelly_fraction
        
        # Ограничения
        kelly_percentage = max(0, min(kelly_percentage, self.max_position_size))
        
        return capital * kelly_percentage
    
    def calculate_from_history(self,
                             trade_history: pd.DataFrame,
                             capital: float) -> float:
        """
        Расчет размера на основе истории торговли
        
        Args:
            trade_history: история сделок
            capital: текущий капитал
            
        Returns:
            Размер позиции
        """
        if len(trade_history) < self.lookback_periods:
            # Недостаточно истории - используем минимальный размер
            return capital * self.min_position_size
        
        # Анализ последних N сделок
        recent_trades = trade_history.tail(self.lookback_periods)
        
        # Расчет статистики
        wins = recent_trades[recent_trades['pnl'] > 0]
        losses = recent_trades[recent_trades['pnl'] <= 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return capital * self.min_position_size
        
        win_probability = len(wins) / len(recent_trades)
        avg_win = wins['pnl'].mean()
        avg_loss = abs(losses['pnl'].mean())
        
        return self.calculate_kelly_size(win_probability, avg_win, avg_loss, capital)


class VolatilityBasedSizer(PositionSizer):
    """Расчет размера позиции на основе волатильности"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.target_volatility = config['risk_management'].get('target_volatility', 0.02)
        self.volatility_lookback = config['risk_management'].get('volatility_lookback', 20)
        
    def calculate_position_size(self,
                              capital: float,
                              entry_price: float,
                              stop_loss: float,
                              volatility: float,
                              confidence: float = 1.0,
                              existing_positions: Optional[Dict] = None) -> PositionInfo:
        """
        Расчет с учетом волатильности
        
        Args:
            capital: капитал
            entry_price: цена входа
            stop_loss: стоп-лосс
            volatility: текущая волатильность (ATR или std)
            confidence: уверенность
            existing_positions: открытые позиции
            
        Returns:
            Информация о позиции
        """
        # Нормализация размера по волатильности
        volatility_ratio = self.target_volatility / (volatility / entry_price)
        volatility_ratio = np.clip(volatility_ratio, 0.5, 2.0)
        
        # Базовый расчет
        base_position = super().calculate_position_size(
            capital, entry_price, stop_loss, confidence, existing_positions
        )
        
        # Корректировка по волатильности
        adjusted_size = base_position.size * volatility_ratio
        adjusted_units = adjusted_size / entry_price
        
        # Пересчет параметров
        return PositionInfo(
            size=adjusted_size,
            units=adjusted_units,
            risk_amount=adjusted_units * abs(entry_price - stop_loss),
            leverage=adjusted_size / capital,
            margin_required=adjusted_size / self.max_leverage,
            position_value=adjusted_size
        )


class RiskParityPositionSizer(PositionSizer):
    """Расчет размера для равновесия риска в портфеле"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.rebalance_threshold = config['risk_management'].get('rebalance_threshold', 0.1)
        
    def calculate_portfolio_weights(self,
                                  symbols: List[str],
                                  returns_data: pd.DataFrame,
                                  current_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Расчет весов для портфеля с равным риском
        
        Args:
            symbols: список символов
            returns_data: данные о доходностях
            current_weights: текущие веса
            
        Returns:
            Оптимальные веса
        """
        # Расчет ковариационной матрицы
        returns = returns_data[symbols].pct_change().dropna()
        cov_matrix = returns.cov()
        
        # Инверсная волатильность как начальное приближение
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol_weights = 1 / volatilities
        inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
        
        # Итеративная оптимизация для равного вклада в риск
        weights = inv_vol_weights
        for _ in range(100):
            # Вклад каждого актива в риск портфеля
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights / portfolio_vol
            contrib_to_risk = weights * marginal_contrib
            
            # Обновление весов
            target_risk = contrib_to_risk.sum() / len(weights)
            weights = weights * target_risk / contrib_to_risk
            weights = weights / weights.sum()
        
        # Проверка необходимости ребалансировки
        weights_dict = dict(zip(symbols, weights))
        
        if current_weights:
            # Расчет отклонения от текущих весов
            max_deviation = max(abs(weights_dict.get(s, 0) - current_weights.get(s, 0)) 
                              for s in symbols)
            
            if max_deviation < self.rebalance_threshold:
                # Сохраняем текущие веса
                return current_weights
        
        return weights_dict


class DynamicPositionSizer(PositionSizer):
    """Динамический расчет размера с учетом рыночных условий"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.regime_lookback = config['risk_management'].get('regime_lookback', 50)
        self.regime_thresholds = {
            'trending': 0.7,
            'ranging': 0.3,
            'volatile': 1.5
        }
        
    def identify_market_regime(self, 
                             price_data: pd.DataFrame,
                             volume_data: Optional[pd.DataFrame] = None) -> str:
        """
        Определение рыночного режима
        
        Args:
            price_data: данные о ценах
            volume_data: данные об объемах
            
        Returns:
            Тип режима: 'trending', 'ranging', 'volatile'
        """
        # Расчет индикаторов режима
        returns = price_data.pct_change().dropna()
        
        # Тренд: ADX или линейная регрессия
        trend_strength = self._calculate_trend_strength(price_data)
        
        # Волатильность
        volatility = returns.rolling(self.regime_lookback).std().iloc[-1]
        avg_volatility = returns.std()
        volatility_ratio = volatility / avg_volatility
        
        # Определение режима
        if trend_strength > self.regime_thresholds['trending']:
            return 'trending'
        elif volatility_ratio > self.regime_thresholds['volatile']:
            return 'volatile'
        else:
            return 'ranging'
    
    def adjust_size_by_regime(self,
                            base_size: float,
                            regime: str,
                            trend_direction: Optional[str] = None) -> float:
        """
        Корректировка размера по режиму рынка
        
        Args:
            base_size: базовый размер
            regime: рыночный режим
            trend_direction: направление тренда
            
        Returns:
            Скорректированный размер
        """
        adjustments = {
            'trending': 1.2,  # Увеличиваем в тренде
            'ranging': 0.8,   # Уменьшаем в боковике
            'volatile': 0.6   # Сильно уменьшаем в волатильности
        }
        
        adjusted_size = base_size * adjustments.get(regime, 1.0)
        
        # Дополнительная корректировка для контртренда
        if regime == 'trending' and trend_direction == 'counter':
            adjusted_size *= 0.5
        
        return adjusted_size
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """Расчет силы тренда"""
        # Простая линейная регрессия
        x = np.arange(len(price_data))
        y = price_data.values
        
        # Нормализация
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        
        # Коэффициент корреляции как мера силы тренда
        correlation = np.corrcoef(x, y)[0, 1]
        
        return abs(correlation)


class OptimalFPositionSizer(PositionSizer):
    """Расчет оптимального f для максимизации роста капитала"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.f_scale = config['risk_management'].get('optimal_f_scale', 0.5)
        self.monte_carlo_runs = config['risk_management'].get('monte_carlo_runs', 1000)
        
    def calculate_optimal_f(self,
                          trade_results: List[float],
                          initial_capital: float = 10000) -> float:
        """
        Расчет оптимального f методом перебора
        
        Args:
            trade_results: список результатов сделок в валюте
            initial_capital: начальный капитал для симуляции
            
        Returns:
            Оптимальное значение f
        """
        if not trade_results or all(r >= 0 for r in trade_results):
            return self.min_position_size
        
        # Нормализация результатов
        worst_loss = min(trade_results)
        if worst_loss >= 0:
            return self.min_position_size
        
        # Поиск оптимального f
        f_values = np.linspace(0.01, 0.99, 100)
        terminal_wealths = []
        
        for f in f_values:
            wealth = initial_capital
            
            for result in trade_results:
                # TWR = Product((1 + f * (-result/worst_loss)))
                holding = 1 + f * (-result / worst_loss)
                wealth *= holding
                
                if wealth <= 0:
                    wealth = 0
                    break
            
            terminal_wealths.append(wealth)
        
        # Находим f с максимальным конечным капиталом
        optimal_idx = np.argmax(terminal_wealths)
        optimal_f = f_values[optimal_idx]
        
        # Применяем масштабирование для безопасности
        return optimal_f * self.f_scale
    
    def monte_carlo_position_size(self,
                                capital: float,
                                trade_stats: Dict,
                                num_simulations: int = None) -> float:
        """
        Расчет размера через Монте-Карло симуляцию
        
        Args:
            capital: текущий капитал
            trade_stats: статистика торговли
            num_simulations: количество симуляций
            
        Returns:
            Оптимальный размер позиции
        """
        if num_simulations is None:
            num_simulations = self.monte_carlo_runs
        
        # Параметры распределения из статистики
        avg_win = trade_stats.get('avg_win', 0.02)
        avg_loss = trade_stats.get('avg_loss', -0.01)
        win_rate = trade_stats.get('win_rate', 0.5)
        
        best_size = self.min_position_size
        best_sharpe = -np.inf
        
        # Тестируем разные размеры позиций
        for position_size in np.linspace(0.01, self.max_position_size, 20):
            returns = []
            
            # Симуляция торговли
            for _ in range(num_simulations):
                equity = capital
                sim_returns = []
                
                for _ in range(100):  # 100 сделок в симуляции
                    if np.random.random() < win_rate:
                        profit = avg_win * position_size
                    else:
                        profit = avg_loss * position_size
                    
                    equity *= (1 + profit)
                    sim_returns.append(profit)
                
                returns.extend(sim_returns)
            
            # Оценка качества через Sharpe ratio
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_size = position_size
        
        return capital * best_size


def create_position_sizer(sizer_type: str, config: Dict) -> PositionSizer:
    """
    Фабрика для создания position sizer'ов
    
    Args:
        sizer_type: тип sizer'а
        config: конфигурация
        
    Returns:
        Экземпляр PositionSizer
    """
    sizers = {
        'fixed': PositionSizer,
        'kelly': KellyPositionSizer,
        'volatility': VolatilityBasedSizer,
        'risk_parity': RiskParityPositionSizer,
        'dynamic': DynamicPositionSizer,
        'optimal_f': OptimalFPositionSizer
    }
    
    if sizer_type not in sizers:
        raise ValueError(f"Unknown position sizer type: {sizer_type}")
    
    return sizers[sizer_type](config)