"""
Система управления рисками для криптотрейдинга
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger

@dataclass
class Position:
    """Класс для представления торговой позиции"""
    symbol: str
    side: str  # 'long' или 'short'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profits: List[float]
    entry_time: datetime
    confidence: float
    atr_value: float
    
    @property
    def position_value(self) -> float:
        return self.entry_price * self.quantity
    
    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.quantity

@dataclass
class RiskMetrics:
    """Метрики риска портфеля"""
    total_exposure: float
    risk_per_trade: float
    portfolio_risk: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float

class RiskManager:
    """Основной класс управления рисками"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config['risk_management']
        self.bybit_config = config.get('bybit', {})
        self.logger = get_logger("RiskManager")
        
        # Параметры риска
        self.max_risk_per_trade = self.risk_config['position_sizing'].get('max_risk_per_trade', 0.5) / 100
        self.max_portfolio_risk = self.risk_config.get('max_portfolio_risk_pct', 10) / 100
        self.max_position_size = self.risk_config['position_sizing']['max_position_pct'] / 100
        self.max_concurrent_positions = self.risk_config['max_concurrent_positions']
        
        # Текущие позиции и история
        self.active_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.equity_curve: List[float] = []
        self.initial_capital = config['backtesting']['initial_capital']
        self.current_capital = self.initial_capital
        
        # Корректировки для разных типов активов
        self.volatility_adjustments = self.risk_config['volatility_adjustment']
        
        # Матрица корреляций для управления риском портфеля
        self.correlation_matrix = None
        
    def calculate_position_size(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_loss: float, 
                              atr_value: float,
                              confidence: float = 1.0,
                              current_volatility: float = 2.5) -> float:
        """Динамический расчет размера позиции с Kelly Criterion"""
        
        # 1. Базовый размер по Kelly Criterion
        win_rate = confidence  # Используем уверенность модели как винрейт
        avg_win = self._get_dynamic_tp_target(current_volatility)
        avg_loss = self._get_dynamic_sl_target(current_volatility)
        
        kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # 2. Коррекция на волатильность
        volatility_adjustment = self._get_volatility_adjustment(current_volatility)
        
        # 3. Коррекция на корреляции
        correlation_adjustment = self._get_correlation_adjustment(symbol)
        
        # 4. Коррекция на тип актива
        risk_multiplier = self._get_risk_multiplier(symbol)
        
        # 5. Финальный расчет
        max_risk_amount = self.current_capital * self.max_risk_per_trade
        kelly_amount = self.current_capital * kelly_fraction
        
        # Берем минимум для безопасности
        risk_amount = min(max_risk_amount, kelly_amount)
        
        # Применяем коррекции
        final_risk_amount = risk_amount * volatility_adjustment * correlation_adjustment * risk_multiplier
        
        # Конвертируем в размер позиции
        stop_loss_pct = self._get_dynamic_sl_target(current_volatility)
        position_size_usd = final_risk_amount / (stop_loss_pct / 100)
        position_size = position_size_usd / entry_price
        
        # Ограничение по максимальному размеру позиции
        max_position_value = self.current_capital * self.max_position_size
        max_position_size = max_position_value / entry_price
        position_size = min(position_size, max_position_size)
        
        self.logger.debug(
            f"Расчет позиции {symbol}: "
            f"Kelly={kelly_fraction:.3f}, "
            f"Vol_adj={volatility_adjustment:.3f}, "
            f"Corr_adj={correlation_adjustment:.3f}, "
            f"финальный_размер={position_size:.4f}"
        )
        
        return position_size
    
    def _get_risk_multiplier(self, symbol: str) -> float:
        """Получение множителя риска для разных типов активов"""
        major_coins = self.volatility_adjustments['major_coins']
        
        if symbol in major_coins:
            return self.volatility_adjustments['major_risk_multiplier']
        elif any(meme in symbol for meme in ['DOGE', 'SHIB', 'PEPE', 'WIF', 'POPCAT', 'PNUT']):
            return self.volatility_adjustments['meme_coin_risk_multiplier']
        else:
            return self.volatility_adjustments['altcoin_risk_multiplier']
    
    def validate_new_position(self, 
                            symbol: str, 
                            position_size: float, 
                            entry_price: float) -> Tuple[bool, str]:
        """Проверка возможности открытия новой позиции"""
        
        # Проверка максимального количества позиций
        if len(self.active_positions) >= self.max_concurrent_positions:
            return False, f"Достигнуто максимальное количество позиций ({self.max_concurrent_positions})"
        
        # Проверка существующей позиции по символу
        if symbol in self.active_positions:
            return False, f"Позиция по {symbol} уже открыта"
        
        # Проверка достаточности капитала
        position_value = position_size * entry_price
        if position_value > self.current_capital * 0.95:  # 95% от капитала
            return False, f"Недостаточно капитала для позиции (требуется {position_value:.2f})"
        
        # Проверка общего риска портфеля
        current_portfolio_risk = self._calculate_portfolio_risk()
        new_position_risk = position_size * entry_price * 0.02  # Примерный риск 2%
        
        if (current_portfolio_risk + new_position_risk) > self.current_capital * self.max_portfolio_risk:
            return False, f"Превышение максимального риска портфеля"
        
        return True, "Позиция одобрена"
    
    def open_position(self, 
                     symbol: str, 
                     side: str, 
                     entry_price: float, 
                     stop_loss: float, 
                     take_profits: List[float],
                     confidence: float,
                     atr_value: float) -> Optional[Position]:
        """Открытие новой позиции"""
        
        # Расчет размера позиции
        position_size = self.calculate_position_size(
            symbol, entry_price, stop_loss, atr_value, confidence
        )
        
        # Валидация позиции
        is_valid, message = self.validate_new_position(symbol, position_size, entry_price)
        
        if not is_valid:
            self.logger.warning(f"Отклонена позиция {symbol}: {message}")
            return None
        
        # Создание позиции
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=position_size,
            stop_loss=stop_loss,
            take_profits=take_profits.copy(),
            entry_time=datetime.now(),
            confidence=confidence,
            atr_value=atr_value
        )
        
        self.active_positions[symbol] = position
        
        self.logger.log_trade(
            symbol=symbol,
            action=f"OPEN_{side.upper()}",
            price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profits[0] if take_profits else 0,
            confidence=confidence
        )
        
        return position
    
    def update_position(self, symbol: str, current_price: float) -> List[Dict]:
        """Обновление позиции и проверка условий закрытия"""
        
        if symbol not in self.active_positions:
            return []
        
        position = self.active_positions[symbol]
        actions = []
        
        # Проверка stop loss
        if self._check_stop_loss(position, current_price):
            actions.append({
                'action': 'close_position',
                'reason': 'stop_loss',
                'price': current_price,
                'quantity': position.quantity
            })
            return actions
        
        # Проверка take profit уровней
        tp_actions = self._check_take_profits(position, current_price)
        actions.extend(tp_actions)
        
        # Обновление trailing stop
        self._update_trailing_stop(position, current_price)
        
        return actions
    
    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Проверка срабатывания stop loss"""
        if position.side == 'long':
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss
    
    def _check_take_profits(self, position: Position, current_price: float) -> List[Dict]:
        """Проверка срабатывания take profit уровней"""
        actions = []
        
        if not position.take_profits:
            return actions
        
        # Конфигурация частичных закрытий из конфига
        partial_sizes = self.risk_config.get('partial_close_sizes', [40, 40, 20])
        tp_percentages = [size / 100 for size in partial_sizes]
        
        for i, tp_price in enumerate(position.take_profits[:]):
            if position.side == 'long' and current_price >= tp_price:
                # Частичное закрытие для long позиции
                close_percentage = tp_percentages[min(i, len(tp_percentages)-1)]
                close_quantity = position.quantity * close_percentage
                
                actions.append({
                    'action': 'partial_close',
                    'reason': f'take_profit_{i+1}',
                    'price': current_price,
                    'quantity': close_quantity,
                    'percentage': close_percentage
                })
                
                # Удаляем сработавший TP
                position.take_profits.remove(tp_price)
                position.quantity -= close_quantity
                
                # Обновляем stop loss к breakeven после первого TP
                if i == 0:
                    position.stop_loss = position.entry_price
                    self.logger.info(f"Stop loss перемещен к breakeven для {position.symbol}")
                
            elif position.side == 'short' and current_price <= tp_price:
                # Аналогично для short позиций
                close_percentage = tp_percentages[min(i, len(tp_percentages)-1)]
                close_quantity = position.quantity * close_percentage
                
                actions.append({
                    'action': 'partial_close',
                    'reason': f'take_profit_{i+1}',
                    'price': current_price,
                    'quantity': close_quantity,
                    'percentage': close_percentage
                })
                
                position.take_profits.remove(tp_price)
                position.quantity -= close_quantity
                
                if i == 0:
                    position.stop_loss = position.entry_price
        
        return actions
    
    def _update_trailing_stop(self, position: Position, current_price: float):
        """Обновление trailing stop loss с учетом волатильности"""
        if position.side == 'long':
            # Trailing stop для long позиций
            unrealized_profit_pct = (current_price - position.entry_price) / position.entry_price * 100
            
            # Динамический trailing stop на основе ATR
            trailing_distance = position.atr_value * 2  # 2 ATR
            trailing_pct = trailing_distance / current_price
            
            if unrealized_profit_pct > 1.5:  # Если прибыль > 1.5%
                new_stop = current_price * (1 - trailing_pct)
                if new_stop > position.stop_loss:
                    old_stop = position.stop_loss
                    position.stop_loss = new_stop
                    self.logger.debug(f"Trailing stop обновлен для {position.symbol}: {old_stop:.4f} -> {new_stop:.4f}")
        
        else:
            # Trailing stop для short позиций
            unrealized_profit_pct = (position.entry_price - current_price) / position.entry_price
            
            if unrealized_profit_pct > 0.02:
                new_stop = current_price * 1.01  # Trailing на 1%
                if new_stop < position.stop_loss:
                    old_stop = position.stop_loss
                    position.stop_loss = new_stop
                    self.logger.debug(f"Trailing stop обновлен для {position.symbol}: {old_stop:.4f} -> {new_stop:.4f}")
    
    def close_position(self, symbol: str, close_price: float, reason: str = "manual"):
        """Полное закрытие позиции"""
        if symbol not in self.active_positions:
            self.logger.warning(f"Попытка закрыть несуществующую позицию {symbol}")
            return
        
        position = self.active_positions[symbol]
        
        # Расчет P&L
        if position.side == 'long':
            pnl = (close_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - close_price) * position.quantity
        
        # Обновление капитала
        self.current_capital += pnl
        self.equity_curve.append(self.current_capital)
        
        # Логирование закрытия
        self.logger.info(
            f"Закрыта позиция {symbol}: "
            f"Entry={position.entry_price:.4f}, "
            f"Exit={close_price:.4f}, "
            f"PnL={pnl:.2f}, "
            f"Reason={reason}"
        )
        
        # Перемещение в закрытые позиции
        self.closed_positions.append(position)
        del self.active_positions[symbol]
    
    def _calculate_portfolio_risk(self) -> float:
        """Расчет текущего риска портфеля"""
        total_risk = 0
        
        for position in self.active_positions.values():
            position_risk = position.risk_amount
            total_risk += position_risk
        
        return total_risk
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Получение метрик риска портфеля"""
        
        if not self.closed_positions and not self.equity_curve:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Общая экспозиция
        total_exposure = sum(pos.position_value for pos in self.active_positions.values())
        
        # Риск на сделку
        current_portfolio_risk = self._calculate_portfolio_risk()
        
        # Максимальная просадка
        max_drawdown = self._calculate_max_drawdown()
        
        # Коэффициент Шарпа
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Win rate
        win_rate = self._calculate_win_rate()
        
        # Profit factor
        profit_factor = self._calculate_profit_factor()
        
        return RiskMetrics(
            total_exposure=total_exposure,
            risk_per_trade=self.max_risk_per_trade * 100,
            portfolio_risk=current_portfolio_risk,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Расчет максимальной просадки"""
        if len(self.equity_curve) < 2:
            return 0
        
        equity_series = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - peak) / peak
        
        return float(np.min(drawdown))
    
    def _calculate_sharpe_ratio(self) -> float:
        """Расчет коэффициента Шарпа"""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))  # Аннуализированный Шарп
    
    def _calculate_win_rate(self) -> float:
        """Расчет win rate"""
        if not self.closed_positions:
            return 0
        
        winning_trades = 0
        for position in self.closed_positions:
            # Примерный расчет - в реальности нужно сохранять PnL
            if hasattr(position, 'pnl') and position.pnl > 0:
                winning_trades += 1
        
        return winning_trades / len(self.closed_positions)
    
    def _calculate_profit_factor(self) -> float:
        """Расчет profit factor"""
        if not self.closed_positions:
            return 0
        
        gross_profit = 0
        gross_loss = 0
        
        for position in self.closed_positions:
            if hasattr(position, 'pnl'):
                if position.pnl > 0:
                    gross_profit += position.pnl
                else:
                    gross_loss += abs(position.pnl)
        
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    def _calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly Criterion: f* = (bp - q) / b"""
        if avg_loss <= 0:
            return 0.0
            
        b = avg_win / avg_loss  # Соотношение выигрыша к проигрышу
        p = win_rate  # Вероятность выигрыша
        q = 1 - p     # Вероятность проигрыша
        
        kelly = (b * p - q) / b
        
        # Ограничиваем Kelly до разумных пределов
        return max(0.0, min(kelly, 0.25))  # Максимум 25% капитала
    
    def _get_dynamic_tp_target(self, volatility: float) -> float:
        """Динамические TP цели на основе волатильности"""
        vol_config = self.risk_config['volatility_adjustment']
        
        if volatility >= vol_config['high_vol_threshold']:
            return np.mean(vol_config['high_vol_multipliers'])
        elif volatility <= vol_config['low_vol_threshold']:
            return np.mean(vol_config['low_vol_multipliers'])
        else:
            return np.mean(self.risk_config['take_profit_targets'])
    
    def _get_dynamic_sl_target(self, volatility: float) -> float:
        """Динамический SL на основе волатильности"""
        base_sl = self.risk_config['stop_loss_pct']
        
        # Увеличиваем SL пропорционально волатильности
        if volatility >= 3.0:
            return base_sl * 1.25  # +25% в высокой волатильности
        elif volatility <= 1.5:
            return base_sl * 0.9   # -10% в низкой волатильности
        else:
            return base_sl
    
    def _get_volatility_adjustment(self, volatility: float) -> float:
        """Коррекция размера позиции на волатильность"""
        # В высокой волатильности уменьшаем размер позиции
        if volatility >= 4.0:
            return 0.5  # Уменьшаем на 50%
        elif volatility >= 3.0:
            return 0.75  # Уменьшаем на 25%
        elif volatility <= 1.0:
            return 1.2   # Увеличиваем на 20%
        else:
            return 1.0
    
    def _get_correlation_adjustment(self, symbol: str) -> float:
        """Коррекция на основе корреляций с открытыми позициями"""
        if not self.active_positions or not self.config.get('position_sizing', {}).get('correlation_adjustment', False):
            return 1.0
        
        total_correlation_risk = 0.0
        
        for open_symbol, position in self.active_positions.items():
            if open_symbol != symbol:
                correlation = self._get_correlation(symbol, open_symbol)
                position_weight = position.position_value / self.current_capital
                correlation_risk = abs(correlation) * position_weight
                total_correlation_risk += correlation_risk
        
        # Уменьшаем размер позиции при высоких корреляциях
        adjustment = max(0.3, 1.0 - total_correlation_risk * 2)
        
        return adjustment
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Получение корреляции между символами"""
        if self.correlation_matrix is None:
            return 0.0  # По умолчанию, если нет данных
            
        try:
            return self.correlation_matrix.loc[symbol1, symbol2]
        except (KeyError, AttributeError):
            return 0.0
    
    def update_correlation_matrix(self, market_data: pd.DataFrame):
        """Обновление матрицы корреляций"""
        self.logger.info("Обновление матрицы корреляций...")
        
        # Pivot данные для расчета корреляций
        pivot_data = market_data.pivot(
            index='datetime', 
            columns='symbol', 
            values='returns'
        )
        
        # Расчет rolling корреляций (за последние 20 дней)
        window = 96 * 20  # 20 дней * 96 периодов в день
        self.correlation_matrix = pivot_data.rolling(window).corr()
        
        # Берем последнюю корреляционную матрицу
        if not self.correlation_matrix.empty:
            self.correlation_matrix = self.correlation_matrix.iloc[-len(pivot_data.columns):]
            self.correlation_matrix.index = pivot_data.columns
            
        self.logger.info("Матрица корреляций обновлена")
    
    def get_portfolio_summary(self) -> Dict:
        """Получение сводки по портфелю"""
        metrics = self.get_risk_metrics()
        
        return {
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'active_positions': len(self.active_positions),
            'total_exposure': metrics.total_exposure,
            'portfolio_risk': metrics.portfolio_risk,
            'max_drawdown': metrics.max_drawdown,
            'sharpe_ratio': metrics.sharpe_ratio,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'total_trades': len(self.closed_positions),
            'correlation_risk': self._calculate_portfolio_correlation_risk()
        }
    
    def _calculate_portfolio_correlation_risk(self) -> float:
        """Расчет корреляционного риска портфеля"""
        if len(self.active_positions) < 2 or self.correlation_matrix is None:
            return 0.0
        
        symbols = list(self.active_positions.keys())
        total_correlation = 0.0
        pairs_count = 0
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = self._get_correlation(symbol1, symbol2)
                weight1 = self.active_positions[symbol1].position_value / self.current_capital
                weight2 = self.active_positions[symbol2].position_value / self.current_capital
                
                total_correlation += abs(correlation) * weight1 * weight2
                pairs_count += 1
        
        return total_correlation / pairs_count if pairs_count > 0 else 0.0