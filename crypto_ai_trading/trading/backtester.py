"""
Система бэктестирования торговых стратегий
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from .risk_manager import RiskManager, Position
from .signals import SignalGenerator, Signal

class Backtester:
    """Основной класс для бэктестирования торговых стратегий"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.backtest_config = config['backtesting']
        self.logger = get_logger("Backtester")
        
        # Инициализация компонентов
        self.risk_manager = RiskManager(config)
        self.signal_generator = SignalGenerator(config)
        
        # Параметры бэктестирования
        self.initial_capital = self.backtest_config['initial_capital']
        self.commission = self.backtest_config['commission']
        self.slippage = self.backtest_config['slippage']
        
        # Результаты бэктестирования
        self.trades_log = []
        self.equity_curve = []
        self.daily_returns = []
        self.drawdown_periods = []
        
    def run_backtest(self, 
                     market_data: pd.DataFrame, 
                     features: pd.DataFrame,
                     model_predictions: Optional[Dict] = None) -> Dict:
        """Запуск полного бэктестирования"""
        
        self.logger.start_stage("backtesting", 
                               start_date=market_data['datetime'].min(),
                               end_date=market_data['datetime'].max(),
                               symbols=market_data['symbol'].nunique())
        
        # Подготовка данных
        data_by_time = self._prepare_time_series_data(market_data, features)
        
        # Основной цикл бэктестирования
        for timestamp, time_data in data_by_time.items():
            self._process_timestamp(timestamp, time_data, model_predictions)
        
        # Закрытие всех открытых позиций в конце
        self._close_all_positions(market_data)
        
        # Расчет результатов
        results = self._calculate_results()
        
        self.logger.end_stage("backtesting", 
                            total_trades=len(self.trades_log),
                            final_capital=self.risk_manager.current_capital)
        
        return results
    
    def _prepare_time_series_data(self, 
                                 market_data: pd.DataFrame, 
                                 features: pd.DataFrame) -> Dict:
        """Подготовка данных по временным меткам"""
        
        # Объединение рыночных данных с признаками
        combined_data = market_data.merge(
            features, 
            on=['symbol', 'datetime'], 
            how='inner'
        )
        
        # Группировка по времени
        data_by_time = {}
        
        for timestamp in combined_data['datetime'].unique():
            timestamp_data = combined_data[combined_data['datetime'] == timestamp]
            data_by_time[timestamp] = timestamp_data
        
        self.logger.info(f"Подготовлено {len(data_by_time)} временных точек для бэктестирования")
        
        return data_by_time
    
    def _process_timestamp(self, 
                          timestamp: pd.Timestamp, 
                          data: pd.DataFrame,
                          model_predictions: Optional[Dict] = None):
        """Обработка одной временной точки"""
        
        # Обновление существующих позиций
        self._update_existing_positions(data)
        
        # Генерация новых сигналов
        if model_predictions:
            signals = self._generate_signals_for_timestamp(data, model_predictions)
            
            # Исполнение сигналов
            for signal in signals:
                self._execute_signal(signal, data)
        
        # Обновление эквити
        current_equity = self._calculate_current_equity(data)
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'active_positions': len(self.risk_manager.active_positions)
        })
        
        # Расчет дневной доходности
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            daily_return = (current_equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)
    
    def _update_existing_positions(self, data: pd.DataFrame):
        """Обновление существующих позиций"""
        
        positions_to_close = []
        
        for symbol, position in self.risk_manager.active_positions.items():
            # Получение текущих данных для символа
            symbol_data = data[data['symbol'] == symbol]
            
            if symbol_data.empty:
                continue
            
            current_price = symbol_data.iloc[0]['close']
            
            # Проверка условий закрытия
            actions = self.risk_manager.update_position(symbol, current_price)
            
            for action in actions:
                if action['action'] == 'close_position':
                    self._close_position(position, current_price, action['reason'])
                    positions_to_close.append(symbol)
                elif action['action'] == 'partial_close':
                    self._partial_close_position(position, current_price, action)
        
        # Удаление закрытых позиций
        for symbol in positions_to_close:
            if symbol in self.risk_manager.active_positions:
                del self.risk_manager.active_positions[symbol]
    
    def _generate_signals_for_timestamp(self, 
                                      data: pd.DataFrame, 
                                      model_predictions: Dict) -> List[Signal]:
        """Генерация сигналов для текущей временной точки"""
        
        # Фиктивные предсказания для демонстрации
        fake_predictions = {
            'tp_probs': np.random.random((len(data), 4)),
            'sl_prob': np.random.random(len(data)),
            'volatility': np.random.random(len(data)) * 0.05,
            'price_pred': np.random.random(len(data)) * 0.04 - 0.02
        }
        
        return self.signal_generator.generate_signals(
            fake_predictions, data, data
        )
    
    def _execute_signal(self, signal: Signal, data: pd.DataFrame):
        """Исполнение торгового сигнала"""
        
        # Получение данных для символа
        symbol_data = data[data['symbol'] == signal.symbol]
        
        if symbol_data.empty:
            return
        
        current_market_data = symbol_data.iloc[0]
        
        # Валидация сигнала
        if not self.signal_generator.validate_signal(signal, current_market_data):
            return
        
        # Применение slippage с учетом размера позиции
        execution_price = self._apply_execution_costs(
            signal.entry_price, 
            signal.side,
            signal.position_size if hasattr(signal, 'position_size') else 0,
            current_market_data
        )
        
        # Открытие позиции через risk manager
        position = self.risk_manager.open_position(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=execution_price,
            stop_loss=signal.stop_loss,
            take_profits=signal.take_profits,
            confidence=signal.confidence,
            atr_value=signal.atr_value
        )
        
        if position:
            # Логирование сделки
            trade_record = {
                'timestamp': current_market_data['datetime'],
                'symbol': signal.symbol,
                'side': signal.side,
                'action': 'open',
                'price': execution_price,
                'quantity': position.quantity,
                'stop_loss': signal.stop_loss,
                'take_profits': signal.take_profits,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            }
            
            self.trades_log.append(trade_record)
    
    def _apply_execution_costs(self, price: float, side: str, position_size: float = 0, market_data: pd.Series = None) -> float:
        """Применение slippage и комиссий к цене исполнения с учетом market impact"""
        
        # Базовый slippage из конфига Bybit
        bybit_config = self.config.get('bybit', {})
        base_slippage = bybit_config.get('slippage', {}).get('base', 0.0005)
        
        # Market impact на основе объема позиции
        market_impact = 0
        if position_size > 0 and market_data is not None and 'volume' in market_data:
            hourly_volume = market_data['volume'] * 4  # 15-мин свечи -> часовой объем
            position_value = position_size * price
            volume_ratio = position_value / (hourly_volume + 1e-8)
            
            # Порог для market impact из конфига
            impact_threshold = bybit_config.get('slippage', {}).get('market_impact_threshold', 0.01)
            
            if volume_ratio > impact_threshold:
                # Квадратичный market impact
                market_impact = min(0.005, (volume_ratio / impact_threshold - 1) ** 2 * 0.001)
        
        total_slippage = base_slippage + market_impact
        
        # Применение slippage
        if side == 'long':
            # При покупке цена хуже (выше)
            execution_price = price * (1 + total_slippage)
        else:
            # При продаже цена хуже (ниже)
            execution_price = price * (1 - total_slippage)
        
        return execution_price
    
    def _close_position(self, position: Position, close_price: float, reason: str):
        """Закрытие позиции"""
        
        # Применение costs к цене закрытия
        execution_price = self._apply_execution_costs(close_price, 
                                                     'short' if position.side == 'long' else 'long',
                                                     position.quantity)
        
        # Расчет P&L
        if position.side == 'long':
            gross_pnl = (execution_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - execution_price) * position.quantity
        
        # Применение комиссий Bybit (maker/taker)
        bybit_fees = self.config.get('bybit', {}).get('fees', {})
        # Предполагаем taker для market orders
        commission_rate = bybit_fees.get('taker', 0.00055)
        
        # Комиссия на вход и выход
        entry_commission = position.entry_price * position.quantity * commission_rate
        exit_commission = execution_price * position.quantity * commission_rate
        total_commission = entry_commission + exit_commission
        
        # Funding rate для позиций > 8 часов
        hold_time_hours = (datetime.now() - position.entry_time).total_seconds() / 3600
        funding_cost = 0
        if hold_time_hours > 8:
            funding_periods = int(hold_time_hours / 8)
            funding_rate = bybit_fees.get('funding_rate', 0.0001)
            funding_cost = position.position_value * funding_rate * funding_periods
        
        net_pnl = gross_pnl - total_commission - funding_cost
        
        # Обновление капитала
        self.risk_manager.current_capital += net_pnl
        
        # Логирование закрытия
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': position.symbol,
            'side': position.side,
            'action': 'close',
            'price': execution_price,
            'quantity': position.quantity,
            'pnl': net_pnl,
            'reason': reason,
            'hold_time': (datetime.now() - position.entry_time).total_seconds() / 3600  # часы
        }
        
        self.trades_log.append(trade_record)
        
        # Сохранение PnL в позиции для статистики
        position.pnl = net_pnl
        self.risk_manager.closed_positions.append(position)
    
    def _partial_close_position(self, position: Position, close_price: float, action: Dict):
        """Частичное закрытие позиции"""
        
        close_quantity = action['quantity']
        close_percentage = action['percentage']
        
        # Применение costs
        execution_price = self._apply_execution_costs(close_price, 
                                                     'short' if position.side == 'long' else 'long',
                                                     close_quantity)
        
        # Расчет P&L для закрываемой части
        if position.side == 'long':
            gross_pnl = (execution_price - position.entry_price) * close_quantity
        else:
            gross_pnl = (position.entry_price - execution_price) * close_quantity
        
        # Комиссии только на закрываемую часть (Bybit taker fee)
        bybit_fees = self.config.get('bybit', {}).get('fees', {})
        commission_rate = bybit_fees.get('taker', 0.00055)
        commission = execution_price * close_quantity * commission_rate
        net_pnl = gross_pnl - commission
        
        # Обновление капитала
        self.risk_manager.current_capital += net_pnl
        
        # Логирование частичного закрытия
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': position.symbol,
            'side': position.side,
            'action': 'partial_close',
            'price': execution_price,
            'quantity': close_quantity,
            'percentage': close_percentage,
            'pnl': net_pnl,
            'reason': action['reason']
        }
        
        self.trades_log.append(trade_record)
    
    def _calculate_current_equity(self, data: pd.DataFrame) -> float:
        """Расчет текущего эквити с учетом открытых позиций"""
        
        equity = self.risk_manager.current_capital
        
        # Добавляем нереализованную прибыль/убыток
        for symbol, position in self.risk_manager.active_positions.items():
            symbol_data = data[data['symbol'] == symbol]
            
            if not symbol_data.empty:
                current_price = symbol_data.iloc[0]['close']
                
                if position.side == 'long':
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position.quantity
                
                equity += unrealized_pnl
        
        return equity
    
    def _close_all_positions(self, market_data: pd.DataFrame):
        """Закрытие всех оставшихся позиций в конце бэктестирования"""
        
        final_data = market_data.groupby('symbol').last().reset_index()
        
        for symbol, position in list(self.risk_manager.active_positions.items()):
            symbol_final_data = final_data[final_data['symbol'] == symbol]
            
            if not symbol_final_data.empty:
                final_price = symbol_final_data.iloc[0]['close']
                self._close_position(position, final_price, "end_of_backtest")
                del self.risk_manager.active_positions[symbol]
    
    def _calculate_results(self) -> Dict:
        """Расчет финальных результатов бэктестирования"""
        
        if not self.trades_log or not self.equity_curve:
            return self._empty_results()
        
        # Базовые метрики
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Анализ сделок
        trades_df = pd.DataFrame(self.trades_log)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Статистика сделок
        completed_trades = trades_df[trades_df['action'] == 'close']
        
        if not completed_trades.empty:
            winning_trades = completed_trades[completed_trades['pnl'] > 0]
            losing_trades = completed_trades[completed_trades['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(completed_trades)
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            profit_factor = (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) 
                           if len(losing_trades) > 0 and losing_trades['pnl'].sum() < 0 else float('inf'))
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Расчет просадки
        equity_series = equity_df['equity']
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Коэффициент Шарпа
        if len(self.daily_returns) > 1:
            sharpe_ratio = (np.mean(self.daily_returns) / np.std(self.daily_returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Коэффициент Сортино
        negative_returns = [r for r in self.daily_returns if r < 0]
        if negative_returns:
            sortino_ratio = (np.mean(self.daily_returns) / np.std(negative_returns)) * np.sqrt(252)
        else:
            sortino_ratio = float('inf') if np.mean(self.daily_returns) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        # Максимальное время в просадке
        in_drawdown = drawdown < 0
        max_drawdown_duration = self._calculate_max_drawdown_duration(in_drawdown)
        
        results = {
            # Основные результаты
            'initial_capital': self.initial_capital,
            'final_capital': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            
            # Риск-метрики
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility': np.std(self.daily_returns) * np.sqrt(252) if self.daily_returns else 0,
            
            # Статистика сделок
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades) if 'winning_trades' in locals() else 0,
            'losing_trades': len(losing_trades) if 'losing_trades' in locals() else 0,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': completed_trades['pnl'].mean() if not completed_trades.empty else 0,
            
            # Временные метрики
            'max_drawdown_duration_days': max_drawdown_duration,
            'avg_trade_duration_hours': completed_trades['hold_time'].mean() if 'hold_time' in completed_trades.columns else 0,
            
            # Детальные данные
            'equity_curve': equity_df.to_dict('records'),
            'trades_log': trades_df.to_dict('records'),
            'daily_returns': self.daily_returns,
            
            # Статистика по символам
            'performance_by_symbol': self._calculate_symbol_performance(completed_trades) if not completed_trades.empty else {}
        }
        
        return results
    
    def _empty_results(self) -> Dict:
        """Возврат пустых результатов при отсутствии данных"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital,
            'total_return': 0,
            'total_return_pct': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'volatility': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'win_rate_pct': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_trade': 0,
            'max_drawdown_duration_days': 0,
            'avg_trade_duration_hours': 0,
            'equity_curve': [],
            'trades_log': [],
            'daily_returns': [],
            'performance_by_symbol': {}
        }
    
    def _calculate_max_drawdown_duration(self, in_drawdown: pd.Series) -> int:
        """Расчет максимальной продолжительности просадки в днях"""
        if not in_drawdown.any():
            return 0
        
        # Группируем последовательные периоды просадки
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        # Добавляем последний период если он не завершился
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_symbol_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Расчет производительности по символам"""
        
        symbol_stats = {}
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            
            total_pnl = symbol_trades['pnl'].sum()
            trade_count = len(symbol_trades)
            winning_trades = len(symbol_trades[symbol_trades['pnl'] > 0])
            
            symbol_stats[symbol] = {
                'total_pnl': total_pnl,
                'trade_count': trade_count,
                'win_rate': winning_trades / trade_count if trade_count > 0 else 0,
                'avg_pnl': symbol_trades['pnl'].mean()
            }
        
        return symbol_stats
    
    def generate_report(self, results: Dict) -> str:
        """Генерация текстового отчета о результатах"""
        
        report = f"""
ОТЧЕТ О БЭКТЕСТИРОВАНИИ
========================

ОБЩИЕ РЕЗУЛЬТАТЫ:
- Начальный капитал: ${results['initial_capital']:,.2f}
- Финальный капитал: ${results['final_capital']:,.2f}
- Общая доходность: {results['total_return_pct']:.2f}%

РИСК-МЕТРИКИ:
- Максимальная просадка: {results['max_drawdown_pct']:.2f}%
- Коэффициент Шарпа: {results['sharpe_ratio']:.2f}
- Коэффициент Сортино: {results['sortino_ratio']:.2f}
- Коэффициент Кальмара: {results['calmar_ratio']:.2f}
- Волатильность (год): {results['volatility']*100:.2f}%

СТАТИСТИКА СДЕЛОК:
- Всего сделок: {results['total_trades']}
- Прибыльных: {results['winning_trades']}
- Убыточных: {results['losing_trades']}
- Win Rate: {results['win_rate_pct']:.2f}%
- Profit Factor: {results['profit_factor']:.2f}
- Средняя прибыль: ${results['avg_win']:.2f}
- Средний убыток: ${results['avg_loss']:.2f}
- Средняя сделка: ${results['avg_trade']:.2f}

ВРЕМЕННЫЕ ХАРАКТЕРИСТИКИ:
- Макс. время в просадке: {results['max_drawdown_duration_days']} дней
- Средняя длительность сделки: {results['avg_trade_duration_hours']:.1f} часов
        """
        
        return report