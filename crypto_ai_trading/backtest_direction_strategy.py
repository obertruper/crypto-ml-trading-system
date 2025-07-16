"""
Бэктестинг стратегии на основе Direction модели
Симуляция реальной торговли с учетом комиссий и проскальзывания
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models.direction_predictor import DirectionPredictor
from train_direction_model import DirectionDatasetAdapter
from data.data_loader import CryptoDataLoader
from utils.logger import get_logger, setup_logging
from utils.config import load_config


class DirectionBacktester:
    """Бэктестер для Direction стратегии"""
    
    def __init__(self, model: nn.Module, config: Dict, initial_capital: float = 10000):
        self.model = model
        self.config = config
        self.logger = get_logger("DirectionBacktester")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Торговые параметры
        self.initial_capital = initial_capital
        self.commission = config['bybit']['fees']['taker']
        self.slippage = config['bybit']['slippage']['base']
        
        # Риск-менеджмент
        self.max_position_size = 0.1  # 10% капитала на позицию
        self.stop_loss_pct = 0.02  # 2% стоп-лосс
        self.take_profit_pct = 0.06  # 6% тейк-профит
        
        # Фильтры для входа
        self.min_confidence = 0.65  # Минимальная уверенность модели
        self.min_volume_ratio = 1.2  # Минимальный объем относительно среднего
        
    def backtest(self, dataloader: DataLoader, 
                timeframe: str = '4h',
                use_filters: bool = True) -> Dict:
        """
        Запуск бэктеста стратегии
        
        Args:
            dataloader: данные для тестирования
            timeframe: основной таймфрейм для торговли
            use_filters: использовать ли дополнительные фильтры
            
        Returns:
            Результаты бэктеста
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Запуск бэктеста Direction стратегии")
        self.logger.info(f"Таймфрейм: {timeframe}, Фильтры: {'Да' if use_filters else 'Нет'}")
        self.logger.info(f"{'='*60}")
        
        # Инициализация портфеля
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},  # symbol -> position info
            'history': [],
            'equity_curve': [self.initial_capital]
        }
        
        all_trades = []
        current_timestamp = None
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(tqdm(dataloader, desc="Backtesting")):
                inputs = inputs.to(self.device)
                
                # Получаем предсказания
                outputs = self.model(inputs, return_features=True)
                
                direction_key = f'direction_{timeframe}'
                if direction_key not in outputs:
                    continue
                
                # Обрабатываем каждый пример в батче
                batch_size = inputs.size(0)
                
                for i in range(batch_size):
                    # Извлекаем информацию
                    symbol = info['symbol'][i] if 'symbol' in info else 'UNKNOWN'
                    timestamp = info['datetime'][i] if 'datetime' in info else None
                    
                    # Текущая цена (из данных)
                    if 'close_price' in info:
                        current_price = info['close_price'][i].item()
                    else:
                        current_price = 100.0  # Default для тестирования
                    
                    # Предсказание и уверенность
                    logits = outputs[direction_key][i]
                    probs = torch.softmax(logits, dim=-1)
                    prediction = logits.argmax().item()
                    confidence = probs.max().item()
                    
                    # Дополнительные фильтры
                    volume_ok = True
                    if use_filters and 'volume_ratio' in info:
                        volume_ratio = info['volume_ratio'][i].item()
                        volume_ok = volume_ratio >= self.min_volume_ratio
                    
                    # Проверяем открытые позиции
                    self._check_positions(portfolio, symbol, current_price, timestamp)
                    
                    # Торговые решения
                    if confidence >= self.min_confidence and volume_ok:
                        position_size = self._calculate_position_size(
                            portfolio, current_price, symbol
                        )
                        
                        if position_size > 0:
                            if prediction == 0:  # UP - открываем LONG
                                trade = self._open_long(
                                    portfolio, symbol, current_price, 
                                    position_size, confidence, timestamp
                                )
                                if trade:
                                    all_trades.append(trade)
                                    
                            elif prediction == 1:  # DOWN - открываем SHORT
                                trade = self._open_short(
                                    portfolio, symbol, current_price,
                                    position_size, confidence, timestamp
                                )
                                if trade:
                                    all_trades.append(trade)
                    
                    # Обновляем equity
                    equity = self._calculate_equity(portfolio, current_price)
                    portfolio['equity_curve'].append(equity)
        
        # Закрываем все открытые позиции
        self._close_all_positions(portfolio, current_price, timestamp)
        
        # Анализ результатов
        results = self._analyze_results(portfolio, all_trades)
        
        return results
    
    def _calculate_position_size(self, portfolio: Dict, 
                               current_price: float,
                               symbol: str) -> float:
        """Расчет размера позиции с учетом риск-менеджмента"""
        
        # Текущий капитал
        equity = self._calculate_equity(portfolio, current_price)
        
        # Максимальный размер позиции
        max_position_value = equity * self.max_position_size
        
        # Проверяем, нет ли уже открытой позиции по символу
        if symbol in portfolio['positions']:
            return 0.0  # Не открываем вторую позицию
        
        # Количество открытых позиций
        num_positions = len(portfolio['positions'])
        if num_positions >= 5:  # Максимум 5 позиций
            return 0.0
        
        # Размер с учетом диверсификации
        position_value = min(max_position_value, equity * 0.02)  # 2% на позицию
        
        return position_value
    
    def _open_long(self, portfolio: Dict, symbol: str, 
                  price: float, size: float, 
                  confidence: float, timestamp) -> Optional[Dict]:
        """Открытие LONG позиции"""
        
        # Комиссия
        commission_paid = size * self.commission
        
        # Проскальзывание
        entry_price = price * (1 + self.slippage)
        
        # Проверяем достаточно ли средств
        total_cost = size + commission_paid
        if portfolio['cash'] < total_cost:
            return None
        
        # Открываем позицию
        portfolio['cash'] -= total_cost
        
        position = {
            'type': 'LONG',
            'symbol': symbol,
            'entry_price': entry_price,
            'size': size,
            'quantity': size / entry_price,
            'stop_loss': entry_price * (1 - self.stop_loss_pct),
            'take_profit': entry_price * (1 + self.take_profit_pct),
            'confidence': confidence,
            'entry_time': timestamp,
            'commission_paid': commission_paid
        }
        
        portfolio['positions'][symbol] = position
        
        trade = {
            'symbol': symbol,
            'type': 'LONG',
            'entry_time': timestamp,
            'entry_price': entry_price,
            'size': size,
            'confidence': confidence
        }
        
        return trade
    
    def _open_short(self, portfolio: Dict, symbol: str,
                   price: float, size: float,
                   confidence: float, timestamp) -> Optional[Dict]:
        """Открытие SHORT позиции"""
        
        # Комиссия
        commission_paid = size * self.commission
        
        # Проскальзывание
        entry_price = price * (1 - self.slippage)
        
        # Проверяем достаточно ли средств (для маржи)
        total_cost = size * 0.1 + commission_paid  # 10% маржа
        if portfolio['cash'] < total_cost:
            return None
        
        # Открываем позицию
        portfolio['cash'] -= commission_paid  # Только комиссия из кэша
        
        position = {
            'type': 'SHORT',
            'symbol': symbol,
            'entry_price': entry_price,
            'size': size,
            'quantity': size / entry_price,
            'stop_loss': entry_price * (1 + self.stop_loss_pct),
            'take_profit': entry_price * (1 - self.take_profit_pct),
            'confidence': confidence,
            'entry_time': timestamp,
            'commission_paid': commission_paid
        }
        
        portfolio['positions'][symbol] = position
        
        trade = {
            'symbol': symbol,
            'type': 'SHORT',
            'entry_time': timestamp,
            'entry_price': entry_price,
            'size': size,
            'confidence': confidence
        }
        
        return trade
    
    def _check_positions(self, portfolio: Dict, symbol: str, 
                        current_price: float, timestamp):
        """Проверка и закрытие позиций по SL/TP"""
        
        if symbol not in portfolio['positions']:
            return
        
        position = portfolio['positions'][symbol]
        
        close_position = False
        close_reason = None
        
        if position['type'] == 'LONG':
            if current_price <= position['stop_loss']:
                close_position = True
                close_reason = 'STOP_LOSS'
            elif current_price >= position['take_profit']:
                close_position = True
                close_reason = 'TAKE_PROFIT'
        
        elif position['type'] == 'SHORT':
            if current_price >= position['stop_loss']:
                close_position = True
                close_reason = 'STOP_LOSS'
            elif current_price <= position['take_profit']:
                close_position = True
                close_reason = 'TAKE_PROFIT'
        
        if close_position:
            self._close_position(portfolio, symbol, current_price, 
                               timestamp, close_reason)
    
    def _close_position(self, portfolio: Dict, symbol: str,
                       current_price: float, timestamp, reason: str):
        """Закрытие позиции"""
        
        if symbol not in portfolio['positions']:
            return
        
        position = portfolio['positions'][symbol]
        
        # Проскальзывание при закрытии
        if position['type'] == 'LONG':
            exit_price = current_price * (1 - self.slippage)
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:  # SHORT
            exit_price = current_price * (1 + self.slippage)
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        # Комиссия при закрытии
        exit_commission = position['size'] * self.commission
        pnl -= exit_commission
        
        # Обновляем портфель
        portfolio['cash'] += position['size'] + pnl
        
        # Сохраняем историю
        trade_result = {
            'symbol': symbol,
            'type': position['type'],
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'pnl_pct': pnl / position['size'],
            'reason': reason,
            'confidence': position['confidence'],
            'commission_total': position['commission_paid'] + exit_commission
        }
        
        portfolio['history'].append(trade_result)
        
        # Удаляем позицию
        del portfolio['positions'][symbol]
    
    def _close_all_positions(self, portfolio: Dict, 
                           current_price: float, timestamp):
        """Закрытие всех открытых позиций"""
        
        symbols_to_close = list(portfolio['positions'].keys())
        
        for symbol in symbols_to_close:
            self._close_position(portfolio, symbol, current_price,
                               timestamp, 'END_OF_TEST')
    
    def _calculate_equity(self, portfolio: Dict, current_price: float) -> float:
        """Расчет текущего капитала"""
        
        equity = portfolio['cash']
        
        # Добавляем стоимость открытых позиций
        for symbol, position in portfolio['positions'].items():
            if position['type'] == 'LONG':
                position_value = position['quantity'] * current_price
            else:  # SHORT
                pnl = (position['entry_price'] - current_price) * position['quantity']
                position_value = position['size'] + pnl
            
            equity += position_value
        
        return equity
    
    def _analyze_results(self, portfolio: Dict, all_trades: List[Dict]) -> Dict:
        """Анализ результатов бэктеста"""
        
        self.logger.info("\n📊 Анализ результатов бэктеста...")
        
        history = portfolio['history']
        equity_curve = np.array(portfolio['equity_curve'])
        
        if len(history) == 0:
            self.logger.warning("⚠️ Нет завершенных сделок для анализа")
            return {
                'total_trades': 0,
                'final_equity': equity_curve[-1] if len(equity_curve) > 0 else self.initial_capital,
                'total_return': 0.0,
                'message': 'No trades executed'
            }
        
        # Конвертируем в DataFrame для удобства
        trades_df = pd.DataFrame(history)
        
        # Основные метрики
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L статистика
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
        
        # Доходность
        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Sharpe Ratio (daily)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if returns.std() > 0:
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Анализ по типам сделок
        long_trades = trades_df[trades_df['type'] == 'LONG']
        short_trades = trades_df[trades_df['type'] == 'SHORT']
        
        # Анализ по причинам закрытия
        close_reasons = trades_df['reason'].value_counts()
        
        # Комиссии
        total_commission = trades_df['commission_total'].sum()
        
        # Временной анализ
        trades_df['hold_time'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
        avg_hold_time = trades_df['hold_time'].mean()
        
        # Вывод результатов
        self.logger.info(f"\n💰 РЕЗУЛЬТАТЫ БЭКТЕСТА:")
        self.logger.info(f"  Начальный капитал: ${self.initial_capital:,.2f}")
        self.logger.info(f"  Финальный капитал: ${final_equity:,.2f}")
        self.logger.info(f"  Общая доходность: {total_return:.2%}")
        self.logger.info(f"  Общий P&L: ${total_pnl:,.2f}")
        
        self.logger.info(f"\n📈 ТОРГОВАЯ СТАТИСТИКА:")
        self.logger.info(f"  Всего сделок: {total_trades}")
        self.logger.info(f"  Прибыльных: {winning_trades} ({win_rate:.1%})")
        self.logger.info(f"  Убыточных: {losing_trades} ({(1-win_rate):.1%})")
        self.logger.info(f"  Средняя прибыль: ${avg_win:,.2f}")
        self.logger.info(f"  Средний убыток: ${avg_loss:,.2f}")
        self.logger.info(f"  Profit Factor: {profit_factor:.2f}")
        
        self.logger.info(f"\n📊 РИСК-МЕТРИКИ:")
        self.logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        self.logger.info(f"  Среднее время в позиции: {avg_hold_time}")
        
        self.logger.info(f"\n💸 КОМИССИИ:")
        self.logger.info(f"  Общие комиссии: ${total_commission:,.2f}")
        self.logger.info(f"  Комиссии от P&L: {(total_commission/abs(total_pnl)*100 if total_pnl != 0 else 0):.1f}%")
        
        self.logger.info(f"\n🎯 АНАЛИЗ ПО ТИПАМ:")
        self.logger.info(f"  LONG сделок: {len(long_trades)} "
                        f"(WR: {(len(long_trades[long_trades['pnl'] > 0])/len(long_trades)*100 if len(long_trades) > 0 else 0):.1f}%)")
        self.logger.info(f"  SHORT сделок: {len(short_trades)} "
                        f"(WR: {(len(short_trades[short_trades['pnl'] > 0])/len(short_trades)*100 if len(short_trades) > 0 else 0):.1f}%)")
        
        self.logger.info(f"\n📍 ПРИЧИНЫ ЗАКРЫТИЯ:")
        for reason, count in close_reasons.items():
            self.logger.info(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")
        
        results = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_commission': total_commission,
            'avg_hold_time': str(avg_hold_time),
            'equity_curve': equity_curve.tolist(),
            'trades_history': trades_df.to_dict('records'),
            'close_reasons': close_reasons.to_dict(),
            'long_stats': {
                'count': len(long_trades),
                'win_rate': len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
            },
            'short_stats': {
                'count': len(short_trades),
                'win_rate': len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
            }
        }
        
        return results
    
    def visualize_results(self, results: Dict, save_path: Path):
        """Визуализация результатов бэктеста"""
        
        if results.get('total_trades', 0) == 0:
            self.logger.warning("⚠️ Нет данных для визуализации")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Direction Strategy Backtest Results', fontsize=16)
        
        # 1. Equity Curve
        ax = axes[0, 0]
        equity_curve = results['equity_curve']
        ax.plot(equity_curve, linewidth=2)
        ax.axhline(y=results['initial_capital'], color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Equity ($)')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax = axes[0, 1]
        equity_curve = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        ax.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown, color='red', linewidth=1)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Chart')
        ax.grid(True, alpha=0.3)
        
        # 3. Trade Distribution
        ax = axes[1, 0]
        trades_df = pd.DataFrame(results['trades_history'])
        
        if not trades_df.empty:
            pnl_pct = trades_df['pnl_pct'] * 100
            
            # Histogram
            n, bins, patches = ax.hist(pnl_pct, bins=30, alpha=0.7, edgecolor='black')
            
            # Color bars
            for i, patch in enumerate(patches):
                if bins[i] >= 0:
                    patch.set_facecolor('green')
                else:
                    patch.set_facecolor('red')
            
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('P&L per Trade (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Trade P&L Distribution')
            
            # Статистика
            ax.text(0.02, 0.95, f"Avg: {pnl_pct.mean():.2f}%", 
                   transform=ax.transAxes, verticalalignment='top')
            ax.text(0.02, 0.90, f"Std: {pnl_pct.std():.2f}%", 
                   transform=ax.transAxes, verticalalignment='top')
        
        # 4. Monthly Returns (если достаточно данных)
        ax = axes[1, 1]
        
        if not trades_df.empty and 'exit_time' in trades_df.columns:
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
            
            monthly_pnl = trades_df.groupby('month')['pnl'].sum()
            
            if len(monthly_pnl) > 0:
                colors = ['green' if x > 0 else 'red' for x in monthly_pnl.values]
                monthly_pnl.plot(kind='bar', ax=ax, color=colors, alpha=0.7)
                ax.set_xlabel('Month')
                ax.set_ylabel('P&L ($)')
                ax.set_title('Monthly P&L')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                # Поворот меток
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path / 'backtest_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Дополнительный график - анализ по символам
        if not trades_df.empty and 'symbol' in trades_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            symbol_pnl = trades_df.groupby('symbol')['pnl'].agg(['sum', 'count', 'mean'])
            symbol_pnl = symbol_pnl.sort_values('sum', ascending=False).head(20)
            
            colors = ['green' if x > 0 else 'red' for x in symbol_pnl['sum'].values]
            symbol_pnl['sum'].plot(kind='bar', ax=ax, color=colors, alpha=0.7)
            
            ax.set_xlabel('Symbol')
            ax.set_ylabel('Total P&L ($)')
            ax.set_title('P&L by Symbol (Top 20)')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Добавляем количество сделок
            for i, (idx, row) in enumerate(symbol_pnl.iterrows()):
                ax.text(i, row['sum'], f"n={int(row['count'])}", 
                       ha='center', va='bottom' if row['sum'] > 0 else 'top',
                       fontsize=8)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path / 'symbol_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"📊 Визуализация сохранена в: {save_path}")
    
    def save_results(self, results: Dict, save_path: Path):
        """Сохранение результатов бэктеста"""
        
        # JSON с основными результатами
        json_path = save_path / 'backtest_results.json'
        
        # Конвертируем numpy в обычные типы
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # CSV с историей сделок
        if 'trades_history' in results and results['trades_history']:
            trades_df = pd.DataFrame(results['trades_history'])
            csv_path = save_path / 'trades_history.csv'
            trades_df.to_csv(csv_path, index=False)
            
        # Текстовый отчет
        report_path = save_path / 'backtest_report.txt'
        with open(report_path, 'w') as f:
            f.write("DIRECTION STRATEGY BACKTEST REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("SUMMARY:\n")
            f.write(f"Initial Capital: ${results['initial_capital']:,.2f}\n")
            f.write(f"Final Equity: ${results['final_equity']:,.2f}\n")
            f.write(f"Total Return: {results['total_return']:.2%}\n")
            f.write(f"Total P&L: ${results['total_pnl']:,.2f}\n\n")
            
            f.write("TRADING STATISTICS:\n")
            f.write(f"Total Trades: {results['total_trades']}\n")
            f.write(f"Win Rate: {results['win_rate']:.2%}\n")
            f.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
            f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown: {results['max_drawdown']:.2%}\n\n")
            
            f.write("RECOMMENDATION:\n")
            if results['sharpe_ratio'] > 1.0 and results['profit_factor'] > 1.5:
                f.write("✅ Strategy shows GOOD performance\n")
            elif results['sharpe_ratio'] > 0.5 and results['profit_factor'] > 1.2:
                f.write("⚠️ Strategy shows MODERATE performance\n")
            else:
                f.write("❌ Strategy shows POOR performance\n")
        
        self.logger.info(f"📄 Результаты сохранены в: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Backtest Direction Strategy')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Specific symbols to test')
    parser.add_argument('--timeframe', type=str, default='4h',
                       choices=['15m', '1h', '4h', '12h'],
                       help='Trading timeframe')
    parser.add_argument('--initial-capital', type=float, default=10000,
                       help='Initial capital for backtest')
    parser.add_argument('--no-filters', action='store_true',
                       help='Disable additional entry filters')
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    
    # Логирование
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/backtest_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir, "backtest")
    logger = get_logger("BacktestDirection")
    
    logger.info("🚀 Запуск бэктеста Direction стратегии")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Initial Capital: ${args.initial_capital:,.2f}")
    
    # Загрузка модели
    logger.info("📂 Загрузка модели...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        model_config = config['model']
    
    model = DirectionPredictor(model_config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("✅ Модель загружена")
    
    # Загрузка данных
    logger.info("📊 Загрузка данных для бэктеста...")
    data_loader = CryptoDataLoader(config)
    
    # Фильтрация по датам и символам
    kwargs = {}
    if args.start_date:
        kwargs['start_date'] = args.start_date
    if args.end_date:
        kwargs['end_date'] = args.end_date
    if args.symbols:
        kwargs['symbols'] = args.symbols
    
    data = data_loader.load_data(**kwargs)
    
    # Используем только test данные для честного бэктеста
    _, _, test_data = data_loader.split_data(data)
    
    logger.info(f"✅ Загружено {len(test_data)} записей для бэктеста")
    
    # Создание dataset
    dataset = DirectionDatasetAdapter(
        test_data,
        context_window=model_config.get('context_window', 168),
        feature_cols=data_loader.feature_columns,
        target_cols=[col for col in data_loader.target_columns if col.startswith('direction_')],
        stride=1
    )
    
    # Добавляем информацию о ценах в dataset
    class BacktestDataset(DirectionDatasetAdapter):
        def __getitem__(self, idx):
            features, targets, info = super().__getitem__(idx)
            
            # Добавляем текущую цену закрытия
            start_idx, end_idx, symbol = self.indices[idx]
            info['close_price'] = torch.FloatTensor([self.data.iloc[end_idx - 1]['close']])
            info['volume_ratio'] = torch.FloatTensor([self.data.iloc[end_idx - 1].get('volume_ratio', 1.0)])
            
            return features, targets, info
    
    # Пересоздаем dataset с дополнительной информацией
    dataset = BacktestDataset(
        test_data,
        context_window=model_config.get('context_window', 168),
        feature_cols=data_loader.feature_columns,
        target_cols=[col for col in data_loader.target_columns if col.startswith('direction_')],
        stride=1
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,  # Меньше батч для последовательной обработки
        shuffle=False,  # Важно для бэктеста!
        num_workers=0   # Для сохранения порядка
    )
    
    # Backtester
    backtester = DirectionBacktester(
        model, config,
        initial_capital=args.initial_capital
    )
    
    # Запуск бэктеста
    logger.info("📈 Запуск симуляции торговли...")
    results = backtester.backtest(
        dataloader,
        timeframe=args.timeframe,
        use_filters=not args.no_filters
    )
    
    # Визуализация и сохранение
    if results.get('total_trades', 0) > 0:
        backtester.visualize_results(results, log_dir)
        backtester.save_results(results, log_dir)
    
    # Итоговый вывод
    logger.info("\n" + "="*60)
    logger.info("ИТОГИ БЭКТЕСТА:")
    logger.info("="*60)
    logger.info(f"Total Return: {results.get('total_return', 0):.2%}")
    logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    logger.info(f"Win Rate: {results.get('win_rate', 0):.2%}")
    
    if results.get('sharpe_ratio', 0) > 1.0:
        logger.info("\n✅ Стратегия показывает хорошие результаты!")
    else:
        logger.info("\n⚠️ Стратегия требует доработки")
    
    logger.info(f"\n📁 Все результаты сохранены в: {log_dir}")


if __name__ == "__main__":
    main()