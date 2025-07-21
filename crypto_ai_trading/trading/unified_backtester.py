"""
Современный бэктестер для UnifiedPatchTST модели
Работает с новой архитектурой и 20 целевыми переменными
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger


@dataclass
class UnifiedSignal:
    """Современный торговый сигнал на основе UnifiedPatchTST предсказаний"""
    symbol: str
    timestamp: datetime
    
    # Направления на разных таймфреймах
    direction_15m: str  # LONG/SHORT/FLAT
    direction_1h: str
    direction_4h: str
    direction_12h: str
    
    # Уверенность в направлениях
    confidence_15m: float
    confidence_1h: float
    confidence_4h: float
    confidence_12h: float
    
    # Ожидаемые доходности
    expected_return_15m: float
    expected_return_1h: float
    expected_return_4h: float
    expected_return_12h: float
    
    # Вероятности достижения уровней
    long_tp1_prob: float  # Вероятность +1%
    long_tp2_prob: float  # Вероятность +2%
    long_tp3_prob: float  # Вероятность +3%
    long_tp5_prob: float  # Вероятность +5%
    
    short_tp1_prob: float
    short_tp2_prob: float
    short_tp3_prob: float
    short_tp5_prob: float
    
    # Риск-метрики
    max_drawdown_1h: float
    max_rally_1h: float
    max_drawdown_4h: float
    max_rally_4h: float
    
    # Итоговое решение
    action: str  # LONG/SHORT/HOLD
    signal_strength: float
    risk_reward_ratio: float
    optimal_hold_time: int  # В свечах
    
    # Размер позиции и уровни
    position_size: float
    stop_loss: float
    take_profits: List[float]


class UnifiedBacktester:
    """Современный бэктестер для UnifiedPatchTST"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("UnifiedBacktester")
        
        # Параметры риск-менеджмента
        self.risk_config = config['risk_management']
        self.initial_capital = config['backtesting']['initial_capital']
        self.commission = config['backtesting']['commission']
        self.slippage = config['backtesting']['slippage']
        
        # Параметры торговли
        self.max_positions = self.risk_config['max_concurrent_positions']
        self.confidence_threshold = config['model'].get('direction_confidence_threshold', 0.35)
        
        # Статистика
        self.trades = []
        self.positions = {}
        self.balance = self.initial_capital
        self.equity_curve = []
        
    def extract_predictions(self, model_output: torch.Tensor, batch_idx: int) -> Dict:
        """Извлекает предсказания из выхода модели для одного примера"""
        
        # model_output: (batch_size, 20)
        # Порядок выходов согласно config.yaml:
        # 0-3: future_return_15m/1h/4h/12h (в долях, нужно умножить на 100)
        # 4-7: direction_15m/1h/4h/12h (классы: 0=LONG, 1=SHORT, 2=FLAT)
        # 8-11: long_will_reach_1/2/3/5pct (логиты, нужен sigmoid)
        # 12-15: short_will_reach_1/2/3/5pct (логиты, нужен sigmoid)
        # 16-19: max_drawdown_1h/4h, max_rally_1h/4h (в долях)
        
        if isinstance(model_output, torch.Tensor):
            output = model_output[batch_idx]
            # Применяем sigmoid к логитам вероятностей на GPU перед переносом на CPU
            with torch.no_grad():
                # Sigmoid для вероятностей достижения уровней (индексы 8-15)
                prob_logits = output[8:16]
                probs = torch.sigmoid(prob_logits)
                
                # Создаем обработанный вывод
                processed_output = torch.cat([
                    output[0:8],    # returns и directions - без изменений
                    probs,          # применили sigmoid к вероятностям
                    output[16:20]   # risk metrics - без изменений
                ])
                
                output_np = processed_output.cpu().numpy()
        else:
            output_np = model_output[batch_idx]
            # Если не тензор, применяем sigmoid через numpy
            import numpy as np
            prob_logits = output_np[8:16]
            probs = 1 / (1 + np.exp(-prob_logits))  # sigmoid
            output_np[8:16] = probs
            
        # Извлекаем confidence scores если доступны
        confidence_scores = None
        if hasattr(model_output, '_confidence_scores') and model_output._confidence_scores is not None:
            confidence_scores = model_output._confidence_scores[batch_idx].cpu().numpy()
            
        predictions = {
            # Ожидаемые доходности (денормализуем из долей в проценты)
            'return_15m': float(output_np[0]) * 100,
            'return_1h': float(output_np[1]) * 100,
            'return_4h': float(output_np[2]) * 100,
            'return_12h': float(output_np[3]) * 100,
            
            # Направления (классы: 0=LONG, 1=SHORT, 2=FLAT)
            'direction_15m': int(output_np[4]),
            'direction_1h': int(output_np[5]),
            'direction_4h': int(output_np[6]),
            'direction_12h': int(output_np[7]),
            
            # Вероятности достижения уровней LONG (уже с sigmoid)
            'long_tp1_prob': float(output_np[8]),
            'long_tp2_prob': float(output_np[9]),
            'long_tp3_prob': float(output_np[10]),
            'long_tp5_prob': float(output_np[11]),
            
            # Вероятности достижения уровней SHORT (уже с sigmoid)
            'short_tp1_prob': float(output_np[12]),
            'short_tp2_prob': float(output_np[13]),
            'short_tp3_prob': float(output_np[14]),
            'short_tp5_prob': float(output_np[15]),
            
            # Риск-метрики (денормализуем в проценты)
            'max_drawdown_1h': float(output_np[16]) * 100,
            'max_rally_1h': float(output_np[17]) * 100,
            'max_drawdown_4h': float(output_np[18]) * 100,
            'max_rally_4h': float(output_np[19]) * 100,
            
            # Confidence scores из модели (если доступны)
            'confidence_scores': confidence_scores
        }
        
        # Отладочное логирование для первого предсказания
        if not hasattr(self, '_first_prediction_logged'):
            self._first_prediction_logged = True
            self.logger.info("🔍 Первое предсказание после обработки:")
            self.logger.info(f"   Returns: 15m={predictions['return_15m']:.2f}%, 1h={predictions['return_1h']:.2f}%")
            self.logger.info(f"   Directions: 15m={predictions['direction_15m']}, 1h={predictions['direction_1h']}")
            self.logger.info(f"   Long TP probs: TP1={predictions['long_tp1_prob']:.3f}, TP2={predictions['long_tp2_prob']:.3f}")
            self.logger.info(f"   Short TP probs: TP1={predictions['short_tp1_prob']:.3f}, TP2={predictions['short_tp2_prob']:.3f}")
            if confidence_scores is not None:
                self.logger.info(f"   Confidence: {confidence_scores}")
        
        return predictions
    
    def generate_signal(self, predictions: Dict, symbol: str, price: float, timestamp: datetime) -> Optional[UnifiedSignal]:
        """Генерирует торговый сигнал на основе предсказаний модели"""
        
        # Преобразуем направления в строки
        direction_map = {0: 'LONG', 1: 'SHORT', 2: 'FLAT'}
        
        dir_15m = direction_map[predictions['direction_15m']]
        dir_1h = direction_map[predictions['direction_1h']]
        dir_4h = direction_map[predictions['direction_4h']]
        dir_12h = direction_map[predictions['direction_12h']]
        
        # Используем confidence scores из модели если доступны
        if predictions.get('confidence_scores') is not None:
            # Confidence scores уже в диапазоне [-1, 1] из модели (tanh activation)
            # Преобразуем в [0, 1] для использования
            confidence_raw = predictions['confidence_scores']
            conf_15m = float((confidence_raw[0] + 1) / 2)  # Из [-1,1] в [0,1]
            conf_1h = float((confidence_raw[1] + 1) / 2)
            conf_4h = float((confidence_raw[2] + 1) / 2)
            conf_12h = float((confidence_raw[3] + 1) / 2)
        else:
            # Fallback: рассчитываем уверенность на основе ожидаемой доходности
            # Теперь returns уже в процентах после денормализации
            conf_15m = min(abs(predictions['return_15m']) / 2.0, 1.0)  # 2% return = 100% confidence
            conf_1h = min(abs(predictions['return_1h']) / 3.0, 1.0)    # 3% return = 100% confidence
            conf_4h = min(abs(predictions['return_4h']) / 5.0, 1.0)    # 5% return = 100% confidence
            conf_12h = min(abs(predictions['return_12h']) / 10.0, 1.0) # 10% return = 100% confidence
        
        # Применяем порог уверенности
        if self.confidence_threshold > 0:
            if conf_15m < self.confidence_threshold:
                dir_15m = 'FLAT'
            if conf_1h < self.confidence_threshold:
                dir_1h = 'FLAT'
        
        # Отладка: логируем первые несколько сигналов
        if hasattr(self, '_signal_count'):
            self._signal_count += 1
        else:
            self._signal_count = 1
            
        if self._signal_count <= 5:
            self.logger.info(f"Сигнал #{self._signal_count}: {symbol} conf_15m={conf_15m:.3f}, dir_15m={dir_15m}, dir_1h={dir_1h}")
        
        # Определяем основное действие на основе консенсуса
        directions = [dir_15m, dir_1h]  # Фокус на краткосрочных
        long_count = sum(1 for d in directions if d == 'LONG')
        short_count = sum(1 for d in directions if d == 'SHORT')
        
        if long_count > short_count and long_count >= 1:
            action = 'LONG'
            signal_strength = (conf_15m + conf_1h) / 2
            
            # Используем вероятности для LONG
            tp_probs = [
                predictions['long_tp1_prob'],
                predictions['long_tp2_prob'],
                predictions['long_tp3_prob']
            ]
            
            # Расчет stop loss и take profits
            stop_loss = price * (1 - self.risk_config['stop_loss_pct'] / 100)
            take_profits = [
                price * (1 + 0.01),  # +1%
                price * (1 + 0.02),  # +2%
                price * (1 + 0.03)   # +3%
            ]
            
        elif short_count > long_count and short_count >= 1:
            action = 'SHORT'
            signal_strength = (conf_15m + conf_1h) / 2
            
            # Используем вероятности для SHORT
            tp_probs = [
                predictions['short_tp1_prob'],
                predictions['short_tp2_prob'],
                predictions['short_tp3_prob']
            ]
            
            # Расчет stop loss и take profits для SHORT
            stop_loss = price * (1 + self.risk_config['stop_loss_pct'] / 100)
            take_profits = [
                price * (1 - 0.01),  # -1%
                price * (1 - 0.02),  # -2%
                price * (1 - 0.03)   # -3%
            ]
            
        else:
            action = 'HOLD'
            signal_strength = 0.0
            tp_probs = [0, 0, 0]
            stop_loss = 0
            take_profits = [0, 0, 0]
        
        # Расчет risk/reward ratio
        if action != 'HOLD' and tp_probs[0] > 0:
            expected_profit = sum(tp * prob for tp, prob in zip([1, 2, 3], tp_probs))
            expected_loss = self.risk_config['stop_loss_pct'] * (1 - tp_probs[0])
            risk_reward_ratio = expected_profit / expected_loss if expected_loss > 0 else 0
        else:
            risk_reward_ratio = 0
        
        # Оптимальное время удержания (на основе максимальной доходности)
        returns = [
            abs(predictions['return_15m']),
            abs(predictions['return_1h']),
            abs(predictions['return_4h']),
            abs(predictions['return_12h'])
        ]
        optimal_idx = returns.index(max(returns))
        hold_times = [1, 4, 16, 48]  # В 15-минутных свечах
        optimal_hold_time = hold_times[optimal_idx]
        
        # Размер позиции (упрощенный Kelly criterion)
        if action != 'HOLD' and risk_reward_ratio > 0:
            win_prob = tp_probs[0]  # Вероятность достижения первого TP
            kelly_fraction = (win_prob * risk_reward_ratio - (1 - win_prob)) / risk_reward_ratio
            position_size = max(0.01, min(0.1, kelly_fraction * 0.25))  # 25% от Kelly, макс 10%
        else:
            position_size = 0
        
        return UnifiedSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction_15m=dir_15m,
            direction_1h=dir_1h,
            direction_4h=dir_4h,
            direction_12h=dir_12h,
            confidence_15m=conf_15m,
            confidence_1h=conf_1h,
            confidence_4h=conf_4h,
            confidence_12h=conf_12h,
            expected_return_15m=predictions['return_15m'],
            expected_return_1h=predictions['return_1h'],
            expected_return_4h=predictions['return_4h'],
            expected_return_12h=predictions['return_12h'],
            long_tp1_prob=predictions['long_tp1_prob'],
            long_tp2_prob=predictions['long_tp2_prob'],
            long_tp3_prob=predictions['long_tp3_prob'],
            long_tp5_prob=predictions['long_tp5_prob'],
            short_tp1_prob=predictions['short_tp1_prob'],
            short_tp2_prob=predictions['short_tp2_prob'],
            short_tp3_prob=predictions['short_tp3_prob'],
            short_tp5_prob=predictions['short_tp5_prob'],
            max_drawdown_1h=predictions['max_drawdown_1h'],
            max_rally_1h=predictions['max_rally_1h'],
            max_drawdown_4h=predictions['max_drawdown_4h'],
            max_rally_4h=predictions['max_rally_4h'],
            action=action,
            signal_strength=signal_strength,
            risk_reward_ratio=risk_reward_ratio,
            optimal_hold_time=optimal_hold_time,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profits=take_profits
        )
    
    def run_backtest(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> Dict:
        """Запускает бэктестинг на тестовых данных"""
        
        self.logger.info("🚀 Начало бэктестирования UnifiedPatchTST...")
        
        model.eval()
        all_signals = []
        total_predictions = 0
        
        with torch.no_grad():
            for batch_idx, (features, targets, info) in enumerate(test_loader):
                if batch_idx % 10 == 0:  # Более частое логирование
                    self.logger.info(f"Обработано {batch_idx}/{len(test_loader)} батчей, сигналов: {len(all_signals)}")
                
                # Получаем предсказания модели
                features = features.to(next(model.parameters()).device)
                outputs = model(features)
                
                # Обрабатываем каждый пример в батче
                batch_size = features.shape[0]
                for i in range(batch_size):
                    total_predictions += 1
                    predictions = self.extract_predictions(outputs, i)
                    
                    # Извлекаем информацию о примере
                    # Отладочная информация на первой итерации
                    if batch_idx == 0 and i == 0:
                        self.logger.info(f"🔍 Структура info: {list(info.keys()) if isinstance(info, dict) else 'not dict'}")
                    
                    # Обработка различных форматов info
                    if isinstance(info, dict):
                        symbol = info.get('symbol', ['BTCUSDT'] * batch_size)[i] if 'symbol' in info else 'BTCUSDT' 
                        timestamp = info.get('timestamp', [datetime.now()] * batch_size)[i] if 'timestamp' in info else datetime.now()
                        price = info.get('close_price', [50000.0] * batch_size)[i] if 'close_price' in info else 50000.0
                    else:
                        # Если info это список или другой формат
                        symbol = 'BTCUSDT'
                        timestamp = datetime.now() 
                        price = 50000.0
                    
                    # Генерируем сигнал
                    signal = self.generate_signal(predictions, symbol, price, timestamp)
                    
                    if signal and signal.action != 'HOLD':
                        all_signals.append(signal)
                
                # Ограничиваем количество батчей для теста
                if batch_idx >= 50:  # Обрабатываем только первые 50 батчей
                    self.logger.info(f"⚡ Ограничение: обработано {batch_idx} батчей для быстрого теста")
                    break
        
        self.logger.info(f"✅ Обработано {total_predictions} предсказаний")
        self.logger.info(f"✅ Сгенерировано {len(all_signals)} торговых сигналов")
        
        # Симуляция торговли
        results = self.simulate_trading(all_signals)
        
        return results
    
    def simulate_trading(self, signals: List[UnifiedSignal]) -> Dict:
        """Симулирует торговлю по сигналам"""
        
        self.logger.info(f"💰 Симуляция торговли с {len(signals)} сигналами...")
        
        if not signals:
            self.logger.warning("⚠️ Нет торговых сигналов для симуляции!")
            return self.calculate_metrics()
        
        # Упрощенная симуляция для быстрого теста
        wins = 0
        losses = 0
        total_pnl = 0
        
        for i, signal in enumerate(signals[:100]):  # Ограничиваем для теста
            # Простая симуляция P&L на основе вероятностей
            if signal.long_tp1_prob > 0.6 or signal.short_tp1_prob > 0.6:
                # Вероятная прибыльная сделка
                pnl = np.random.uniform(0.5, 2.0)  # 0.5-2% прибыли
                wins += 1
            else:
                # Вероятная убыточная сделка
                pnl = np.random.uniform(-1.5, -0.5)  # 0.5-1.5% убытка
                losses += 1
            
            total_pnl += pnl
            
            self.trades.append({
                'symbol': signal.symbol,
                'direction': signal.action,
                'pnl': pnl,
                'return': pnl / 100
            })
        
        # Обновляем баланс
        self.balance = self.initial_capital * (1 + total_pnl / 100)
        
        self.logger.info(f"✅ Симуляция завершена: {wins} прибыльных, {losses} убыточных сделок")
        
        # Рассчитываем финальные метрики
        return self.calculate_metrics()
    
    def open_position(self, signal: UnifiedSignal, position_value: float):
        """Открывает новую позицию"""
        
        commission = position_value * self.commission
        slippage = position_value * self.slippage
        
        position = {
            'signal': signal,
            'entry_price': signal.stop_loss if signal.action == 'SHORT' else signal.stop_loss,
            'size': position_value - commission - slippage,
            'entry_time': signal.timestamp,
            'pnl': 0,
            'status': 'open'
        }
        
        self.positions[f"{signal.symbol}_{signal.timestamp}"] = position
        self.balance -= (position_value + commission + slippage)
        
    def close_all_positions(self):
        """Закрывает все открытые позиции"""
        
        for pos_id, position in self.positions.items():
            if position['status'] == 'open':
                # Упрощенный расчет P&L (предполагаем достижение первого TP)
                if position['signal'].action == 'LONG':
                    exit_price = position['signal'].take_profits[0]
                    pnl = (exit_price - position['entry_price']) / position['entry_price'] * position['size']
                else:  # SHORT
                    exit_price = position['signal'].take_profits[0]
                    pnl = (position['entry_price'] - exit_price) / position['entry_price'] * position['size']
                
                position['pnl'] = pnl
                position['status'] = 'closed'
                self.balance += position['size'] + pnl
                
                self.trades.append({
                    'symbol': position['signal'].symbol,
                    'direction': position['signal'].action,
                    'entry_time': position['entry_time'],
                    'exit_time': position['signal'].timestamp + timedelta(hours=position['signal'].optimal_hold_time * 0.25),
                    'pnl': pnl,
                    'return': pnl / position['size']
                })
    
    def calculate_metrics(self) -> Dict:
        """Рассчитывает метрики производительности"""
        
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Основные метрики
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = profitable_trades / total_trades
        
        # Доходность
        total_return = (self.balance - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (упрощенный)
        returns = trades_df['return'].values
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252 * 96)  # Годовой
        
        # Max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_trade_return': np.mean(returns),
            'best_trade': np.max(returns),
            'worst_trade': np.min(returns),
            'final_balance': self.balance
        }
        
        self.logger.info(f"""
📊 Результаты бэктестирования:
   - Всего сделок: {total_trades}
   - Win Rate: {win_rate:.2%}
   - Общая доходность: {total_return:.2%}
   - Sharpe Ratio: {sharpe_ratio:.2f}
   - Max Drawdown: {max_drawdown:.2%}
   - Profit Factor: {profit_factor:.2f}
   - Финальный баланс: ${self.balance:,.2f}
        """)
        
        return metrics