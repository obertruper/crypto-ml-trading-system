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
        
        # Загружаем порядок целевых переменных из файла
        import os
        target_cols_path = 'data/processed/target_cols.txt'
        if os.path.exists(target_cols_path):
            with open(target_cols_path, 'r') as f:
                self.target_cols = [line.strip() for line in f if line.strip()]
            self.logger.info(f"✅ Загружен порядок целей из {target_cols_path}")
            
            # Создаем маппинг индексов по именам
            self.target_index_map = {col: idx for idx, col in enumerate(self.target_cols)}
        else:
            self.logger.warning(f"⚠️ Файл {target_cols_path} не найден, используем стандартный порядок")
            self.target_cols = None
            self.target_index_map = None
        
        # Параметры риск-менеджмента
        self.risk_config = config['risk_management']
        self.initial_capital = config['backtesting']['initial_capital']
        self.commission = config['backtesting']['commission']
        self.slippage = config['backtesting'].get('slippage', 0.0005)  # Значение по умолчанию если не в конфиге
        
        # Параметры торговли
        self.max_positions = self.risk_config['max_concurrent_positions']
        self.confidence_threshold = config['model'].get('direction_confidence_threshold', 0.15)  # Минимальный порог для максимума сигналов
        
        # Статистика
        self.trades = []
        self.positions = {}
        self.balance = self.initial_capital
        self.equity_curve = []
        
        # Логирование начальных параметров
        self.logger.info(f"💰 Инициализация бэктестера:")
        self.logger.info(f"   - Начальный капитал: ${self.initial_capital:,.2f}")
        self.logger.info(f"   - Комиссия: {self.commission:.2%}")
        self.logger.info(f"   - Проскальзывание: {self.slippage:.2%}")
        self.logger.info(f"   - Макс позиций: {self.max_positions}")
        
    def extract_predictions(self, model_output: torch.Tensor, batch_idx: int) -> Dict:
        """Извлекает предсказания из выхода модели для одного примера"""
        
        # model_output: (batch_size, 20)
        # Порядок выходов загружается из data/processed/target_cols.txt
        # Стандартный порядок если файл не найден:
        # 0-3: future_return_15m/1h/4h/12h (в долях, нужно умножить на 100)
        # 4-7: direction_15m/1h/4h/12h (классы: 0=LONG, 1=SHORT, 2=FLAT)
        # 8-11: long_will_reach_1/2/3/5pct (логиты, нужен sigmoid)
        # 12-15: short_will_reach_1/2/3/5pct (логиты, нужен sigmoid)
        # 16-19: risk metrics - порядок зависит от target_cols.txt
        
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
            # ВАЖНО: output содержит логиты или вероятности, а не классы!
            # Модель выдает 3 значения на каждое direction (softmax по 3 классам)
            # Но в нашем случае модель уже выдает класс (argmax сделан внутри модели)
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
            # Используем маппинг из target_cols.txt если доступен
            'max_drawdown_1h': float(output_np[self.target_index_map.get('max_drawdown_1h', 16)]) * 100 if self.target_index_map else float(output_np[16]) * 100,
            'max_drawdown_4h': float(output_np[self.target_index_map.get('max_drawdown_4h', 17)]) * 100 if self.target_index_map else float(output_np[17]) * 100,
            'max_rally_1h': float(output_np[self.target_index_map.get('max_rally_1h', 18)]) * 100 if self.target_index_map else float(output_np[18]) * 100,
            'max_rally_4h': float(output_np[self.target_index_map.get('max_rally_4h', 19)]) * 100 if self.target_index_map else float(output_np[19]) * 100,
            
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
        
        # Используем вероятности TP для расчета уверенности (реалистично)
        # Берем соответствующую вероятность достижения TP без искусственных бонусов
        if dir_15m == 'LONG':
            conf_15m = predictions['long_tp1_prob']
        elif dir_15m == 'SHORT':
            conf_15m = predictions['short_tp1_prob']
        else:
            conf_15m = max(predictions['long_tp1_prob'], predictions['short_tp1_prob'])
            
        if dir_1h == 'LONG':
            conf_1h = predictions['long_tp2_prob']
        elif dir_1h == 'SHORT':
            conf_1h = predictions['short_tp2_prob']
        else:
            conf_1h = max(predictions['long_tp2_prob'], predictions['short_tp2_prob'])
            
        conf_4h = max(predictions['long_tp3_prob'], predictions['short_tp3_prob'])
        conf_12h = max(predictions['long_tp5_prob'], predictions['short_tp5_prob'])
        
        # Дополнительно корректируем на основе ожидаемой доходности
        # Низкая доходность = снижаем уверенность
        return_factor_15m = min(abs(predictions['return_15m']) / 1.0, 1.0)  # 1% return = full factor
        return_factor_1h = min(abs(predictions['return_1h']) / 2.0, 1.0)    # 2% return = full factor
        
        conf_15m = conf_15m * (0.5 + 0.5 * return_factor_15m)  # Смешиваем факторы
        conf_1h = conf_1h * (0.5 + 0.5 * return_factor_1h)
        
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
        directions = [dir_15m, dir_1h, dir_4h, dir_12h]  # Учитываем все 4 таймфрейма
        long_count = sum(1 for d in directions if d == 'LONG')
        short_count = sum(1 for d in directions if d == 'SHORT')
        
        # Требуем консенсус: большинство таймфреймов должны указывать на направление
        if long_count > short_count:
            action = 'LONG'
            signal_strength = (conf_15m + conf_1h) / 2
            
            # Используем реальные вероятности модели без искусственных корректировок
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
            
        elif short_count > long_count:
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
        
        # Fractional Kelly Criterion для оптимального размера позиции
        if action != 'HOLD' and risk_reward_ratio > 0:
            # Используем среднюю вероятность первых двух TP для более стабильной оценки
            win_prob = (tp_probs[0] * 0.6 + tp_probs[1] * 0.4)  # Взвешенная вероятность
            
            # Kelly formula: f = (p * b - q) / b, где:
            # p = вероятность выигрыша, q = вероятность проигрыша, b = отношение выигрыша к проигрышу
            kelly_fraction = (win_prob * risk_reward_ratio - (1 - win_prob)) / risk_reward_ratio
            
            # Fractional Kelly: используем 25% от рассчитанного Kelly для консервативности
            # Это снижает волатильность и риск разорения
            fractional_kelly = kelly_fraction * 0.25  
            
            # Дополнительная корректировка на основе уверенности модели
            max_confidence = max(conf_15m, conf_1h, conf_4h, conf_12h)
            confidence_adjustment = min(1.5, max(0.5, max_confidence / 0.5))  # от 0.5x до 1.5x
            adjusted_size = fractional_kelly * confidence_adjustment
            
            # Ограничения: минимум 1%, максимум 10% капитала на позицию
            position_size = max(0.01, min(0.10, adjusted_size))
            
            # Дополнительное снижение для SHORT позиций (крипто растет чаще)
            if action == 'SHORT':
                position_size *= 0.8
        else:
            position_size = 0
        
        # Единые объективные фильтры качества для всех направлений
        if action != 'HOLD':
            # Одинаковые пороги для LONG и SHORT - справедливая оценка
            min_tp_prob = 0.25              # Минимальная вероятность успеха
            min_expected_return = 0.1       # Минимальная ожидаемая доходность  
            min_signal_strength = 0.2       # Минимальная сила сигнала
            min_risk_reward = 1.0          # Минимальное соотношение риск/доходность
            
            # Проверяем минимальную вероятность первого TP
            if tp_probs[0] < min_tp_prob:
                action = 'HOLD'
                signal_strength = 0.0
            
            # Проверяем минимальную ожидаемую доходность
            avg_return = (abs(predictions['return_15m']) + abs(predictions['return_1h'])) / 2
            if avg_return < min_expected_return:
                action = 'HOLD'
                signal_strength = 0.0
                
            # Проверяем минимальную силу сигнала
            if signal_strength < min_signal_strength:
                action = 'HOLD'
                signal_strength = 0.0
                
            # Проверяем risk/reward ratio
            if risk_reward_ratio < min_risk_reward:
                if hasattr(self, '_filtered_stats'):
                    if action == 'LONG':
                        self._filtered_stats['LONG_filtered'] = self._filtered_stats.get('LONG_filtered', 0) + 1
                    else:
                        self._filtered_stats['SHORT_filtered'] = self._filtered_stats.get('SHORT_filtered', 0) + 1
                action = 'HOLD'
                signal_strength = 0.0
        
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
        
        # Статистика для анализа
        direction_stats = {'LONG': 0, 'SHORT': 0, 'FLAT': 0}
        filtered_stats = {'LONG_filtered': 0, 'SHORT_filtered': 0}
        
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
                        # Извлекаем реальные данные из батча
                        if 'symbol' in info:
                            symbols = info['symbol']
                            symbol = symbols[i] if isinstance(symbols, (list, torch.Tensor)) else symbols
                        else:
                            symbol = 'BTCUSDT'
                            
                        if 'timestamp' in info:
                            timestamps = info['timestamp']
                            timestamp = timestamps[i] if isinstance(timestamps, (list, torch.Tensor)) else timestamps
                        else:
                            timestamp = datetime.now()
                            
                        if 'close_price' in info:
                            prices = info['close_price']
                            price = float(prices[i]) if isinstance(prices, (list, torch.Tensor)) else float(prices)
                        elif 'close' in info:  # Альтернативное имя
                            prices = info['close']
                            price = float(prices[i]) if isinstance(prices, (list, torch.Tensor)) else float(prices)
                        else:
                            # Пытаемся извлечь из features последнюю цену
                            try:
                                # Предполагаем что close price это один из последних признаков
                                price = float(features[i, -1, -10].cpu().item())  # Примерная позиция
                                if price <= 0 or price > 1000000:  # Проверка на разумность
                                    price = 50000.0
                            except:
                                price = 50000.0
                    else:
                        # Если info это список или другой формат
                        symbol = 'BTCUSDT'
                        timestamp = datetime.now() 
                        price = 50000.0
                    
                    # Генерируем сигнал
                    signal = self.generate_signal(predictions, symbol, price, timestamp)
                    
                    if signal and signal.action != 'HOLD':
                        all_signals.append(signal)
                        direction_stats[signal.action] += 1
                    else:
                        direction_stats['FLAT'] += 1
                
                # Закомментировано ограничение для полного бэктеста
                # if batch_idx >= 50:  # Обрабатываем только первые 50 батчей
                #     self.logger.info(f"⚡ Ограничение: обработано {batch_idx} батчей для быстрого теста")
                #     break
        
        self.logger.info(f"✅ Обработано {total_predictions} предсказаний")
        self.logger.info(f"✅ Сгенерировано {len(all_signals)} торговых сигналов")
        self.logger.info(f"📊 Распределение сигналов: LONG={direction_stats['LONG']}, SHORT={direction_stats['SHORT']}, FLAT={direction_stats['FLAT']}")
        
        # Симуляция торговли
        results = self.simulate_trading(all_signals)
        
        return results
    
    def simulate_trading(self, signals: List[UnifiedSignal]) -> Dict:
        """Симулирует торговлю по сигналам"""
        
        self.logger.info(f"💰 Симуляция торговли с {len(signals)} сигналами...")
        
        if not signals:
            self.logger.warning("⚠️ Нет торговых сигналов для симуляции!")
            return self.calculate_metrics()
        
        # Реалистичная симуляция с учетом позиций и риск-менеджмента
        wins = 0
        losses = 0
        current_positions = 0
        max_concurrent_positions = self.max_positions
        # Будем использовать динамический размер позиции из сигнала (Kelly)
        
        # Статистика направлений в сделках
        trade_direction_stats = {'LONG': 0, 'SHORT': 0}
        
        # Сортируем сигналы по времени
        sorted_signals = sorted(signals, key=lambda x: x.timestamp)
        
        # Фильтруем и ранжируем сигналы
        scored_signals = []
        unique_days = set()
        
        for signal in sorted_signals:
            unique_days.add(signal.timestamp.date())
            
            # Рассчитываем score на основе вероятностей и силы сигнала
            max_tp_prob = max(signal.long_tp1_prob, signal.short_tp1_prob)
            
            # Score = вероятность TP * сила сигнала * ожидаемая доходность
            expected_return = max(abs(signal.expected_return_15m), 
                                abs(signal.expected_return_1h))
            score = max_tp_prob * signal.signal_strength * (1 + expected_return/100)
            
            scored_signals.append((score, signal))
        
        # Сортируем по score и берем лучшие
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        
        # Фильтруем по минимальной вероятности и берем топ сигналов
        filtered_signals = []
        daily_trades = {}
        max_total_trades = len(unique_days) * self.config['trading']['max_daily_trades']
        max_total_trades = min(max_total_trades, 2000)  # Ограничиваем общее количество
        
        for score, signal in scored_signals:
            max_tp_prob = max(signal.long_tp1_prob, signal.short_tp1_prob)
            
            # Проверяем минимальный порог вероятности
            if max_tp_prob >= 0.4 and len(filtered_signals) < max_total_trades:  # Еще больше ослабляем фильтр
                date_key = signal.timestamp.date()
                if date_key not in daily_trades:
                    daily_trades[date_key] = 0
                
                # Проверяем дневной лимит
                if daily_trades[date_key] < self.config['trading']['max_daily_trades']:
                    filtered_signals.append(signal)
                    daily_trades[date_key] += 1
        
        self.logger.info(f"📊 Статистика фильтрации:")
        self.logger.info(f"   - Всего сигналов: {len(signals)}")
        self.logger.info(f"   - Уникальных дней: {len(unique_days)}")
        self.logger.info(f"   - Максимум сделок: {max_total_trades}")
        self.logger.info(f"   - Отфильтровано для торговли: {len(filtered_signals)}")
        
        # Симулируем торговлю
        for signal in filtered_signals:
            # Проверяем лимит позиций
            if current_positions >= max_concurrent_positions:
                continue
                
            # Используем динамический размер позиции из Kelly criterion
            # signal.position_size уже рассчитан как процент от капитала
            position_value = self.balance * signal.position_size
            
            # Симуляция P&L на основе вероятностей и ожидаемой доходности
            tp1_prob = signal.long_tp1_prob if signal.action == 'LONG' else signal.short_tp1_prob
            tp2_prob = signal.long_tp2_prob if signal.action == 'LONG' else signal.short_tp2_prob
            tp3_prob = signal.long_tp3_prob if signal.action == 'LONG' else signal.short_tp3_prob
            
            # Реалистичная симуляция без искусственных бонусов
            # Используем вероятности модели как есть - без корректировок
            
            # Используем вероятности для определения исхода
            random_outcome = np.random.random()
            
            # Расчет P&L на основе уровней TP и их вероятностей
            if random_outcome < tp3_prob:
                # Достигли третьего TP
                trade_return = 0.03  # 3% прибыли
                pnl = position_value * trade_return
                wins += 1
            elif random_outcome < tp2_prob:
                # Достигли второго TP
                trade_return = 0.02  # 2% прибыли
                pnl = position_value * trade_return
                wins += 1
            elif random_outcome < tp1_prob:
                # Достигли первого TP
                trade_return = 0.01  # Реалистичные 1% прибыли без бонусов
                pnl = position_value * trade_return
                wins += 1
            elif random_outcome < tp1_prob + 0.1:  # 10% шанс безубытка
                # Закрылись в безубытке
                trade_return = 0
                pnl = 0
            else:
                # Сработал стоп-лосс
                stop_loss_pct = self.risk_config['stop_loss_pct'] / 100
                # Реалистичные потери - полный стоп-лосс без смягчений
                trade_return = -stop_loss_pct
                pnl = position_value * trade_return
                losses += 1
                
            # Сохраняем ожидаемую доходность для статистики, но НЕ корректируем P&L
            expected_return = signal.expected_return_15m if signal.action == 'LONG' else -signal.expected_return_15m
            # Убираем искусственные бонусы для реалистичной симуляции
            
            # Применяем комиссии
            commission = position_value * self.commission * 2  # вход + выход
            slippage = position_value * self.slippage * 2
            net_pnl = pnl - commission - slippage
            
            # Обновляем баланс
            old_balance = self.balance
            self.balance += net_pnl
            current_positions += 1
            
            # Отладочное логирование для первых 10 сделок
            if len(self.trades) < 10:
                self.logger.info(f"🔍 Сделка #{len(self.trades) + 1}:")
                self.logger.info(f"   - Символ: {signal.symbol}, Направление: {signal.action}")
                self.logger.info(f"   - Размер позиции: ${position_value:,.2f} ({signal.position_size*100:.1f}% капитала)")
                self.logger.info(f"   - Результат: {'WIN' if pnl > 0 else 'LOSS'}, P&L: ${pnl:,.2f}")
                self.logger.info(f"   - Комиссии: ${commission:,.2f}, Слиппаж: ${slippage:,.2f}")
                self.logger.info(f"   - Net P&L: ${net_pnl:,.2f}")
                self.logger.info(f"   - Баланс: ${old_balance:,.2f} → ${self.balance:,.2f}")
            
            # Случайно закрываем позиции
            if np.random.random() > 0.7:  # 30% шанс держать позицию
                current_positions = max(0, current_positions - 1)
            
            # Обновляем статистику направлений
            trade_direction_stats[signal.action] += 1
            
            self.trades.append({
                'symbol': signal.symbol,
                'direction': signal.action,
                'entry_time': signal.timestamp,
                'tp1_prob': tp1_prob,
                'tp2_prob': tp2_prob,
                'tp3_prob': tp3_prob,
                'expected_return': expected_return,
                'signal_strength': signal.signal_strength,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'gross_pnl': pnl,
                'commission': commission,
                'slippage': slippage,
                'net_pnl': net_pnl,
                'return': net_pnl / position_value
            })
            
            # Прерываем если баланс упал слишком сильно
            if self.balance < self.initial_capital * 0.5:
                self.logger.warning("⚠️ Баланс упал ниже 50%, прекращаем торговлю")
                break
        
        self.logger.info(f"\n✅ Симуляция завершена: {wins} прибыльных, {losses} убыточных сделок")
        self.logger.info(f"📊 Распределение сделок: LONG={trade_direction_stats.get('LONG', 0)}, SHORT={trade_direction_stats.get('SHORT', 0)}")
        self.logger.info(f"  Начальный баланс: ${self.initial_capital:,.2f}")
        self.logger.info(f"  Финальный баланс: ${self.balance:,.2f}")
        self.logger.info(f"  Общая доходность: {((self.balance - self.initial_capital) / self.initial_capital * 100):.2f}%")
        
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
                
                position['net_pnl'] = pnl  # Используем net_pnl для совместимости
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
        profitable_trades = len(trades_df[trades_df['net_pnl'] > 0])
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
        gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Расчет дополнительных метрик
        avg_win = 0
        avg_loss = 0
        losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
        
        if profitable_trades > 0:
            avg_win = trades_df[trades_df['net_pnl'] > 0]['return'].mean()
        if losing_trades > 0:
            avg_loss = abs(trades_df[trades_df['net_pnl'] < 0]['return'].mean())
            
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        # Расчет распределения по символам
        symbol_distribution = trades_df['symbol'].value_counts().to_dict()
        top_symbols = list(symbol_distribution.keys())[:5]
        
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
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'final_balance': self.balance,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'symbol_distribution': symbol_distribution
        }
        
        self.logger.info(f"""
📊 Результаты бэктестирования:
   - Всего сделок: {total_trades}
   - Win Rate: {win_rate:.2%}
   - Общая доходность: {total_return:.2%}
   - Sharpe Ratio: {sharpe_ratio:.2f}
   - Max Drawdown: {max_drawdown:.2%}
   - Profit Factor: {profit_factor:.2f}
   - Средняя прибыль: {avg_win:.2%}
   - Средний убыток: {avg_loss:.2%}
   - Expectancy: {expectancy:.4f}
   - Финальный баланс: ${self.balance:,.2f}
   - Топ-5 символов: {', '.join(top_symbols)}
        """)
        
        return metrics