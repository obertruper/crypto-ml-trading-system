"""
Улучшенный генератор торговых сигналов для работы с новой архитектурой
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from utils.logger import get_logger


@dataclass
class TradingSignal:
    """Улучшенный класс торгового сигнала"""
    symbol: str
    direction: str  # 'LONG' или 'SHORT'
    entry_price: float
    optimal_entry_time: int  # Количество свечей до оптимального входа
    stop_loss: float
    take_profits: List[float]  # TP1, TP2, TP3
    tp_probabilities: List[float]  # Вероятности достижения каждого TP
    sl_probability: float  # Вероятность достижения SL
    confidence: float  # Общая уверенность в сигнале
    expected_value: float  # Ожидаемая доходность с учетом вероятностей
    timestamp: datetime
    metadata: Dict  # Дополнительная информация


class ImprovedSignalGenerator:
    """Улучшенный генератор торговых сигналов"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("ImprovedSignalGenerator")
        self.risk_config = config['risk_management']
        
        # Параметры фильтрации сигналов
        self.min_confidence = 0.65
        self.min_expected_value = 0.5  # Минимальная ожидаемая доходность в %
        self.max_sl_probability = 0.4  # Максимальная вероятность SL
        self.min_tp1_probability = 0.6  # Минимальная вероятность первого TP
        
        # Уровни TP/SL из конфигурации
        self.tp_levels = self.risk_config['take_profit_targets']  # [1.5, 2.5, 4.0]
        self.sl_percentage = self.risk_config['stop_loss_pct']  # 2.0
        self.partial_close_sizes = self.risk_config['partial_close_sizes']  # [40, 40, 20]
        
    def generate_signals(self, 
                        model_outputs: Dict[str, torch.Tensor],
                        market_data: pd.DataFrame,
                        features: pd.DataFrame) -> List[TradingSignal]:
        """
        Генерация торговых сигналов на основе выходов модели
        
        Args:
            model_outputs: Выходы модели PatchTSTForTrading
            market_data: Текущие рыночные данные
            features: Рассчитанные признаки
            
        Returns:
            Список торговых сигналов
        """
        signals = []
        
        # Преобразуем тензоры в numpy для удобства
        outputs_np = {k: v.detach().cpu().numpy() for k, v in model_outputs.items()}
        
        # Обрабатываем каждый символ
        unique_symbols = market_data['symbol'].unique()
        
        for i, symbol in enumerate(unique_symbols):
            # Текущие данные символа
            symbol_data = market_data[market_data['symbol'] == symbol].iloc[-1]
            symbol_features = features[features['symbol'] == symbol].iloc[-1]
            
            # Извлекаем предсказания для символа
            direction_probs = outputs_np['direction_probs'][i]  # [long, short, neutral]
            
            # Определяем лучшее направление
            best_direction_idx = np.argmax(direction_probs)
            direction_confidence = direction_probs[best_direction_idx]
            
            if best_direction_idx == 2:  # NEUTRAL
                continue  # Пропускаем, если модель рекомендует не торговать
            
            # Генерируем сигнал для выбранного направления
            if best_direction_idx == 0:  # LONG
                signal = self._create_long_signal(
                    symbol, symbol_data, symbol_features,
                    outputs_np, i, direction_confidence
                )
            else:  # SHORT
                signal = self._create_short_signal(
                    symbol, symbol_data, symbol_features,
                    outputs_np, i, direction_confidence
                )
            
            if signal and self._validate_signal(signal):
                signals.append(signal)
        
        # Ранжируем и фильтруем сигналы
        filtered_signals = self._rank_and_filter_signals(signals)
        
        self.logger.info(
            f"Сгенерировано {len(filtered_signals)} сигналов из {len(signals)} кандидатов"
        )
        
        return filtered_signals
    
    def _create_long_signal(self, 
                           symbol: str,
                           market_data: pd.Series,
                           features: pd.Series,
                           outputs: Dict[str, np.ndarray],
                           idx: int,
                           direction_confidence: float) -> Optional[TradingSignal]:
        """Создание LONG сигнала"""
        
        current_price = market_data['close']
        
        # Извлекаем предсказания для LONG
        tp_probs = outputs['long_tp_probs'][idx]  # Вероятности TP1, TP2, TP3
        sl_prob = outputs['long_sl_prob'][idx][0]  # Вероятность SL
        optimal_entry_time = int(outputs['long_entry_time'][idx][0])
        
        # Расчет уровней
        stop_loss = current_price * (1 - self.sl_percentage / 100)
        take_profits = [
            current_price * (1 + tp / 100) for tp in self.tp_levels
        ]
        
        # Расчет ожидаемой доходности
        expected_value = self._calculate_expected_value(
            tp_probs, sl_prob, self.tp_levels, self.sl_percentage,
            self.partial_close_sizes
        )
        
        # Общая уверенность
        confidence = self._calculate_confidence(
            direction_confidence, tp_probs, sl_prob, features
        )
        
        return TradingSignal(
            symbol=symbol,
            direction='LONG',
            entry_price=current_price,
            optimal_entry_time=optimal_entry_time,
            stop_loss=stop_loss,
            take_profits=take_profits,
            tp_probabilities=tp_probs.tolist(),
            sl_probability=sl_prob,
            confidence=confidence,
            expected_value=expected_value,
            timestamp=datetime.now(),
            metadata={
                'atr': features.get('atr', 0),
                'rsi': features.get('rsi', 50),
                'volume_ratio': features.get('volume_ratio', 1),
                'rally_score': features.get('rally_detection_score', 0)
            }
        )
    
    def _create_short_signal(self, 
                            symbol: str,
                            market_data: pd.Series,
                            features: pd.Series,
                            outputs: Dict[str, np.ndarray],
                            idx: int,
                            direction_confidence: float) -> Optional[TradingSignal]:
        """Создание SHORT сигнала"""
        
        current_price = market_data['close']
        
        # Извлекаем предсказания для SHORT
        tp_probs = outputs['short_tp_probs'][idx]
        sl_prob = outputs['short_sl_prob'][idx][0]
        optimal_entry_time = int(outputs['short_entry_time'][idx][0])
        
        # Расчет уровней (для SHORT направления обратные)
        stop_loss = current_price * (1 + self.sl_percentage / 100)
        take_profits = [
            current_price * (1 - tp / 100) for tp in self.tp_levels
        ]
        
        # Расчет ожидаемой доходности
        expected_value = self._calculate_expected_value(
            tp_probs, sl_prob, self.tp_levels, self.sl_percentage,
            self.partial_close_sizes
        )
        
        # Общая уверенность
        confidence = self._calculate_confidence(
            direction_confidence, tp_probs, sl_prob, features
        )
        
        return TradingSignal(
            symbol=symbol,
            direction='SHORT',
            entry_price=current_price,
            optimal_entry_time=optimal_entry_time,
            stop_loss=stop_loss,
            take_profits=take_profits,
            tp_probabilities=tp_probs.tolist(),
            sl_probability=sl_prob,
            confidence=confidence,
            expected_value=expected_value,
            timestamp=datetime.now(),
            metadata={
                'atr': features.get('atr', 0),
                'rsi': features.get('rsi', 50),
                'volume_ratio': features.get('volume_ratio', 1),
                'rally_score': features.get('rally_detection_score', 0)
            }
        )
    
    def _calculate_expected_value(self,
                                 tp_probs: np.ndarray,
                                 sl_prob: float,
                                 tp_levels: List[float],
                                 sl_level: float,
                                 partial_sizes: List[float]) -> float:
        """
        Расчет математического ожидания с учетом частичных закрытий
        
        Returns:
            Ожидаемая доходность в процентах
        """
        # Нормализуем размеры частичных закрытий
        partial_weights = np.array(partial_sizes) / 100.0
        
        # Расчет прибыли от каждого TP с учетом частичных закрытий
        tp_returns = 0.0
        remaining_position = 1.0
        
        for i, (tp_prob, tp_level, partial_weight) in enumerate(
            zip(tp_probs, tp_levels, partial_weights)
        ):
            # Прибыль от частичного закрытия на этом уровне
            tp_returns += tp_prob * tp_level * partial_weight * remaining_position
            remaining_position -= partial_weight * remaining_position
        
        # Убыток от SL (на оставшейся позиции)
        sl_return = -sl_prob * sl_level
        
        # Общее математическое ожидание
        expected_value = tp_returns + sl_return
        
        return expected_value
    
    def _calculate_confidence(self,
                            direction_confidence: float,
                            tp_probs: np.ndarray,
                            sl_prob: float,
                            features: pd.Series) -> float:
        """
        Расчет общей уверенности в сигнале
        """
        # Базовая уверенность от модели
        base_confidence = direction_confidence
        
        # Корректировка на основе вероятностей TP/SL
        tp_confidence = np.max(tp_probs) * 0.7 + np.mean(tp_probs) * 0.3
        sl_adjustment = 1 - sl_prob
        
        # Корректировка на основе технических индикаторов
        technical_score = self._score_technical_indicators(features)
        
        # Корректировка на основе качества сигнала
        signal_quality = features.get('signal_quality_score', 0.5)
        
        # Взвешенная уверенность
        confidence = (
            base_confidence * 0.3 +
            tp_confidence * 0.25 +
            sl_adjustment * 0.2 +
            technical_score * 0.15 +
            signal_quality * 0.1
        )
        
        return min(confidence, 0.95)  # Ограничиваем максимум 95%
    
    def _score_technical_indicators(self, features: pd.Series) -> float:
        """Оценка технических индикаторов"""
        score = 0.5
        
        # RSI
        rsi = features.get('rsi', 50)
        if 30 < rsi < 70:
            score += 0.1
        
        # ADX (сила тренда)
        adx = features.get('adx', 20)
        if adx > 25:
            score += 0.1
        
        # Объем
        volume_ratio = features.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            score += 0.1
        
        # Волатильность
        volatility_regime = features.get('volatility_regime', 'normal')
        if volatility_regime == 'normal':
            score += 0.1
        
        # Rally detection
        rally_score = features.get('rally_detection_score', 0)
        score += rally_score * 0.1
        
        return min(score, 1.0)
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Валидация сигнала по критериям"""
        
        # Минимальная уверенность
        if signal.confidence < self.min_confidence:
            return False
        
        # Минимальная ожидаемая доходность
        if signal.expected_value < self.min_expected_value:
            return False
        
        # Максимальная вероятность SL
        if signal.sl_probability > self.max_sl_probability:
            return False
        
        # Минимальная вероятность первого TP
        if signal.tp_probabilities[0] < self.min_tp1_probability:
            return False
        
        # Проверка разумности уровней
        price_change_to_sl = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        if price_change_to_sl > 0.1:  # SL больше 10%
            return False
        
        return True
    
    def _rank_and_filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Ранжирование и фильтрация сигналов"""
        
        if not signals:
            return []
        
        # Сортируем по комбинации ожидаемой доходности и уверенности
        def signal_score(signal: TradingSignal) -> float:
            return (
                signal.expected_value * 0.6 +  # Больший вес на доходность
                signal.confidence * 10 * 0.4    # Уверенность (умножаем на 10 для масштаба)
            )
        
        ranked_signals = sorted(signals, key=signal_score, reverse=True)
        
        # Ограничиваем количество сигналов
        max_signals = self.config['trading'].get('max_concurrent_positions', 10)
        filtered_signals = ranked_signals[:max_signals]
        
        # Логируем топ сигналы
        for i, signal in enumerate(filtered_signals[:5]):
            self.logger.info(
                f"Сигнал #{i+1}: {signal.symbol} {signal.direction} "
                f"EV: {signal.expected_value:.2f}% Conf: {signal.confidence:.2%} "
                f"Entry in: {signal.optimal_entry_time} candles"
            )
        
        return filtered_signals
    
    def format_signal_for_execution(self, signal: TradingSignal) -> Dict:
        """Форматирование сигнала для исполнения"""
        
        return {
            'symbol': signal.symbol,
            'side': signal.direction,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profits': signal.take_profits,
            'position_size': None,  # Будет рассчитан position sizer'ом
            'metadata': {
                'confidence': signal.confidence,
                'expected_value': signal.expected_value,
                'tp_probabilities': signal.tp_probabilities,
                'sl_probability': signal.sl_probability,
                'optimal_entry_time': signal.optimal_entry_time,
                'timestamp': signal.timestamp.isoformat()
            }
        }