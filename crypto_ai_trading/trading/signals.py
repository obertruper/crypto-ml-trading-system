"""
Генерация торговых сигналов на основе предсказаний модели
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger

@dataclass
class Signal:
    """Класс торгового сигнала"""
    symbol: str
    side: str  # 'long' или 'short'
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    confidence: float
    signal_strength: float
    timestamp: datetime
    atr_value: float
    volatility: float
    reasoning: str

class SignalGenerator:
    """Генератор торговых сигналов"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config['risk_management']
        self.logger = get_logger("SignalGenerator")
        
        # Пороги для сигналов
        self.min_confidence = 0.6
        self.min_signal_strength = 0.7
        self.volatility_threshold = 0.05
        
        # Уровни take profit из конфигурации
        self.tp_levels = self.risk_config['take_profit_targets']
        self.sl_percentage = self.risk_config['stop_loss_pct'] / 100
        
    def generate_signals(self, 
                        predictions: Dict, 
                        market_data: pd.DataFrame,
                        features: pd.DataFrame) -> List[Signal]:
        """Генерация списка торговых сигналов"""
        
        signals = []
        
        # Обработка предсказаний для каждого символа
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].iloc[-1]
            symbol_features = features[features['symbol'] == symbol].iloc[-1]
            
            # Извлечение предсказаний для символа
            symbol_predictions = self._extract_symbol_predictions(predictions, symbol)
            
            if symbol_predictions is None:
                continue
            
            # Генерация сигнала
            signal = self._generate_symbol_signal(
                symbol, 
                symbol_data, 
                symbol_features, 
                symbol_predictions
            )
            
            if signal:
                signals.append(signal)
        
        # Фильтрация и ранжирование сигналов
        filtered_signals = self._filter_signals(signals)
        ranked_signals = self._rank_signals(filtered_signals)
        
        self.logger.info(f"Сгенерировано {len(ranked_signals)} сигналов из {len(signals)} кандидатов")
        
        return ranked_signals
    
    def _extract_symbol_predictions(self, predictions: Dict, symbol: str) -> Optional[Dict]:
        """Извлечение реальных предсказаний модели для конкретного символа"""
        if not predictions:
            self.logger.warning(f"Получены пустые предсказания")
            return None
            
        if symbol not in predictions:
            self.logger.warning(f"Отсутствуют предсказания для символа {symbol}")
            return None
        
        symbol_predictions = predictions[symbol]
        
        # Валидация структуры предсказаний
        required_keys = ['tp_probs', 'sl_prob', 'volatility', 'price_pred']
        missing_keys = [key for key in required_keys if key not in symbol_predictions]
        
        if missing_keys:
            self.logger.error(f"Отсутствуют ключи в предсказаниях для {symbol}: {missing_keys}")
            return None
        
        # Валидация значений
        tp_probs = symbol_predictions['tp_probs']
        if not isinstance(tp_probs, (list, np.ndarray)) or len(tp_probs) != len(self.tp_levels):
            self.logger.error(f"Неверный формат tp_probs для {symbol}: ожидалось {len(self.tp_levels)} значений")
            return None
        
        # Проверка на валидность вероятностей
        if isinstance(tp_probs, np.ndarray):
            tp_probs = tp_probs.tolist()
        
        for prob in tp_probs:
            if not 0 <= prob <= 1:
                self.logger.error(f"Невалидная вероятность TP для {symbol}: {prob}")
                return None
        
        sl_prob = symbol_predictions['sl_prob']
        if not 0 <= sl_prob <= 1:
            self.logger.error(f"Невалидная вероятность SL для {symbol}: {sl_prob}")
            return None
        
        return symbol_predictions
    
    def _generate_symbol_signal(self, 
                               symbol: str, 
                               market_data: pd.Series, 
                               features: pd.Series,
                               predictions: Dict) -> Optional[Signal]:
        """Генерация сигнала для одного символа"""
        
        current_price = market_data['close']
        atr_value = features.get('atr', current_price * 0.02)
        
        # Анализ предсказаний модели
        tp_probs = predictions['tp_probs']
        sl_prob = predictions['sl_prob']
        predicted_volatility = predictions['volatility']
        price_direction = predictions['price_pred']
        
        # Определение направления сигнала
        if price_direction > 0 and max(tp_probs) > self.min_confidence:
            side = 'long'
            signal_strength = max(tp_probs)
        elif price_direction < 0 and max(tp_probs) > self.min_confidence:
            side = 'short'
            signal_strength = max(tp_probs)
        else:
            return None  # Недостаточно сильный сигнал
        
        # Проверка рыночных условий
        if not self._check_market_conditions(features, predicted_volatility):
            return None
        
        # Расчет уровней входа и выхода
        entry_price = current_price
        stop_loss = self._calculate_stop_loss(entry_price, side, atr_value)
        take_profits = self._calculate_take_profits(entry_price, side, tp_probs)
        
        # Расчет общей уверенности
        confidence = self._calculate_confidence(
            signal_strength, sl_prob, features, predicted_volatility
        )
        
        # Генерация обоснования сигнала
        reasoning = self._generate_reasoning(
            symbol, side, signal_strength, tp_probs, features
        )
        
        signal = Signal(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profits=take_profits,
            confidence=confidence,
            signal_strength=signal_strength,
            timestamp=datetime.now(),
            atr_value=atr_value,
            volatility=predicted_volatility,
            reasoning=reasoning
        )
        
        return signal
    
    def _check_market_conditions(self, features: pd.Series, predicted_volatility: float) -> bool:
        """Проверка рыночных условий для торговли"""
        
        # Проверка волатильности
        if predicted_volatility > self.volatility_threshold:
            return False
        
        # Проверка RSI (избежание перекупленности/перепроданности)
        rsi = features.get('rsi', 50)
        if rsi > 80 or rsi < 20:
            return False
        
        # Проверка объема
        volume_ratio = features.get('volume_ratio', 1)
        if volume_ratio < 0.5:  # Низкий объем
            return False
        
        # Проверка спреда
        hl_spread = features.get('hl_spread', 0.01)
        if hl_spread > 0.05:  # Слишком широкий спред
            return False
        
        # Проверка торговой сессии
        hour = features.get('hour', 12)
        if hour < 6 or hour > 22:  # Избегаем ночных часов
            return False
        
        return True
    
    def _calculate_stop_loss(self, entry_price: float, side: str, atr_value: float) -> float:
        """Расчет уровня stop loss"""
        
        # Базовый stop loss из конфигурации
        base_sl_distance = entry_price * self.sl_percentage
        
        # ATR-основанный stop loss
        atr_sl_distance = atr_value * 2
        
        # Используем больший из двух для лучшей защиты
        sl_distance = max(base_sl_distance, atr_sl_distance)
        
        if side == 'long':
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance
    
    def _calculate_take_profits(self, 
                              entry_price: float, 
                              side: str, 
                              tp_probs: np.ndarray) -> List[float]:
        """Расчет уровней take profit"""
        
        take_profits = []
        
        for i, (tp_pct, prob) in enumerate(zip(self.tp_levels, tp_probs)):
            # Учитываем только уровни с достаточной вероятностью
            if prob > 0.5:
                tp_distance = entry_price * (tp_pct / 100)
                
                if side == 'long':
                    tp_price = entry_price + tp_distance
                else:
                    tp_price = entry_price - tp_distance
                
                take_profits.append(tp_price)
        
        return take_profits
    
    def _calculate_confidence(self, 
                            signal_strength: float, 
                            sl_prob: float, 
                            features: pd.Series,
                            predicted_volatility: float) -> float:
        """Расчет общей уверенности в сигнале"""
        
        # Базовая уверенность от силы сигнала
        base_confidence = signal_strength
        
        # Корректировка на вероятность stop loss
        sl_adjustment = 1 - sl_prob
        
        # Корректировка на рыночные условия
        market_conditions_score = self._score_market_conditions(features)
        
        # Корректировка на волатильность
        volatility_adjustment = 1 - min(predicted_volatility / 0.1, 1)
        
        # Итоговая уверенность
        confidence = (
            base_confidence * 0.4 +
            sl_adjustment * 0.2 +
            market_conditions_score * 0.2 +
            volatility_adjustment * 0.2
        )
        
        return min(confidence, 0.95)  # Максимум 95%
    
    def _score_market_conditions(self, features: pd.Series) -> float:
        """Оценка рыночных условий (0-1)"""
        
        score = 0.5  # Базовая оценка
        
        # RSI в нормальном диапазоне
        rsi = features.get('rsi', 50)
        if 30 < rsi < 70:
            score += 0.1
        
        # Хороший объем
        volume_ratio = features.get('volume_ratio', 1)
        if volume_ratio > 1.2:
            score += 0.1
        
        # Низкий спред
        hl_spread = features.get('hl_spread', 0.01)
        if hl_spread < 0.02:
            score += 0.1
        
        # Активная торговая сессия
        session_overlap = features.get('session_overlap', 0)
        if session_overlap:
            score += 0.1
        
        # Трендовые условия
        adx = features.get('adx', 20)
        if adx > 25:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_reasoning(self, 
                          symbol: str, 
                          side: str, 
                          signal_strength: float,
                          tp_probs: np.ndarray, 
                          features: pd.Series) -> str:
        """Генерация текстового обоснования сигнала"""
        
        reasons = []
        
        # Основной сигнал
        reasons.append(f"Модель предсказывает {side} движение с силой {signal_strength:.2%}")
        
        # Take profit вероятности
        best_tp_idx = np.argmax(tp_probs)
        best_tp_level = self.tp_levels[best_tp_idx]
        reasons.append(f"Наибольшая вероятность достижения TP {best_tp_level}%")
        
        # Технические факторы
        rsi = features.get('rsi', 50)
        if side == 'long' and rsi < 50:
            reasons.append(f"RSI {rsi:.1f} поддерживает покупку")
        elif side == 'short' and rsi > 50:
            reasons.append(f"RSI {rsi:.1f} поддерживает продажу")
        
        # MACD
        macd_diff = features.get('macd_diff', 0)
        if side == 'long' and macd_diff > 0:
            reasons.append("MACD показывает бычий сигнал")
        elif side == 'short' and macd_diff < 0:
            reasons.append("MACD показывает медвежий сигнал")
        
        # Объем
        volume_ratio = features.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            reasons.append(f"Высокий объем ({volume_ratio:.1f}x)")
        
        return "; ".join(reasons)
    
    def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Фильтрация сигналов по качеству"""
        
        filtered = []
        
        for signal in signals:
            # Минимальная уверенность
            if signal.confidence < self.min_confidence:
                continue
            
            # Минимальная сила сигнала
            if signal.signal_strength < self.min_signal_strength:
                continue
            
            # Наличие take profit уровней
            if not signal.take_profits:
                continue
            
            # Адекватность stop loss
            sl_distance_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            if sl_distance_pct > 0.05:  # Больше 5%
                continue
            
            filtered.append(signal)
        
        return filtered
    
    def _rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Ранжирование сигналов по качеству"""
        
        def signal_score(signal: Signal) -> float:
            # Комплексная оценка сигнала
            score = (
                signal.confidence * 0.4 +
                signal.signal_strength * 0.3 +
                len(signal.take_profits) / len(self.tp_levels) * 0.2 +
                (1 - signal.volatility / 0.1) * 0.1
            )
            return score
        
        # Сортировка по убыванию оценки
        ranked = sorted(signals, key=signal_score, reverse=True)
        
        # Логирование топ сигналов
        for i, signal in enumerate(ranked[:5]):
            self.logger.info(
                f"Сигнал #{i+1}: {signal.symbol} {signal.side.upper()} "
                f"@ {signal.entry_price:.4f} (conf: {signal.confidence:.2%}, "
                f"strength: {signal.signal_strength:.2%})"
            )
        
        return ranked
    
    def validate_signal(self, signal: Signal, current_market_data: pd.Series) -> bool:
        """Финальная валидация сигнала перед исполнением"""
        
        current_price = current_market_data['close']
        
        # Проверка изменения цены с момента генерации
        price_change = abs(current_price - signal.entry_price) / signal.entry_price
        if price_change > 0.005:  # Цена изменилась более чем на 0.5%
            self.logger.warning(f"Цена {signal.symbol} изменилась на {price_change:.2%} с момента генерации сигнала")
            return False
        
        # Проверка времени актуальности
        signal_age = (datetime.now() - signal.timestamp).total_seconds()
        if signal_age > 300:  # Более 5 минут
            self.logger.warning(f"Сигнал {signal.symbol} устарел ({signal_age:.0f} сек)")
            return False
        
        # Проверка спреда
        current_spread = (current_market_data['high'] - current_market_data['low']) / current_price
        if current_spread > 0.03:  # Спред больше 3%
            self.logger.warning(f"Слишком широкий спред для {signal.symbol}: {current_spread:.2%}")
            return False
        
        return True
    
    def get_signal_summary(self, signals: List[Signal]) -> Dict:
        """Получение сводки по сигналам"""
        
        if not signals:
            return {'total': 0, 'by_side': {}, 'avg_confidence': 0}
        
        by_side = {}
        total_confidence = 0
        
        for signal in signals:
            if signal.side not in by_side:
                by_side[signal.side] = 0
            by_side[signal.side] += 1
            total_confidence += signal.confidence
        
        return {
            'total': len(signals),
            'by_side': by_side,
            'avg_confidence': total_confidence / len(signals),
            'symbols': [s.symbol for s in signals]
        }