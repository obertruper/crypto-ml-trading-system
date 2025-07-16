#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Критерии оценки качества торговых сигналов
Межрыночные корреляции и структурные паттерны
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Состояние рынка"""
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


class SignalQuality(Enum):
    """Качество торгового сигнала"""
    EXCELLENT = 5  # > 90% уверенность
    GOOD = 4      # 70-90% уверенность
    MODERATE = 3  # 50-70% уверенность
    WEAK = 2      # 30-50% уверенность
    POOR = 1      # < 30% уверенность


@dataclass
class SignalQualityCriteria:
    """Критерии оценки качества сигнала"""
    
    # Минимальные требования для каждого типа сигнала
    MIN_INDICATORS_AGREEMENT = 0.6  # 60% индикаторов должны согласовываться
    MIN_CATEGORY_AGREEMENT = 0.5    # 50% категорий должны подтверждать
    MIN_VOLUME_CONFIRMATION = 1.2   # Объем должен быть выше среднего
    MIN_TREND_STRENGTH = 25         # ADX для трендовой торговли
    
    # Фильтры шума
    PRICE_NOISE_THRESHOLD = 0.001   # 0.1% движение считается шумом
    INDICATOR_NOISE_MULTIPLIER = 2  # Умножитель для фильтра шума индикатора
    
    # Временные ограничения
    SIGNAL_VALIDITY_BARS = 4        # Сигнал действителен 4 бара (1 час)
    CONFIRMATION_WINDOW = 2         # Окно подтверждения 2 бара
    
    # Веса для оценки качества
    WEIGHTS = {
        'indicator_agreement': 0.25,
        'category_consensus': 0.20,
        'volume_confirmation': 0.15,
        'market_condition': 0.15,
        'time_alignment': 0.10,
        'risk_reward': 0.15
    }


class SignalQualityEvaluator:
    """Оценщик качества торговых сигналов"""
    
    def __init__(self, criteria: SignalQualityCriteria = None):
        self.criteria = criteria or SignalQualityCriteria()
        
    def evaluate_signal_quality(self, signal_data: Dict) -> Dict:
        """
        Комплексная оценка качества сигнала
        
        Args:
            signal_data: Данные сигнала от TechnicalIndicatorPatterns
            
        Returns:
            Dict с оценкой качества
        """
        quality_scores = {}
        
        # 1. Согласованность индикаторов
        quality_scores['indicator_agreement'] = self._evaluate_indicator_agreement(
            signal_data.get('category_signals', {})
        )
        
        # 2. Консенсус категорий
        quality_scores['category_consensus'] = self._evaluate_category_consensus(
            signal_data.get('category_agreement', 0)
        )
        
        # 3. Подтверждение объемом
        quality_scores['volume_confirmation'] = self._evaluate_volume_confirmation(
            signal_data.get('category_signals', {}).get('volume', {})
        )
        
        # 4. Состояние рынка
        quality_scores['market_condition'] = self._evaluate_market_condition(
            signal_data.get('category_signals', {})
        )
        
        # 5. Временное выравнивание
        quality_scores['time_alignment'] = self._evaluate_time_alignment(
            signal_data.get('category_signals', {}).get('time', {})
        )
        
        # 6. Соотношение риск/прибыль
        quality_scores['risk_reward'] = self._evaluate_risk_reward(
            signal_data.get('expected_return', 0),
            signal_data.get('stop_loss', 1.1)
        )
        
        # Рассчитываем общий балл качества
        total_score = sum(
            score * self.criteria.WEIGHTS.get(criterion, 0)
            for criterion, score in quality_scores.items()
        )
        
        # Определяем уровень качества
        if total_score >= 0.9:
            quality_level = SignalQuality.EXCELLENT
        elif total_score >= 0.7:
            quality_level = SignalQuality.GOOD
        elif total_score >= 0.5:
            quality_level = SignalQuality.MODERATE
        elif total_score >= 0.3:
            quality_level = SignalQuality.WEAK
        else:
            quality_level = SignalQuality.POOR
        
        return {
            'quality_level': quality_level,
            'total_score': total_score,
            'scores': quality_scores,
            'recommendation': self._get_recommendation(quality_level, signal_data),
            'filters_passed': self._check_filters(signal_data, quality_scores)
        }
    
    def _evaluate_indicator_agreement(self, category_signals: Dict) -> float:
        """Оценивает согласованность индикаторов"""
        all_signals = []
        
        for category_data in category_signals.values():
            for signal in category_data.get('signals', []):
                all_signals.append(signal.get('signal', 0))
        
        if not all_signals:
            return 0
        
        # Подсчитываем согласованность
        signal_values = [s.value if hasattr(s, 'value') else s for s in all_signals]
        positive_signals = sum(1 for s in signal_values if s > 0)
        negative_signals = sum(1 for s in signal_values if s < 0)
        
        agreement = max(positive_signals, negative_signals) / len(signal_values)
        
        return min(1.0, agreement / self.criteria.MIN_INDICATORS_AGREEMENT)
    
    def _evaluate_category_consensus(self, category_agreement: float) -> float:
        """Оценивает консенсус между категориями"""
        return min(1.0, category_agreement / self.criteria.MIN_CATEGORY_AGREEMENT)
    
    def _evaluate_volume_confirmation(self, volume_data: Dict) -> float:
        """Оценивает подтверждение объемом"""
        volume_signals = volume_data.get('signals', [])
        
        if not volume_signals:
            return 0.5  # Нейтральная оценка при отсутствии данных
        
        # Ищем volume_ratio
        volume_ratio = None
        for signal in volume_signals:
            if signal.get('indicator') == 'volume_ratio':
                volume_ratio = signal.get('value', 1.0)
                break
        
        if volume_ratio is None:
            return 0.5
        
        # Оцениваем качество объема
        if volume_ratio >= self.criteria.MIN_VOLUME_CONFIRMATION:
            return min(1.0, volume_ratio / 2.0)  # Максимум при 2x объеме
        else:
            return volume_ratio / self.criteria.MIN_VOLUME_CONFIRMATION
    
    def _evaluate_market_condition(self, category_signals: Dict) -> float:
        """Оценивает состояние рынка"""
        # Получаем данные о тренде
        trend_data = category_signals.get('trend', {}).get('signals', [])
        volatility_data = category_signals.get('volatility', {}).get('signals', [])
        
        # Находим ADX
        adx_value = None
        for signal in trend_data:
            if signal.get('indicator') == 'adx_val':
                adx_value = signal.get('value')
                break
        
        # Находим ATR
        atr_value = None
        for signal in volatility_data:
            if signal.get('indicator') == 'atr_norm':
                atr_value = signal.get('value')
                break
        
        # Определяем состояние рынка
        if adx_value and adx_value >= self.criteria.MIN_TREND_STRENGTH:
            if atr_value and atr_value > 0.02:
                market_condition = MarketCondition.VOLATILE
                score = 0.7  # Волатильный тренд - осторожно
            else:
                market_condition = MarketCondition.STRONG_TREND
                score = 1.0  # Идеально для трендовой торговли
        else:
            if atr_value and atr_value > 0.02:
                market_condition = MarketCondition.VOLATILE
                score = 0.3  # Волатильный боковик - опасно
            else:
                market_condition = MarketCondition.RANGING
                score = 0.5  # Боковик - нейтрально
        
        return score
    
    def _evaluate_time_alignment(self, time_data: Dict) -> float:
        """Оценивает временные факторы"""
        time_signals = time_data.get('signals', [])
        
        if not time_signals:
            return 0.7  # Нейтральная оценка
        
        # Проверяем активные часы и дни
        good_time_factors = 0
        total_time_factors = 0
        
        for signal in time_signals:
            if signal.get('indicator') in ['hour', 'day_of_week', 'is_weekend']:
                total_time_factors += 1
                if signal.get('quality', 0) > 0.5:
                    good_time_factors += 1
        
        if total_time_factors > 0:
            return good_time_factors / total_time_factors
        
        return 0.7
    
    def _evaluate_risk_reward(self, expected_return: float, stop_loss: float) -> float:
        """Оценивает соотношение риск/прибыль"""
        if expected_return <= 0:
            return 0
        
        risk_reward_ratio = expected_return / stop_loss
        
        # Идеальное соотношение 3:1 или выше
        if risk_reward_ratio >= 3:
            return 1.0
        elif risk_reward_ratio >= 2:
            return 0.8
        elif risk_reward_ratio >= 1.5:
            return 0.6
        elif risk_reward_ratio >= 1:
            return 0.4
        else:
            return 0.2
    
    def _check_filters(self, signal_data: Dict, quality_scores: Dict) -> Dict[str, bool]:
        """Проверяет прохождение фильтров"""
        filters = {
            'min_agreement': quality_scores.get('indicator_agreement', 0) >= 0.6,
            'volume_confirm': quality_scores.get('volume_confirmation', 0) >= 0.5,
            'market_suitable': quality_scores.get('market_condition', 0) >= 0.5,
            'time_suitable': quality_scores.get('time_alignment', 0) >= 0.5,
            'risk_reward_ok': quality_scores.get('risk_reward', 0) >= 0.4
        }
        
        filters['all_passed'] = all(filters.values())
        
        return filters
    
    def _get_recommendation(self, quality_level: SignalQuality, signal_data: Dict) -> str:
        """Возвращает рекомендацию на основе качества"""
        action = signal_data.get('action', 'hold')
        
        if quality_level == SignalQuality.EXCELLENT:
            if action in ['strong_buy', 'strong_sell']:
                return f"EXECUTE {action.upper()} - Excellent signal quality"
            else:
                return f"CONSIDER {action.upper()} - Very good signal"
        
        elif quality_level == SignalQuality.GOOD:
            if action in ['strong_buy', 'strong_sell']:
                return f"CONSIDER {action.upper()} - Good signal quality"
            else:
                return f"MONITOR for {action} - Good potential"
        
        elif quality_level == SignalQuality.MODERATE:
            return f"WAIT for confirmation - {action} signal needs improvement"
        
        elif quality_level == SignalQuality.WEAK:
            return "AVOID - Signal quality too weak"
        
        else:  # POOR
            return "SKIP - Poor signal quality, high risk"


class StructuralPatternDetector:
    """Детектор структурных паттернов на рынке"""
    
    @staticmethod
    def detect_support_resistance(highs: List[float], lows: List[float], 
                                closes: List[float], window: int = 20) -> Dict:
        """
        Определяет уровни поддержки и сопротивления
        
        Args:
            highs: Список максимумов
            lows: Список минимумов
            closes: Список цен закрытия
            window: Окно для анализа
            
        Returns:
            Dict с уровнями
        """
        if len(highs) < window:
            return {'support': [], 'resistance': []}
        
        # Находим локальные экстремумы
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(highs) - window):
            # Сопротивление - локальный максимум
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance_levels.append({
                    'level': highs[i],
                    'strength': sum(1 for h in highs[i-window:i+window+1] 
                                  if abs(h - highs[i]) / highs[i] < 0.002)
                })
            
            # Поддержка - локальный минимум
            if lows[i] == min(lows[i-window:i+window+1]):
                support_levels.append({
                    'level': lows[i],
                    'strength': sum(1 for l in lows[i-window:i+window+1] 
                                  if abs(l - lows[i]) / lows[i] < 0.002)
                })
        
        # Группируем близкие уровни
        def group_levels(levels, threshold=0.005):
            if not levels:
                return []
            
            grouped = []
            levels_sorted = sorted(levels, key=lambda x: x['level'])
            
            current_group = [levels_sorted[0]]
            
            for level in levels_sorted[1:]:
                if abs(level['level'] - current_group[-1]['level']) / current_group[-1]['level'] < threshold:
                    current_group.append(level)
                else:
                    # Объединяем группу
                    avg_level = np.mean([l['level'] for l in current_group])
                    total_strength = sum(l['strength'] for l in current_group)
                    grouped.append({
                        'level': avg_level,
                        'strength': total_strength,
                        'touches': len(current_group)
                    })
                    current_group = [level]
            
            # Последняя группа
            if current_group:
                avg_level = np.mean([l['level'] for l in current_group])
                total_strength = sum(l['strength'] for l in current_group)
                grouped.append({
                    'level': avg_level,
                    'strength': total_strength,
                    'touches': len(current_group)
                })
            
            return grouped
        
        return {
            'support': group_levels(support_levels),
            'resistance': group_levels(resistance_levels),
            'current_price': closes[-1] if closes else 0
        }
    
    @staticmethod
    def detect_chart_patterns(ohlc_data: pd.DataFrame, min_bars: int = 20) -> List[Dict]:
        """
        Определяет графические паттерны
        
        Args:
            ohlc_data: DataFrame с OHLC данными
            min_bars: Минимум баров для паттерна
            
        Returns:
            List найденных паттернов
        """
        patterns = []
        
        if len(ohlc_data) < min_bars:
            return patterns
        
        # Двойная вершина/дно
        double_top_bottom = StructuralPatternDetector._detect_double_top_bottom(ohlc_data)
        if double_top_bottom:
            patterns.extend(double_top_bottom)
        
        # Треугольники
        triangles = StructuralPatternDetector._detect_triangles(ohlc_data)
        if triangles:
            patterns.extend(triangles)
        
        # Флаги и вымпелы
        flags = StructuralPatternDetector._detect_flags(ohlc_data)
        if flags:
            patterns.extend(flags)
        
        return patterns
    
    @staticmethod
    def _detect_double_top_bottom(df: pd.DataFrame) -> List[Dict]:
        """Определяет паттерны двойная вершина/дно"""
        patterns = []
        window = 10
        
        for i in range(window * 2, len(df) - 5):
            # Двойная вершина
            high1_idx = df['high'].iloc[i-window*2:i-window].idxmax()
            high2_idx = df['high'].iloc[i-window:i].idxmax()
            
            if high1_idx != high2_idx:
                high1 = df.loc[high1_idx, 'high']
                high2 = df.loc[high2_idx, 'high']
                
                # Проверяем, что вершины примерно на одном уровне
                if abs(high1 - high2) / high1 < 0.02:
                    # Находим минимум между вершинами
                    valley_idx = df['low'].iloc[high1_idx:high2_idx].idxmin()
                    valley = df.loc[valley_idx, 'low']
                    
                    # Проверяем глубину коррекции
                    if (high1 - valley) / high1 > 0.03:
                        patterns.append({
                            'type': 'double_top',
                            'start_idx': high1_idx,
                            'end_idx': high2_idx,
                            'resistance': (high1 + high2) / 2,
                            'support': valley,
                            'target': valley - (high1 - valley),
                            'confidence': 0.7
                        })
            
            # Двойное дно (аналогично)
            low1_idx = df['low'].iloc[i-window*2:i-window].idxmin()
            low2_idx = df['low'].iloc[i-window:i].idxmin()
            
            if low1_idx != low2_idx:
                low1 = df.loc[low1_idx, 'low']
                low2 = df.loc[low2_idx, 'low']
                
                if abs(low1 - low2) / low1 < 0.02:
                    peak_idx = df['high'].iloc[low1_idx:low2_idx].idxmax()
                    peak = df.loc[peak_idx, 'high']
                    
                    if (peak - low1) / low1 > 0.03:
                        patterns.append({
                            'type': 'double_bottom',
                            'start_idx': low1_idx,
                            'end_idx': low2_idx,
                            'support': (low1 + low2) / 2,
                            'resistance': peak,
                            'target': peak + (peak - low1),
                            'confidence': 0.7
                        })
        
        return patterns
    
    @staticmethod
    def _detect_triangles(df: pd.DataFrame) -> List[Dict]:
        """Определяет треугольные формации"""
        patterns = []
        min_points = 4
        
        for end_idx in range(min_points * 5, len(df)):
            start_idx = max(0, end_idx - 50)
            
            # Получаем экстремумы
            highs = []
            lows = []
            
            for i in range(start_idx + 2, end_idx - 2):
                if df.iloc[i]['high'] > df.iloc[i-1]['high'] and df.iloc[i]['high'] > df.iloc[i+1]['high']:
                    highs.append((i, df.iloc[i]['high']))
                if df.iloc[i]['low'] < df.iloc[i-1]['low'] and df.iloc[i]['low'] < df.iloc[i+1]['low']:
                    lows.append((i, df.iloc[i]['low']))
            
            if len(highs) >= min_points and len(lows) >= min_points:
                # Проверяем сходящийся треугольник
                high_slope = np.polyfit([h[0] for h in highs[-min_points:]], 
                                      [h[1] for h in highs[-min_points:]], 1)[0]
                low_slope = np.polyfit([l[0] for l in lows[-min_points:]], 
                                     [l[1] for l in lows[-min_points:]], 1)[0]
                
                if high_slope < 0 and low_slope > 0:
                    patterns.append({
                        'type': 'symmetrical_triangle',
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'apex': end_idx + int(abs(highs[-1][1] - lows[-1][1]) / (abs(high_slope) + abs(low_slope))),
                        'upper_line': (highs[0], highs[-1]),
                        'lower_line': (lows[0], lows[-1]),
                        'confidence': 0.6
                    })
                elif high_slope < -0.0001 and abs(low_slope) < 0.0001:
                    patterns.append({
                        'type': 'descending_triangle',
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'support': np.mean([l[1] for l in lows[-min_points:]]),
                        'confidence': 0.65
                    })
                elif abs(high_slope) < 0.0001 and low_slope > 0.0001:
                    patterns.append({
                        'type': 'ascending_triangle',
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'resistance': np.mean([h[1] for h in highs[-min_points:]]),
                        'confidence': 0.65
                    })
        
        return patterns
    
    @staticmethod
    def _detect_flags(df: pd.DataFrame) -> List[Dict]:
        """Определяет флаги и вымпелы"""
        patterns = []
        
        # Параметры для определения импульса и коррекции
        impulse_min_change = 0.03  # 3% минимум для импульса
        correction_max = 0.5       # Коррекция не более 50% импульса
        
        for i in range(20, len(df) - 10):
            # Ищем импульс вверх
            for j in range(i - 20, i - 5):
                impulse_change = (df.iloc[i]['high'] - df.iloc[j]['low']) / df.iloc[j]['low']
                
                if impulse_change > impulse_min_change:
                    # Проверяем коррекцию
                    correction_low = df['low'].iloc[i:min(i+10, len(df))].min()
                    correction_size = (df.iloc[i]['high'] - correction_low) / (df.iloc[i]['high'] - df.iloc[j]['low'])
                    
                    if 0.2 < correction_size < correction_max:
                        patterns.append({
                            'type': 'bull_flag',
                            'impulse_start': j,
                            'impulse_end': i,
                            'flag_start': i,
                            'impulse_size': impulse_change,
                            'correction_size': correction_size,
                            'target': df.iloc[i]['high'] + (df.iloc[i]['high'] - df.iloc[j]['low']) * 0.8,
                            'confidence': 0.6
                        })
                        break
            
            # Ищем импульс вниз (аналогично для медвежьего флага)
            for j in range(i - 20, i - 5):
                impulse_change = (df.iloc[j]['high'] - df.iloc[i]['low']) / df.iloc[j]['high']
                
                if impulse_change > impulse_min_change:
                    correction_high = df['high'].iloc[i:min(i+10, len(df))].max()
                    correction_size = (correction_high - df.iloc[i]['low']) / (df.iloc[j]['high'] - df.iloc[i]['low'])
                    
                    if 0.2 < correction_size < correction_max:
                        patterns.append({
                            'type': 'bear_flag',
                            'impulse_start': j,
                            'impulse_end': i,
                            'flag_start': i,
                            'impulse_size': impulse_change,
                            'correction_size': correction_size,
                            'target': df.iloc[i]['low'] - (df.iloc[j]['high'] - df.iloc[i]['low']) * 0.8,
                            'confidence': 0.6
                        })
                        break
        
        return patterns


class IntermarketCorrelationAnalyzer:
    """Анализатор межрыночных корреляций"""
    
    @staticmethod
    def calculate_correlations(symbol_data: Dict[str, pd.DataFrame], 
                             window: int = 100) -> pd.DataFrame:
        """
        Рассчитывает корреляции между символами
        
        Args:
            symbol_data: Dict с данными символов
            window: Окно для расчета корреляции
            
        Returns:
            DataFrame с матрицей корреляций
        """
        # Собираем цены закрытия
        closes = pd.DataFrame()
        
        for symbol, data in symbol_data.items():
            if 'close' in data.columns:
                closes[symbol] = data['close']
        
        # Рассчитываем доходности
        returns = closes.pct_change()
        
        # Скользящая корреляция
        rolling_corr = returns.rolling(window).corr()
        
        # Последняя корреляционная матрица
        latest_corr = returns.iloc[-window:].corr()
        
        return latest_corr
    
    @staticmethod
    def find_leading_indicators(symbol: str, correlations: pd.DataFrame, 
                              threshold: float = 0.7) -> List[Dict]:
        """
        Находит опережающие индикаторы для символа
        
        Args:
            symbol: Целевой символ
            correlations: Матрица корреляций
            threshold: Порог корреляции
            
        Returns:
            List опережающих индикаторов
        """
        if symbol not in correlations.columns:
            return []
        
        # Получаем корреляции для символа
        symbol_corr = correlations[symbol]
        
        # Находим сильные корреляции
        strong_correlations = []
        
        for other_symbol, corr_value in symbol_corr.items():
            if other_symbol != symbol and abs(corr_value) >= threshold:
                strong_correlations.append({
                    'symbol': other_symbol,
                    'correlation': corr_value,
                    'type': 'positive' if corr_value > 0 else 'negative',
                    'strength': 'very_strong' if abs(corr_value) > 0.9 else 'strong'
                })
        
        return sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    @staticmethod
    def detect_divergences(symbol1_data: pd.DataFrame, symbol2_data: pd.DataFrame,
                         window: int = 20) -> List[Dict]:
        """
        Определяет дивергенции между коррелированными активами
        
        Args:
            symbol1_data: Данные первого символа
            symbol2_data: Данные второго символа  
            window: Окно для анализа
            
        Returns:
            List найденных дивергенций
        """
        divergences = []
        
        if len(symbol1_data) < window or len(symbol2_data) < window:
            return divergences
        
        # Выравниваем данные по времени
        merged = pd.merge(
            symbol1_data[['timestamp', 'close']].rename(columns={'close': 'close1'}),
            symbol2_data[['timestamp', 'close']].rename(columns={'close': 'close2'}),
            on='timestamp'
        )
        
        if len(merged) < window:
            return divergences
        
        # Рассчитываем изменения
        merged['change1'] = merged['close1'].pct_change(window)
        merged['change2'] = merged['close2'].pct_change(window)
        
        # Находим дивергенции
        for i in range(window, len(merged)):
            change1 = merged.iloc[i]['change1']
            change2 = merged.iloc[i]['change2']
            
            # Бычья дивергенция: symbol1 растет, symbol2 падает
            if change1 > 0.02 and change2 < -0.02:
                divergences.append({
                    'type': 'bullish_divergence',
                    'timestamp': merged.iloc[i]['timestamp'],
                    'symbol1_change': change1,
                    'symbol2_change': change2,
                    'strength': abs(change1 - change2)
                })
            
            # Медвежья дивергенция: symbol1 падает, symbol2 растет
            elif change1 < -0.02 and change2 > 0.02:
                divergences.append({
                    'type': 'bearish_divergence',
                    'timestamp': merged.iloc[i]['timestamp'],
                    'symbol1_change': change1,
                    'symbol2_change': change2,
                    'strength': abs(change1 - change2)
                })
        
        return divergences


# Пример использования
if __name__ == "__main__":
    # Инициализация оценщика качества
    evaluator = SignalQualityEvaluator()
    
    # Пример данных сигнала
    sample_signal = {
        'action': 'buy',
        'confidence': 0.75,
        'signal_value': 1.2,
        'category_agreement': 0.8,
        'expected_return': 3.5,
        'stop_loss': 1.1,
        'category_signals': {
            'trend': {
                'signals': [
                    {'indicator': 'adx_val', 'value': 32, 'signal': 1, 'quality': 0.8},
                    {'indicator': 'macd_val', 'value': 0.002, 'signal': 1, 'quality': 0.7}
                ],
                'avg_quality': 0.75,
                'consensus': 1
            },
            'volume': {
                'signals': [
                    {'indicator': 'volume_ratio', 'value': 1.8, 'signal': 1, 'quality': 0.9}
                ],
                'avg_quality': 0.9,
                'consensus': 1
            },
            'volatility': {
                'signals': [
                    {'indicator': 'atr_norm', 'value': 0.015, 'signal': 0, 'quality': 0.6}
                ],
                'avg_quality': 0.6,
                'consensus': 0
            }
        }
    }
    
    # Оценка качества сигнала
    quality_result = evaluator.evaluate_signal_quality(sample_signal)
    
    print(f"Качество сигнала: {quality_result['quality_level'].name}")
    print(f"Общий балл: {quality_result['total_score']:.2f}")
    print(f"Рекомендация: {quality_result['recommendation']}")
    print(f"\nДетали оценки:")
    for criterion, score in quality_result['scores'].items():
        print(f"  {criterion}: {score:.2f}")
    print(f"\nФильтры:")
    for filter_name, passed in quality_result['filters_passed'].items():
        print(f"  {filter_name}: {'✓' if passed else '✗'}")
    
    # Пример определения структурных паттернов
    print("\n" + "="*50)
    print("СТРУКТУРНЫЕ ПАТТЕРНЫ")
    
    # Создаем примерные данные
    sample_ohlc = pd.DataFrame({
        'high': np.random.randn(100).cumsum() + 100 + np.abs(np.random.randn(100)) * 0.5,
        'low': np.random.randn(100).cumsum() + 100 - np.abs(np.random.randn(100)) * 0.5,
        'close': np.random.randn(100).cumsum() + 100
    })
    sample_ohlc['open'] = sample_ohlc['close'].shift(1).fillna(sample_ohlc['close'].iloc[0])
    
    # Определяем уровни поддержки/сопротивления
    sr_levels = StructuralPatternDetector.detect_support_resistance(
        sample_ohlc['high'].tolist(),
        sample_ohlc['low'].tolist(),
        sample_ohlc['close'].tolist()
    )
    
    print(f"\nУровни поддержки: {len(sr_levels['support'])}")
    for level in sr_levels['support'][:3]:
        print(f"  {level['level']:.2f} (сила: {level['strength']}, касаний: {level['touches']})")
    
    print(f"\nУровни сопротивления: {len(sr_levels['resistance'])}")
    for level in sr_levels['resistance'][:3]:
        print(f"  {level['level']:.2f} (сила: {level['strength']}, касаний: {level['touches']})")