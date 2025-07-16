#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Детальные паттерны оценки для технических индикаторов
Определяет оптимальные диапазоны, сигналы и комбинации для всех 49 индикаторов
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SignalStrength(Enum):
    """Сила торгового сигнала"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class IndicatorPattern:
    """Паттерн для оценки индикатора"""
    name: str
    category: str
    optimal_range: Tuple[float, float]
    buy_thresholds: Dict[str, float]
    sell_thresholds: Dict[str, float]
    noise_filter: float
    relevance_window: int  # количество баров
    weight: float  # вес в общей оценке
    combinations: List[str]  # список индикаторов для комбинирования


class TechnicalIndicatorPatterns:
    """Класс с паттернами оценки для всех технических индикаторов"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, IndicatorPattern]:
        """Инициализирует все паттерны индикаторов"""
        patterns = {}
        
        # ==========================================
        # 1. ТРЕНДОВЫЕ ИНДИКАТОРЫ (Trend Indicators)
        # ==========================================
        
        # EMA 15
        patterns['ema_15'] = IndicatorPattern(
            name='EMA 15',
            category='trend',
            optimal_range=(0.95, 1.05),  # относительно цены
            buy_thresholds={
                'price_above': 1.002,  # цена выше EMA на 0.2%
                'slope_positive': 0.0001  # положительный наклон
            },
            sell_thresholds={
                'price_below': 0.998,  # цена ниже EMA на 0.2%
                'slope_negative': -0.0001  # отрицательный наклон
            },
            noise_filter=0.001,  # игнорировать движения < 0.1%
            relevance_window=20,  # 5 часов для 15м
            weight=0.8,
            combinations=['adx_val', 'macd_val', 'volume_ratio']
        )
        
        # ADX (Average Directional Index)
        patterns['adx_val'] = IndicatorPattern(
            name='ADX',
            category='trend',
            optimal_range=(25, 60),  # сильный тренд
            buy_thresholds={
                'adx_min': 25,  # минимум для трендовой торговли
                'adx_rising': True,  # ADX растет
                'di_diff_positive': 5  # +DI выше -DI на 5
            },
            sell_thresholds={
                'adx_min': 25,
                'adx_rising': True,
                'di_diff_negative': -5  # -DI выше +DI на 5
            },
            noise_filter=2,  # изменения < 2 пунктов игнорируются
            relevance_window=16,  # 4 часа
            weight=0.9,
            combinations=['ema_15', 'macd_val', 'rsi_val']
        )
        
        # ADX +DI
        patterns['adx_plus_di'] = IndicatorPattern(
            name='ADX +DI',
            category='trend',
            optimal_range=(20, 40),
            buy_thresholds={
                'di_min': 20,
                'above_minus_di': True,
                'di_rising': True
            },
            sell_thresholds={
                'di_max': 15,
                'below_minus_di': True
            },
            noise_filter=1.5,
            relevance_window=16,
            weight=0.7,
            combinations=['adx_minus_di', 'adx_val']
        )
        
        # ADX -DI
        patterns['adx_minus_di'] = IndicatorPattern(
            name='ADX -DI',
            category='trend',
            optimal_range=(20, 40),
            buy_thresholds={
                'di_max': 15,
                'below_plus_di': True
            },
            sell_thresholds={
                'di_min': 20,
                'above_plus_di': True,
                'di_rising': True
            },
            noise_filter=1.5,
            relevance_window=16,
            weight=0.7,
            combinations=['adx_plus_di', 'adx_val']
        )
        
        # MACD Value
        patterns['macd_val'] = IndicatorPattern(
            name='MACD',
            category='trend',
            optimal_range=(-0.002, 0.002),  # относительно цены
            buy_thresholds={
                'macd_positive': 0,
                'above_signal': True,
                'histogram_positive': 0.0001
            },
            sell_thresholds={
                'macd_negative': 0,
                'below_signal': True,
                'histogram_negative': -0.0001
            },
            noise_filter=0.00005,
            relevance_window=24,  # 6 часов
            weight=0.85,
            combinations=['macd_signal', 'macd_hist', 'rsi_val']
        )
        
        # MACD Signal
        patterns['macd_signal'] = IndicatorPattern(
            name='MACD Signal',
            category='trend',
            optimal_range=(-0.002, 0.002),
            buy_thresholds={
                'below_macd': True,
                'signal_rising': True
            },
            sell_thresholds={
                'above_macd': True,
                'signal_falling': True
            },
            noise_filter=0.00005,
            relevance_window=24,
            weight=0.7,
            combinations=['macd_val', 'macd_hist']
        )
        
        # MACD Histogram
        patterns['macd_hist'] = IndicatorPattern(
            name='MACD Histogram',
            category='trend',
            optimal_range=(-0.001, 0.001),
            buy_thresholds={
                'histogram_positive': 0,
                'histogram_rising': True,
                'min_value': 0.00005
            },
            sell_thresholds={
                'histogram_negative': 0,
                'histogram_falling': True,
                'max_value': -0.00005
            },
            noise_filter=0.00002,
            relevance_window=20,
            weight=0.75,
            combinations=['macd_val', 'macd_signal', 'volume_ratio']
        )
        
        # CCI (Commodity Channel Index)
        patterns['cci_val'] = IndicatorPattern(
            name='CCI',
            category='trend',
            optimal_range=(-100, 100),
            buy_thresholds={
                'oversold_exit': -100,  # выход из перепроданности
                'bullish_cross': 0,
                'strong_buy': 100
            },
            sell_thresholds={
                'overbought_exit': 100,  # выход из перекупленности
                'bearish_cross': 0,
                'strong_sell': -100
            },
            noise_filter=10,
            relevance_window=20,
            weight=0.65,
            combinations=['rsi_val', 'stoch_k', 'williams_r']
        )
        
        # Ichimoku Conversion Line
        patterns['ichimoku_conv'] = IndicatorPattern(
            name='Ichimoku Conversion',
            category='trend',
            optimal_range=(0.98, 1.02),  # относительно цены
            buy_thresholds={
                'above_base': True,
                'price_above': True,
                'conv_rising': True
            },
            sell_thresholds={
                'below_base': True,
                'price_below': True,
                'conv_falling': True
            },
            noise_filter=0.001,
            relevance_window=26,
            weight=0.7,
            combinations=['ichimoku_base', 'ema_15']
        )
        
        # Ichimoku Base Line
        patterns['ichimoku_base'] = IndicatorPattern(
            name='Ichimoku Base',
            category='trend',
            optimal_range=(0.98, 1.02),
            buy_thresholds={
                'price_above': True,
                'base_rising': True,
                'conv_above': True
            },
            sell_thresholds={
                'price_below': True,
                'base_falling': True,
                'conv_below': True
            },
            noise_filter=0.001,
            relevance_window=52,
            weight=0.75,
            combinations=['ichimoku_conv', 'adx_val']
        )
        
        # Parabolic SAR
        patterns['sar'] = IndicatorPattern(
            name='Parabolic SAR',
            category='trend',
            optimal_range=(0.97, 1.03),
            buy_thresholds={
                'sar_below_price': True,
                'sar_flip_up': True  # SAR перешел под цену
            },
            sell_thresholds={
                'sar_above_price': True,
                'sar_flip_down': True  # SAR перешел над ценой
            },
            noise_filter=0.002,
            relevance_window=12,
            weight=0.8,
            combinations=['adx_val', 'ema_15']
        )
        
        # Aroon Up
        patterns['aroon_up'] = IndicatorPattern(
            name='Aroon Up',
            category='trend',
            optimal_range=(70, 100),
            buy_thresholds={
                'aroon_strong': 70,
                'above_down': True,
                'cross_up': 50
            },
            sell_thresholds={
                'aroon_weak': 30,
                'below_down': True
            },
            noise_filter=5,
            relevance_window=14,
            weight=0.65,
            combinations=['aroon_down', 'adx_val']
        )
        
        # Aroon Down
        patterns['aroon_down'] = IndicatorPattern(
            name='Aroon Down',
            category='trend',
            optimal_range=(70, 100),
            buy_thresholds={
                'aroon_weak': 30,
                'below_up': True
            },
            sell_thresholds={
                'aroon_strong': 70,
                'above_up': True,
                'cross_up': 50
            },
            noise_filter=5,
            relevance_window=14,
            weight=0.65,
            combinations=['aroon_up', 'adx_val']
        )
        
        # DPO (Detrended Price Oscillator)
        patterns['dpo'] = IndicatorPattern(
            name='DPO',
            category='trend',
            optimal_range=(-0.01, 0.01),
            buy_thresholds={
                'dpo_positive': 0,
                'dpo_rising': True,
                'min_value': 0.002
            },
            sell_thresholds={
                'dpo_negative': 0,
                'dpo_falling': True,
                'max_value': -0.002
            },
            noise_filter=0.001,
            relevance_window=20,
            weight=0.6,
            combinations=['roc', 'macd_val']
        )
        
        # Vortex Positive
        patterns['vortex_vip'] = IndicatorPattern(
            name='Vortex VI+',
            category='trend',
            optimal_range=(0.9, 1.3),
            buy_thresholds={
                'vip_min': 1.0,
                'above_vin': True,
                'vip_rising': True
            },
            sell_thresholds={
                'vip_max': 0.9,
                'below_vin': True
            },
            noise_filter=0.02,
            relevance_window=14,
            weight=0.7,
            combinations=['vortex_vin', 'adx_val']
        )
        
        # Vortex Negative
        patterns['vortex_vin'] = IndicatorPattern(
            name='Vortex VI-',
            category='trend',
            optimal_range=(0.9, 1.3),
            buy_thresholds={
                'vin_max': 0.9,
                'below_vip': True
            },
            sell_thresholds={
                'vin_min': 1.0,
                'above_vip': True,
                'vin_rising': True
            },
            noise_filter=0.02,
            relevance_window=14,
            weight=0.7,
            combinations=['vortex_vip', 'adx_val']
        )
        
        # ==========================================
        # 2. ОСЦИЛЛЯТОРЫ (Oscillators)
        # ==========================================
        
        # RSI
        patterns['rsi_val'] = IndicatorPattern(
            name='RSI',
            category='oscillator',
            optimal_range=(30, 70),
            buy_thresholds={
                'oversold': 30,
                'oversold_extreme': 20,
                'bullish_divergence': True,
                'rsi_rising': True
            },
            sell_thresholds={
                'overbought': 70,
                'overbought_extreme': 80,
                'bearish_divergence': True,
                'rsi_falling': True
            },
            noise_filter=2,
            relevance_window=14,
            weight=0.85,
            combinations=['stoch_k', 'macd_hist', 'volume_ratio']
        )
        
        # Stochastic %K
        patterns['stoch_k'] = IndicatorPattern(
            name='Stochastic K',
            category='oscillator',
            optimal_range=(20, 80),
            buy_thresholds={
                'oversold': 20,
                'k_above_d': True,
                'k_rising': True
            },
            sell_thresholds={
                'overbought': 80,
                'k_below_d': True,
                'k_falling': True
            },
            noise_filter=3,
            relevance_window=14,
            weight=0.75,
            combinations=['stoch_d', 'rsi_val', 'williams_r']
        )
        
        # Stochastic %D
        patterns['stoch_d'] = IndicatorPattern(
            name='Stochastic D',
            category='oscillator',
            optimal_range=(20, 80),
            buy_thresholds={
                'oversold': 20,
                'd_rising': True
            },
            sell_thresholds={
                'overbought': 80,
                'd_falling': True
            },
            noise_filter=3,
            relevance_window=14,
            weight=0.7,
            combinations=['stoch_k', 'rsi_val']
        )
        
        # Williams %R
        patterns['williams_r'] = IndicatorPattern(
            name='Williams %R',
            category='oscillator',
            optimal_range=(-80, -20),
            buy_thresholds={
                'oversold': -80,
                'oversold_extreme': -95,
                'w_rising': True
            },
            sell_thresholds={
                'overbought': -20,
                'overbought_extreme': -5,
                'w_falling': True
            },
            noise_filter=3,
            relevance_window=14,
            weight=0.7,
            combinations=['rsi_val', 'stoch_k', 'cci_val']
        )
        
        # ROC (Rate of Change)
        patterns['roc'] = IndicatorPattern(
            name='ROC',
            category='oscillator',
            optimal_range=(-5, 5),
            buy_thresholds={
                'roc_positive': 0,
                'roc_rising': True,
                'min_value': 0.5
            },
            sell_thresholds={
                'roc_negative': 0,
                'roc_falling': True,
                'max_value': -0.5
            },
            noise_filter=0.2,
            relevance_window=12,
            weight=0.65,
            combinations=['macd_val', 'dpo']
        )
        
        # Ultimate Oscillator
        patterns['ult_osc'] = IndicatorPattern(
            name='Ultimate Oscillator',
            category='oscillator',
            optimal_range=(30, 70),
            buy_thresholds={
                'oversold': 30,
                'bullish_divergence': True,
                'osc_rising': True
            },
            sell_thresholds={
                'overbought': 70,
                'bearish_divergence': True,
                'osc_falling': True
            },
            noise_filter=2,
            relevance_window=28,
            weight=0.7,
            combinations=['rsi_val', 'stoch_k', 'williams_r']
        )
        
        # ==========================================
        # 3. ИНДИКАТОРЫ ВОЛАТИЛЬНОСТИ (Volatility)
        # ==========================================
        
        # ATR
        patterns['atr_val'] = IndicatorPattern(
            name='ATR',
            category='volatility',
            optimal_range=(0.001, 0.03),  # 0.1% - 3% от цены
            buy_thresholds={
                'atr_expanding': True,
                'min_atr': 0.002,  # минимальная волатильность
                'max_atr': 0.025   # не торговать при экстремальной волатильности
            },
            sell_thresholds={
                'atr_expanding': True,
                'min_atr': 0.002,
                'max_atr': 0.025
            },
            noise_filter=0.0001,
            relevance_window=10,
            weight=0.8,
            combinations=['bb_upper', 'bb_lower', 'volume_ratio']
        )
        
        # Bollinger Upper Band
        patterns['bb_upper'] = IndicatorPattern(
            name='Bollinger Upper',
            category='volatility',
            optimal_range=(1.01, 1.04),  # 1-4% выше цены
            buy_thresholds={
                'price_below_upper': True,
                'band_expanding': True
            },
            sell_thresholds={
                'price_at_upper': 0.998,  # цена около верхней полосы
                'band_contracting': True
            },
            noise_filter=0.001,
            relevance_window=20,
            weight=0.75,
            combinations=['bb_lower', 'bb_basis', 'rsi_val']
        )
        
        # Bollinger Lower Band
        patterns['bb_lower'] = IndicatorPattern(
            name='Bollinger Lower',
            category='volatility',
            optimal_range=(0.96, 0.99),  # 1-4% ниже цены
            buy_thresholds={
                'price_at_lower': 1.002,  # цена около нижней полосы
                'band_contracting': True
            },
            sell_thresholds={
                'price_above_lower': True,
                'band_expanding': True
            },
            noise_filter=0.001,
            relevance_window=20,
            weight=0.75,
            combinations=['bb_upper', 'bb_basis', 'rsi_val']
        )
        
        # Bollinger Basis (Middle Band)
        patterns['bb_basis'] = IndicatorPattern(
            name='Bollinger Basis',
            category='volatility',
            optimal_range=(0.99, 1.01),
            buy_thresholds={
                'price_above_basis': True,
                'basis_rising': True
            },
            sell_thresholds={
                'price_below_basis': True,
                'basis_falling': True
            },
            noise_filter=0.0005,
            relevance_window=20,
            weight=0.7,
            combinations=['bb_upper', 'bb_lower', 'ema_15']
        )
        
        # Donchian Upper Channel
        patterns['donchian_upper'] = IndicatorPattern(
            name='Donchian Upper',
            category='volatility',
            optimal_range=(1.005, 1.03),
            buy_thresholds={
                'price_break_upper': True,
                'channel_expanding': True
            },
            sell_thresholds={
                'price_at_upper': 0.998,
                'channel_stable': True
            },
            noise_filter=0.001,
            relevance_window=20,
            weight=0.7,
            combinations=['donchian_lower', 'atr_val']
        )
        
        # Donchian Lower Channel
        patterns['donchian_lower'] = IndicatorPattern(
            name='Donchian Lower',
            category='volatility',
            optimal_range=(0.97, 0.995),
            buy_thresholds={
                'price_at_lower': 1.002,
                'channel_stable': True
            },
            sell_thresholds={
                'price_break_lower': True,
                'channel_expanding': True
            },
            noise_filter=0.001,
            relevance_window=20,
            weight=0.7,
            combinations=['donchian_upper', 'atr_val']
        )
        
        # ==========================================
        # 4. ОБЪЕМНЫЕ ИНДИКАТОРЫ (Volume)
        # ==========================================
        
        # OBV (On Balance Volume)
        patterns['obv'] = IndicatorPattern(
            name='OBV',
            category='volume',
            optimal_range=(-np.inf, np.inf),  # нет фиксированного диапазона
            buy_thresholds={
                'obv_rising': True,
                'obv_divergence_bullish': True,
                'obv_acceleration': True
            },
            sell_thresholds={
                'obv_falling': True,
                'obv_divergence_bearish': True,
                'obv_deceleration': True
            },
            noise_filter=1000,  # зависит от объемов
            relevance_window=20,
            weight=0.8,
            combinations=['cmf', 'mfi', 'volume_ratio']
        )
        
        # CMF (Chaikin Money Flow)
        patterns['cmf'] = IndicatorPattern(
            name='CMF',
            category='volume',
            optimal_range=(-0.25, 0.25),
            buy_thresholds={
                'cmf_positive': 0.05,
                'cmf_strong': 0.15,
                'cmf_rising': True
            },
            sell_thresholds={
                'cmf_negative': -0.05,
                'cmf_weak': -0.15,
                'cmf_falling': True
            },
            noise_filter=0.02,
            relevance_window=20,
            weight=0.75,
            combinations=['obv', 'mfi', 'volume_ratio']
        )
        
        # MFI (Money Flow Index)
        patterns['mfi'] = IndicatorPattern(
            name='MFI',
            category='volume',
            optimal_range=(20, 80),
            buy_thresholds={
                'oversold': 20,
                'mfi_rising': True,
                'volume_confirm': True
            },
            sell_thresholds={
                'overbought': 80,
                'mfi_falling': True,
                'volume_confirm': True
            },
            noise_filter=3,
            relevance_window=14,
            weight=0.8,
            combinations=['rsi_val', 'cmf', 'volume_ratio']
        )
        
        # ==========================================
        # 5. ПРОИЗВОДНЫЕ ИНДИКАТОРЫ (Derived)
        # ==========================================
        
        # MACD Signal Ratio
        patterns['macd_signal_ratio'] = IndicatorPattern(
            name='MACD Signal Ratio',
            category='derived',
            optimal_range=(0.8, 1.2),
            buy_thresholds={
                'ratio_above_1': 1.0,
                'ratio_rising': True
            },
            sell_thresholds={
                'ratio_below_1': 1.0,
                'ratio_falling': True
            },
            noise_filter=0.02,
            relevance_window=20,
            weight=0.65,
            combinations=['macd_val', 'macd_signal']
        )
        
        # ADX Difference (+DI - -DI)
        patterns['adx_diff'] = IndicatorPattern(
            name='ADX Difference',
            category='derived',
            optimal_range=(-20, 20),
            buy_thresholds={
                'diff_positive': 5,
                'diff_rising': True
            },
            sell_thresholds={
                'diff_negative': -5,
                'diff_falling': True
            },
            noise_filter=2,
            relevance_window=14,
            weight=0.7,
            combinations=['adx_val', 'adx_plus_di', 'adx_minus_di']
        )
        
        # Bollinger Position
        patterns['bb_position'] = IndicatorPattern(
            name='BB Position',
            category='derived',
            optimal_range=(0.2, 0.8),
            buy_thresholds={
                'position_low': 0.3,  # ближе к нижней полосе
                'position_rising': True
            },
            sell_thresholds={
                'position_high': 0.7,  # ближе к верхней полосе
                'position_falling': True
            },
            noise_filter=0.05,
            relevance_window=20,
            weight=0.7,
            combinations=['bb_upper', 'bb_lower', 'rsi_val']
        )
        
        # RSI Distance from Middle
        patterns['rsi_dist_from_mid'] = IndicatorPattern(
            name='RSI Distance',
            category='derived',
            optimal_range=(0, 0.6),
            buy_thresholds={
                'oversold_distance': 0.4,  # RSI далеко от 50 в зоне перепроданности
                'distance_decreasing': True
            },
            sell_thresholds={
                'overbought_distance': 0.4,  # RSI далеко от 50 в зоне перекупленности
                'distance_decreasing': True
            },
            noise_filter=0.02,
            relevance_window=14,
            weight=0.6,
            combinations=['rsi_val', 'stoch_k']
        )
        
        # Stochastic Difference (K - D)
        patterns['stoch_diff'] = IndicatorPattern(
            name='Stochastic Diff',
            category='derived',
            optimal_range=(-10, 10),
            buy_thresholds={
                'diff_positive': 3,
                'diff_rising': True
            },
            sell_thresholds={
                'diff_negative': -3,
                'diff_falling': True
            },
            noise_filter=1,
            relevance_window=14,
            weight=0.65,
            combinations=['stoch_k', 'stoch_d']
        )
        
        # Vortex Ratio (VI+ / VI-)
        patterns['vortex_ratio'] = IndicatorPattern(
            name='Vortex Ratio',
            category='derived',
            optimal_range=(0.8, 1.2),
            buy_thresholds={
                'ratio_above_1': 1.05,
                'ratio_rising': True
            },
            sell_thresholds={
                'ratio_below_1': 0.95,
                'ratio_falling': True
            },
            noise_filter=0.02,
            relevance_window=14,
            weight=0.7,
            combinations=['vortex_vip', 'vortex_vin', 'adx_val']
        )
        
        # Ichimoku Difference (Conversion - Base)
        patterns['ichimoku_diff'] = IndicatorPattern(
            name='Ichimoku Diff',
            category='derived',
            optimal_range=(-0.01, 0.01),
            buy_thresholds={
                'diff_positive': 0.001,
                'diff_rising': True
            },
            sell_thresholds={
                'diff_negative': -0.001,
                'diff_falling': True
            },
            noise_filter=0.0005,
            relevance_window=26,
            weight=0.7,
            combinations=['ichimoku_conv', 'ichimoku_base']
        )
        
        # ATR Normalized
        patterns['atr_norm'] = IndicatorPattern(
            name='ATR Normalized',
            category='derived',
            optimal_range=(0.001, 0.03),
            buy_thresholds={
                'volatility_optimal': (0.005, 0.02),  # оптимальная волатильность
                'volatility_stable': True
            },
            sell_thresholds={
                'volatility_optimal': (0.005, 0.02),
                'volatility_stable': True
            },
            noise_filter=0.0002,
            relevance_window=10,
            weight=0.65,
            combinations=['atr_val', 'bb_position']
        )
        
        # ==========================================
        # 6. ВРЕМЕННЫЕ ПРИЗНАКИ (Time Features)
        # ==========================================
        
        # Hour of Day
        patterns['hour'] = IndicatorPattern(
            name='Hour',
            category='time',
            optimal_range=(0, 23),
            buy_thresholds={
                'active_hours': [1, 2, 3, 8, 9, 14, 15],  # активные часы торговли
                'avoid_hours': [22, 23, 0]  # избегать
            },
            sell_thresholds={
                'active_hours': [1, 2, 3, 8, 9, 14, 15],
                'avoid_hours': [22, 23, 0]
            },
            noise_filter=0,
            relevance_window=1,
            weight=0.3,
            combinations=['day_of_week', 'volume_ratio']
        )
        
        # Day of Week
        patterns['day_of_week'] = IndicatorPattern(
            name='Day of Week',
            category='time',
            optimal_range=(0, 6),
            buy_thresholds={
                'active_days': [1, 2, 3, 4],  # вторник-пятница
                'avoid_days': [0, 6]  # понедельник, воскресенье
            },
            sell_thresholds={
                'active_days': [1, 2, 3, 4],
                'avoid_days': [0, 6]
            },
            noise_filter=0,
            relevance_window=1,
            weight=0.2,
            combinations=['hour', 'is_weekend']
        )
        
        # Is Weekend
        patterns['is_weekend'] = IndicatorPattern(
            name='Is Weekend',
            category='time',
            optimal_range=(0, 1),
            buy_thresholds={
                'not_weekend': 0
            },
            sell_thresholds={
                'not_weekend': 0
            },
            noise_filter=0,
            relevance_window=1,
            weight=0.2,
            combinations=['day_of_week', 'volume_ratio']
        )
        
        # ==========================================
        # 7. ЦЕНОВЫЕ ПАТТЕРНЫ (Price Patterns)
        # ==========================================
        
        # Price Change 1 bar
        patterns['price_change_1'] = IndicatorPattern(
            name='Price Change 1',
            category='price',
            optimal_range=(-0.02, 0.02),  # -2% до +2%
            buy_thresholds={
                'min_change': -0.01,  # коррекция не более 1%
                'max_change': 0.005,  # не гнаться за ростом
                'momentum_shift': True
            },
            sell_thresholds={
                'max_change': 0.01,
                'min_change': -0.005,
                'momentum_shift': True
            },
            noise_filter=0.001,
            relevance_window=4,
            weight=0.6,
            combinations=['price_change_4', 'volume_ratio']
        )
        
        # Price Change 4 bars (1 hour)
        patterns['price_change_4'] = IndicatorPattern(
            name='Price Change 4',
            category='price',
            optimal_range=(-0.03, 0.03),
            buy_thresholds={
                'min_change': -0.02,
                'max_change': 0.01,
                'trend_continuation': True
            },
            sell_thresholds={
                'max_change': 0.02,
                'min_change': -0.01,
                'trend_continuation': True
            },
            noise_filter=0.002,
            relevance_window=8,
            weight=0.7,
            combinations=['price_change_16', 'macd_val']
        )
        
        # Price Change 16 bars (4 hours)
        patterns['price_change_16'] = IndicatorPattern(
            name='Price Change 16',
            category='price',
            optimal_range=(-0.05, 0.05),
            buy_thresholds={
                'trend_up': 0.01,
                'not_overbought': 0.04
            },
            sell_thresholds={
                'trend_down': -0.01,
                'not_oversold': -0.04
            },
            noise_filter=0.003,
            relevance_window=16,
            weight=0.75,
            combinations=['adx_val', 'macd_val']
        )
        
        # Volatility 4 bars
        patterns['volatility_4'] = IndicatorPattern(
            name='Volatility 4',
            category='price',
            optimal_range=(0.001, 0.02),
            buy_thresholds={
                'volatility_increasing': True,
                'optimal_range': (0.003, 0.015)
            },
            sell_thresholds={
                'volatility_increasing': True,
                'optimal_range': (0.003, 0.015)
            },
            noise_filter=0.0005,
            relevance_window=8,
            weight=0.6,
            combinations=['volatility_16', 'atr_val']
        )
        
        # Volatility 16 bars
        patterns['volatility_16'] = IndicatorPattern(
            name='Volatility 16',
            category='price',
            optimal_range=(0.002, 0.025),
            buy_thresholds={
                'volatility_stable': True,
                'optimal_range': (0.005, 0.02)
            },
            sell_thresholds={
                'volatility_stable': True,
                'optimal_range': (0.005, 0.02)
            },
            noise_filter=0.0005,
            relevance_window=16,
            weight=0.65,
            combinations=['atr_val', 'bb_position']
        )
        
        # Volume SMA
        patterns['volume_sma'] = IndicatorPattern(
            name='Volume SMA',
            category='volume',
            optimal_range=(0, np.inf),
            buy_thresholds={
                'volume_above_avg': 1.2,
                'volume_increasing': True
            },
            sell_thresholds={
                'volume_above_avg': 1.2,
                'volume_increasing': True
            },
            noise_filter=0.1,
            relevance_window=20,
            weight=0.7,
            combinations=['volume_ratio', 'cmf']
        )
        
        # Volume Ratio
        patterns['volume_ratio'] = IndicatorPattern(
            name='Volume Ratio',
            category='volume',
            optimal_range=(0.5, 3.0),
            buy_thresholds={
                'high_volume': 1.5,  # объем выше среднего в 1.5 раза
                'volume_spike': 2.5,  # всплеск объема
                'volume_confirm': True
            },
            sell_thresholds={
                'high_volume': 1.5,
                'volume_spike': 2.5,
                'volume_confirm': True
            },
            noise_filter=0.1,
            relevance_window=20,
            weight=0.8,
            combinations=['obv', 'cmf', 'mfi']
        )
        
        return patterns
    
    def evaluate_indicator(self, indicator_name: str, value: float, 
                          price: float = None, history: List[float] = None) -> Dict[str, any]:
        """
        Оценивает значение индикатора и возвращает торговый сигнал
        
        Args:
            indicator_name: Название индикатора
            value: Текущее значение
            price: Текущая цена (для относительных индикаторов)
            history: История значений для анализа тренда
            
        Returns:
            Dict с оценкой: signal_strength, quality, recommendation
        """
        if indicator_name not in self.patterns:
            return {'signal_strength': SignalStrength.NEUTRAL, 'quality': 0, 'recommendation': 'unknown'}
        
        pattern = self.patterns[indicator_name]
        
        # Фильтрация шума
        if history and len(history) > 1:
            recent_change = abs(value - history[-1])
            if recent_change < pattern.noise_filter:
                return {'signal_strength': SignalStrength.NEUTRAL, 'quality': 0.5, 'recommendation': 'noise'}
        
        # Проверка оптимального диапазона
        in_range = pattern.optimal_range[0] <= value <= pattern.optimal_range[1]
        range_quality = 1.0 if in_range else 0.7
        
        # Определение сигнала
        signal = SignalStrength.NEUTRAL
        confidence = 0.5
        
        # Проверка условий покупки
        buy_score = 0
        buy_conditions = len(pattern.buy_thresholds)
        
        for condition, threshold in pattern.buy_thresholds.items():
            if self._check_condition(condition, threshold, value, history, price):
                buy_score += 1
        
        # Проверка условий продажи
        sell_score = 0
        sell_conditions = len(pattern.sell_thresholds)
        
        for condition, threshold in pattern.sell_thresholds.items():
            if self._check_condition(condition, threshold, value, history, price):
                sell_score += 1
        
        # Определение силы сигнала
        if buy_conditions > 0:
            buy_confidence = buy_score / buy_conditions
            if buy_confidence >= 0.8:
                signal = SignalStrength.STRONG_BUY
                confidence = buy_confidence
            elif buy_confidence >= 0.6:
                signal = SignalStrength.BUY
                confidence = buy_confidence
        
        if sell_conditions > 0:
            sell_confidence = sell_score / sell_conditions
            if sell_confidence >= 0.8:
                signal = SignalStrength.STRONG_SELL
                confidence = sell_confidence
            elif sell_confidence >= 0.6:
                signal = SignalStrength.SELL
                confidence = sell_confidence
        
        # Качество сигнала
        quality = confidence * range_quality * pattern.weight
        
        # Рекомендация
        if signal == SignalStrength.STRONG_BUY:
            recommendation = 'strong_buy'
        elif signal == SignalStrength.BUY:
            recommendation = 'buy'
        elif signal == SignalStrength.STRONG_SELL:
            recommendation = 'strong_sell'
        elif signal == SignalStrength.SELL:
            recommendation = 'sell'
        else:
            recommendation = 'hold'
        
        return {
            'signal_strength': signal,
            'quality': quality,
            'confidence': confidence,
            'recommendation': recommendation,
            'in_optimal_range': in_range,
            'pattern_weight': pattern.weight
        }
    
    def _check_condition(self, condition: str, threshold: any, 
                        value: float, history: List[float], price: float) -> bool:
        """Проверяет условие для индикатора"""
        
        # Простые пороговые условия
        if condition in ['oversold', 'overbought', 'min_value', 'max_value',
                        'adx_min', 'di_min', 'di_max', 'strong_buy', 'strong_sell']:
            if isinstance(threshold, (int, float)):
                return value >= threshold if 'min' in condition else value <= threshold
        
        # Условия сравнения
        if condition in ['price_above', 'price_below', 'above_signal', 'below_signal']:
            if price and isinstance(threshold, (int, float)):
                return (price > value * threshold) if 'above' in condition else (price < value * threshold)
        
        # Условия тренда
        if condition in ['rising', 'falling', 'slope_positive', 'slope_negative']:
            if history and len(history) >= 3:
                recent_trend = value - history[-1]
                return recent_trend > 0 if 'rising' in condition or 'positive' in condition else recent_trend < 0
        
        # Булевы условия
        if isinstance(threshold, bool):
            return threshold
        
        # Условия диапазона
        if condition == 'optimal_range' and isinstance(threshold, tuple):
            return threshold[0] <= value <= threshold[1]
        
        # Условия списка (для временных признаков)
        if condition in ['active_hours', 'active_days', 'avoid_hours', 'avoid_days']:
            if isinstance(threshold, list):
                return value in threshold if 'active' in condition else value not in threshold
        
        return False
    
    def get_category_signals(self, indicators: Dict[str, float], 
                           price: float = None) -> Dict[str, Dict]:
        """
        Получает сигналы по категориям индикаторов
        
        Args:
            indicators: Словарь с текущими значениями индикаторов
            price: Текущая цена
            
        Returns:
            Dict с оценками по категориям
        """
        category_signals = {
            'trend': {'signals': [], 'avg_quality': 0, 'consensus': SignalStrength.NEUTRAL},
            'oscillator': {'signals': [], 'avg_quality': 0, 'consensus': SignalStrength.NEUTRAL},
            'volatility': {'signals': [], 'avg_quality': 0, 'consensus': SignalStrength.NEUTRAL},
            'volume': {'signals': [], 'avg_quality': 0, 'consensus': SignalStrength.NEUTRAL},
            'derived': {'signals': [], 'avg_quality': 0, 'consensus': SignalStrength.NEUTRAL},
            'time': {'signals': [], 'avg_quality': 0, 'consensus': SignalStrength.NEUTRAL},
            'price': {'signals': [], 'avg_quality': 0, 'consensus': SignalStrength.NEUTRAL}
        }
        
        for ind_name, value in indicators.items():
            if ind_name in self.patterns:
                pattern = self.patterns[ind_name]
                evaluation = self.evaluate_indicator(ind_name, value, price)
                
                category_signals[pattern.category]['signals'].append({
                    'indicator': ind_name,
                    'value': value,
                    'signal': evaluation['signal_strength'],
                    'quality': evaluation['quality'],
                    'confidence': evaluation['confidence']
                })
        
        # Рассчитываем консенсус по категориям
        for category, data in category_signals.items():
            if data['signals']:
                # Средняя качество
                data['avg_quality'] = np.mean([s['quality'] for s in data['signals']])
                
                # Взвешенный консенсус
                weighted_signal = 0
                total_weight = 0
                
                for signal in data['signals']:
                    weight = signal['quality']
                    weighted_signal += signal['signal'].value * weight
                    total_weight += weight
                
                if total_weight > 0:
                    consensus_value = weighted_signal / total_weight
                    
                    # Преобразуем в SignalStrength
                    if consensus_value >= 1.5:
                        data['consensus'] = SignalStrength.STRONG_BUY
                    elif consensus_value >= 0.5:
                        data['consensus'] = SignalStrength.BUY
                    elif consensus_value <= -1.5:
                        data['consensus'] = SignalStrength.STRONG_SELL
                    elif consensus_value <= -0.5:
                        data['consensus'] = SignalStrength.SELL
                    else:
                        data['consensus'] = SignalStrength.NEUTRAL
        
        return category_signals
    
    def get_combined_signal(self, indicators: Dict[str, float], 
                          price: float = None) -> Dict[str, any]:
        """
        Получает комбинированный сигнал на основе всех индикаторов
        
        Args:
            indicators: Словарь с текущими значениями индикаторов
            price: Текущая цена
            
        Returns:
            Dict с финальной рекомендацией
        """
        category_signals = self.get_category_signals(indicators, price)
        
        # Веса категорий
        category_weights = {
            'trend': 1.0,
            'oscillator': 0.8,
            'volatility': 0.7,
            'volume': 0.9,
            'derived': 0.6,
            'time': 0.3,
            'price': 0.7
        }
        
        # Рассчитываем общий сигнал
        total_signal = 0
        total_weight = 0
        
        for category, data in category_signals.items():
            if data['avg_quality'] > 0:
                weight = category_weights.get(category, 0.5) * data['avg_quality']
                total_signal += data['consensus'].value * weight
                total_weight += weight
        
        if total_weight > 0:
            final_signal_value = total_signal / total_weight
            
            # Определяем силу сигнала
            if final_signal_value >= 1.5:
                final_signal = SignalStrength.STRONG_BUY
                action = 'strong_buy'
            elif final_signal_value >= 0.5:
                final_signal = SignalStrength.BUY
                action = 'buy'
            elif final_signal_value <= -1.5:
                final_signal = SignalStrength.STRONG_SELL
                action = 'strong_sell'
            elif final_signal_value <= -0.5:
                final_signal = SignalStrength.SELL
                action = 'sell'
            else:
                final_signal = SignalStrength.NEUTRAL
                action = 'hold'
            
            # Рассчитываем уверенность
            confidence = min(abs(final_signal_value) / 2.0, 1.0)
            
            # Проверяем согласованность категорий
            category_agreement = sum(1 for cat in category_signals.values() 
                                   if cat['consensus'].value * final_signal_value > 0) / len(category_signals)
            
            return {
                'signal': final_signal,
                'action': action,
                'confidence': confidence,
                'signal_value': final_signal_value,
                'category_agreement': category_agreement,
                'category_signals': category_signals,
                'quality_score': np.mean([cat['avg_quality'] for cat in category_signals.values()])
            }
        
        return {
            'signal': SignalStrength.NEUTRAL,
            'action': 'hold',
            'confidence': 0,
            'signal_value': 0,
            'category_agreement': 0,
            'category_signals': category_signals,
            'quality_score': 0
        }
    
    def get_indicator_combinations(self, indicator_name: str) -> List[str]:
        """Возвращает список индикаторов для комбинирования"""
        if indicator_name in self.patterns:
            return self.patterns[indicator_name].combinations
        return []
    
    def evaluate_combination(self, indicators: Dict[str, float], 
                           combination: List[str], price: float = None) -> float:
        """
        Оценивает качество комбинации индикаторов
        
        Args:
            indicators: Значения индикаторов
            combination: Список индикаторов для комбинирования
            price: Текущая цена
            
        Returns:
            float: Оценка качества комбинации (0-1)
        """
        if not combination:
            return 0
        
        signals = []
        for ind_name in combination:
            if ind_name in indicators:
                eval_result = self.evaluate_indicator(ind_name, indicators[ind_name], price)
                signals.append(eval_result['signal_strength'].value)
        
        if not signals:
            return 0
        
        # Проверяем согласованность сигналов
        signal_std = np.std(signals)
        signal_mean = np.mean(signals)
        
        # Высокая согласованность = низкая дисперсия + сильный сигнал
        if signal_std < 0.5 and abs(signal_mean) > 0.5:
            return min(1.0, abs(signal_mean) / 2.0 * (1 - signal_std))
        
        return max(0, 0.5 - signal_std)


# Пример использования
if __name__ == "__main__":
    # Инициализация паттернов
    patterns = TechnicalIndicatorPatterns()
    
    # Пример данных индикаторов
    sample_indicators = {
        'ema_15': 100.5,
        'adx_val': 28,
        'adx_plus_di': 25,
        'adx_minus_di': 18,
        'rsi_val': 35,
        'macd_val': 0.0015,
        'macd_signal': 0.0012,
        'macd_hist': 0.0003,
        'volume_ratio': 1.8,
        'bb_position': 0.25,
        'atr_val': 0.015
    }
    
    current_price = 100.0
    
    # Получение комбинированного сигнала
    result = patterns.get_combined_signal(sample_indicators, current_price)
    
    print(f"Финальный сигнал: {result['action']}")
    print(f"Уверенность: {result['confidence']:.2%}")
    print(f"Качество сигнала: {result['quality_score']:.2f}")
    print(f"Согласованность категорий: {result['category_agreement']:.2%}")
    
    # Детали по категориям
    print("\nСигналы по категориям:")
    for category, data in result['category_signals'].items():
        if data['signals']:
            print(f"\n{category.upper()}:")
            print(f"  Консенсус: {data['consensus'].name}")
            print(f"  Среднее качество: {data['avg_quality']:.2f}")
            print(f"  Индикаторы:")
            for signal in data['signals']:
                print(f"    - {signal['indicator']}: {signal['signal'].name} (уверенность: {signal['confidence']:.2%})")