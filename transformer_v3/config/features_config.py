"""
Конфигурация признаков для XGBoost v3.0
"""

from typing import Dict, List, Tuple

# Технические индикаторы и их допустимые диапазоны
TECHNICAL_INDICATORS_BOUNDS = {
    'rsi_val': (0, 100),
    'adx_val': (0, 100),
    'adx_plus_di': (0, 100),
    'adx_minus_di': (0, 100),
    'aroon_up': (0, 100),
    'aroon_down': (0, 100),
    'williams_r': (-100, 0),
    'bb_position': (0, 1),
    'stoch_k': (0, 100),
    'stoch_d': (0, 100),
    'mfi': (0, 100),
    'cmf': (-1, 1),
}

# Группы признаков
FEATURE_GROUPS = {
    'technical_indicators': [
        # Trend indicators
        'ema_15', 'adx_val', 'adx_plus_di', 'adx_minus_di', 'adx_diff',
        'macd_val', 'macd_signal', 'macd_hist', 'macd_signal_ratio',
        'sar', 'sar_trend', 'sar_distance',
        'ich_tenkan', 'ich_kijun', 'ich_senkou_a', 'ich_senkou_b',
        'ich_chikou', 'ich_tenkan_kijun_signal', 'ich_price_kumo',
        'aroon_up', 'aroon_down', 'aroon_oscillator',
        
        # Oscillators
        'rsi_val', 'rsi_ma', 'stoch_k', 'stoch_d', 'stoch_signal',
        'cci', 'williams_r',
        
        # Volume indicators
        'obv', 'obv_slope', 'cmf', 'mfi', 'volume_ratio',
        
        # Volatility indicators
        'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'bb_position', 'kc_upper', 'kc_lower', 'dc_upper', 'dc_lower'
    ],
    
    'market_features': [
        'btc_correlation_20', 'btc_correlation_60',
        'btc_return_1h', 'btc_return_4h', 'btc_volatility',
        'relative_strength_btc',
        'market_regime_low_vol', 'market_regime_med_vol', 'market_regime_high_vol',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ],
    
    'ohlc_features': [
        'open_ratio', 'high_ratio', 'low_ratio',
        'hl_spread', 'body_size', 'upper_shadow', 'lower_shadow',
        'is_bullish', 'log_return', 'log_volume',
        'price_to_ema15', 'price_to_ema50', 'price_to_vwap'
    ],
    
    'symbol_features': [
        'is_weekend', 'is_hammer', 'is_doji',
        'is_btc', 'is_eth', 'is_bnb', 'is_xrp', 'is_ada',
        'is_doge', 'is_sol', 'is_dot', 'is_matic', 'is_shib',
        'is_avax', 'is_ltc', 'is_uni', 'is_link'
    ],
    
    'binary_features': [
        'rsi_oversold', 'rsi_overbought', 'macd_bullish',
        'strong_trend', 'volume_spike', 'bb_near_lower', 'bb_near_upper'
    ],
    
    'engineered_features': [
        'rsi_macd_interaction', 'volume_volatility_interaction',
        'rsi_to_adx', 'volume_to_volatility', 'price_momentum_ratio'
    ],
    
    'divergence_features': [
        'rsi_bullish_divergence', 'rsi_bearish_divergence', 'volume_price_divergence'
    ],
    
    'candle_patterns': [
        'bullish_engulfing', 'bearish_engulfing'
    ],
    
    'volume_profile': [
        'vwap_distance'
    ]
}

# Целевые переменные
TARGET_COLUMNS = ['buy_expected_return', 'sell_expected_return']

# Колонки для исключения из признаков
EXCLUDE_COLUMNS = [
    'buy_expected_return', 'sell_expected_return',
    'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'processed_at', 'created_at', 
    'datetime', 'hour', 'day_of_week',  # временные колонки
    'prev_close', 'prev_open', 'btc_close',  # временные колонки для расчетов
    'technical_indicators', 'expected_returns'  # JSON колонки
]

# Конфигурация признаков
FEATURE_CONFIG = {
    # Параметры для расчета бинарных признаков
    'binary_thresholds': {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'macd_bullish': 0,
        'strong_trend': 25,  # ADX > 25
        'volume_spike': 2.0,  # volume > 2 * avg_volume
        'bb_near_lower': 0.2,  # в нижних 20% BB
        'bb_near_upper': 0.8   # в верхних 20% BB
    },
    
    # Параметры для взвешенных признаков
    'weighted_features': {
        'weights': {
            'rsi': 0.3,
            'macd': 0.3,
            'volume': 0.2,
            'volatility': 0.2
        }
    },
    
    # Параметры для скользящих статистик
    'rolling_windows': [5, 10, 20, 60],
    
    # Параметры для дивергенций
    'divergence_window': 14,
    
    # Минимальная корреляция для утечки данных
    'max_feature_target_correlation': 0.95
}

def get_all_features() -> List[str]:
    """Получить полный список всех признаков"""
    all_features = []
    for group_features in FEATURE_GROUPS.values():
        all_features.extend(group_features)
    return list(set(all_features))  # Убираем дубликаты

def get_feature_count() -> Dict[str, int]:
    """Получить количество признаков по группам"""
    return {group: len(features) for group, features in FEATURE_GROUPS.items()}

def validate_features(df_columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Валидация признаков в датафрейме
    Возвращает (найденные_признаки, отсутствующие_признаки)
    """
    expected_features = get_all_features()
    df_features = [col for col in df_columns if col not in EXCLUDE_COLUMNS]
    
    found_features = [f for f in expected_features if f in df_features]
    missing_features = [f for f in expected_features if f not in df_features]
    
    return found_features, missing_features