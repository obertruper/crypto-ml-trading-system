"""
Явный маппинг признаков по категориям для XGBoost v3.0
Это решает проблему неточной категоризации признаков
"""

# Явный маппинг всех известных признаков по категориям
FEATURE_CATEGORY_MAPPING = {
    'technical': [
        # Основные технические индикаторы
        'rsi_val', 'rsi_ma', 'rsi_val_ma_5', 'rsi_val_ma_10', 'rsi_val_ma_20', 'rsi_val_ma_60',
        'rsi_val_std_5', 'rsi_val_std_10', 'rsi_val_std_20', 'rsi_val_std_60',
        'macd_val', 'macd_signal', 'macd_hist', 'macd_signal_ratio',
        'bb_position', 'bb_width', 'bb_upper', 'bb_middle', 'bb_lower',
        'adx_val', 'adx_plus_di', 'adx_minus_di', 'adx_diff',
        'adx_val_ma_5', 'adx_val_ma_10', 'adx_val_ma_20', 'adx_val_ma_60',
        'adx_val_std_5', 'adx_val_std_10', 'adx_val_std_20', 'adx_val_std_60',
        'atr', 'atr_percent',
        'volume_ratio', 'volume_ratio_ma', 
        'volume_ratio_ma_5', 'volume_ratio_ma_10', 'volume_ratio_ma_20', 'volume_ratio_ma_60',
        'volume_ratio_std_5', 'volume_ratio_std_10', 'volume_ratio_std_20', 'volume_ratio_std_60',
        'stoch_k', 'stoch_d', 'stoch_signal',
        'williams_r', 'mfi', 'cci', 'cmf', 'obv', 'obv_slope',
        'ema_15', 'sma_20', 'vwap', 'price_to_vwap', 'price_to_ema15', 'price_to_ema50',
        'sar', 'sar_distance', 'sar_trend',
        'ich_tenkan', 'ich_kijun', 'ich_senkou_a', 'ich_senkou_b', 'ich_chikou',
        'ich_tenkan_kijun_signal', 'ich_price_kumo',
        'aroon_up', 'aroon_down', 'aroon_oscillator',
        'kc_upper', 'kc_lower', 'dc_upper', 'dc_lower',
        
        # OHLC признаки
        'open_ratio', 'high_ratio', 'low_ratio',
        'hl_spread', 'body_size', 'upper_shadow', 'lower_shadow',
        'is_bullish', 'log_return', 'log_volume',
        
        # Паттерны свечей
        'is_hammer', 'is_doji', 'bullish_engulfing', 'bearish_engulfing',
        'higher_high', 'lower_low', 'consecutive_hh', 'consecutive_ll',
        'inside_bar', 'pin_bar_bull', 'pin_bar_bear',
        
        # Бинарные технические
        'rsi_oversold', 'rsi_overbought', 'macd_bullish',
        'strong_trend', 'volume_spike', 'bb_near_lower', 'bb_near_upper',
        
        # Взвешенные и взаимодействия
        'rsi_macd_interaction', 'volume_volatility_interaction',
        'rsi_to_adx', 'volume_to_volatility', 'price_momentum_ratio',
        
        # Дивергенции
        'rsi_bullish_divergence', 'rsi_bearish_divergence', 'volume_price_divergence',
        
        # Volume profile
        'volume_position_20', 'cumulative_volume_ratio',
        
        # Микроструктурные
        'spread_approximation', 'price_efficiency', 'volume_price_corr', 'gk_volatility',
        
        # Межтаймфреймовые
        'position_in_1h_range', 'trend_1h', 'trend_4h',
        
        # Рыночные режимы (переносим в технические!)
        'market_regime_low_vol', 'market_regime_med_vol', 'market_regime_high_vol',
        
        # НОВЫЕ продвинутые технические признаки
        'roc_5', 'roc_10', 'roc_20',
        'momentum_10', 'momentum_20',
        'price_velocity', 'price_acceleration',
        'volatility_ma_20', 'volatility_ratio', 'volatility_expanding', 'volatility_contracting',
        'volume_weighted_momentum', 'adl', 'adl_slope',
        'high_20', 'low_20', 'distance_to_high_20', 'distance_to_low_20', 'position_in_range_20',
        'trend_slope_10', 'trend_slope_20', 'trend_strength_10', 'trend_strength_20',
        'zscore_20', 'zscore_50', 'bb_squeeze',
        'close_position', 'buy_pressure', 'sell_pressure', 'order_flow_imbalance',
        'hurst_exponent', 'is_trending', 'is_mean_reverting',
        
        # Простые ценовые изменения
        'price_change_1', 'price_change_3', 'price_change_5', 'price_change_10',
        'momentum_3', 'momentum_5',
        
        # Простые бинарные сигналы
        'rsi_extreme', 'bb_breakout', 'bb_breakout_up', 'bb_breakout_down',
        'volume_spike', 'volume_spike_large',
        
        # Базовый час как технический признак (для внутридневных паттернов)
        'hour'
    ],
    
    'temporal': [
        # ТОЛЬКО чистые временные признаки
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend'
    ],
    
    'btc_related': [
        # Все что связано с BTC
        'btc_correlation_5', 'btc_correlation_20', 'btc_correlation_60',
        'btc_return_1h', 'btc_return_4h', 'btc_volatility',
        'btc_volume_ratio', 'btc_price_ratio',
        'relative_strength_btc'
    ],
    
    'symbol': [
        # One-hot encoding символов
        'is_btc', 'is_eth', 'is_bnb', 'is_xrp', 'is_ada',
        'is_doge', 'is_sol', 'is_dot', 'is_matic', 'is_shib',
        'is_avax', 'is_ltc', 'is_uni', 'is_link'
    ]
}

def get_feature_category(feature_name: str) -> str:
    """
    Получить категорию признака по его имени
    
    Args:
        feature_name: Имя признака
        
    Returns:
        Категория признака: 'technical', 'temporal', 'btc_related', 'symbol', 'other'
    """
    # Приводим к нижнему регистру для сравнения
    feature_lower = feature_name.lower()
    
    # Проверяем явный маппинг
    for category, features in FEATURE_CATEGORY_MAPPING.items():
        if feature_name in features:
            return category
            
    # Если не нашли в явном маппинге, пробуем по паттернам
    # Это нужно для новых признаков, которые могут появиться
    
    # Символы
    if feature_lower.startswith('is_') and any(coin in feature_lower for coin in ['btc', 'eth', 'bnb', 'xrp', 'ada', 'doge', 'sol', 'dot', 'matic', 'shib']):
        return 'symbol'
    
    # BTC related
    if 'btc_' in feature_lower or 'bitcoin' in feature_lower:
        return 'btc_related'
    
    # Временные
    if any(time_word in feature_lower for time_word in ['hour', 'dow', 'day_of_week', 'weekend']):
        # Исключаем паттерны которые не являются временными
        exclude_patterns = ['consecutive', 'higher', 'lower', 'pattern']
        if not any(exc in feature_lower for exc in exclude_patterns):
            return 'temporal'
    
    # Все остальное считаем техническими если содержит технические паттерны
    technical_keywords = [
        'rsi', 'macd', 'bb_', 'adx', 'atr', 'stoch', 'williams',
        'mfi', 'cci', 'cmf', 'obv', 'ema', 'sma', 'vwap', 'sar',
        'aroon', 'kc_', 'dc_', 'volume', 'price', 'ratio', 'ma_', 'std_',
        'divergence', 'signal', 'position', 'trend', 'regime',
        'open', 'high', 'low', 'close', 'body', 'shadow', 'spread'
    ]
    
    for keyword in technical_keywords:
        if keyword in feature_lower:
            return 'technical'
    
    # Если ничего не подошло
    return 'other'

def get_category_targets() -> dict:
    """Получить целевые проценты для каждой категории"""
    return {
        'technical': 85,     # 85% технические индикаторы (увеличено)
        'temporal': 2,       # 2% временные (резко уменьшено!)
        'btc_related': 10,   # 10% BTC корреляции
        'symbol': 3,         # 3% символы (уменьшено)
        'other': 0           # 0% другие
    }

def get_temporal_blacklist() -> list:
    """Получить список временных признаков которые нужно полностью исключить"""
    return [
        'dow_sin', 'dow_cos',           # День недели - основной источник переобучения
        'is_weekend',                   # Выходные дни
        'hour_sin', 'hour_cos',         # Час sin/cos - вызывает переобучение
        'session_asia',                 # Скрытые временные признаки
        'session_europe',               # Скрытые временные признаки  
        'session_america',              # Скрытые временные признаки
        'day_of_week',                  # День недели
        # hour НЕ в blacklist - используется как технический признак
    ]