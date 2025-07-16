#!/usr/bin/env python3
"""
Функция для корректного создания бинарных признаков из непрерывных после SMOTE
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clip_technical_indicators(X, feature_names):
    """
    Ограничивает значения технических индикаторов их естественными диапазонами
    после применения SMOTE
    
    Args:
        X: массив признаков после SMOTE
        feature_names: список имен признаков
        
    Returns:
        X с ограниченными значениями
    """
    X_clipped = X.copy()
    
    # Определяем диапазоны для технических индикаторов
    indicator_ranges = {
        'rsi_val': (0, 100),
        'adx_val': (0, 100),
        'adx_plus_di': (0, 100),
        'adx_minus_di': (0, 100),
        'aroon_up': (0, 100),
        'aroon_down': (0, 100),
        'cci_val': (-200, 200),  # CCI может выходить за -100/+100
        'williams_r': (-100, 0),
        'roc_val': (-100, 100),
        'bb_position': (0, 1),  # Позиция между полосами Боллинджера
        'stoch_k': (0, 100),
        'stoch_d': (0, 100),
        'mfi': (0, 100),
        'obv_ratio': (0, None),  # OBV ratio не имеет верхней границы
        'cmf': (-1, 1),
        'percent_k': (0, 100),
        'percent_d': (0, 100),
    }
    
    clipped_indicators = []
    
    for indicator, (min_val, max_val) in indicator_ranges.items():
        if indicator in feature_names:
            idx = feature_names.index(indicator)
            old_values = X_clipped[:, idx].copy()
            
            # Применяем ограничения
            if min_val is not None:
                X_clipped[:, idx] = np.maximum(X_clipped[:, idx], min_val)
            if max_val is not None:
                X_clipped[:, idx] = np.minimum(X_clipped[:, idx], max_val)
            
            # Проверяем, были ли изменения
            if not np.array_equal(old_values, X_clipped[:, idx]):
                clipped_indicators.append(indicator)
                out_of_range = np.sum((old_values < min_val) | (old_values > max_val if max_val is not None else False))
                logger.info(f"   📊 {indicator}: ограничено {out_of_range} значений в диапазон [{min_val}, {max_val}]")
    
    if clipped_indicators:
        logger.info(f"   ✂️ Ограничены значения для {len(clipped_indicators)} индикаторов после SMOTE")
    
    return X_clipped


def recreate_binary_features(X, feature_names, direction='buy'):
    """
    Пересоздает бинарные признаки из непрерывных после применения SMOTE
    
    Args:
        X: массив признаков после SMOTE
        feature_names: список имен признаков
        direction: 'buy' или 'sell' для логирования
        
    Returns:
        X с пересозданными бинарными признаками
    """
    X_copy = X.copy()
    
    # Создаем DataFrame для удобства
    df = pd.DataFrame(X_copy, columns=feature_names)
    
    # Пересоздаем бинарные признаки из непрерывных
    recreated = []
    
    # RSI признаки
    if 'rsi_val' in df.columns and 'rsi_oversold' in df.columns:
        old_oversold = df['rsi_oversold'].copy()
        df['rsi_oversold'] = (df['rsi_val'] < 30).astype(int)
        if not np.array_equal(old_oversold.values, df['rsi_oversold'].values):
            recreated.append('rsi_oversold')
    
    if 'rsi_val' in df.columns and 'rsi_overbought' in df.columns:
        old_overbought = df['rsi_overbought'].copy()
        df['rsi_overbought'] = (df['rsi_val'] > 70).astype(int)
        if not np.array_equal(old_overbought.values, df['rsi_overbought'].values):
            recreated.append('rsi_overbought')
    
    # MACD признаки
    if 'macd_hist' in df.columns and 'macd_bullish' in df.columns:
        old_macd = df['macd_bullish'].copy()
        df['macd_bullish'] = (df['macd_hist'] > 0).astype(int)
        if not np.array_equal(old_macd.values, df['macd_bullish'].values):
            recreated.append('macd_bullish')
    
    # ADX признаки
    if 'adx_val' in df.columns and 'strong_trend' in df.columns:
        old_trend = df['strong_trend'].copy()
        df['strong_trend'] = (df['adx_val'] > 25).astype(int)
        if not np.array_equal(old_trend.values, df['strong_trend'].values):
            recreated.append('strong_trend')
    
    # Volume признаки
    if 'volume_ratio' in df.columns and 'volume_spike' in df.columns:
        old_spike = df['volume_spike'].copy()
        df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
        if not np.array_equal(old_spike.values, df['volume_spike'].values):
            recreated.append('volume_spike')
    
    # Bollinger bands признаки
    if 'bb_position' in df.columns:
        if 'bb_near_lower' in df.columns:
            old_lower = df['bb_near_lower'].copy()
            df['bb_near_lower'] = (df['bb_position'] < 0.2).astype(int)
            if not np.array_equal(old_lower.values, df['bb_near_lower'].values):
                recreated.append('bb_near_lower')
        
        if 'bb_near_upper' in df.columns:
            old_upper = df['bb_near_upper'].copy()
            df['bb_near_upper'] = (df['bb_position'] > 0.8).astype(int)
            if not np.array_equal(old_upper.values, df['bb_near_upper'].values):
                recreated.append('bb_near_upper')
    
    # is_bullish
    if 'close' in df.columns and 'open' in df.columns and 'is_bullish' in df.columns:
        old_bullish = df['is_bullish'].copy()
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        if not np.array_equal(old_bullish.values, df['is_bullish'].values):
            recreated.append('is_bullish')
    
    if recreated:
        logger.info(f"   ♻️ Пересозданы бинарные признаки после SMOTE для {direction}: {recreated}")
    
    # Логируем статистику
    binary_features = ['rsi_oversold', 'rsi_overbought', 'macd_bullish', 'strong_trend', 
                      'volume_spike', 'bb_near_lower', 'bb_near_upper', 'is_bullish']
    
    stats = []
    for feat in binary_features:
        if feat in df.columns:
            pct = df[feat].mean() * 100
            if pct > 0:
                stats.append(f"{feat}: {pct:.1f}%")
    
    if stats:
        logger.info(f"   📊 Статистика бинарных признаков после SMOTE: {', '.join(stats[:4])}")
        if len(stats) > 4:
            logger.info(f"      {', '.join(stats[4:])}")
    
    return df.values


def separate_features_for_smote(X, feature_names):
    """
    Разделяет признаки на непрерывные и бинарные для правильного применения SMOTE
    
    Returns:
        continuous_indices: индексы непрерывных признаков
        binary_indices: индексы бинарных признаков
        continuous_features: имена непрерывных признаков
        binary_features: имена бинарных признаков
    """
    binary_feature_names = [
        'rsi_oversold', 'rsi_overbought', 'macd_bullish',
        'bb_near_lower', 'bb_near_upper', 'strong_trend', 'volume_spike',
        'is_bullish', 'is_weekend', 'is_major', 'is_meme', 'is_defi', 'is_alt',
        'market_regime_low_vol', 'market_regime_med_vol', 'market_regime_high_vol'
    ]
    
    # Добавляем one-hot закодированные символы и паттерны
    binary_feature_names.extend([f for f in feature_names if f.startswith('is_')])
    binary_feature_names.extend([
        'is_hammer', 'is_doji', 'bullish_engulfing', 'bearish_engulfing',
        'pin_bar', 'rsi_bullish_divergence', 'rsi_bearish_divergence',
        'macd_bullish_divergence', 'macd_bearish_divergence', 'volume_price_divergence'
    ])
    
    # Убираем дубликаты
    binary_feature_names = list(set(binary_feature_names))
    
    binary_indices = []
    continuous_indices = []
    binary_features = []
    continuous_features = []
    
    for i, feature in enumerate(feature_names):
        if feature in binary_feature_names:
            binary_indices.append(i)
            binary_features.append(feature)
        else:
            continuous_indices.append(i)
            continuous_features.append(feature)
    
    return continuous_indices, binary_indices, continuous_features, binary_features