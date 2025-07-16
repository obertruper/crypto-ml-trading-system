"""
Инженерия признаков для анализа Bitcoin с моделью
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Рассчитать дополнительные признаки для модели"""
    logger.info("🔧 Расчет дополнительных признаков...")
    
    # Копируем DataFrame чтобы не изменять оригинал
    df = df.copy()
    
    # Проверяем наличие базовых колонок
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"❌ Отсутствует колонка {col}")
            return df
    
    # 1. Spread и volatility признаки
    df['hl_spread'] = (df['high'] - df['low']) / df['close'] * 100
    df['gk_volatility'] = np.sqrt(0.5 * np.log(df['high'] / df['low'])**2) * 100
    
    # 2. Volume-based features
    df['cumulative_volume_ratio'] = df['volume'].rolling(20).sum() / df['volume'].rolling(60).sum()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # 3. VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_to_vwap'] = df['close'] / df['vwap']
    
    # 4. MA-based features
    for period in [5, 10, 20, 60]:
        df[f'volume_ratio_ma_{period}'] = df['volume_ratio'].rolling(period).mean()
        df[f'volume_ratio_std_{period}'] = df['volume_ratio'].rolling(period).std()
        
        if 'adx_val' in df.columns:
            df[f'adx_val_ma_{period}'] = df['adx_val'].rolling(period).mean()
            df[f'adx_val_std_{period}'] = df['adx_val'].rolling(period).std()
            
        if 'rsi_val' in df.columns:
            df[f'rsi_val_ma_{period}'] = df['rsi_val'].rolling(period).mean()
            df[f'rsi_val_std_{period}'] = df['rsi_val'].rolling(period).std()
    
    # 5. Price momentum и efficiency
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['price_momentum_ratio'] = df['close'] / df['close'].shift(20)
    df['price_efficiency'] = df['close'].diff().abs().rolling(20).sum() / (df['high'] - df['low']).rolling(20).sum()
    
    # 6. RSI/ADX взаимодействие
    if 'rsi_val' in df.columns and 'adx_val' in df.columns:
        df['rsi_to_adx'] = df['rsi_val'] / (df['adx_val'] + 1)
        df['rsi_macd_interaction'] = df['rsi_val'] * df.get('macd_hist', 0)
    
    # 7. Временные признаки
    df['hour'] = pd.to_datetime(df['timestamp'], unit='ms').dt.hour
    df['dow'] = pd.to_datetime(df['timestamp'], unit='ms').dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    
    # 8. Price patterns
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    df['is_hammer'] = ((df['lower_shadow'] > 2 * (df['high'] - df['low'])) & 
                       (df['close'] > df['open'])).astype(int)
    
    # 9. Market regime
    df['market_regime_med_vol'] = (df['volume'] > df['volume'].rolling(50).median()).astype(int)
    
    # 10. Price position features
    df['high_ratio'] = df['high'] / df['close']
    df['low_ratio'] = df['low'] / df['close']
    df['open_ratio'] = df['open'] / df['close']
    
    rolling_high = df['high'].rolling(4).max()
    rolling_low = df['low'].rolling(4).min()
    df['position_in_1h_range'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
    
    # 11. Volume position
    df['volume_position_20'] = df['volume'].rolling(20).apply(lambda x: (x.iloc[-1] > x.median()).astype(int))
    df['log_volume'] = np.log(df['volume'] + 1)
    
    # 12. Consecutive patterns
    df['consecutive_hh'] = (df['high'] > df['high'].shift(1)).rolling(3).sum()
    
    # 13. Volume-price correlation
    df['volume_price_corr'] = df['close'].rolling(20).corr(df['volume'])
    
    # 14. BTC features (заглушки, так как у нас только BTC)
    df['btc_volatility'] = df['gk_volatility']  # Используем собственную волатильность
    df['btc_correlation_20'] = 1.0
    df['btc_correlation_60'] = 1.0
    df['btc_return_1h'] = df['close'].pct_change(4) * 100
    df['btc_return_4h'] = df['close'].pct_change(16) * 100
    
    # 15. Price to EMA
    if 'ema_15' in df.columns:
        df['price_to_ema15'] = df['close'] / df['ema_15']
    
    ema50 = df['close'].ewm(span=50).mean()
    df['price_to_ema50'] = df['close'] / ema50
    
    # 16. Additional indicators that might be missing
    if 'macd_signal' in df.columns and 'macd_val' in df.columns:
        df['macd_signal_ratio'] = df['macd_val'] / (df['macd_signal'] + 1e-8)
    
    # Заполняем пропуски
    df = df.fillna(method='ffill').fillna(0)
    
    # Заменяем бесконечности
    df = df.replace([np.inf, -np.inf], 0)
    
    logger.info(f"✅ Рассчитано дополнительных признаков: {len(df.columns) - len(required_cols)}")
    
    return df