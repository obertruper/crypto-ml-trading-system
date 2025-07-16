"""
Расширенные признаки для улучшения точности предсказаний направления
Включает market regime, microstructure и cross-asset features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger


class EnhancedFeatureEngineer:
    """Создание продвинутых признаков для direction prediction"""
    
    def __init__(self):
        self.logger = get_logger("EnhancedFeatures")
        self.btc_data = None  # Для cross-asset features
        
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Определение рыночного режима"""
        self.logger.info("🏛️ Добавляем market regime features...")
        
        # 1. Trend Regime (Trending vs Ranging)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # ADX для силы тренда
        df['trend_strength'] = self._calculate_adx(df, period=14)
        
        # Классификация режима
        df['regime_trend'] = 0  # 0=Ranging, 1=Uptrend, 2=Downtrend
        uptrend_mask = (df['ema_20'] > df['sma_50']) & (df['trend_strength'] > 25)
        downtrend_mask = (df['ema_20'] < df['sma_50']) & (df['trend_strength'] > 25)
        df.loc[uptrend_mask, 'regime_trend'] = 1
        df.loc[downtrend_mask, 'regime_trend'] = 2
        
        # 2. Volatility Regime
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()
        df['volatility_50'] = df['close'].pct_change().rolling(50).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_50'].replace(0, 1)
        
        # Классификация волатильности
        df['regime_volatility'] = pd.qcut(df['volatility_ratio'], q=3, labels=[0, 1, 2])  # Low, Medium, High
        
        # 3. Volume Regime
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20'].replace(0, 1)
        
        # Volume spike detection
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
        
        # 4. Wyckoff Phases
        df = self._identify_wyckoff_phases(df)
        
        # 5. Market Structure
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_lows'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['market_structure'] = df['higher_highs'].rolling(10).sum() + df['higher_lows'].rolling(10).sum()
        
        # 6. Momentum Regime
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        df['momentum_strength'] = df['momentum_10'] / df['momentum_20'].replace(0, 1)
        
        return df
    
    def add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Микроструктурные признаки из order flow"""
        self.logger.info("🔬 Добавляем microstructure features...")
        
        # 1. Order Flow Imbalance (OFI)
        df['price_change'] = df['close'].diff()
        df['volume_imbalance'] = df.apply(
            lambda x: x['volume'] if x['price_change'] > 0 else -x['volume'], 
            axis=1
        )
        df['ofi'] = df['volume_imbalance'].rolling(10).sum()
        df['ofi_normalized'] = df['ofi'] / df['volume'].rolling(10).sum().replace(0, 1)
        
        # 2. Trade Intensity
        df['trade_intensity'] = df['volume'] / df['quote_volume'].replace(0, 1)
        df['trade_intensity_ma'] = df['trade_intensity'].rolling(20).mean()
        
        # 3. Aggressive Trades (приближение через размер свечи)
        df['candle_range'] = df['high'] - df['low']
        df['candle_body'] = abs(df['close'] - df['open'])
        df['wick_ratio'] = df['candle_body'] / df['candle_range'].replace(0, 1)
        
        # Агрессивные покупки/продажи
        df['aggressive_buying'] = ((df['close'] > df['open']) & (df['wick_ratio'] > 0.7)).astype(int)
        df['aggressive_selling'] = ((df['close'] < df['open']) & (df['wick_ratio'] > 0.7)).astype(int)
        
        # 4. Price Impact
        df['price_impact'] = abs(df['price_change']) / df['volume'].replace(0, 1)
        df['price_impact_ma'] = df['price_impact'].rolling(20).mean()
        
        # 5. Volume Profile Analysis
        df['volume_at_high'] = df['volume'] * (df['close'] / df['high'])
        df['volume_at_low'] = df['volume'] * (df['low'] / df['close'].replace(0, 1))
        df['volume_distribution'] = df['volume_at_high'] / df['volume_at_low'].replace(0, 1)
        
        # 6. Tick Rule (приближение)
        df['tick_direction'] = np.sign(df['price_change'])
        df['tick_volume'] = df['volume'] * df['tick_direction']
        df['cumulative_tick_volume'] = df['tick_volume'].rolling(20).sum()
        
        # 7. Large Trade Detection
        volume_percentile_90 = df['volume'].rolling(100).quantile(0.9)
        df['large_trade'] = (df['volume'] > volume_percentile_90).astype(int)
        df['large_trade_direction'] = df['large_trade'] * np.sign(df['price_change'])
        
        return df
    
    def add_cross_asset_features(self, df: pd.DataFrame, all_symbols_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Cross-asset корреляции и лидирующие индикаторы"""
        self.logger.info("🔗 Добавляем cross-asset features...")
        
        # Получаем данные BTC как лидирующий индикатор
        if 'BTCUSDT' in all_symbols_data and df['symbol'].iloc[0] != 'BTCUSDT':
            btc_data = all_symbols_data['BTCUSDT'].set_index('datetime')
            current_data = df.set_index('datetime')
            
            # 1. Корреляция с BTC
            for window in [20, 50, 100]:
                btc_returns = btc_data['close'].pct_change()
                current_returns = current_data['close'].pct_change()
                
                # Выравниваем индексы
                aligned_btc = btc_returns.reindex(current_data.index, method='ffill')
                
                correlation = current_returns.rolling(window).corr(aligned_btc)
                df[f'btc_correlation_{window}'] = correlation.values
            
            # 2. BTC как лидирующий индикатор
            # Проверяем, предсказывает ли движение BTC наше движение
            btc_future_return = btc_data['close'].pct_change().shift(-4)  # BTC через час
            current_return = current_data['close'].pct_change()
            
            lead_correlation = current_return.rolling(20).corr(btc_future_return.reindex(current_data.index, method='ffill'))
            df['btc_lead_indicator'] = lead_correlation.values
            
            # 3. Divergence с BTC
            btc_ma = btc_data['close'].rolling(20).mean()
            current_ma = current_data['close'].rolling(20).mean()
            
            btc_ma_direction = (btc_ma.pct_change() > 0).astype(int)
            current_ma_direction = (current_ma.pct_change() > 0).astype(int)
            
            divergence = (btc_ma_direction.reindex(current_data.index, method='ffill') != current_ma_direction).astype(int)
            df['btc_divergence'] = divergence.values
            
            df = df.reset_index(drop=True)
        
        # 4. Сила сектора (если есть информация о секторе)
        if 'sector' in df.columns:
            sector = df['sector'].iloc[0]
            sector_symbols = [s for s, data in all_symbols_data.items() 
                            if 'sector' in data.columns and data['sector'].iloc[0] == sector]
            
            if len(sector_symbols) > 1:
                # Средний return сектора
                sector_returns = []
                for symbol in sector_symbols:
                    if symbol != df['symbol'].iloc[0]:
                        symbol_data = all_symbols_data[symbol].set_index('datetime')
                        returns = symbol_data['close'].pct_change()
                        sector_returns.append(returns)
                
                if sector_returns:
                    avg_sector_return = pd.concat(sector_returns, axis=1).mean(axis=1)
                    df['sector_strength'] = avg_sector_return.rolling(20).mean().reindex(
                        df.set_index('datetime').index, method='ffill'
                    ).values
                    df = df.reset_index(drop=True)
        
        # 5. Market Beta
        if 'BTCUSDT' in all_symbols_data and df['symbol'].iloc[0] != 'BTCUSDT':
            # Рассчитываем бету относительно BTC
            market_returns = btc_data['close'].pct_change()
            asset_returns = df.set_index('datetime')['close'].pct_change()
            
            # Rolling beta
            for window in [20, 50]:
                cov = asset_returns.rolling(window).cov(
                    market_returns.reindex(asset_returns.index, method='ffill')
                )
                var = market_returns.rolling(window).var()
                beta = cov / var.reindex(asset_returns.index, method='ffill').replace(0, 1)
                df[f'beta_{window}'] = beta.values
            
            df = df.reset_index(drop=True)
        
        return df
    
    def add_advanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Продвинутые технические индикаторы для direction prediction"""
        self.logger.info("📈 Добавляем advanced technical features...")
        
        # 1. Multi-timeframe Features
        # Создаем признаки на разных таймфреймах
        for tf_multiplier in [4, 12, 48]:  # 1h, 3h, 12h на 15m данных
            # Resample to higher timeframe
            high_tf_ohlc = df['close'].rolling(tf_multiplier).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last'
            })
            
            # Higher timeframe тренд
            df[f'htf_{tf_multiplier}_trend'] = (
                high_tf_ohlc['close'] > high_tf_ohlc['open']
            ).astype(int)
            
            # Higher timeframe momentum
            df[f'htf_{tf_multiplier}_momentum'] = high_tf_ohlc['close'].pct_change()
        
        # 2. Volume-Price Analysis
        # VWAP Bands
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_std'] = (df['close'] - df['vwap']).rolling(20).std()
        df['vwap_upper'] = df['vwap'] + 2 * df['vwap_std']
        df['vwap_lower'] = df['vwap'] - 2 * df['vwap_std']
        df['price_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Volume-weighted momentum
        df['volume_weighted_momentum'] = (
            df['close'].diff() * df['volume']
        ).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # 3. Market Internals
        # Advance/Decline (using highs/lows as proxy)
        df['new_highs'] = (df['high'] == df['high'].rolling(50).max()).astype(int)
        df['new_lows'] = (df['low'] == df['low'].rolling(50).min()).astype(int)
        df['advance_decline'] = df['new_highs'].rolling(20).sum() - df['new_lows'].rolling(20).sum()
        
        # 4. Momentum Divergence
        # Price vs RSI divergence
        price_higher = df['close'] > df['close'].shift(20)
        rsi_lower = df['rsi'] < df['rsi'].shift(20) if 'rsi' in df.columns else False
        df['bearish_divergence'] = (price_higher & rsi_lower).astype(int)
        
        price_lower = df['close'] < df['close'].shift(20)
        rsi_higher = df['rsi'] > df['rsi'].shift(20) if 'rsi' in df.columns else False
        df['bullish_divergence'] = (price_lower & rsi_higher).astype(int)
        
        # 5. Support/Resistance Levels
        # Dynamic S/R based on recent pivots
        df['pivot_high'] = df['high'].rolling(10).max()
        df['pivot_low'] = df['low'].rolling(10).min()
        df['distance_to_resistance'] = (df['pivot_high'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['pivot_low']) / df['close']
        
        # 6. Fractal Dimension (измерение "шумности" рынка)
        df['fractal_dimension'] = df['close'].rolling(30).apply(
            lambda x: self._calculate_fractal_dimension(x)
        )
        
        return df
    
    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy для sentiment через price action и volume"""
        self.logger.info("😊 Добавляем sentiment proxy features...")
        
        # 1. Fear & Greed приближение
        # Momentum
        momentum_score = (df['close'].pct_change(20) > 0).astype(int)
        
        # Volatility (inverse - low vol = greed)
        vol_score = (df['volatility_ratio'] < 1).astype(int) if 'volatility_ratio' in df.columns else 0
        
        # Volume
        volume_score = (df['volume_ratio'] > 1.5).astype(int) if 'volume_ratio' in df.columns else 0
        
        # Market breadth (new highs)
        breadth_score = (df['new_highs'] > df['new_lows']).astype(int) if 'new_highs' in df.columns else 0
        
        # Combined Fear & Greed
        df['fear_greed_index'] = (momentum_score + vol_score + volume_score + breadth_score) / 4
        
        # 2. Panic/Euphoria Detection
        # Panic: high volume + sharp decline
        df['panic_selling'] = (
            (df['close'].pct_change() < -0.03) &  # 3% drop
            (df['volume_ratio'] > 2.0) if 'volume_ratio' in df.columns else False
        ).astype(int)
        
        # Euphoria: successive gains with increasing volume
        df['euphoria_buying'] = (
            (df['close'].pct_change().rolling(3).sum() > 0.05) &  # 5% gain over 3 periods
            (df['volume'].diff() > 0)
        ).astype(int)
        
        # 3. Accumulation/Distribution Phases
        # Based on volume and price relationship
        df['accumulation'] = (
            (df['close'] > df['open']) &
            (df['volume'] > df['volume'].rolling(20).mean()) &
            (df['close'].rolling(5).std() < df['close'].rolling(20).std())  # Decreasing volatility
        ).astype(int)
        
        df['distribution'] = (
            (df['close'] < df['open']) &
            (df['volume'] > df['volume'].rolling(20).mean()) &
            (df['high'] == df['high'].rolling(20).max())  # At resistance
        ).astype(int)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет ADX для определения силы тренда"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional movements
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = pd.Series(0.0, index=df.index)
        neg_dm = pd.Series(0.0, index=df.index)
        
        pos_dm[(up_move > down_move) & (up_move > 0)] = up_move
        neg_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        # Smooth DM
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        # ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di).replace(0, 1)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _identify_wyckoff_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Определение фаз Wyckoff"""
        # Упрощенная версия определения фаз
        # 0: Accumulation, 1: Markup, 2: Distribution, 3: Markdown
        
        # Используем комбинацию volume и price patterns
        df['wyckoff_phase'] = 0
        
        # Accumulation: low volatility, ranging price
        accumulation_mask = (
            (df['trend_strength'] < 25) &  # Low trend
            (df['volume_ratio'] < 1.2) &   # Normal volume
            (df['close'].rolling(20).std() < df['close'].rolling(50).std())  # Decreasing volatility
        )
        
        # Markup: strong uptrend
        markup_mask = (
            (df['regime_trend'] == 1) &  # Uptrend
            (df['trend_strength'] > 30) &  # Strong trend
            (df['volume_ratio'] > 1.0)  # Good volume
        )
        
        # Distribution: high volatility at tops
        distribution_mask = (
            (df['high'] == df['high'].rolling(50).max()) &  # At highs
            (df['volume_ratio'] > 1.5) &  # High volume
            (df['close'].rolling(20).std() > df['close'].rolling(50).std())  # Increasing volatility
        )
        
        # Markdown: strong downtrend
        markdown_mask = (
            (df['regime_trend'] == 2) &  # Downtrend
            (df['trend_strength'] > 30)  # Strong trend
        )
        
        df.loc[accumulation_mask, 'wyckoff_phase'] = 0
        df.loc[markup_mask, 'wyckoff_phase'] = 1
        df.loc[distribution_mask, 'wyckoff_phase'] = 2
        df.loc[markdown_mask, 'wyckoff_phase'] = 3
        
        return df
    
    def _calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """Расчет фрактальной размерности для измерения сложности ценового движения"""
        if len(prices) < 10:
            return 1.5  # Default value
        
        try:
            # Hurst exponent через R/S анализ
            lags = range(2, min(20, len(prices) // 2))
            rs_values = []
            
            for lag in lags:
                # Разбиваем на подпериоды
                sub_series = [prices[i:i+lag] for i in range(0, len(prices)-lag+1, lag)]
                
                rs_lag = []
                for sub in sub_series:
                    if len(sub) < 2:
                        continue
                    mean = sub.mean()
                    std = sub.std()
                    if std == 0:
                        continue
                    
                    # Cumulative deviations
                    cum_dev = (sub - mean).cumsum()
                    R = cum_dev.max() - cum_dev.min()
                    rs_lag.append(R / std)
                
                if rs_lag:
                    rs_values.append(np.mean(rs_lag))
            
            if len(rs_values) > 2:
                # Log-log regression
                log_lags = np.log(list(lags[:len(rs_values)]))
                log_rs = np.log(rs_values)
                
                # Hurst exponent
                H = np.polyfit(log_lags, log_rs, 1)[0]
                
                # Fractal dimension
                FD = 2 - H
                return np.clip(FD, 1.0, 2.0)
            
        except:
            pass
        
        return 1.5  # Default
    
    def create_enhanced_features(self, df: pd.DataFrame, 
                               all_symbols_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Главная функция для создания всех enhanced features"""
        self.logger.info("🚀 Создаем enhanced features для улучшения direction prediction...")
        
        # Сохраняем оригинальные колонки
        original_columns = df.columns.tolist()
        
        # Последовательно добавляем все группы признаков
        df = self.add_market_regime_features(df)
        df = self.add_microstructure_features(df)
        df = self.add_advanced_technical_features(df)
        df = self.add_sentiment_features(df)
        
        # Cross-asset features если есть данные других символов
        if all_symbols_data:
            df = self.add_cross_asset_features(df, all_symbols_data)
        
        # Очистка NaN
        new_columns = [col for col in df.columns if col not in original_columns]
        df[new_columns] = df[new_columns].fillna(method='ffill').fillna(0)
        
        self.logger.info(f"✅ Добавлено {len(new_columns)} новых признаков")
        
        return df