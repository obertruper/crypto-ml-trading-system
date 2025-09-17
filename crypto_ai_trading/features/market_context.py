"""
Контекст рынка - индикаторы на основе внутренних данных
Этап 1: Только данные OHLCV от топ-N монет без внешних источников
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from utils.logger import get_logger


class MarketContextFeatures:
    """
    Создание индикаторов контекста рынка на основе внутренних данных
    Безопасные индикаторы для определения типа рынка без внешних источников
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("MarketContextFeatures")
        
        # Конфигурация
        context_config = config.get('features', {}).get('market_context', {})
        self.internal_only = context_config.get('internal_only', True)
        self.fear_greed_components = context_config.get('fear_greed_components', True)
        self.breadth_indicators = context_config.get('breadth_indicators', True)
        
        # Топ монеты для расчета рыночных метрик
        self.top_coins = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'AAVEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT'
        ]
        
        self.logger.info(f"📊 Market Context инициализирован:")
        self.logger.info(f"   - Internal only: {self.internal_only}")
        self.logger.info(f"   - Fear&Greed: {self.fear_greed_components}")
        self.logger.info(f"   - Breadth indicators: {self.breadth_indicators}")
        
    def create_market_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание индикаторов контекста рынка
        
        Args:
            df: DataFrame со всеми символами (multi-symbol)
            
        Returns:
            DataFrame с добавленными контекстными признаками
        """
        if df['symbol'].nunique() < 2:
            self.logger.warning("⚠️ Недостаточно символов для контекстных признаков")
            return self._add_default_context(df)
        
        df = df.copy()
        
        # Создаем компоненты
        if self.fear_greed_components:
            df = self._create_fear_greed_index(df)
            
        if self.breadth_indicators:
            df = self._create_market_breadth_indicators(df)
            
        # Дополнительные контекстные индикаторы
        df = self._create_volatility_regime_indicators(df)
        df = self._create_momentum_indicators(df)
        df = self._create_volume_indicators(df)
        df = self._create_correlation_indicators(df)
        
        # Комбинированные индикаторы
        df = self._create_market_regime_classification(df)
        
        # Логируем результат
        context_features = [col for col in df.columns if col.startswith('ctx_')]
        self.logger.info(f"🎯 Создано контекстных признаков: {len(context_features)}")
        
        return df
    
    def _create_fear_greed_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Криптовалютный аналог Fear & Greed Index
        4 компонента по 25% каждый: Volatility, Momentum, Volume, Breadth
        """
        self.logger.info("📈 Создание Fear & Greed компонентов...")
        
        # Предварительные расчеты для каждого символа
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            if len(symbol_data) < 100:
                continue
            
            # Волатильность (25% веса)
            # ---------------------------
            symbol_data['hourly_returns'] = symbol_data['close'].pct_change()
            symbol_data['realized_volatility'] = symbol_data['hourly_returns'].rolling(24).std() * np.sqrt(24)
            
            # Volatility Index = текущая волатильность / MA30 волатильности  
            vol_ma30 = symbol_data['realized_volatility'].rolling(30*24, min_periods=24*7).mean()
            symbol_data['volatility_index'] = symbol_data['realized_volatility'] / vol_ma30
            
            # Volatility Percentile (90-дневный)
            symbol_data['volatility_percentile'] = (
                symbol_data['realized_volatility']
                .rolling(90*24, min_periods=30*24)
                .apply(lambda x: stats.percentileofscore(x, x.iloc[-1], kind='rank') / 100, raw=False)
            )
            
            # Обратно записываем в основной DataFrame
            df.loc[mask, 'realized_volatility'] = symbol_data['realized_volatility']
            df.loc[mask, 'volatility_index'] = symbol_data['volatility_index']
            df.loc[mask, 'volatility_percentile'] = symbol_data['volatility_percentile']
        
        # Агрегируем по времени для получения рыночных метрик
        market_vol = df.groupby('datetime').agg({
            'volatility_index': ['mean', 'median'],
            'volatility_percentile': ['mean', 'std']
        }).round(4)
        
        market_vol.columns = ['_'.join(col) for col in market_vol.columns]
        market_vol = market_vol.reset_index()
        
        # Merge обратно в основной DataFrame
        df = df.merge(market_vol, on='datetime', how='left')
        
        # Компонент 1: Volatility (0-100, где 0 = extreme fear, 100 = extreme greed)
        df['ctx_fg_volatility'] = (
            100 - df['volatility_percentile_mean'] * 100
        ).clip(0, 100)
        
        # Momentum компонент (25% веса)
        # -----------------------------
        df = self._add_momentum_component(df)
        
        # Volume компонент (25% веса)  
        # ---------------------------
        df = self._add_volume_component(df)
        
        # Market Breadth компонент (25% веса)
        # -----------------------------------
        df = self._add_breadth_component(df)
        
        # Итоговый Fear & Greed Index (0-100)
        df['ctx_fear_greed_index'] = (
            0.25 * df['ctx_fg_volatility'] +
            0.25 * df['ctx_fg_momentum'] +
            0.25 * df['ctx_fg_volume'] +
            0.25 * df['ctx_fg_breadth']
        ).clip(0, 100)
        
        # Интерпретация
        df['ctx_market_sentiment'] = pd.cut(
            df['ctx_fear_greed_index'],
            bins=[0, 25, 45, 55, 75, 100],
            labels=['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'],
            include_lowest=True
        )
        
        return df
    
    def _add_momentum_component(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum компонент Fear & Greed"""
        
        # Momentum Score для каждого символа
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            if len(symbol_data) < 200:
                continue
                
            # Momentum = (close - MA50) / MA50 * 100
            ma50 = symbol_data['close'].rolling(50*4, min_periods=25*4).mean()  # 50 дней
            momentum_score = (symbol_data['close'] / ma50 - 1) * 100
            
            # RSI weighted
            rsi = symbol_data['rsi'] if 'rsi' in symbol_data.columns else 50
            
            df.loc[mask, 'momentum_score'] = momentum_score
            df.loc[mask, 'rsi_weighted'] = rsi
        
        # Рыночные momentum метрики
        market_momentum = df.groupby('datetime').agg({
            'momentum_score': ['mean', 'std'],
            'rsi_weighted': 'mean'
        }).round(4)
        
        market_momentum.columns = ['_'.join(col) for col in market_momentum.columns]
        market_momentum = market_momentum.reset_index()
        
        df = df.merge(market_momentum, on='datetime', how='left')
        
        # Momentum Component (0-100)
        df['ctx_fg_momentum'] = (
            50 + df['momentum_score_mean'].clip(-25, 25) * 2
        ).clip(0, 100)
        
        return df
    
    def _add_volume_component(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume компонент Fear & Greed"""
        
        # Volume surge для каждого символа
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            if len(symbol_data) < 100:
                continue
                
            # Volume Surge = volume / MA30_volume
            volume_ma30 = symbol_data['volume'].rolling(30*24, min_periods=7*24).mean()
            volume_surge = symbol_data['volume'] / volume_ma30
            
            # Volume trend (линейная регрессия за 7 дней)
            volume_trend = symbol_data['volume'].rolling(7*24, min_periods=3*24).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 10 else 0
            )
            
            df.loc[mask, 'volume_surge'] = volume_surge
            df.loc[mask, 'volume_trend'] = volume_trend
        
        # Агрегация
        market_volume = df.groupby('datetime').agg({
            'volume_surge': ['mean', 'median'],
            'volume_trend': 'mean'
        }).round(4)
        
        market_volume.columns = ['_'.join(col) for col in market_volume.columns]
        market_volume = market_volume.reset_index()
        
        df = df.merge(market_volume, on='datetime', how='left')
        
        # Volume Component (0-100)
        df['ctx_fg_volume'] = (
            50 + df['volume_surge_mean'].clip(0.5, 2.0) * 25
        ).clip(0, 100)
        
        return df
        
    def _add_breadth_component(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Breadth компонент Fear & Greed"""
        
        # Подсчитываем для каждого timestamp
        breadth_metrics = []
        
        for timestamp in df['datetime'].unique():
            timestamp_data = df[df['datetime'] == timestamp]
            
            if len(timestamp_data) < 5:
                continue
                
            # Процент монет выше разных MA
            pct_above_ma20 = (timestamp_data['close'] > timestamp_data['sma_20']).mean() * 100
            pct_above_ma50 = (timestamp_data['close'] > timestamp_data['sma_50']).mean() * 100
            
            # Advance/Decline Ratio
            returns = timestamp_data['returns']
            advancing = (returns > 0.001).sum()  # >0.1%
            declining = (returns < -0.001).sum()  # <-0.1%
            
            adv_decl_ratio = advancing / max(declining, 1)
            
            breadth_metrics.append({
                'datetime': timestamp,
                'pct_above_ma20': pct_above_ma20,
                'pct_above_ma50': pct_above_ma50,
                'adv_decl_ratio': adv_decl_ratio,
                'advancing_count': advancing,
                'declining_count': declining
            })
        
        if breadth_metrics:
            breadth_df = pd.DataFrame(breadth_metrics)
            df = df.merge(breadth_df, on='datetime', how='left')
        else:
            # Если метрики не рассчитаны, создаем пустые колонки
            df['pct_above_ma20'] = 50.0
            df['pct_above_ma50'] = 50.0
            df['adv_decl_ratio'] = 1.0
            df['advancing_count'] = 0
            df['declining_count'] = 0
        
        # Breadth Component (0-100)
        df['ctx_fg_breadth'] = (
            0.4 * df['pct_above_ma50'] +
            0.3 * df['pct_above_ma20'] +
            0.3 * df['adv_decl_ratio'].clip(0, 2) * 50
        ).clip(0, 100)
        
        return df
    
    def _create_market_breadth_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Дополнительные индикаторы ширины рынка"""
        
        breadth_metrics = []
        
        for timestamp in df['datetime'].unique():
            timestamp_data = df[df['datetime'] == timestamp]
            
            if len(timestamp_data) < 5:
                continue
            
            # Market Participation Rate (% активно торгуемых монет)
            volume_threshold = timestamp_data['volume'].quantile(0.3)
            participation_rate = (timestamp_data['volume'] > volume_threshold).mean()
            
            # New Highs vs New Lows (за 30 дней)
            high_30d = timestamp_data['high'].rolling(30*24, min_periods=10*24).max()
            low_30d = timestamp_data['low'].rolling(30*24, min_periods=10*24).min()
            
            new_highs = (timestamp_data['high'] >= high_30d * 0.999).sum()
            new_lows = (timestamp_data['low'] <= low_30d * 1.001).sum()
            
            hl_ratio = new_highs / max(new_lows, 1)
            
            breadth_metrics.append({
                'datetime': timestamp,
                'ctx_market_participation': participation_rate,
                'ctx_new_highs_count': new_highs,
                'ctx_new_lows_count': new_lows,
                'ctx_hl_ratio': hl_ratio
            })
        
        if breadth_metrics:
            breadth_df = pd.DataFrame(breadth_metrics)
            df = df.merge(breadth_df, on='datetime', how='left')
        else:
            # Если метрики не рассчитаны, создаем пустые колонки
            df['pct_above_ma20'] = 50.0
            df['pct_above_ma50'] = 50.0
            df['adv_decl_ratio'] = 1.0
            df['advancing_count'] = 0
            df['declining_count'] = 0
        
        # Market Breadth Oscillator (кумулятивная сумма advance-decline)
        df['ctx_breadth_oscillator'] = (
            df['advancing_count'] - df['declining_count']
        ).fillna(0).cumsum()
        
        return df
    
    def _create_volatility_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикаторы волатильного режима рынка"""
        
        # Рыночная волатильность (среднее по топ монетам)
        top_symbols_mask = df['symbol'].isin(self.top_coins[:5])
        market_vol_data = df[top_symbols_mask].groupby('datetime').agg({
            'realized_volatility': 'mean',
            'atr_pct': 'mean'
        }).reset_index()
        
        market_vol_data.columns = ['datetime', 'ctx_market_volatility', 'ctx_market_atr']
        df = df.merge(market_vol_data, on='datetime', how='left')
        
        # Режимы волатильности
        vol_quantiles = df['ctx_market_volatility'].quantile([0.33, 0.66])
        df['ctx_volatility_regime'] = pd.cut(
            df['ctx_market_volatility'],
            bins=[0, vol_quantiles.iloc[0], vol_quantiles.iloc[1], np.inf],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        
        # VIX-подобный индекс (на основе краткосрочной волатильности)
        df['ctx_vix_crypto'] = (df['ctx_market_volatility'] * np.sqrt(365) * 100).clip(0, 200)
        
        return df
    
    def _create_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикаторы моментума рынка"""
        
        # Консистентность моментума (% монет в тренде)
        momentum_metrics = []
        
        for timestamp in df['datetime'].unique():
            timestamp_data = df[df['datetime'] == timestamp]
            
            if len(timestamp_data) < 5:
                continue
            
            # Монеты в тренде (ADX > 25)
            if 'adx' in timestamp_data.columns:
                trending_pct = (timestamp_data['adx'] > 25).mean()
            else:
                trending_pct = 0.5
            
            # Согласованность трендов (все в одном направлении)
            if 'momentum_score' in timestamp_data.columns:
                positive_momentum = (timestamp_data['momentum_score'] > 0).mean()
                trend_alignment = abs(positive_momentum - 0.5) * 2  # 0 = разнонаправленность, 1 = согласованность
            else:
                trend_alignment = 0.5
            
            momentum_metrics.append({
                'datetime': timestamp,
                'ctx_trending_pct': trending_pct,
                'ctx_trend_alignment': trend_alignment,
                'ctx_positive_momentum_pct': positive_momentum if 'positive_momentum' in locals() else 0.5
            })
        
        momentum_df = pd.DataFrame(momentum_metrics)
        df = df.merge(momentum_df, on='datetime', how='left')
        
        return df
    
    def _create_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Объемные индикаторы рынка"""
        
        # Общий объем рынка
        total_volume = df.groupby('datetime')['volume'].sum().reset_index()
        total_volume.columns = ['datetime', 'ctx_total_market_volume']
        df = df.merge(total_volume, on='datetime', how='left')
        
        # Volume surge на уровне рынка
        df['ctx_market_volume_surge'] = (
            df['ctx_total_market_volume'] / 
            df['ctx_total_market_volume'].rolling(30*24, min_periods=7*24).mean()
        )
        
        # Распределение объемов (концентрация vs распределенность)
        volume_metrics = []
        
        for timestamp in df['datetime'].unique():
            timestamp_data = df[df['datetime'] == timestamp]
            
            if len(timestamp_data) < 5:
                continue
                
            # Gini коэффициент для объемов (концентрация)
            volumes = timestamp_data['volume'].sort_values()
            n = len(volumes)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * volumes)) / (n * np.sum(volumes)) - (n + 1) / n
            
            volume_metrics.append({
                'datetime': timestamp,
                'ctx_volume_concentration': gini
            })
        
        volume_df = pd.DataFrame(volume_metrics)
        df = df.merge(volume_df, on='datetime', how='left')
        
        return df
    
    def _create_correlation_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Корреляционные индикаторы рынка"""
        
        # Средняя корреляция между активами
        correlation_metrics = []
        
        # Получаем матрицу returns для расчета корреляций
        returns_pivot = df.pivot_table(
            index='datetime', 
            columns='symbol', 
            values='returns'
        )
        
        # Rolling корреляция (7 дней)
        window = 7 * 24
        
        for i in range(window, len(returns_pivot)):
            window_data = returns_pivot.iloc[i-window:i]
            
            if window_data.shape[1] < 3:
                continue
                
            # Корреляционная матрица
            corr_matrix = window_data.corr()
            
            # Средняя корреляция (исключая диагональ)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            avg_correlation = corr_matrix.where(mask).stack().mean()
            
            correlation_metrics.append({
                'datetime': returns_pivot.index[i],
                'ctx_avg_correlation': avg_correlation,
                'ctx_correlation_regime': 'high' if avg_correlation > 0.7 else 'medium' if avg_correlation > 0.4 else 'low'
            })
        
        correlation_df = pd.DataFrame(correlation_metrics)
        df = df.merge(correlation_df, on='datetime', how='left')
        
        return df
    
    def _create_market_regime_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Классификация режима рынка на основе всех индикаторов"""
        
        # Bull Market Score (0-100)
        bull_score = (
            0.25 * (df['ctx_fg_momentum'].fillna(50) - 50) / 50 * 100 +
            0.25 * (df['ctx_pct_above_ma50'].fillna(50)) +
            0.25 * (df['ctx_fg_breadth'].fillna(50)) +
            0.25 * (100 - df['ctx_vix_crypto'].fillna(30).clip(0, 100))
        ).clip(0, 100)
        
        # Bear Market Score (0-100) 
        bear_score = (100 - bull_score).clip(0, 100)
        
        # Sideways Score (0-100)
        sideways_score = (
            100 - abs(bull_score - 50) * 2
        ).clip(0, 100)
        
        df['ctx_bull_market_score'] = bull_score
        df['ctx_bear_market_score'] = bear_score  
        df['ctx_sideways_score'] = sideways_score
        
        # Финальная классификация режима
        df['ctx_market_regime'] = np.where(
            bull_score > 60, 'bull',
            np.where(bear_score > 60, 'bear', 'sideways')
        )
        
        # Confidence в классификации
        max_scores = pd.DataFrame({
            'bull': bull_score,
            'bear': bear_score, 
            'sideways': sideways_score
        }).max(axis=1)
        
        df['ctx_regime_confidence'] = (max_scores - 33.33) / 66.67  # Нормализуем в [0,1]
        
        return df
    
    def _add_default_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление дефолтных значений при недостатке данных"""
        
        default_features = {
            'ctx_fear_greed_index': 50,
            'ctx_market_sentiment': 'neutral',
            'ctx_bull_market_score': 33,
            'ctx_bear_market_score': 33,
            'ctx_sideways_score': 34,
            'ctx_market_regime': 'sideways',
            'ctx_regime_confidence': 0.1,
            'ctx_market_volatility': 0.02,
            'ctx_volatility_regime': 'medium',
            'ctx_vix_crypto': 30
        }
        
        for feature, default_value in default_features.items():
            df[feature] = default_value
        
        self.logger.warning("⚠️ Использованы дефолтные значения контекста (недостаточно символов)")
        return df
    
    def get_feature_names(self) -> List[str]:
        """Получить список всех контекстных признаков"""
        features = [
            # Fear & Greed компоненты
            'ctx_fear_greed_index', 'ctx_market_sentiment',
            'ctx_fg_volatility', 'ctx_fg_momentum', 'ctx_fg_volume', 'ctx_fg_breadth',
            
            # Market Regime
            'ctx_bull_market_score', 'ctx_bear_market_score', 'ctx_sideways_score',
            'ctx_market_regime', 'ctx_regime_confidence',
            
            # Volatility
            'ctx_market_volatility', 'ctx_volatility_regime', 'ctx_vix_crypto',
            
            # Breadth
            'ctx_market_participation', 'ctx_hl_ratio', 'ctx_breadth_oscillator',
            
            # Momentum
            'ctx_trending_pct', 'ctx_trend_alignment',
            
            # Volume
            'ctx_total_market_volume', 'ctx_market_volume_surge', 'ctx_volume_concentration',
            
            # Correlation
            'ctx_avg_correlation', 'ctx_correlation_regime'
        ]
        
        return features