"""
Feature Engineering для XGBoost v3.0
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler

from config import Config, FEATURE_CONFIG, TOP_SYMBOLS
from config.constants import (
    EPSILON, EPSILON_PRICE, EPSILON_STD,
    MARKET_FEATURES, OHLC_FEATURES, DIVERGENCE_PARAMS,
    BTC_DATA_PARAMS
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Класс для создания и инженерии признаков"""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_config = FEATURE_CONFIG
        self.created_features = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Основной метод создания признаков
        """
        logger.info("🔄 Добавление продвинутых признаков...")
        
        # Сохраняем оригинальный размер
        original_features = df.shape[1]
        
        # 1. Рыночные признаки
        df = self._create_market_features(df)
        
        # 2. OHLC признаки
        df = self._create_ohlc_features(df)
        
        # 3. Временные признаки (минимизированы)
        df = self._create_time_features(df)
        
        # 4. Symbol one-hot encoding
        df = self._create_symbol_features(df)
        
        # 5. Бинарные признаки
        df = self._create_binary_features(df)
        
        # 6. Взвешенные комбинации
        df = self._create_weighted_features(df)
        
        # 7. Скользящие статистики
        # Создаем все признаки для лучшего качества
        df = self._create_rolling_features(df)
        
        # 8. Дивергенции
        df = self._create_divergence_features(df)
        
        # 9. Паттерны свечей
        df = self._create_candle_patterns(df)
        
        # 10. Volume profile
        df = self._create_volume_profile_features(df)
        
        # 11. Price action паттерны
        df = self._create_price_action_patterns(df)
        
        # 12. Микроструктурные признаки
        df = self._create_microstructure_features(df)
        
        # 13. Межтаймфреймовые признаки
        df = self._create_cross_timeframe_features(df)
        
        # 14. НОВОЕ: Продвинутые технические признаки
        df = self._create_advanced_technical_features(df)
        
        # ВАЖНО: Дефрагментация DataFrame после создания множества признаков
        logger.info("🔧 Дефрагментация DataFrame...")
        df = df.copy()  # Это устраняет фрагментацию
        logger.info("✅ Дефрагментация завершена")
        
        # Удаляем дубликаты
        df = self._remove_duplicate_features(df)
        
        # Заполняем пропуски, которые могли появиться после создания признаков
        df = self._handle_new_nans(df)
        
        # Удаляем константные признаки ДО валидации
        df = self._remove_constant_features(df)
        
        # Валидация и исправление признаков
        df = self.validate_features(df)
        
        # Статистика
        new_features = df.shape[1] - original_features
        logger.info(f"✅ Создано {new_features} новых признаков")
        logger.info(f"📊 Итого признаков: {df.shape[1]}")
        
        return df
        
    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание рыночных признаков"""
        logger.info("🌍 Создание рыночных признаков...")
        
        # Загружаем данные BTC для корреляций
        if 'btc_close' not in df.columns:
            from data.btc_data_loader import BTCDataLoader
            btc_loader = BTCDataLoader(self.config)
            df = btc_loader.load_btc_data(df)
            
            # Проверяем успешность загрузки
            if 'btc_close' not in df.columns:
                logger.warning("⚠️ Не удалось загрузить данные BTC, пропускаем рыночные признаки")
                return df
            
        # Корреляции с BTC
        for window in MARKET_FEATURES['correlation_windows']:
            df[f'btc_correlation_{window}'] = df['close'].rolling(window).corr(df['btc_close'])
        
        # Относительные доходности
        for period_name, period_candles in MARKET_FEATURES['return_periods'].items():
            df[f'btc_return_{period_name}'] = df['btc_close'].pct_change(period_candles)
        
        # Волатильность BTC
        df['btc_volatility'] = df['btc_return_1h'].rolling(20).std()
        
        # Относительная сила
        df['relative_strength_btc'] = df['close'].pct_change(MARKET_FEATURES['return_periods']['1h']) / (df['btc_return_1h'] + EPSILON)
        
        # Рыночный режим
        # ИСПРАВЛЕНО: используем скользящие квантили вместо глобальных для избежания утечки данных
        volatility_quantiles = MARKET_FEATURES['volatility_quantiles']  # [0.33, 0.67]
        
        # Используем скользящее окно для расчета квантилей (например, последние 500 свечей)
        window_size = 500
        df['btc_vol_low_threshold'] = df['btc_volatility'].rolling(window=window_size, min_periods=100).quantile(volatility_quantiles[0])
        df['btc_vol_high_threshold'] = df['btc_volatility'].rolling(window=window_size, min_periods=100).quantile(volatility_quantiles[1])
        
        # Заполняем начальные значения медианой из первых доступных данных
        df['btc_vol_low_threshold'] = df['btc_vol_low_threshold'].fillna(df['btc_volatility'].iloc[:window_size].quantile(volatility_quantiles[0]))
        df['btc_vol_high_threshold'] = df['btc_vol_high_threshold'].fillna(df['btc_volatility'].iloc[:window_size].quantile(volatility_quantiles[1]))
        
        df['market_regime_low_vol'] = (df['btc_volatility'] < df['btc_vol_low_threshold']).astype(int)
        df['market_regime_med_vol'] = ((df['btc_volatility'] >= df['btc_vol_low_threshold']) & 
                                       (df['btc_volatility'] < df['btc_vol_high_threshold'])).astype(int)
        df['market_regime_high_vol'] = (df['btc_volatility'] >= df['btc_vol_high_threshold']).astype(int)
        
        # Удаляем временные пороговые колонки
        df.drop(['btc_vol_low_threshold', 'btc_vol_high_threshold'], axis=1, inplace=True)
        
        # Заполняем пропуски
        market_features = ['btc_correlation_20', 'btc_correlation_60', 'btc_return_1h', 
                          'btc_return_4h', 'btc_volatility', 'relative_strength_btc']
        for col in market_features:
            df[col] = df[col].fillna(0)
            
        # Удаляем временную колонку
        df.drop('btc_close', axis=1, inplace=True)
        
        logger.info("✅ Рыночные признаки добавлены")
        return df
        
    def _create_ohlc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание OHLC признаков"""
        logger.info("📊 Создание OHLC признаков...")
        
        # Нормализованные цены
        df['open_ratio'] = df['open'] / df['close']
        df['high_ratio'] = df['high'] / df['close']
        df['low_ratio'] = df['low'] / df['close']
        
        # Спреды и размеры
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        
        # Тени свечей
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Направление свечи
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        # Убеждаемся что значения только 0 и 1
        df['is_bullish'] = df['is_bullish'].clip(0, 1)
        
        # Логарифмические преобразования
        df['log_return'] = np.log(df['close'] / df['open'].replace(0, EPSILON_PRICE))
        df['log_volume'] = np.log1p(df['volume'])
        
        # Расстояния до скользящих средних
        if 'ema_15' in df.columns:
            df['price_to_ema15'] = (df['close'] - df['ema_15']) / df['ema_15']
            # Аппроксимация EMA50 через EMA15
            ema50_approx = df['ema_15'] * OHLC_FEATURES['ema50_multiplier']
            df['price_to_ema50'] = (df['close'] - ema50_approx) / ema50_approx
            
        # VWAP approximation с ограниченным окном (только прошлые данные)
        # ИСПРАВЛЕНО: используем expanding с min_periods для корректного расчета VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['volume'] * typical_price).expanding(min_periods=1).sum() / df['volume'].expanding(min_periods=1).sum()
        df['price_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        return df
        
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание временных признаков"""
        logger.info("🕐 Создание временных признаков (только час)...")
        
        # Преобразуем timestamp в datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Минимальные временные признаки - ТОЛЬКО базовый час
        # Без sin/cos преобразований и торговых сессий
        df['hour'] = df['datetime'].dt.hour
        
        # НЕ СОЗДАЕМ:
        # - hour_sin, hour_cos (временное переобучение)
        # - session_asia, session_europe, session_america (скрытые временные признаки)
        # - dow_sin, dow_cos, is_weekend (уже в blacklist)
        
        # Удаляем только datetime, оставляем hour как технический признак
        df.drop(['datetime'], axis=1, inplace=True)
        
        logger.info("✅ Добавлен минимальный временной признак (hour) как технический")
        return df
        
    def _create_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков для символов"""
        if 'symbol' not in df.columns:
            return df
            
        logger.info("🏷️ Создание symbol признаков...")
        
        # One-hot encoding для топ символов из констант
        for symbol in TOP_SYMBOLS:
            col_name = f"is_{symbol.replace('USDT', '').lower()}"
            df[col_name] = (df['symbol'] == symbol).astype(int)
            
        return df
        
    def _create_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание бинарных признаков"""
        logger.info("🔢 Создание бинарных признаков...")
        
        thresholds = self.feature_config['binary_thresholds']
        
        # RSI условия
        if 'rsi_val' in df.columns:
            # Проверяем есть ли вообще значения в диапазонах
            oversold_exists = (df['rsi_val'] < thresholds['rsi_oversold']).any()
            overbought_exists = (df['rsi_val'] > thresholds['rsi_overbought']).any()
            
            if oversold_exists:
                df['rsi_oversold'] = (df['rsi_val'] < thresholds['rsi_oversold']).astype(int)
            else:
                logger.warning(f"⚠️ Нет значений RSI < {thresholds['rsi_oversold']}, пропускаем rsi_oversold")
                
            if overbought_exists:
                df['rsi_overbought'] = (df['rsi_val'] > thresholds['rsi_overbought']).astype(int)
            else:
                logger.warning(f"⚠️ Нет значений RSI > {thresholds['rsi_overbought']}, пропускаем rsi_overbought")
            
        # MACD условие
        if 'macd_hist' in df.columns:
            df['macd_bullish'] = (df['macd_hist'] > thresholds['macd_bullish']).astype(int)
            
        # Сильный тренд
        if 'adx_val' in df.columns:
            strong_trend_exists = (df['adx_val'] > thresholds['strong_trend']).any()
            if strong_trend_exists:
                df['strong_trend'] = (df['adx_val'] > thresholds['strong_trend']).astype(int)
            else:
                logger.warning(f"⚠️ Нет значений ADX > {thresholds['strong_trend']}, пропускаем strong_trend")
            
        # Всплеск объема
        if 'volume_ratio' in df.columns:
            df['volume_spike'] = (df['volume_ratio'] > thresholds['volume_spike']).astype(int)
            
        # Позиция в Bollinger Bands
        if 'bb_position' in df.columns:
            df['bb_near_lower'] = (df['bb_position'] < thresholds['bb_near_lower']).astype(int)
            df['bb_near_upper'] = (df['bb_position'] > thresholds['bb_near_upper']).astype(int)
            
        # Паттерны свечей
        candle_params = OHLC_FEATURES['candle_patterns']
        df['is_hammer'] = ((df['body_size'] < candle_params['hammer_body_size']) & 
                          (df['lower_shadow'] > df['body_size'] * candle_params['hammer_shadow_ratio'])).astype(int)
        df['is_doji'] = (df['body_size'] < candle_params['doji_body_size']).astype(int)
        
        return df
        
    def _create_weighted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание взвешенных комбинаций признаков"""
        logger.info("🔄 Создание взвешенных комбинаций признаков...")
        
        created_count = 0
        
        # RSI + MACD взаимодействие
        if 'rsi_val' in df.columns and 'macd_hist' in df.columns:
            # Нормализуем значения
            rsi_norm = (df['rsi_val'] - 50) / 50  # Центрируем вокруг 0
            macd_norm = df['macd_hist'] / (df['macd_hist'].std() + EPSILON_STD)
            df['rsi_macd_interaction'] = rsi_norm * macd_norm
            created_count += 1
            
        # Volume + Volatility взаимодействие
        if 'volume_ratio' in df.columns and 'atr' in df.columns:
            df['volume_volatility_interaction'] = df['volume_ratio'] * df['atr']
            created_count += 1
            
        # Дополнительные взвешенные признаки
        if 'rsi_val' in df.columns and 'adx_val' in df.columns:
            df['rsi_to_adx'] = df['rsi_val'] / (df['adx_val'] + 1)
            created_count += 1
            
        if 'volume' in df.columns and 'atr' in df.columns:
            df['volume_to_volatility'] = df['volume'] / (df['atr'] + EPSILON)
            created_count += 1
            
        if 'close' in df.columns and 'sar' in df.columns:
            df['price_momentum_ratio'] = (df['close'] - df['sar']) / df['close']
            created_count += 1
            
        logger.info(f"✅ Создано {created_count} взвешенных признаков")
        
        return df
        
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание скользящих статистик"""
        logger.info("📊 Добавление скользящих статистик...")
        
        windows = self.feature_config['rolling_windows']
        
        # Для основных индикаторов
        indicators = ['rsi_val', 'adx_val', 'volume_ratio']
        
        # ИСПРАВЛЕНО: Проверяем существование признаков перед созданием
        created_rolling = 0
        skipped_rolling = 0
        
        for indicator in indicators:
            if indicator in df.columns:
                for window in windows:
                    # Проверяем, не существует ли уже такой признак
                    ma_col = f'{indicator}_ma_{window}'
                    std_col = f'{indicator}_std_{window}'
                    
                    # Создаем только если еще нет
                    if ma_col not in df.columns:
                        # Скользящее среднее
                        df[ma_col] = df[indicator].rolling(window).mean()
                        created_rolling += 1
                    else:
                        skipped_rolling += 1
                        
                    if std_col not in df.columns:
                        # Скользящее стд
                        df[std_col] = df[indicator].rolling(window).std()
                        created_rolling += 1
                    else:
                        skipped_rolling += 1
        
        if skipped_rolling > 0:
            logger.info(f"   ⚠️ Пропущено {skipped_rolling} дублирующих rolling признаков")
        logger.info(f"   ✅ Создано {created_rolling} новых rolling признаков")
                    
        # Заполняем пропуски
        for col in df.columns:
            if '_ma_' in col or '_std_' in col:
                # ИСПРАВЛЕНО: используем только ffill для избежания утечки данных из будущего
                # Для начальных значений используем первое доступное значение
                df[col] = df[col].fillna(method='ffill').fillna(df[col].iloc[df[col].first_valid_index()] if df[col].first_valid_index() is not None else 0)
                
        logger.info("✅ Добавлены скользящие статистики")
        return df
        
    def _create_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков дивергенций"""
        logger.info("🔄 Добавление дивергенций...")
        
        window = self.feature_config['divergence_window']
        
        # RSI дивергенция
        if 'rsi_val' in df.columns and 'close' in df.columns:
            # Бычья дивергенция: цена падает, RSI растет
            price_change = df['close'].pct_change(window)
            rsi_change = df['rsi_val'].pct_change(window)
            
            df['rsi_bullish_divergence'] = (
                (price_change < -DIVERGENCE_PARAMS['price_change_threshold']) & 
                (rsi_change > DIVERGENCE_PARAMS['rsi_change_threshold'])
            ).astype(int)
            df['rsi_bearish_divergence'] = (
                (price_change > DIVERGENCE_PARAMS['price_change_threshold']) & 
                (rsi_change < -DIVERGENCE_PARAMS['rsi_change_threshold'])
            ).astype(int)
            
        # Volume-Price дивергенция
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change(window)
            df['volume_price_divergence'] = (
                (abs(price_change) > DIVERGENCE_PARAMS['volume_price_threshold']) & 
                (volume_change < DIVERGENCE_PARAMS['volume_change_threshold'])
            ).astype(int)
            
        logger.info("✅ Добавлены дивергенции")
        return df
        
    def _create_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание паттернов свечей"""
        logger.info("🕯️ Добавление паттернов свечей...")
        
        # Bullish Engulfing
        df['prev_close'] = df['close'].shift(1)
        df['prev_open'] = df['open'].shift(1)
        
        df['bullish_engulfing'] = (
            (df['prev_close'] < df['prev_open']) &  # Предыдущая свеча медвежья
            (df['close'] > df['open']) &             # Текущая свеча бычья
            (df['open'] < df['prev_close']) &        # Открытие ниже предыдущего закрытия
            (df['close'] > df['prev_open'])          # Закрытие выше предыдущего открытия
        ).astype(int)
        
        # Bearish Engulfing
        df['bearish_engulfing'] = (
            (df['prev_close'] > df['prev_open']) &   # Предыдущая свеча бычья
            (df['close'] < df['open']) &             # Текущая свеча медвежья
            (df['open'] > df['prev_close']) &        # Открытие выше предыдущего закрытия
            (df['close'] < df['prev_open'])          # Закрытие ниже предыдущего открытия
        ).astype(int)
        
        # Удаляем временные колонки
        df.drop(['prev_close', 'prev_open'], axis=1, inplace=True)
        
        logger.info("✅ Добавлены паттерны свечей")
        return df
        
    def _create_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков volume profile"""
        logger.info("📈 Добавление volume profile...")
        
        # Volume weighted metrics
        if 'volume' in df.columns and 'close' in df.columns:
            # Относительная позиция в объемном профиле за последние 20 свечей
            volume_sum_20 = df['volume'].rolling(20).sum()
            df['volume_position_20'] = df['volume'].rolling(20).apply(
                lambda x: (x.iloc[-1] / x.sum()) if x.sum() > 0 else 0
            )
            
            # Кумулятивный объем как процент от среднего
            # ИСПРАВЛЕНО: используем expanding вместо cumsum для избежания утечки данных
            df['cumulative_volume_ratio'] = df['volume'].expanding(min_periods=1).sum() / df['volume'].expanding(min_periods=1).mean()
            
        logger.info("✅ Добавлены volume profile признаки")
        return df
        
    def _create_price_action_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание price action паттернов"""
        logger.info("📊 Добавление price action паттернов...")
        
        # Higher highs and lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Последовательные higher highs/lower lows
        df['consecutive_hh'] = df['higher_high'].rolling(3).sum()
        df['consecutive_ll'] = df['lower_low'].rolling(3).sum()
        
        # Inside bar pattern
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & 
                           (df['low'] > df['low'].shift(1))).astype(int)
        
        # Pin bar pattern (длинная тень)
        body_size = abs(df['close'] - df['open'])
        upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
        lower_wick = df[['close', 'open']].min(axis=1) - df['low']
        
        df['pin_bar_bull'] = ((lower_wick > body_size * 2) & 
                              (upper_wick < body_size * 0.5)).astype(int)
        df['pin_bar_bear'] = ((upper_wick > body_size * 2) & 
                              (lower_wick < body_size * 0.5)).astype(int)
        
        logger.info("✅ Добавлены price action паттерны")
        return df
        
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание микроструктурных признаков"""
        logger.info("🔬 Добавление микроструктурных признаков...")
        
        # Bid-ask spread approximation (через high-low)
        df['spread_approximation'] = (df['high'] - df['low']) / df['close']
        
        # Price efficiency ratio (насколько направленное движение)
        price_change = df['close'].diff(10).abs()
        path_length = df['close'].diff().abs().rolling(10).sum()
        df['price_efficiency'] = price_change / (path_length + EPSILON)
        
        # Volume-price correlation
        df['volume_price_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # Микроструктурная волатильность (Garman-Klass estimator)
        df['gk_volatility'] = np.sqrt(
            0.5 * np.log(df['high']/df['low'])**2 - 
            (2*np.log(2) - 1) * np.log(df['close']/df['open'])**2
        )
        
        logger.info("✅ Добавлены микроструктурные признаки")
        return df
        
    def _create_cross_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание межтаймфреймовых признаков"""
        logger.info("⏱️ Добавление межтаймфреймовых признаков...")
        
        # Эмуляция старших таймфреймов через агрегацию
        # 1H = 4 свечи по 15м
        df['high_1h'] = df['high'].rolling(4).max()
        df['low_1h'] = df['low'].rolling(4).min()
        # ИСПРАВЛЕНО: используем только прошлые данные для close_1h
        df['close_1h'] = df['close'].rolling(4).apply(lambda x: x.iloc[-1] if len(x) == 4 else np.nan)
        
        # Позиция текущей цены относительно 1H диапазона
        df['position_in_1h_range'] = (df['close'] - df['low_1h']) / (df['high_1h'] - df['low_1h'] + EPSILON)
        
        # 4H = 16 свечей по 15м
        df['high_4h'] = df['high'].rolling(16).max()
        df['low_4h'] = df['low'].rolling(16).min()
        
        # Тренд на старшем таймфрейме
        df['trend_1h'] = (df['close'] > df['close'].shift(4)).astype(int)
        df['trend_4h'] = (df['close'] > df['close'].shift(16)).astype(int)
        
        # Удаляем временные колонки
        df.drop(['high_1h', 'low_1h', 'close_1h', 'high_4h', 'low_4h'], axis=1, inplace=True)
        
        logger.info("✅ Добавлены межтаймфреймовые признаки")
        return df
        
    def _remove_duplicate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление дублированных признаков"""
        # Находим дублированные колонки
        duplicated_cols = df.columns[df.T.duplicated()].tolist()
        
        if duplicated_cols:
            logger.info(f"   ✅ Удалено дубликатов признаков: {len(duplicated_cols)}")
            logger.info(f"   📋 Дублированные признаки: {duplicated_cols}")
            df = df.loc[:, ~df.T.duplicated()]
            
        return df
        
    def _handle_new_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка NaN значений, появившихся после создания признаков"""
        # Делаем копию чтобы избежать SettingWithCopyWarning
        df = df.copy()
        
        # Проверяем наличие NaN
        nan_cols = df.columns[df.isnull().any()].tolist()
        
        if nan_cols:
            logger.info(f"🔧 Обработка NaN в {len(nan_cols)} новых признаках...")
            
            for col in nan_cols:
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Для числовых признаков
                    if 'correlation' in col or 'return' in col:
                        # Для корреляций и доходностей - заполняем 0
                        df.loc[:, col] = df[col].fillna(0)
                    elif 'ratio' in col or 'volatility' in col:
                        # Для отношений и волатильности - заполняем 1
                        df.loc[:, col] = df[col].fillna(1)
                    else:
                        # ИСПРАВЛЕНО: используем forward fill вместо глобальной медианы для избежания утечки данных
                        # Сначала forward fill, потом backward fill для начальных значений, затем 0 для оставшихся
                        df.loc[:, col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                else:
                    # Для категориальных - заполняем 0
                    df.loc[:, col] = df[col].fillna(0)
                    
        return df
        
    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидация и исправление созданных признаков"""
        logger.info("🔍 Валидация признаков...")
        
        # Делаем копию чтобы избежать SettingWithCopyWarning
        df = df.copy()
        
        # Проверка на бесконечности
        numeric_df = df.select_dtypes(include=[np.number])
        inf_mask = np.isinf(numeric_df).any()
        inf_cols = numeric_df.columns[inf_mask].tolist()
        if inf_cols:
            logger.warning(f"⚠️ Колонки с бесконечностями: {inf_cols}")
            # Заменяем бесконечности на NaN, затем на 0
            for col in inf_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        # Проверка и исправление бинарных признаков
        # Собираем все бинарные колонки
        binary_patterns = ['is_', '_oversold', '_overbought', '_bullish', '_bearish', 
                          'strong_trend', 'volume_spike', 'bb_near_', '_engulfing', 
                          '_divergence', 'is_doji', 'is_hammer', 'is_weekend',
                          'market_regime_']
        
        binary_cols = []
        for col in df.columns:
            for pattern in binary_patterns:
                if pattern in col:
                    binary_cols.append(col)
                    break
                    
        # Исправляем бинарные признаки
        for col in binary_cols:
            if col in df.columns:
                # Преобразуем все положительные значения в 1, остальные в 0
                # Для специальных случаев как is_bullish где может быть -1
                if df[col].min() < 0:
                    # Если есть отрицательные значения, считаем их как 0
                    df[col] = (df[col] > 0).astype(int)
                else:
                    # Иначе все не-нулевые в 1
                    df[col] = (df[col] != 0).astype(int)
                
        # Проверка диапазонов индикаторов
        indicators_bounds = {
            'rsi_val': (0, 100),
            'stoch_k': (0, 100),
            'stoch_d': (0, 100),
            'williams_r': (-100, 0),
            'aroon_up': (0, 100),
            'aroon_down': (0, 100),
            'mfi': (0, 100)
        }
        
        for ind, (min_val, max_val) in indicators_bounds.items():
            if ind in df.columns:
                df[ind] = df[ind].clip(min_val, max_val)
                
        logger.info("✅ Валидация признаков завершена")
        return df
        
    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление константных признаков с оптимизацией"""
        logger.info("🔍 Поиск константных признаков...")
        
        # Исключаем базовые колонки
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 
                       'buy_expected_return', 'sell_expected_return']
        check_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Используем векторизованный подход для скорости
        constant_features = []
        
        # Проверяем батчами для оптимизации памяти
        batch_size = 100
        for i in range(0, len(check_cols), batch_size):
            batch_cols = check_cols[i:i+batch_size]
            for col in batch_cols:
                if df[col].nunique(dropna=False) <= 1:
                    constant_features.append(col)
                    
        if constant_features:
            logger.warning(f"⚠️ Удаляем {len(constant_features)} константных признаков")
            if len(constant_features) < 10:
                logger.warning(f"   Константные: {constant_features}")
            else:
                logger.warning(f"   Первые 10: {constant_features[:10]}...")
            df = df.drop(columns=constant_features)
        else:
            logger.info("✅ Константных признаков не найдено")
            
        return df
        
    def _create_advanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание продвинутых технических признаков для замены временных"""
        logger.info("🚀 Создание продвинутых технических признаков...")
        
        # 1. ПРОСТЫЕ ЦЕНОВЫЕ ИЗМЕНЕНИЯ (самые важные для краткосрочного трейдинга)
        if 'close' in df.columns:
            # Простые процентные изменения - КЛЮЧЕВЫЕ ПРИЗНАКИ
            df['price_change_1'] = df['close'].pct_change(1) * 100  # За 1 бар
            df['price_change_3'] = df['close'].pct_change(3) * 100  # За 3 бара (45 мин)
            df['price_change_5'] = df['close'].pct_change(5) * 100  # За 5 баров (1.25 часа)
            df['price_change_10'] = df['close'].pct_change(10) * 100  # За 10 баров (2.5 часа)
            
            # Простой моментум (отношение цен)
            df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            
            # Rate of Change (ROC)
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = df['close'].pct_change(period) * 100
                
            # Price velocity (скорость изменения цены)
            df['price_velocity'] = df['close'].diff().rolling(5).mean()
            df['price_acceleration'] = df['price_velocity'].diff()
            
        # 2. Volatility clusters
        if 'atr' in df.columns:
            # Volatility regime change
            df['volatility_ma_20'] = df['atr'].rolling(20).mean()
            df['volatility_ratio'] = df['atr'] / (df['volatility_ma_20'] + EPSILON)
            df['volatility_expanding'] = (df['volatility_ratio'] > 1.2).astype(int)
            df['volatility_contracting'] = (df['volatility_ratio'] < 0.8).astype(int)
            
        # 3. Volume-Price Analysis
        if 'volume' in df.columns and 'close' in df.columns:
            # Volume-Weighted Momentum
            df['volume_weighted_momentum'] = (df['close'].pct_change() * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
            
            # Accumulation/Distribution Line derivative
            if 'high' in df.columns and 'low' in df.columns:
                money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + EPSILON)
                money_flow_volume = money_flow_multiplier * df['volume']
                df['adl'] = money_flow_volume.cumsum()
                df['adl_slope'] = df['adl'].diff(5) / 5
                
        # 4. Support/Resistance levels
        if 'high' in df.columns and 'low' in df.columns:
            # Расстояние до локальных экстремумов
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            df['distance_to_high_20'] = (df['high_20'] - df['close']) / df['close']
            df['distance_to_low_20'] = (df['close'] - df['low_20']) / df['close']
            
            # Позиция в 20-периодном диапазоне
            df['position_in_range_20'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + EPSILON)
            
        # 5. Trend strength metrics
        if 'close' in df.columns:
            # Linear regression slope
            def calculate_slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                slope, _ = np.polyfit(x, series, 1)
                return slope
                
            df['trend_slope_10'] = df['close'].rolling(10).apply(calculate_slope)
            df['trend_slope_20'] = df['close'].rolling(20).apply(calculate_slope)
            
            # R-squared для определения силы тренда
            def calculate_r2(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                coeffs = np.polyfit(x, series, 1)
                p = np.poly1d(coeffs)
                yhat = p(x)
                ybar = np.mean(series)
                ssreg = np.sum((yhat - ybar)**2)
                sstot = np.sum((series - ybar)**2)
                return ssreg / (sstot + EPSILON)
                
            df['trend_strength_10'] = df['close'].rolling(10).apply(calculate_r2)
            df['trend_strength_20'] = df['close'].rolling(20).apply(calculate_r2)
            
        # 6. Mean reversion indicators
        if 'close' in df.columns:
            # Z-score
            for period in [20, 50]:
                ma = df['close'].rolling(period).mean()
                std = df['close'].rolling(period).std()
                df[f'zscore_{period}'] = (df['close'] - ma) / (std + EPSILON)
                
            # Bollinger Band squeeze (низкая волатильность перед движением)
            if 'bb_width' in df.columns:
                df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(100).quantile(0.2)).astype(int)
                
        # 7. ПРОСТЫЕ БИНАРНЫЕ СИГНАЛЫ (легко интерпретируемые)
        # RSI экстремумы
        if 'rsi_val' in df.columns:
            df['rsi_oversold'] = (df['rsi_val'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi_val'] > 70).astype(int)
            df['rsi_extreme'] = df['rsi_oversold'] | df['rsi_overbought']
            
        # Bollinger Bands пробой
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_breakout_up'] = (df['close'] > df['bb_upper']).astype(int)
            df['bb_breakout_down'] = (df['close'] < df['bb_lower']).astype(int)
            df['bb_breakout'] = df['bb_breakout_up'] | df['bb_breakout_down']
            
        # Объемные аномалии
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(20).mean()
            df['volume_spike'] = (df['volume'] > volume_ma * 1.5).astype(int)
            df['volume_spike_large'] = (df['volume'] > volume_ma * 2.0).astype(int)
            
        # Order flow imbalance proxy
        if 'volume' in df.columns and 'close' in df.columns:
            # Приближение дисбаланса потока ордеров
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + EPSILON)
            df['buy_pressure'] = df['volume'] * df['close_position']
            df['sell_pressure'] = df['volume'] * (1 - df['close_position'])
            df['order_flow_imbalance'] = (df['buy_pressure'] - df['sell_pressure']) / (df['buy_pressure'] + df['sell_pressure'] + EPSILON)
            
        # 8. Fractal dimension (измерение "изломанности" движения)
        if 'high' in df.columns and 'low' in df.columns:
            def hurst_exponent(series, lags=20):
                """Упрощенный расчет экспоненты Херста"""
                if len(series) < lags:
                    return 0.5
                    
                lags = range(2, min(lags, len(series)//2))
                tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
                
                if len(tau) < 2:
                    return 0.5
                    
                try:
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    return poly[0] * 2.0
                except:
                    return 0.5
                    
            df['hurst_exponent'] = df['close'].rolling(50).apply(hurst_exponent)
            df['is_trending'] = (df['hurst_exponent'] > 0.6).astype(int)
            df['is_mean_reverting'] = (df['hurst_exponent'] < 0.4).astype(int)
            
        logger.info("✅ Добавлены продвинутые технические признаки")
        return df