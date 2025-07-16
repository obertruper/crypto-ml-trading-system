"""
Инженерия признаков для криптовалютных данных
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
# Для совместимости с логированием
from tqdm.contrib.logging import logging_redirect_tqdm
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

class FeatureEngineer:
    """Создание признаков для модели прогнозирования"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("FeatureEngineer")
        self.feature_config = config['features']
        self.scalers = {}
        self.process_position = None  # Позиция для прогресс-баров при параллельной обработке
        self.disable_progress = False  # Флаг для отключения прогресс-баров
    
    @staticmethod
    def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value=0.0, max_value=1000.0, min_denominator=1e-8) -> pd.Series:
        """ИСПРАВЛЕНО: Безопасное деление с правильной обработкой малых значений"""
        # Создаем безопасный знаменатель
        safe_denominator = denominator.copy()
        
        # Заменяем очень маленькие значения
        mask_small = (safe_denominator.abs() < min_denominator)
        safe_denominator[mask_small] = min_denominator
        
        # Выполняем деление
        result = numerator / safe_denominator
        
        # Клиппинг результата для предотвращения экстремальных значений
        result = result.clip(lower=-max_value, upper=max_value)
        
        # Обработка inf и nan
        result = result.replace([np.inf, -np.inf], [fill_value, fill_value])
        result = result.fillna(fill_value)
        
        return result
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Улучшенный расчет VWAP с дополнительными проверками"""
        # Базовый расчет VWAP
        vwap = self.safe_divide(
            df['turnover'], 
            df['volume'], 
            fill_value=df['close']
        )
        
        # Дополнительная проверка: VWAP не должен сильно отличаться от close
        # Если VWAP слишком отличается от close (более чем в 2 раза), используем close
        mask_invalid = (vwap < df['close'] * 0.5) | (vwap > df['close'] * 2.0)
        vwap[mask_invalid] = df['close'][mask_invalid]
        
        return vwap
        
    def create_features(self, df: pd.DataFrame, train_end_date: Optional[str] = None, 
                       use_enhanced_features: bool = False) -> pd.DataFrame:
        """Создание всех признаков для датасета с walk-forward валидацией
        
        Args:
            df: DataFrame с raw данными
            train_end_date: дата окончания обучения для walk-forward нормализации
            use_enhanced_features: использовать ли расширенные признаки для direction prediction
        """
        if not self.disable_progress:
            self.logger.start_stage("feature_engineering", 
                                   symbols=df['symbol'].nunique())
        
        # Валидация данных
        self._validate_data(df)
        
        featured_dfs = []
        all_symbols_data = {}  # Для enhanced features
        
        # Первый проход - базовые признаки
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('datetime')
            
            symbol_data = self._create_basic_features(symbol_data)
            symbol_data = self._create_technical_indicators(symbol_data)
            symbol_data = self._create_microstructure_features(symbol_data)
            symbol_data = self._create_rally_detection_features(symbol_data)
            symbol_data = self._create_signal_quality_features(symbol_data)
            symbol_data = self._create_futures_specific_features(symbol_data)
            symbol_data = self._create_ml_optimized_features(symbol_data)
            symbol_data = self._create_temporal_features(symbol_data)
            symbol_data = self._create_target_variables(symbol_data)
            
            featured_dfs.append(symbol_data)
            
            # Сохраняем для enhanced features
            if use_enhanced_features:
                all_symbols_data[symbol] = symbol_data.copy()
        
        result_df = pd.concat(featured_dfs, ignore_index=True)
        
        # ИСПРАВЛЕНО: cross-asset features нужны все символы, но если обрабатываем по одному - пропускаем
        # Если в df больше одного символа - создаем cross-asset features
        if df['symbol'].nunique() > 1:
            result_df = self._create_cross_asset_features(result_df)
        
        # Добавляем enhanced features если запрошено
        if use_enhanced_features:
            result_df = self._add_enhanced_features(result_df, all_symbols_data)
        
        # Обработка NaN значений
        result_df = self._handle_missing_values(result_df)
        
        # Walk-forward нормализация только если указана дата (иначе нормализация будет в prepare_trading_data.py)
        if train_end_date:
            result_df = self._normalize_walk_forward(result_df, train_end_date)
        
        self._log_feature_statistics(result_df)
        
        if not self.disable_progress:
            self.logger.end_stage("feature_engineering", 
                                total_features=len(result_df.columns))
        
        return result_df
    
    def _validate_data(self, df: pd.DataFrame):
        """Валидация целостности данных"""
        # ИСПРАВЛЕНО: Конвертация числовых колонок в правильные типы
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_columns:
            if col in df.columns:
                # Конвертируем в числовой тип, заменяя ошибки на NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Заполняем NaN значения предыдущими значениями
                df[col] = df[col].ffill().bfill()
        
        # Проверка на отсутствующие значения
        if df.isnull().any().any():
            if not self.disable_progress:
                self.logger.warning("Обнаружены пропущенные значения в данных")
            
        # Проверка на аномальные цены
        price_changes = df.groupby('symbol')['close'].pct_change()
        extreme_moves = abs(price_changes) > 0.15  # >15% за 15 минут
        
        if extreme_moves.sum() > 0:
            if not self.disable_progress:
                self.logger.warning(f"Обнаружено {extreme_moves.sum()} экстремальных движений цены")
            
        # Проверка временных гэпов (только значительные разрывы > 2 часов)
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            time_diff = symbol_data['datetime'].diff()
            expected_diff = pd.Timedelta('15 minutes')
            # Считаем большими только разрывы больше 2 часов (8 интервалов)
            large_gaps = time_diff > expected_diff * 8
            
            if large_gaps.sum() > 0:
                if not self.disable_progress:
                    self.logger.warning(f"Символ {symbol}: обнаружено {large_gaps.sum()} значительных временных разрывов (> 2 часов)")
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Базовые признаки из OHLCV данных без look-ahead bias"""
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Доходности за разные периоды
        for period in [5, 10, 20]:
            df[f'returns_{period}'] = np.log(
                df['close'] / df['close'].shift(period)
            )
        
        # Ценовые соотношения
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Позиция закрытия в диапазоне
        df['close_position'] = (
            (df['close'] - df['low']) / 
            (df['high'] - df['low'] + 1e-10)
        )
        
        # Объемные соотношения с использованием только исторических данных
        df['volume_ratio'] = self.safe_divide(
            df['volume'], 
            df['volume'].rolling(20, min_periods=20).mean(),
            fill_value=1.0
        )
        df['turnover_ratio'] = self.safe_divide(
            df['turnover'], 
            df['turnover'].rolling(20, min_periods=20).mean(),
            fill_value=1.0
        )
        
        # VWAP с улучшенным расчетом
        df['vwap'] = self.calculate_vwap(df)
        
        # Более надежный расчет close_vwap_ratio
        # Нормальное соотношение close/vwap должно быть около 1.0
        # VWAP уже проверен и исправлен в calculate_vwap()
        
        # Простой и надежный расчет
        df['close_vwap_ratio'] = df['close'] / df['vwap']
        
        # ИСПРАВЛЕНО: Расширенные границы для криптовалют (±30%)
        # Криптовалюты могут отклоняться от VWAP на 20-50% в периоды высокой волатильности
        df['close_vwap_ratio'] = df['close_vwap_ratio'].clip(lower=0.7, upper=1.3)
        
        # Добавляем индикатор экстремального отклонения от VWAP
        df['vwap_extreme_deviation'] = (
            (df['close_vwap_ratio'] < 0.85) | (df['close_vwap_ratio'] > 1.15)
        ).astype(int)
        
        # Дополнительная проверка на аномалии
        # Если ratio все еще выходит за разумные пределы, заменяем на 1.0
        mask_invalid = (df['close_vwap_ratio'] < 0.95) | (df['close_vwap_ratio'] > 1.05)
        if mask_invalid.sum() > 0:
            self.logger.debug(f"Заменено {mask_invalid.sum()} аномальных close_vwap_ratio на 1.0")
            df.loc[mask_invalid, 'close_vwap_ratio'] = 1.0
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Технические индикаторы"""
        tech_config = self.feature_config['technical']
        
        # SMA
        sma_config = next((c for c in tech_config if c['name'] == 'sma'), None)
        if sma_config:
            for period in sma_config['periods']:
                df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], period)
                df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # EMA
        ema_config = next((c for c in tech_config if c['name'] == 'ema'), None)
        if ema_config:
            for period in ema_config['periods']:
                df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], period)
                df[f'close_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
        
        # RSI
        rsi_config = next((c for c in tech_config if c['name'] == 'rsi'), None)
        if rsi_config:
            df['rsi'] = ta.momentum.RSIIndicator(
                df['close'], 
                window=rsi_config['period']
            ).rsi()
            
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        macd_config = next((c for c in tech_config if c['name'] == 'macd'), None)
        if macd_config:
            macd = ta.trend.MACD(
                df['close'],
                window_slow=macd_config['slow'],
                window_fast=macd_config['fast'],
                window_sign=macd_config['signal']
            )
            # Нормализуем MACD относительно цены для сравнимости между активами
            # MACD в абсолютных значениях может быть очень большим для дорогих активов
            df['macd'] = macd.macd() / df['close'] * 100  # В процентах от цены
            df['macd_signal'] = macd.macd_signal() / df['close'] * 100
            df['macd_diff'] = macd.macd_diff() / df['close'] * 100
        
        # Bollinger Bands
        bb_config = next((c for c in tech_config if c['name'] == 'bollinger_bands'), None)
        if bb_config:
            bb = ta.volatility.BollingerBands(
                df['close'],
                window=bb_config['period'],
                window_dev=bb_config['std_dev']
            )
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            # ИСПРАВЛЕНО: bb_width как процент от цены
            df['bb_width'] = self.safe_divide(
                df['bb_high'] - df['bb_low'],
                df['close'],
                fill_value=0.02,  # 2% по умолчанию
                max_value=0.5   # Максимум 50% от цены
            )
            
            # ИСПРАВЛЕНО: bb_position теперь корректно рассчитывается с использованием абсолютной ширины
            # bb_position показывает где находится цена внутри канала Bollinger
            bb_range = df['bb_high'] - df['bb_low']
            df['bb_position'] = self.safe_divide(
                df['close'] - df['bb_low'],
                bb_range,
                fill_value=0.5,
                max_value=2.0  # Позволяем выходы за пределы для отслеживания прорывов
            )
            
            # Создаем индикаторы прорывов ПЕРЕД клиппингом
            df['bb_breakout_upper'] = (df['bb_position'] > 1).astype(int)
            df['bb_breakout_lower'] = (df['bb_position'] < 0).astype(int)
            df['bb_breakout_strength'] = np.abs(df['bb_position'] - 0.5) * 2  # Сила отклонения от центра
            
            # Теперь ограничиваем для совместимости
            df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # ATR
        atr_config = next((c for c in tech_config if c['name'] == 'atr'), None)
        if atr_config:
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], 
                df['low'], 
                df['close'],
                window=atr_config['period']
            ).average_true_range()
            
            # ATR в процентах от цены с ограничением экстремальных значений
            df['atr_pct'] = self.safe_divide(
                df['atr'], 
                df['close'],
                fill_value=0.01,  # 1% по умолчанию
                max_value=0.2     # Максимум 20% от цены
            )
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            df['high'], 
            df['low'], 
            df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Parabolic SAR
        psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
        df['psar'] = psar.psar()
        # Вместо отдельных psar_up и psar_down, создаем индикатор направления
        df['psar_trend'] = (df['close'] > df['psar']).astype(float)
        
        # ИСПРАВЛЕНО: Нормализованное расстояние PSAR по волатильности
        # Деление на ATR делает метрику сравнимой между активами
        df['psar_distance'] = (df['close'] - df['psar']) / df['close']
        if 'atr' in df.columns:
            df['psar_distance_normalized'] = (df['close'] - df['psar']) / (df['atr'] + 1e-10)
        else:
            df['psar_distance_normalized'] = df['psar_distance']
        
        # ===== НОВЫЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (2024 best practices) =====
        
        # 1. Ichimoku Cloud - популярный в крипто
        try:
            ichimoku = ta.trend.IchimokuIndicator(
                high=df['high'],
                low=df['low'],
                window1=9,     # Tenkan-sen
                window2=26,    # Kijun-sen  
                window3=52     # Senkou Span B
            )
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_span_a'] = ichimoku.ichimoku_a()
            df['ichimoku_span_b'] = ichimoku.ichimoku_b()
            # Облако - расстояние между span A и B
            df['ichimoku_cloud_thickness'] = (df['ichimoku_span_a'] - df['ichimoku_span_b']) / df['close']
            # Позиция цены относительно облака
            df['price_vs_cloud'] = ((df['close'] - (df['ichimoku_span_a'] + df['ichimoku_span_b']) / 2) / df['close'])
        except:
            pass
        
        # 2. Keltner Channels - альтернатива Bollinger Bands
        try:
            keltner = ta.volatility.KeltnerChannel(
                high=df['high'],
                low=df['low'], 
                close=df['close'],
                window=20,
                window_atr=10
            )
            df['keltner_upper'] = keltner.keltner_channel_hband()
            df['keltner_middle'] = keltner.keltner_channel_mband()
            df['keltner_lower'] = keltner.keltner_channel_lband()
            df['keltner_position'] = (df['close'] - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'])
        except:
            pass
        
        # 3. Donchian Channels - для определения прорывов
        try:
            donchian = ta.volatility.DonchianChannel(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=20
            )
            df['donchian_upper'] = donchian.donchian_channel_hband()
            df['donchian_middle'] = donchian.donchian_channel_mband()
            df['donchian_lower'] = donchian.donchian_channel_lband()
            # Индикатор прорыва
            df['donchian_breakout'] = ((df['close'] > df['donchian_upper'].shift(1)) | 
                                       (df['close'] < df['donchian_lower'].shift(1))).astype(int)
        except:
            pass
        
        # 4. Volume Weighted Moving Average (VWMA)
        df['vwma_20'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['close_vwma_ratio'] = df['close'] / df['vwma_20']
        
        # 5. Money Flow Index (MFI) - объемный осциллятор
        try:
            mfi = ta.volume.MFIIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=14
            )
            df['mfi'] = mfi.money_flow_index()
            df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
            df['mfi_oversold'] = (df['mfi'] < 20).astype(int)
        except:
            pass
        
        # 6. Commodity Channel Index (CCI)
        try:
            cci = ta.trend.CCIIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=20
            )
            df['cci'] = cci.cci()
            df['cci_overbought'] = (df['cci'] > 100).astype(int)
            df['cci_oversold'] = (df['cci'] < -100).astype(int)
        except:
            pass
        
        # 7. Williams %R
        try:
            williams = ta.momentum.WilliamsRIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                lbp=14
            )
            df['williams_r'] = williams.williams_r()
        except:
            pass
        
        # 8. Ultimate Oscillator - комбинирует несколько периодов
        try:
            ultimate = ta.momentum.UltimateOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window1=7,
                window2=14,
                window3=28
            )
            df['ultimate_oscillator'] = ultimate.ultimate_oscillator()
        except:
            pass
        
        # 9. Accumulation/Distribution Index
        try:
            adl = ta.volume.AccDistIndexIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            df['accumulation_distribution'] = adl.acc_dist_index()
        except:
            pass
        
        # 10. On Balance Volume (OBV)
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'],
                volume=df['volume']
            )
            df['obv'] = obv.on_balance_volume()
            # OBV trend
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            df['obv_trend'] = (df['obv'] > df['obv_ema']).astype(int)
        except:
            pass
        
        # 11. Chaikin Money Flow (CMF)
        try:
            cmf = ta.volume.ChaikinMoneyFlowIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=20
            )
            df['cmf'] = cmf.chaikin_money_flow()
        except:
            pass
        
        # 12. Average Directional Movement Index Rating (ADXR)
        try:
            adxr = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            df['adxr'] = adxr.adx().rolling(14).mean()  # ADXR = среднее ADX
        except:
            pass
        
        # 13. Aroon Indicator
        try:
            aroon = ta.trend.AroonIndicator(
                close=df['close'],
                window=25
            )
            df['aroon_up'] = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
        except:
            pass
        
        # 14. Pivot Points (поддержка/сопротивление)
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance1'] = 2 * df['pivot'] - df['low']
        df['support1'] = 2 * df['pivot'] - df['high']
        df['resistance2'] = df['pivot'] + (df['high'] - df['low'])
        df['support2'] = df['pivot'] - (df['high'] - df['low'])
        
        # Расстояние до уровней
        df['dist_to_resistance1'] = (df['resistance1'] - df['close']) / df['close']
        df['dist_to_support1'] = (df['close'] - df['support1']) / df['close']
        
        # 15. Rate of Change (ROC)
        try:
            roc = ta.momentum.ROCIndicator(
                close=df['close'],
                window=10
            )
            df['roc'] = roc.roc()
        except:
            pass
        
        # 16. Trix - тройное экспоненциальное сглаживание
        try:
            trix = ta.trend.TRIXIndicator(
                close=df['close'],
                window=15
            )
            df['trix'] = trix.trix()
        except:
            pass
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Признаки микроструктуры рынка"""
        # Спред high-low
        df['hl_spread'] = self.safe_divide(df['high'] - df['low'], df['close'], fill_value=0.0)
        df['hl_spread_ma'] = df['hl_spread'].rolling(20).mean()
        
        # Направление цены и объем
        df['price_direction'] = np.sign(df['close'] - df['open'])
        df['directed_volume'] = df['volume'] * df['price_direction']
        df['volume_imbalance'] = df['directed_volume'].rolling(10).sum() / \
                                 df['volume'].rolling(10).sum()
        
        # Ценовое воздействие - улучшенная формула
        # ИСПРАВЛЕНО: Используем dollar volume для более точной оценки
        df['dollar_volume'] = df['volume'] * df['close']
        # ИСПРАВЛЕНО v3: Масштабируем price_impact для криптовалют
        # где dollar_volume может быть от $10K до $100M+
        # log10($10K) ≈ 4, log10($1M) ≈ 6, log10($100M) ≈ 8
        # Умножаем на 100 для получения значимых значений price_impact
        df['price_impact'] = self.safe_divide(
            df['returns'].abs() * 100,  # Умножаем на 100 для правильного масштаба
            np.log10(df['dollar_volume'] + 100),  # log10 для правильного масштаба
            fill_value=0.0,
            max_value=0.1,  # Лимит для нового масштаба
        )
        
        # Альтернативная формула с логарифмом объема
        df['price_impact_log'] = self.safe_divide(
            df['returns'].abs(),
            np.log(df['volume'] + 10),  # Увеличен сдвиг для стабильности
            fill_value=0.0,
            max_value=10.0
        )
        
        # ИСПРАВЛЕНО v3: Используем экспоненциальную формулу для toxicity
        # toxicity = exp(-price_impact * 20)
        # С новым масштабированием price_impact:
        # При price_impact=0.04: toxicity≈0.45
        # При price_impact=0.02: toxicity≈0.67
        # При price_impact=0.01: toxicity≈0.82
        df['toxicity'] = np.exp(-df['price_impact'] * 20)
        df['toxicity'] = df['toxicity'].clip(0.3, 1.0)
        
        # Амихуд неликвидность - скорректированная формула
        # Традиционная формула: |returns| / dollar_volume
        # Но мы масштабируем на миллион для получения значимых значений
        df['amihud_illiquidity'] = self.safe_divide(
            df['returns'].abs() * 1e6,  # Масштабируем на миллион
            df['turnover'],
            fill_value=0.0,
            max_value=100.0  # Ограничиваем разумным максимумом
        )
        df['amihud_ma'] = df['amihud_illiquidity'].rolling(20).mean()
        
        # Кайл лямбда - правильная формула
        # ИСПРАВЛЕНО: |price_change| / volume, а не отношение std
        df['kyle_lambda'] = self.safe_divide(
            df['returns'].abs(),
            np.log(df['volume'] + 1),
            fill_value=0.0,
            max_value=10.0
        )
        
        # Альтернативная версия - отношение волатильностей
        df['volatility_volume_ratio'] = self.safe_divide(
            df['returns'].rolling(10).std(),
            df['volume'].rolling(10).std(),
            fill_value=0.0,
            max_value=10.0
        )
        
        # Реализованная волатильность - правильная аннуализация
        # ИСПРАВЛЕНО: Разные периоды аннуализации
        # Для 15-минутных данных: 96 периодов в день, 365 дней в году
        df['realized_vol_1h'] = df['returns'].rolling(4).std() * np.sqrt(96)  # Часовая волатильность -> дневная
        df['realized_vol_daily'] = df['returns'].rolling(96).std() * np.sqrt(96)  # Дневная волатильность
        df['realized_vol_annual'] = df['returns'].rolling(96).std() * np.sqrt(96 * 365)  # Годовая волатильность
        
        # Для совместимости оставляем старое имя
        df['realized_vol'] = df['realized_vol_daily']
        
        # Соотношение объема к волатильности
        # ИСПРАВЛЕНО: Используем log объема и нормализуем на средний объем
        avg_volume = df['volume'].rolling(96).mean()
        normalized_volume = df['volume'] / (avg_volume + 1)  # Нормализованный объем
        
        df['volume_volatility_ratio'] = self.safe_divide(
            normalized_volume,
            df['realized_vol'] * 100,  # Волатильность в процентах
            fill_value=1.0,
            max_value=100.0  # Разумный лимит
        )
        
        return df
    
    def _create_rally_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Признаки для определения ралли и крупных движений"""
        if not self.disable_progress:
            self.logger.info("Создание признаков для определения ралли...")
        initial_cols = len(df.columns)
        features_created = []
        
        # 1. Накопленный объем за разные периоды (8 признаков)
        # ИСПРАВЛЕНО: Используем log-трансформацию для больших объемов
        for hours in [4, 8, 12, 24]:
            periods = hours * 4  # 15-минутные свечи
            col_cumsum = f'volume_cumsum_{hours}h'
            col_ratio = f'volume_cumsum_{hours}h_ratio'
            
            # Используем log1p для безопасной трансформации
            # log1p(x) = log(1 + x), безопасен для x=0
            df[col_cumsum] = np.log1p(df['volume'].rolling(periods).sum())
            
            # Отношение к среднему объему за более длинный период
            avg_volume_long = df['volume'].rolling(periods * 4).mean()
            df[col_ratio] = self.safe_divide(
                df['volume'].rolling(periods).sum(),
                avg_volume_long * periods,  # Нормализуем на ожидаемую сумму
                fill_value=1.0,
                max_value=10.0  # Ограничиваем экстремальные всплески
            )
            features_created.extend([col_cumsum, col_ratio])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Накопленный объем: создано {len([f for f in features_created if 'volume_cumsum' in f])} признаков")
        
        # 2. Аномальные всплески объема (3 признака)
        volume_mean = df['volume'].rolling(96).mean()  # средний объем за 24ч
        volume_std = df['volume'].rolling(96).std()
        df['volume_zscore'] = self.safe_divide(
            df['volume'] - volume_mean,
            volume_std,
            fill_value=0.0,
            max_value=50.0  # ИСПРАВЛЕНО: В крипто Z-score может достигать 20-50
        )
        df['volume_spike'] = (df['volume_zscore'] > 3).astype(int)
        df['volume_spike_magnitude'] = df['volume_zscore'].clip(0, 10)
        features_created.extend(['volume_zscore', 'volume_spike', 'volume_spike_magnitude'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Аномальные всплески объема: создано 3 признака")
        
        # 3. Уровни поддержки/сопротивления (15 признаков)
        # Локальные минимумы и максимумы
        for window in [20, 50, 100]:  # 5ч, 12.5ч, 25ч
            df[f'local_high_{window}'] = df['high'].rolling(window).max()
            df[f'local_low_{window}'] = df['low'].rolling(window).min()
            df[f'distance_from_high_{window}'] = (df['close'] - df[f'local_high_{window}']) / df['close']
            df[f'distance_from_low_{window}'] = (df['close'] - df[f'local_low_{window}']) / df['close']
            df[f'position_in_range_{window}'] = self.safe_divide(
                df['close'] - df[f'local_low_{window}'],
                df[f'local_high_{window}'] - df[f'local_low_{window}'],
                fill_value=0.5,  # Середина диапазона
                max_value=1.0   # Позиция в диапазоне от 0 до 1
            )
            features_created.extend([f'local_high_{window}', f'local_low_{window}', 
                                   f'distance_from_high_{window}', f'distance_from_low_{window}', 
                                   f'position_in_range_{window}'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Уровни поддержки/сопротивления: создано 15 признаков")
        
        # 4. Сжатие волатильности (признак будущего прорыва) (2 признака)
        # Bollinger Bands уже есть, добавим Keltner Channels для сравнения
        atr_multiplier = 2.0
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        kc_upper = ema20 + atr_multiplier * df['atr']
        kc_lower = ema20 - atr_multiplier * df['atr']
        # ИСПРАВЛЕНО: сравниваем относительные ширины каналов
        kc_width = (kc_upper - kc_lower) / df['close']

        df['volatility_squeeze'] = (df['bb_width'] < kc_width).astype(int)
        # ИСПРАВЛЕНО: продолжительность сжатия считается только для периодов squeeze
        squeeze_group = (df['volatility_squeeze'] != df['volatility_squeeze'].shift()).cumsum()
        df['volatility_squeeze_duration'] = (
            df['volatility_squeeze']
            .groupby(squeeze_group)
            .cumsum()
        )
        df.loc[df['volatility_squeeze'] == 0, 'volatility_squeeze_duration'] = 0
        features_created.extend(['volatility_squeeze', 'volatility_squeeze_duration'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Сжатие волатильности: создано 2 признака")
        
        # 5. Дивергенции RSI/MACD с ценой (4 признака)
        # RSI дивергенция
        price_higher = (df['close'] > df['close'].shift(14)) & (df['close'].shift(14) > df['close'].shift(28))
        rsi_lower = (df['rsi'] < df['rsi'].shift(14)) & (df['rsi'].shift(14) < df['rsi'].shift(28))
        df['bearish_divergence_rsi'] = (price_higher & rsi_lower).astype(int)
        
        price_lower = (df['close'] < df['close'].shift(14)) & (df['close'].shift(14) < df['close'].shift(28))
        rsi_higher = (df['rsi'] > df['rsi'].shift(14)) & (df['rsi'].shift(14) > df['rsi'].shift(28))
        df['bullish_divergence_rsi'] = (price_lower & rsi_higher).astype(int)
        
        # MACD дивергенция
        macd_lower = (df['macd'] < df['macd'].shift(14)) & (df['macd'].shift(14) < df['macd'].shift(28))
        df['bearish_divergence_macd'] = (price_higher & macd_lower).astype(int)
        
        macd_higher = (df['macd'] > df['macd'].shift(14)) & (df['macd'].shift(14) > df['macd'].shift(28))
        df['bullish_divergence_macd'] = (price_lower & macd_higher).astype(int)
        features_created.extend(['bearish_divergence_rsi', 'bullish_divergence_rsi', 
                               'bearish_divergence_macd', 'bullish_divergence_macd'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Дивергенции RSI/MACD: создано 4 признака")
        
        # 6. Паттерны накопления/распределения (4 признака)
        # On-Balance Volume (OBV)
        # ИСПРАВЛЕНО: Используем log-трансформацию для контроля масштаба
        obv_change = df['volume'] * ((df['close'] > df['close'].shift(1)) * 2 - 1)
        
        # Используем скользящее окно с log-трансформацией
        obv_raw = obv_change.rolling(100).sum()  # 100 периодов (25 часов)
        
        # Применяем log-трансформацию для контроля масштаба
        df['obv'] = np.sign(obv_raw) * np.log1p(np.abs(obv_raw))
        
        # Нормализуем OBV относительно среднего объема для сравнимости между активами
        avg_volume = df['volume'].rolling(100).mean()
        df['obv_normalized'] = self.safe_divide(
            df['obv'],
            np.log1p(avg_volume),  # Логарифмируем и средний объем
            fill_value=0.0,
            max_value=20.0
        )
        
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        df['obv_divergence'] = df['obv'] - df['obv_ema']
        
        # Chaikin Money Flow
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        mfv = mfm * df['volume']
        df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
        features_created.extend(['obv', 'obv_normalized', 'obv_ema', 'obv_divergence', 'cmf'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Паттерны накопления/распределения: создано 5 признаков")
        
        # 7. Momentum и ускорение (4 признака)
        # Используем groupby для правильного расчета по символам
        df['momentum_1h'] = df.groupby('symbol')['close'].pct_change(4) * 100  # 1 час
        df['momentum_4h'] = df.groupby('symbol')['close'].pct_change(16) * 100  # 4 часа
        df['momentum_24h'] = df.groupby('symbol')['close'].pct_change(96) * 100  # 24 часа
        
        # Ускорение (изменение momentum)
        df['momentum_acceleration'] = df.groupby('symbol')['momentum_1h'].transform(
            lambda x: x - x.shift(4)
        )
        features_created.extend(['momentum_1h', 'momentum_4h', 'momentum_24h', 'momentum_acceleration'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Momentum индикаторы: создано 4 признака")
        
        # 8. Паттерн "пружина" - сильное сжатие перед движением (1 признак)
        df['spring_pattern'] = (
            (df['volatility_squeeze'] == 1) & 
            (df['volume_spike'] == 1) &
            (df['atr_pct'].rolling(20).mean() < df['atr_pct'].rolling(100).mean() * 0.7)
        ).astype(int)
        features_created.append('spring_pattern')
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Паттерн 'пружина': создан 1 признак")
        
        # Итоговая статистика
        total_created = len(features_created)
        if not self.disable_progress:
            self.logger.info(f"✅ Rally detection features: всего создано {total_created} признаков")
            self.logger.info(f"   Детализация: {features_created}")
        
        return df
    
    def _create_signal_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Признаки для оценки качества торговых сигналов"""
        if not self.disable_progress:
            self.logger.info("Создание признаков качества сигналов...")
        initial_cols = len(df.columns)
        features_created = []
        
        # 1. Согласованность индикаторов
        # Определяем сигналы от разных индикаторов
        indicators_long = []
        indicators_short = []
        indicators_used = []
        
        # RSI
        if 'rsi' in df.columns:
            indicators_long.append((df['rsi'] < 30).astype(int))
            indicators_short.append((df['rsi'] > 70).astype(int))
            indicators_used.append('RSI')
        
        # MACD
        if 'macd_diff' in df.columns:
            indicators_long.append((df['macd_diff'] > 0).astype(int))
            indicators_short.append((df['macd_diff'] < 0).astype(int))
            indicators_used.append('MACD')
        
        # Bollinger Bands
        if 'bb_position' in df.columns:
            indicators_long.append((df['bb_position'] < 0.2).astype(int))
            indicators_short.append((df['bb_position'] > 0.8).astype(int))
            indicators_used.append('Bollinger Bands')
        
        # Stochastic
        if 'stoch_k' in df.columns:
            indicators_long.append((df['stoch_k'] < 20).astype(int))
            indicators_short.append((df['stoch_k'] > 80).astype(int))
            indicators_used.append('Stochastic')
        
        # ADX (сила тренда)
        if 'adx' in df.columns:
            strong_trend = (df['adx'] > 25).astype(int)
            indicators_long.append(strong_trend & (df['adx_pos'] > df['adx_neg']))
            indicators_short.append(strong_trend & (df['adx_neg'] > df['adx_pos']))
            indicators_used.append('ADX')
        
        # Moving averages
        if 'close_sma_20_ratio' in df.columns and 'close_sma_50_ratio' in df.columns:
            indicators_long.append((df['close_sma_20_ratio'] > df['close_sma_50_ratio']).astype(int))
            indicators_short.append((df['close_sma_20_ratio'] < df['close_sma_50_ratio']).astype(int))
            indicators_used.append('Moving Averages')
        
        # Считаем согласованность
        if indicators_long:
            df['indicators_consensus_long'] = sum(indicators_long) / len(indicators_long)
            df['indicators_count_long'] = sum(indicators_long)
        else:
            df['indicators_consensus_long'] = 0
            df['indicators_count_long'] = 0
            
        if indicators_short:
            df['indicators_consensus_short'] = sum(indicators_short) / len(indicators_short)
            df['indicators_count_short'] = sum(indicators_short)
        else:
            df['indicators_consensus_short'] = 0
            df['indicators_count_short'] = 0
        
        features_created.extend(['indicators_consensus_long', 'indicators_count_long', 
                               'indicators_consensus_short', 'indicators_count_short'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Согласованность индикаторов: создано 4 признака")
            self.logger.info(f"    Используемые индикаторы: {', '.join(indicators_used)}")
        
        # 2. Сила тренда на старших таймфреймах (4 признака)
        # ИСПРАВЛЕНО: Нормализуем тренды относительно цены для избежания больших значений
        # Эмулируем 1-часовой таймфрейм (4 свечи по 15 мин)
        ma_1h = df['close'].rolling(4).mean()
        ma_1h_prev = ma_1h.shift(4)
        df['trend_1h'] = self.safe_divide(
            ma_1h - ma_1h_prev,
            ma_1h_prev,
            fill_value=0.0,
            max_value=0.1  # Максимум 10% изменение
        ) * 100  # В процентах
        
        df['trend_1h_strength'] = self.safe_divide(
            df['trend_1h'],
            df['atr_pct'].rolling(4).mean() * 100,  # ATR уже в процентах
            fill_value=0.0,
            max_value=10.0
        )
        
        # Эмулируем 4-часовой таймфрейм (16 свечей)
        ma_4h = df['close'].rolling(16).mean()
        ma_4h_prev = ma_4h.shift(16)
        df['trend_4h'] = self.safe_divide(
            ma_4h - ma_4h_prev,
            ma_4h_prev,
            fill_value=0.0,
            max_value=0.2  # Максимум 20% изменение
        ) * 100  # В процентах
        
        df['trend_4h_strength'] = self.safe_divide(
            df['trend_4h'],
            df['atr_pct'].rolling(16).mean() * 100,  # ATR уже в процентах
            fill_value=0.0,
            max_value=10.0
        )
        
        features_created.extend(['trend_1h', 'trend_1h_strength', 'trend_4h', 'trend_4h_strength'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Сила тренда на старших ТФ: создано 4 признака")
        
        # 3. Позиция в дневном диапазоне (7 признаков)
        # Дневной диапазон (96 свечей = 24 часа)
        df['daily_high'] = df['high'].rolling(96).max()
        df['daily_low'] = df['low'].rolling(96).min()
        # ИСПРАВЛЕНО: daily_range как процент от цены
        df['daily_range'] = self.safe_divide(
            df['daily_high'] - df['daily_low'],
            df['close'],
            fill_value=0.02,  # 2% по умолчанию
            max_value=0.5   # Максимум 50% от цены
        )
        # ИСПРАВЛЕНО: Правильный расчет позиции в дневном диапазоне
        # daily_range уже в процентах, поэтому используем абсолютные цены
        daily_range_abs = df['daily_high'] - df['daily_low']
        df['position_in_daily_range'] = self.safe_divide(
            df['close'] - df['daily_low'],
            daily_range_abs,
            fill_value=0.5,
            max_value=1.0
        )
        
        # Близость к экстремумам
        df['near_daily_high'] = (df['position_in_daily_range'] > 0.9).astype(int)
        df['near_daily_low'] = (df['position_in_daily_range'] < 0.1).astype(int)
        features_created.extend(['daily_high', 'daily_low', 'daily_range', 
                               'position_in_daily_range', 'near_daily_high', 'near_daily_low'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Позиция в дневном диапазоне: создано 6 признаков")
        
        # 4. Качество структуры рынка (2 признака)
        # Higher highs и higher lows для uptrend
        hh = (df['high'] > df['high'].shift(4)) & (df['high'].shift(4) > df['high'].shift(8))
        hl = (df['low'] > df['low'].shift(4)) & (df['low'].shift(4) > df['low'].shift(8))
        df['uptrend_structure'] = (hh & hl).astype(int)
        
        # Lower highs и lower lows для downtrend
        lh = (df['high'] < df['high'].shift(4)) & (df['high'].shift(4) < df['high'].shift(8))
        ll = (df['low'] < df['low'].shift(4)) & (df['low'].shift(4) < df['low'].shift(8))
        df['downtrend_structure'] = (lh & ll).astype(int)
        features_created.extend(['uptrend_structure', 'downtrend_structure'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Качество структуры рынка: создано 2 признака")
        
        # 5. Риск новостных событий (1 признак)
        # Аномальный объем часто связан с новостями
        df['news_risk'] = (df['volume_spike'] == 1).astype(int)
        features_created.append('news_risk')
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Риск новостных событий: создан 1 признак")
        
        # 6. Оценка ликвидности (3 признака)
        # Средний спред и объем за последний час
        # Используем high-low спред вместо bid-ask
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # ИСПРАВЛЕНО: Безопасный расчет liquidity_score с ограничениями
        hl_spread_mean = df['hl_spread'].rolling(4).mean()
        volume_mean = df['volume'].rolling(4).mean()
        
        # Клиппинг hl_spread для избежания деления на очень малые числа
        # Минимальный спред 0.01% (0.0001) для стейблкоинов
        hl_spread_clipped = np.clip(hl_spread_mean, 0.0001, 1.0)
        
        # Используем log-трансформацию для контроля масштаба
        # liquidity_score теперь в диапазоне примерно [0, 20]
        df['liquidity_score'] = np.log1p(volume_mean / (hl_spread_clipped * 1000))
        
        # Ранжирование по ликвидности
        df['liquidity_rank'] = df.groupby('datetime')['liquidity_score'].rank(pct=True)
        features_created.extend(['hl_spread', 'liquidity_score', 'liquidity_rank'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Оценка ликвидности: создано 3 признака")
        
        # 6. Signal Strength - комбинированная сила сигнала на основе исторических данных
        # БЕЗ УТЕЧЕК: используем только исторические индикаторы
        
        # Компоненты signal_strength:
        # 1. Сила тренда (ADX)
        trend_strength = df['adx'] / 100 if 'adx' in df.columns else pd.Series(0.5, index=df.index)
        
        # 2. Momentum (RSI отклонение от 50)
        momentum_strength = np.abs(df['rsi'] - 50) / 50 if 'rsi' in df.columns else pd.Series(0.5, index=df.index)
        
        # 3. Волатильность (нормализованная историческая)
        if 'volatility_20' in df.columns:
            # Используем историческое среднее, НЕ будущие данные
            vol_mean_hist = df.groupby('symbol')['volatility_20'].transform(
                lambda x: x.rolling(100, min_periods=20).mean()
            )
            vol_strength = df['volatility_20'] / (vol_mean_hist + 1e-6)
            vol_strength = np.clip(vol_strength, 0, 2) / 2
        else:
            vol_strength = pd.Series(0.5, index=df.index)
        
        # 4. Volume (нормализованный исторический)
        if 'volume' in df.columns:
            # Используем историческое среднее, НЕ будущие данные
            vol_mean_hist = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(100, min_periods=20).mean()
            )
            volume_strength = df['volume'] / (vol_mean_hist + 1e-6)
            volume_strength = np.clip(volume_strength, 0, 2) / 2
        else:
            volume_strength = pd.Series(0.5, index=df.index)
        
        # Комбинированная сила сигнала (признак, не целевая переменная)
        df['signal_strength'] = (
            0.3 * trend_strength +
            0.3 * momentum_strength +
            0.2 * vol_strength +
            0.2 * volume_strength
        )
        df['signal_strength'] = np.clip(df['signal_strength'], 0, 1)
        features_created.append('signal_strength')
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Signal strength: создан 1 признак (без утечек)")
        
        # Итоговая статистика
        total_created = len(features_created)
        if not self.disable_progress:
            self.logger.info(f"✅ Signal quality features: всего создано {total_created} признаков")
        
        return df
    
    def _create_futures_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Признаки специфичные для торговли фьючерсами с плечом"""
        if not self.disable_progress:
            self.logger.info("Создание признаков для фьючерсной торговли...")
        initial_cols = len(df.columns)
        features_created = []
        
        # Динамическое плечо на основе волатильности
        # Базовое плечо 5x, но корректируем на основе ATR
        base_leverage = 5
        
        # Корректируем плечо на основе волатильности
        # Чем выше волатильность, тем меньше плечо
        # ATR в процентах уже есть в df['atr_pct']
        volatility_factor = df['atr_pct'].rolling(24).mean()  # Средняя волатильность за 6 часов
        
        # Плечо от 3x до 10x в зависимости от волатильности
        # При волатильности 0.5% -> leverage = 10
        # При волатильности 2% -> leverage = 3
        dynamic_leverage = base_leverage * (0.01 / (volatility_factor + 0.001))
        dynamic_leverage = dynamic_leverage.clip(3, 10)  # Ограничиваем диапазон
        
        # 1. Расчет ликвидационной цены
        # Для LONG: Liq Price = Entry Price * (1 - 1/leverage + fees)
        # Для SHORT: Liq Price = Entry Price * (1 + 1/leverage - fees)
        maintenance_margin = 0.5 / 100  # 0.5% для Bybit
        
        df['long_liquidation_price'] = df['close'] * (1 - 1/dynamic_leverage + maintenance_margin)
        df['short_liquidation_price'] = df['close'] * (1 + 1/dynamic_leverage - maintenance_margin)
        
        # Расстояние до ликвидации в процентах
        df['long_liquidation_distance_pct'] = ((df['close'] - df['long_liquidation_price']) / df['close']) * 100
        df['short_liquidation_distance_pct'] = ((df['short_liquidation_price'] - df['close']) / df['close']) * 100
        
        # Сохраняем текущее динамическое плечо
        df['current_leverage'] = dynamic_leverage
        features_created.extend(['long_liquidation_price', 'short_liquidation_price',
                               'long_liquidation_distance_pct', 'short_liquidation_distance_pct',
                               'current_leverage'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Ликвидационные цены: создано 4 признака")
        
        # 2. Вероятность касания ликвидационной цены
        # Основано на исторической волатильности
        # Используем максимальное отклонение за последние 24 часа
        max_drawdown_24h = df['low'].rolling(96).min() / df['close'].shift(96) - 1
        max_rally_24h = df['high'].rolling(96).max() / df['close'].shift(96) - 1
        
        # Простая оценка вероятности на основе исторических движений
        df['long_liquidation_risk'] = (abs(max_drawdown_24h) > df['long_liquidation_distance_pct'] / 100).rolling(96).mean()
        df['short_liquidation_risk'] = (max_rally_24h > df['short_liquidation_distance_pct'] / 100).rolling(96).mean()
        features_created.extend(['long_liquidation_risk', 'short_liquidation_risk'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Вероятность ликвидации: создано 2 признака")
        
        # 3. Оптимальное плечо для текущей волатильности
        # Правило: максимальное плечо = 20% / (дневная волатильность)
        daily_volatility = df['returns'].rolling(96).std() * np.sqrt(96)  # Приведение к дневной
        df['optimal_leverage'] = (0.2 / (daily_volatility + 0.01)).clip(1, 10)  # От 1x до 10x
        
        # Безопасное плечо (консервативное)
        df['safe_leverage'] = (0.1 / (daily_volatility + 0.01)).clip(1, 5)  # От 1x до 5x
        features_created.extend(['optimal_leverage', 'safe_leverage'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Оптимальное плечо: создано 2 признака")
        
        # 4. Риск каскадных ликвидаций
        # Когда много позиций могут быть ликвидированы одновременно
        # Индикатор: резкие движения + высокий объем
        df['cascade_risk'] = (
            (df['volume_spike'] == 1) & 
            (abs(df['returns']) > df['returns'].rolling(96).std() * 2)
        ).astype(int)
        features_created.append('cascade_risk')
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Риск каскадных ликвидаций: создан 1 признак")
        
        # 5. Funding rate влияние (для удержания позиций)
        # Примерный funding rate (в реальности нужно загружать с биржи)
        # Положительный funding = лонги платят шортам
        # Используем разницу между спот и фьючерс как прокси
        df['funding_proxy'] = df['momentum_1h'] * 0.01  # Упрощенная оценка
        
        # Стоимость удержания позиции на день (3 funding периода)
        df['long_holding_cost_daily'] = df['funding_proxy'] * 3
        df['short_holding_cost_daily'] = -df['funding_proxy'] * 3
        features_created.extend(['funding_proxy', 'long_holding_cost_daily', 'short_holding_cost_daily'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Funding rate влияние: создано 3 признака")
        
        # 6. Метрики риска для размера позиции
        # Value at Risk (VaR) - максимальная потеря с 95% вероятностью
        returns_sorted = df['returns'].rolling(96).apply(lambda x: np.percentile(x, 5))
        df['var_95'] = abs(returns_sorted)
        
        # Рекомендуемый размер позиции относительно VaR
        max_loss_per_trade = 2.0  # 2% максимальная потеря как в конфиге
        df['recommended_position_size'] = max_loss_per_trade / (df['var_95'] * dynamic_leverage)
        features_created.extend(['var_95', 'recommended_position_size'])
        
        if not self.disable_progress:
            self.logger.info(f"  ✓ Метрики риска для позиций: создано 2 признака")
        
        # Итоговая статистика
        total_created = len(features_created)
        if not self.disable_progress:
            self.logger.info(f"✅ Futures-specific features: всего создано {total_created} признаков")
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Временные признаки"""
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        
        # Циклическое кодирование времени
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Торговые сессии
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['american_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        # Пересечение сессий
        df['session_overlap'] = (
            (df['asian_session'] + df['european_session'] + df['american_session']) > 1
        ).astype(int)
        
        return df
    
    def _create_ml_optimized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание ML-оптимизированных признаков для 2024-2025"""
        if not self.disable_progress:
            self.logger.info("Создание ML-оптимизированных признаков...")
        
        # 1. Hurst Exponent - мера персистентности рынка
        # >0.5 = тренд, <0.5 = возврат к среднему, ~0.5 = случайное блуждание
        def hurst_exponent(ts, max_lag=20):
            """Вычисление экспоненты Херста"""
            lags = range(2, min(max_lag, len(ts) // 2))
            tau = []
            
            for lag in lags:
                pp = np.array(ts[:-lag])
                pn = np.array(ts[lag:])
                diff = pn - pp
                tau.append(np.sqrt(np.nanmean(diff**2)))
            
            if len(tau) > 0 and all(t > 0 for t in tau):
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            return 0.5
        
        # Применяем Hurst для close с окном 50
        df['hurst_exponent'] = df['close'].rolling(50).apply(
            lambda x: hurst_exponent(x) if len(x) == 50 else 0.5
        )
        
        # 2. Fractal Dimension - сложность ценового движения
        # 1 = прямая линия, 2 = заполняет плоскость
        def fractal_dimension(ts):
            """Вычисление фрактальной размерности методом Хигучи"""
            N = len(ts)
            if N < 10:
                return 1.5
            
            kmax = min(5, N // 2)
            L = []
            
            for k in range(1, kmax + 1):
                Lk = 0
                for m in range(k):
                    Lmk = 0
                    for i in range(1, int((N - m) / k)):
                        Lmk += abs(ts[m + i * k] - ts[m + (i - 1) * k])
                    if int((N - m) / k) > 0:
                        Lmk = Lmk * (N - 1) / (k * int((N - m) / k))
                    Lk += Lmk
                L.append(Lk / k)
            
            if len(L) > 0 and all(l > 0 for l in L):
                x = np.log(range(1, kmax + 1))
                y = np.log(L)
                poly = np.polyfit(x, y, 1)
                return poly[0]
            return 1.5
        
        df['fractal_dimension'] = df['close'].rolling(30).apply(
            lambda x: fractal_dimension(x.values) if len(x) == 30 else 1.5
        )
        
        # 3. Market Efficiency Ratio - эффективность движения цены
        # Высокие значения = сильный тренд, низкие = боковик
        df['efficiency_ratio'] = self.safe_divide(
            (df['close'] - df['close'].shift(20)).abs(),
            df['close'].diff().abs().rolling(20).sum()
        )
        
        # 4. Trend Quality Index - качество тренда
        # Комбинация ADX, направления и волатильности
        df['trend_quality'] = (
            df['adx'] / 100 *  # Сила тренда
            ((df['close'] > df['sma_50']).astype(float) * 2 - 1) *  # Направление
            (1 - df['bb_width'] / df['bb_width'].rolling(50).max())  # Нормализованная волатильность
        )
        
        # 5. Regime Detection Features
        # Определение рыночного режима (тренд/флэт/высокая волатильность)
        returns = df['close'].pct_change()
        
        # Realized volatility
        df['realized_vol_5m'] = returns.rolling(20).std() * np.sqrt(20)
        df['realized_vol_15m'] = returns.rolling(60).std() * np.sqrt(60)
        df['realized_vol_1h'] = returns.rolling(240).std() * np.sqrt(240)
        
        # GARCH-подобная волатильность (упрощенная)
        df['garch_vol'] = returns.rolling(20).apply(
            lambda x: np.sqrt(0.94 * x.var() + 0.06 * x.iloc[-1]**2) if len(x) > 0 else 0
        )
        
        # Режим волатильности
        atr_q25 = df['atr'].rolling(1000).quantile(0.25)
        atr_q75 = df['atr'].rolling(1000).quantile(0.75)
        df['vol_regime'] = 0  # Нормальная
        df.loc[df['atr'] < atr_q25, 'vol_regime'] = -1  # Низкая
        df.loc[df['atr'] > atr_q75, 'vol_regime'] = 1   # Высокая
        
        # 6. Information-theoretic features
        # Энтропия распределения доходностей
        def shannon_entropy(series, bins=10):
            """Вычисление энтропии Шеннона"""
            if len(series) < bins:
                return 0
            counts, _ = np.histogram(series, bins=bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return -np.sum(probs * np.log(probs))
        
        df['return_entropy'] = returns.rolling(100).apply(
            lambda x: shannon_entropy(x)
        )
        
        # 7. Microstructure features
        # Amihud illiquidity
        df['amihud_illiquidity'] = self.safe_divide(
            returns.abs(),
            df['turnover']
        ).rolling(20).mean()
        
        # Kyle's lambda (price impact)
        df['kyle_lambda'] = self.safe_divide(
            returns.abs().rolling(20).mean(),
            df['volume'].rolling(20).mean()
        )
        
        # 8. Cross-sectional features (если есть данные BTC)
        if 'btc_returns' in df.columns:
            # Beta к BTC
            df['btc_beta'] = returns.rolling(100).cov(df['btc_returns']) / df['btc_returns'].rolling(100).var()
            
            # Идиосинкратическая волатильность
            df['idio_vol'] = (returns - df['btc_beta'] * df['btc_returns']).rolling(50).std()
        
        # 9. Autocorrelation features
        # Автокорреляция доходностей на разных лагах
        df['returns_ac_1'] = returns.rolling(50).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
        df['returns_ac_5'] = returns.rolling(50).apply(lambda x: x.autocorr(lag=5) if len(x) > 5 else 0)
        df['returns_ac_10'] = returns.rolling(50).apply(lambda x: x.autocorr(lag=10) if len(x) > 10 else 0)
        
        # 10. Jump detection
        # Обнаружение прыжков в цене
        df['price_jump'] = (
            returns.abs() > returns.rolling(100).std() * 3
        ).astype(int)
        
        df['jump_intensity'] = df['price_jump'].rolling(50).mean()
        
        # 11. Order flow imbalance persistence
        if 'order_flow_imbalance' in df.columns:
            df['ofi_persistence'] = df['order_flow_imbalance'].rolling(20).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )
        
        # 12. Volume-synchronized probability of informed trading (VPIN)
        # Упрощенная версия
        df['vpin'] = self.safe_divide(
            (df['volume'] * ((df['close'] > df['open']).astype(float) - 0.5)).rolling(50).sum().abs(),
            df['volume'].rolling(50).sum()
        )
        
        # 13. Liquidity-adjusted returns
        df['liquidity_adj_returns'] = returns * (1 - df['amihud_illiquidity'] / df['amihud_illiquidity'].rolling(100).max())
        
        # 14. Tail risk measures
        # Conditional Value at Risk (CVaR)
        df['cvar_5pct'] = returns.rolling(100).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else x.quantile(0.05)
        )
        
        # Заполнение пропусков
        ml_features = [
            'hurst_exponent', 'fractal_dimension', 'efficiency_ratio', 'trend_quality',
            'realized_vol_5m', 'realized_vol_15m', 'realized_vol_1h', 'garch_vol',
            'vol_regime', 'return_entropy', 'amihud_illiquidity', 'kyle_lambda',
            'returns_ac_1', 'returns_ac_5', 'returns_ac_10', 'price_jump',
            'jump_intensity', 'vpin', 'liquidity_adj_returns', 'cvar_5pct'
        ]
        
        # Добавляем условные признаки если они были созданы
        if 'btc_beta' in df.columns:
            ml_features.extend(['btc_beta', 'idio_vol'])
        if 'ofi_persistence' in df.columns:
            ml_features.append('ofi_persistence')
        
        # Заполняем пропуски
        for feature in ml_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(method='ffill').fillna(0)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений"""
        if not self.disable_progress:
            self.logger.info("Обработка пропущенных значений...")
        
        # Сохраняем информационные колонки
        info_cols = ['id', 'symbol', 'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        
        # Группируем по символам для правильной обработки
        processed_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Для каждой колонки применяем соответствующий метод заполнения
            for col in symbol_data.columns:
                if col in info_cols:
                    continue
                    
                if symbol_data[col].isna().any():
                    # Для категориальных переменных (Categorical dtype)
                    if hasattr(symbol_data[col], 'cat'):
                        # Для категориальных переменных используем наиболее частую категорию или 'FLAT'/'HOLD'
                        if 'direction' in col:
                            symbol_data[col] = symbol_data[col].fillna('FLAT')
                        else:
                            # Используем моду (наиболее частое значение)
                            mode = symbol_data[col].mode()
                            if len(mode) > 0:
                                symbol_data[col] = symbol_data[col].fillna(mode.iloc[0])
                    # Для технических индикаторов используем forward fill
                    elif any(indicator in col for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'adx']):
                        symbol_data[col] = symbol_data[col].ffill()
                    # Для остальных используем 0
                    else:
                        symbol_data[col] = symbol_data[col].fillna(0)
            
            # Удаляем первые строки где могут быть NaN из-за расчета индикаторов
            # Находим максимальный период среди всех индикаторов
            max_period = 50  # SMA50 требует минимум 50 периодов
            symbol_data = symbol_data.iloc[max_period:].copy()
            
            processed_dfs.append(symbol_data)
        
        result_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Финальная проверка
        nan_count = result_df.isna().sum().sum()
        if nan_count > 0:
            if not self.disable_progress:
                self.logger.warning(f"Остались {nan_count} NaN значений после обработки")
            # Принудительно заполняем оставшиеся NaN
            for col in result_df.columns:
                if result_df[col].isna().any():
                    # Для категориальных переменных
                    if hasattr(result_df[col], 'cat'):
                        if 'direction' in col:
                            result_df[col] = result_df[col].fillna('FLAT')
                        else:
                            mode = result_df[col].mode()
                            if len(mode) > 0:
                                result_df[col] = result_df[col].fillna(mode.iloc[0])
                    # Для числовых колонок
                    elif pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(0)
        
        # Проверка на бесконечные значения
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(result_df[numeric_cols]).sum().sum()
        if inf_count > 0:
            if not self.disable_progress:
                self.logger.warning(f"Обнаружены {inf_count} бесконечных значений, заменяем на конечные")
            # ИСПРАВЛЕНО: Заменяем бесконечности на 99-й персентиль для каждой колонки
            for col in numeric_cols:
                if np.isinf(result_df[col]).any():
                    # Вычисляем персентили на конечных значениях
                    finite_vals = result_df[col][np.isfinite(result_df[col])]
                    if len(finite_vals) > 0:
                        p99 = finite_vals.quantile(0.99)
                        p1 = finite_vals.quantile(0.01)
                        result_df[col] = result_df[col].replace([np.inf, -np.inf], [p99, p1])
                    else:
                        result_df[col] = result_df[col].replace([np.inf, -np.inf], [0, 0])
        
        if not self.disable_progress:
            self.logger.info(f"Обработка завершена. Итоговый размер: {len(result_df)} записей")
        return result_df
    
    def _create_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кросс-активные признаки"""
        if not self.disable_progress:
            self.logger.info("Создание кросс-активных признаков...")
        
        # BTC как базовый актив
        btc_data = df[df['symbol'] == 'BTCUSDT'][['datetime', 'close', 'returns']].copy()
        if len(btc_data) > 0:
            btc_data.rename(columns={
                'close': 'btc_close',
                'returns': 'btc_returns'
            }, inplace=True)
            
            df = df.merge(btc_data, on='datetime', how='left')
            
            # Корреляция с BTC
            for symbol in df['symbol'].unique():
                if symbol != 'BTCUSDT':
                    mask = df['symbol'] == symbol
                    # ИСПРАВЛЕНО: используем min_periods для корреляции
                    df.loc[mask, 'btc_correlation'] = (
                        df.loc[mask, 'returns']
                        .rolling(window=96, min_periods=50)
                        .corr(df.loc[mask, 'btc_returns'])
                    )
            
            df.loc[df['symbol'] == 'BTCUSDT', 'btc_correlation'] = 1.0
            
            # Относительная сила к BTC
            df['relative_strength_btc'] = df['close'] / df['btc_close']
            df['rs_btc_ma'] = df.groupby('symbol')['relative_strength_btc'].transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            )
            
            # ИСПРАВЛЕНО: заполняем NaN значения для BTC-связанных признаков
            df['btc_close'] = df['btc_close'].fillna(method='ffill').fillna(method='bfill')
            df['btc_returns'] = df['btc_returns'].fillna(0.0)
            df['btc_correlation'] = df['btc_correlation'].fillna(0.5)  # нейтральная корреляция
            df['relative_strength_btc'] = df['relative_strength_btc'].fillna(1.0)
            df['rs_btc_ma'] = df['rs_btc_ma'].fillna(1.0)
        else:
            # Заполняем нулями если нет данных BTC
            df['btc_close'] = 0
            df['btc_returns'] = 0
            df['btc_correlation'] = 0
            df['relative_strength_btc'] = 0
            df['rs_btc_ma'] = 0
        
        # Определяем сектора
        defi_tokens = ['AAVEUSDT', 'UNIUSDT', 'CAKEUSDT', 'DYDXUSDT']
        layer1_tokens = ['ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT', 'NEARUSDT']
        meme_tokens = ['DOGEUSDT', 'FARTCOINUSDT', 'MELANIAUSDT', 'TRUMPUSDT', 
                      'POPCATUSDT', 'PNUTUSDT', 'ZEREBROUSDT', 'WIFUSDT']
        
        df['sector'] = 'other'
        df.loc[df['symbol'].isin(defi_tokens), 'sector'] = 'defi'
        df.loc[df['symbol'].isin(layer1_tokens), 'sector'] = 'layer1'
        df.loc[df['symbol'].isin(meme_tokens), 'sector'] = 'meme'
        df.loc[df['symbol'] == 'BTCUSDT', 'sector'] = 'btc'
        
        # Секторные доходности
        df['sector_returns'] = df.groupby(['datetime', 'sector'])['returns'].transform('mean')
        
        # Относительная доходность к сектору
        df['relative_to_sector'] = df['returns'] - df['sector_returns']
        
        # Ранк доходности
        df['returns_rank'] = df.groupby('datetime')['returns'].rank(pct=True)
        
        # 24-часовой моментум - уже рассчитан в rally_detection_features
        # Здесь только заполняем NaN значения если есть
        if 'momentum_24h' in df.columns and df['momentum_24h'].isna().any():
            df['momentum_24h'] = df['momentum_24h'].fillna(0)
        df['is_momentum_leader'] = (
            df.groupby('datetime')['momentum_24h']
            .rank(ascending=False) <= 5
        ).astype(int)
        
        return df
    
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание целевых переменных БЕЗ УТЕЧЕК ДАННЫХ - версия 4.0"""
        if not self.disable_progress:
            self.logger.info("🎯 Создание целевых переменных v4.0 (без утечек)...")
        
        # Периоды для расчета будущих возвратов (в свечах по 15 минут)
        return_periods = {
            '15m': 1,    # 15 минут
            '1h': 4,     # 1 час
            '4h': 16,    # 4 часа
            '12h': 48    # 12 часов
        }
        
        # Пороги для классификации направления
        # ОПТИМИЗИРОВАНЫ для баланса между качеством и количеством сигналов
        direction_thresholds = {
            '15m': 0.0015,  # 0.15% - уменьшает шум от мелких движений
            '1h': 0.003,    # 0.3% - фильтрует случайные колебания
            '4h': 0.007,    # 0.7% - фокус на значимых движениях
            '12h': 0.01     # 1% - долгосрочные тренды
        }
        
        # Уровни прибыли для бинарных целевых
        profit_levels = {
            '1pct_4h': (0.01, 16),    # 1% за 4 часа
            '2pct_4h': (0.02, 16),    # 2% за 4 часа
            '3pct_12h': (0.03, 48),   # 3% за 12 часов
            '5pct_12h': (0.05, 48)    # 5% за 12 часов
        }
        
        # Commission and costs
        commission_rate = 0.0006  # 0.06%
        slippage = 0.0005         # 0.05%
        
        # A. Базовые возвраты (4)
        for period_name, n_candles in return_periods.items():
            df[f'future_return_{period_name}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.shift(-n_candles) / x - 1
            )
        
        # B. Направление движения (4)
        for period_name in return_periods.keys():
            future_return = df[f'future_return_{period_name}']
            threshold = direction_thresholds[period_name]
        
            df[f'direction_{period_name}'] = pd.cut(
                future_return,
                bins=[-np.inf, -threshold, threshold, np.inf],
                labels=['DOWN', 'FLAT', 'UP']
            )
        
        # C. Достижение уровней прибыли LONG (4) - используем только shift для будущих цен
        for level_name, (profit_threshold, n_candles) in profit_levels.items():
            # Для каждой строки проверяем достигнет ли максимальная цена нужного уровня
            max_future_returns = pd.DataFrame()
            for i in range(1, n_candles + 1):
                future_high = df.groupby('symbol')['high'].transform(lambda x: x.shift(-i))
                future_return = (future_high / df['close'] - 1)
                max_future_returns[f'return_{i}'] = future_return
        
            # Максимальный return за период
            max_return = max_future_returns.max(axis=1)
            df[f'long_will_reach_{level_name}'] = (max_return >= profit_threshold).astype(int)
        
        # D. Достижение уровней прибыли SHORT (4)
        for level_name, (profit_threshold, n_candles) in profit_levels.items():
            # Для SHORT: проверяем минимальную цену
            min_future_returns = pd.DataFrame()
            for i in range(1, n_candles + 1):
                future_low = df.groupby('symbol')['low'].transform(lambda x: x.shift(-i))
                future_return = (df['close'] / future_low - 1)  # Для SHORT инвертируем
                min_future_returns[f'return_{i}'] = future_return
        
            # Максимальный return для SHORT за период
            max_return = min_future_returns.max(axis=1)
            df[f'short_will_reach_{level_name}'] = (max_return >= profit_threshold).astype(int)
        
        # E. Риск-метрики (4)
        # Максимальная просадка за период (для LONG)
        for period_name, n_candles in [('1h', 4), ('4h', 16)]:
            min_prices = pd.DataFrame()
            for i in range(1, n_candles + 1):
                future_low = df.groupby('symbol')['low'].transform(lambda x: x.shift(-i))
                min_prices[f'low_{i}'] = future_low
        
            # Минимальная цена за период
            min_price = min_prices.min(axis=1)
            df[f'max_drawdown_{period_name}'] = (df['close'] / min_price - 1).fillna(0)
        
        # Максимальный рост за период (для SHORT)
        for period_name, n_candles in [('1h', 4), ('4h', 16)]:
            max_prices = pd.DataFrame()
            for i in range(1, n_candles + 1):
                future_high = df.groupby('symbol')['high'].transform(lambda x: x.shift(-i))
                max_prices[f'high_{i}'] = future_high
        
            # Максимальная цена за период
            max_price = max_prices.max(axis=1)
            df[f'max_rally_{period_name}'] = (max_price / df['close'] - 1).fillna(0)
        
        # ИСПРАВЛЕНО: Убираем торговые сигналы с утечками данных
        # best_action, risk_reward_ratio и optimal_hold_time будут генерироваться
        # в trading/signal_generator.py на основе предсказаний модели
        
        # ПЕРЕНЕСЕНО В ПРИЗНАКИ: signal_strength теперь feature, не target
        # Это основано на исторических данных, без утечек
        
        # УДАЛЕНО: risk_reward_ratio и optimal_hold_time содержали утечки данных
        # Эти переменные будут генерироваться в trading/signal_generator.py
        # на основе предсказаний модели, а не реальных будущих данных
        
        # УДАЛЕНО: best_action и все legacy переменные (best_direction, reached, hit)
        # В версии 4.0 используем только 20 целевых переменных без утечек данных
        # Все необходимые целевые переменные уже созданы выше
        
        # Фиктивные временные переменные для совместимости
        df['long_tp1_time'] = 16  # 4 часа
        df['long_tp2_time'] = 16
        df['long_tp3_time'] = 48  # 12 часов
        df['long_sl_time'] = 100
        df['short_tp1_time'] = 16
        df['short_tp2_time'] = 16
        df['short_tp3_time'] = 48
        df['short_sl_time'] = 100
        
        # Expected value для совместимости
        df['long_expected_value'] = df['future_return_4h'] * df['long_will_reach_2pct_4h'] * 2.0
        df['short_expected_value'] = -df['future_return_4h'] * df['short_will_reach_2pct_4h'] * 2.0
        
        # Optimal entry фиктивные переменные
        df['long_optimal_entry_time'] = 1
        df['long_optimal_entry_price'] = df['close']
        df['long_optimal_entry_improvement'] = 0
        df['short_optimal_entry_time'] = 1
        df['short_optimal_entry_price'] = df['close']
        df['short_optimal_entry_improvement'] = 0
        
        # Итоговая статистика
        if not self.disable_progress:
            self.logger.info(f"  ✅ Создано 20 целевых переменных без утечек данных")
            self.logger.info(f"  📊 Распределение направлений:")
            for period in ['15m', '1h', '4h', '12h']:
                if f'direction_{period}' in df.columns:
                    dist = df[f'direction_{period}'].value_counts(normalize=True) * 100
                    self.logger.info(f"     {period}: UP={dist.get('UP', 0):.1f}%, DOWN={dist.get('DOWN', 0):.1f}%, FLAT={dist.get('FLAT', 0):.1f}%")
        
        
        return df
    
    
    
    def _normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Нормализация признаков с поддержкой режима fit/transform
        
        Args:
            df: данные для нормализации
            fit: если True - обучает scaler, если False - использует существующий
        """
        if fit:
            if not self.disable_progress:
                self.logger.info("📊 Обучение и применение нормализации...")
        else:
            if not self.disable_progress:
                self.logger.info("📊 Применение существующей нормализации...")
        
        # Столбцы для исключения из нормализации
        exclude_cols = [
            'id', 'symbol', 'timestamp', 'datetime', 'sector',
            'open', 'high', 'low', 'close', 'volume', 'turnover'
        ]
        
        # Целевые переменные и индикаторы направления
        target_cols = [col for col in df.columns if any(pattern in col for pattern in [
            'target_', 'future_', 'optimal_', '_reached', '_tp', '_sl', 
            'expected_value', 'best_direction', 'signal_strength'
        ])]
        exclude_cols.extend(target_cols)
        
        # Временные и категориальные колонки
        time_cols = ['hour', 'minute', 'dayofweek', 'day', 'month', 'is_weekend',
                    'asian_session', 'european_session', 'american_session', 'session_overlap']
        exclude_cols.extend(time_cols)
        
        # Признаки-соотношения, которые уже нормализованы по своей природе
        ratio_cols = ['close_vwap_ratio', 'close_open_ratio', 'high_low_ratio', 
                      'close_position', 'bb_position', 'position_in_range_20',
                      'position_in_range_50', 'position_in_range_100']
        exclude_cols.extend(ratio_cols)
        
        # ИСПРАВЛЕНО: Технические индикаторы с естественными диапазонами НЕ нормализуем
        technical_indicators = ['rsi', 'stoch_k', 'stoch_d', 'adx', 'adx_pos', 'adx_neg',
                              'rsi_oversold', 'rsi_overbought', 'toxicity', 'psar_trend',
                              'cci', 'williams_r', 'roc', 'momentum', 'kama', 'trix',
                              'ppo', 'macd', 'macd_signal', 'macd_diff']
        exclude_cols.extend(technical_indicators)
        
        # Определяем только числовые признаки для нормализации
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Логирование для отладки
        if not self.disable_progress:
            self.logger.debug(f"Колонки для нормализации ({len(feature_cols)}): {feature_cols[:10]}...")
            excluded_technical = [col for col in ['toxicity', 'bb_position', 'close_position', 'psar_trend', 
                                                  'rsi_oversold', 'rsi_overbought'] if col in numeric_cols]
            self.logger.debug(f"Технические индикаторы в исключениях: {excluded_technical}")
        
        if not feature_cols:
            self.logger.warning("⚠️ Нет признаков для нормализации!")
            return df
        
        # Нормализация по символам
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            
            if symbol_mask.sum() > 0:
                if fit:
                    # Создаем новый scaler для символа
                    if symbol not in self.scalers:
                        self.scalers[symbol] = RobustScaler()
                    
                    # Обучаем scaler
                    symbol_data = df.loc[symbol_mask, feature_cols]
                    valid_data = symbol_data.dropna()
                    
                    if len(valid_data) > 0:
                        self.scalers[symbol].fit(valid_data)
                        if not self.disable_progress:
                            self.logger.debug(f"✅ Scaler обучен для {symbol} на {len(valid_data)} записях")
                
                # Применяем scaler (если он существует)
                if symbol in self.scalers:
                    valid_mask = symbol_mask & df[feature_cols].notna().all(axis=1)
                    if valid_mask.sum() > 0:
                        df.loc[valid_mask, feature_cols] = self.scalers[symbol].transform(
                            df.loc[valid_mask, feature_cols]
                        )
                else:
                    if not self.disable_progress:
                        self.logger.warning(f"⚠️ Scaler не найден для {symbol}")
        
        return df
    
    def _normalize_walk_forward(self, df: pd.DataFrame, train_end_date: str) -> pd.DataFrame:
        """Walk-forward нормализация без data leakage"""
        if not self.disable_progress:
            self.logger.info(f"Walk-forward нормализация до {train_end_date}...")
        
        # Столбцы для исключения из нормализации
        exclude_cols = [
            'id', 'symbol', 'timestamp', 'datetime', 'sector',
            'open', 'high', 'low', 'close', 'volume', 'turnover'
        ]
        
        # Целевые переменные
        target_cols = [col for col in df.columns if col.startswith(('target_', 'future_', 'optimal_'))]
        exclude_cols.extend(target_cols)
        
        # Временные колонки
        time_cols = ['hour', 'minute', 'dayofweek', 'day', 'month', 'is_weekend',
                    'asian_session', 'european_session', 'american_session', 'session_overlap']
        exclude_cols.extend(time_cols)
        
        # Признаки-соотношения, которые уже нормализованы по своей природе
        ratio_cols = ['close_vwap_ratio', 'close_open_ratio', 'high_low_ratio', 
                      'close_position', 'bb_position', 'position_in_range_20',
                      'position_in_range_50', 'position_in_range_100']
        exclude_cols.extend(ratio_cols)
        
        # ИСПРАВЛЕНО: Технические индикаторы с естественными диапазонами НЕ нормализуем
        technical_indicators = ['rsi', 'stoch_k', 'stoch_d', 'adx', 'adx_pos', 'adx_neg',
                              'rsi_oversold', 'rsi_overbought', 'toxicity', 'psar_trend',
                              'cci', 'williams_r', 'roc', 'momentum', 'kama', 'trix',
                              'ppo', 'macd', 'macd_signal', 'macd_diff']
        exclude_cols.extend(technical_indicators)
        
        # Определяем признаки для нормализации
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Маска для обучающих данных
        train_mask = df['datetime'] <= pd.to_datetime(train_end_date)
        
        # Нормализация по символам
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            train_symbol_mask = symbol_mask & train_mask
            
            if train_symbol_mask.sum() > 0:
                if symbol not in self.scalers:
                    self.scalers[symbol] = StandardScaler()
                
                # Обучаем scaler только на train данных
                train_data = df.loc[train_symbol_mask, feature_cols].dropna()
                if len(train_data) > 0:
                    self.scalers[symbol].fit(train_data)
                    
                    # Применяем ко всем данным символа
                    valid_mask = symbol_mask & df[feature_cols].notna().all(axis=1)
                    if valid_mask.sum() > 0:
                        df.loc[valid_mask, feature_cols] = self.scalers[symbol].transform(
                            df.loc[valid_mask, feature_cols]
                        )
        
        return df
    
    def _log_feature_statistics(self, df: pd.DataFrame):
        """Логирование статистики по признакам"""
        if not self.disable_progress:
            feature_counts = {
                'basic': len([col for col in df.columns if col in [
                    'returns', 'high_low_ratio', 'close_open_ratio', 'volume_ratio'
                ]]),
                'technical': len([col for col in df.columns if any(
                    ind in col for ind in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr']
                )]),
                'microstructure': len([col for col in df.columns if any(
                    ms in col for ms in ['spread', 'imbalance', 'toxicity', 'illiquidity']
                )]),
                'temporal': len([col for col in df.columns if any(
                    t in col for t in ['hour', 'day', 'month', 'session']
                )]),
                'cross_asset': len([col for col in df.columns if any(
                    ca in col for ca in ['btc_', 'sector', 'rank', 'momentum']
                )])
            }
            
            self.logger.info(f"📊 Создано признаков по категориям: {feature_counts}")
            
            # Проверка пропущенных значений
            missing_counts = df.isnull().sum()
            if missing_counts.sum() > 0:
                self.logger.warning(
                    f"⚠️ Обнаружены пропущенные значения в {missing_counts[missing_counts > 0].shape[0]} признаках"
                )
    
    def get_feature_names(self, include_targets: bool = False) -> List[str]:
        """Получение списка названий признаков"""
        # TODO: Реализовать правильное хранение названий признаков
        return []
    
    def save_scalers(self, path: str):
        """Сохранение скейлеров для использования в продакшене"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        if not self.disable_progress:
            self.logger.info(f"Скейлеры сохранены в {path}")
    
    def load_scalers(self, path: str):
        """Загрузка сохраненных скейлеров"""
        import pickle
        with open(path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        if not self.disable_progress:
            self.logger.info(f"Скейлеры загружены из {path}")
    
    def create_features_with_train_split(self, 
                                       df: pd.DataFrame, 
                                       train_ratio: float = 0.6,
                                       val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ИСПРАВЛЕННЫЙ метод создания признаков БЕЗ DATA LEAKAGE
        
        Args:
            df: исходные данные
            train_ratio: доля обучающих данных
            val_ratio: доля валидационных данных
            
        Returns:
            Tuple[train_data, val_data, test_data] - правильно нормализованные данные
        """
        if not self.disable_progress:
            self.logger.start_stage("feature_engineering_no_leakage", 
                                   symbols=df['symbol'].nunique())
        
        # 1. Создание признаков (без нормализации)
        if not self.disable_progress:
            self.logger.info("1/5 - Создание базовых признаков...")
        featured_dfs = []
        
        symbols = df['symbol'].unique()
        if not self.disable_progress:
            self.logger.info(f"Обработка {len(symbols)} символов...")
        
        # В многопроцессорном режиме прогресс-бары не нужны
        disable_progress = hasattr(self, 'disable_progress') and self.disable_progress
        
        if disable_progress:
            symbols_iterator = symbols
        else:
            symbols_iterator = tqdm(symbols, desc="Создание признаков", unit="символ")
        
        for symbol in symbols_iterator:
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('datetime')
            
            symbol_data = self._create_basic_features(symbol_data)
            symbol_data = self._create_technical_indicators(symbol_data)
            symbol_data = self._create_microstructure_features(symbol_data)
            symbol_data = self._create_rally_detection_features(symbol_data)
            symbol_data = self._create_signal_quality_features(symbol_data)
            symbol_data = self._create_futures_specific_features(symbol_data)
            symbol_data = self._create_ml_optimized_features(symbol_data)
            symbol_data = self._create_temporal_features(symbol_data)
            symbol_data = self._create_target_variables(symbol_data)
            
            featured_dfs.append(symbol_data)
        
        if not self.disable_progress:
            self.logger.info("2/5 - Объединение кросс-активных признаков...")
        result_df = pd.concat(featured_dfs, ignore_index=True)
        result_df = self._create_cross_asset_features(result_df)
        
        if not self.disable_progress:
            self.logger.info("3/5 - Обработка пропущенных значений...")
        result_df = self._handle_missing_values(result_df)
        
        # 2. Разделение данных ПО ВРЕМЕНИ (критично для предотвращения data leakage)
        if not self.disable_progress:
            self.logger.info("4/5 - Временное разделение данных...")
        train_data_list = []
        val_data_list = []
        test_data_list = []
        
        for symbol in result_df['symbol'].unique():
            symbol_data = result_df[result_df['symbol'] == symbol].sort_values('datetime')
            n = len(symbol_data)
            
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_data_list.append(symbol_data.iloc[:train_end])
            val_data_list.append(symbol_data.iloc[train_end:val_end])
            test_data_list.append(symbol_data.iloc[val_end:])
        
        train_data = pd.concat(train_data_list, ignore_index=True)
        val_data = pd.concat(val_data_list, ignore_index=True)
        test_data = pd.concat(test_data_list, ignore_index=True)
        
        # 3. ПРАВИЛЬНАЯ нормализация БЕЗ DATA LEAKAGE
        if not self.disable_progress:
            self.logger.info("5/5 - Нормализация без data leakage...")


        # Определяем признаки для нормализации
        exclude_cols = [
            'id', 'symbol', 'timestamp', 'datetime', 'sector',
            'open', 'high', 'low', 'close', 'volume', 'turnover'
        ]
        
        # Целевые переменные
        target_cols = [col for col in train_data.columns if col.startswith(('target_', 'future_', 'optimal_'))]
        exclude_cols.extend(target_cols)
        
        # Временные колонки (уже нормализованы)
        time_cols = ['hour', 'minute', 'dayofweek', 'day', 'month', 'is_weekend',
                    'asian_session', 'european_session', 'american_session', 'session_overlap']
        exclude_cols.extend(time_cols)
        
        # Признаки-соотношения, которые уже нормализованы по своей природе
        ratio_cols = ['close_vwap_ratio', 'close_open_ratio', 'high_low_ratio', 
                      'close_position', 'bb_position', 'position_in_range_20',
                      'position_in_range_50', 'position_in_range_100']
        exclude_cols.extend(ratio_cols)
        
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        # Нормализация по символам
        unique_symbols = train_data['symbol'].unique()
        # В многопроцессорном режиме отключаем прогресс-бары
        if disable_progress:
            norm_iterator = unique_symbols
        else:
            norm_iterator = tqdm(unique_symbols, desc="Нормализация", unit="символ")
        
        for symbol in norm_iterator:
            
            # Маски для каждого символа
            train_mask = train_data['symbol'] == symbol
            val_mask = val_data['symbol'] == symbol
            test_mask = test_data['symbol'] == symbol
            
            if train_mask.sum() == 0:
                continue
            
            # Обучаем scaler ТОЛЬКО на train данных
            if symbol not in self.scalers:
                self.scalers[symbol] = RobustScaler()
            
            # Получаем только валидные train данные
            train_symbol_data = train_data.loc[train_mask, feature_cols].dropna()
            
            # Сохраняем числовые колонки для использования во всем цикле
            numeric_feature_cols = []
            
            if len(train_symbol_data) > 0:
                # Очистка экстремальных значений в train данных
                train_cleaned = train_symbol_data.copy()
                
                # Проверяем типы данных и фильтруем только числовые колонки
                for col in feature_cols:
                    if col in train_cleaned.columns and pd.api.types.is_numeric_dtype(train_cleaned[col]):
                        numeric_feature_cols.append(col)
                    else:
                        if not self.disable_progress:
                            self.logger.warning(f"Колонка '{col}' не является числовой или отсутствует, пропускаем")
                
                for col in numeric_feature_cols:
                    # ИСПРАВЛЕНО: Дополнительная проверка и конвертация перед квантилями
                    # Конвертируем в числовой тип на случай если есть строки
                    train_cleaned[col] = pd.to_numeric(train_cleaned[col], errors='coerce')
                    
                    # Пропускаем колонки с только NaN значениями
                    if train_cleaned[col].notna().sum() == 0:
                        if not self.disable_progress:
                            self.logger.warning(f"Колонка '{col}' содержит только NaN значения, пропускаем")
                        continue
                    
                    # Клиппинг экстремальных значений
                    q01 = train_cleaned[col].quantile(0.01)
                    q99 = train_cleaned[col].quantile(0.99)
                    train_cleaned[col] = train_cleaned[col].clip(lower=q01, upper=q99)
                    
                    # Замена inf на конечные значения
                    train_cleaned[col] = train_cleaned[col].replace([np.inf, -np.inf], [q99, q01])
                    train_cleaned[col] = train_cleaned[col].fillna(train_cleaned[col].median())
                
                # Обучаем scaler на очищенных train данных только по числовым колонкам
                self.scalers[symbol].fit(train_cleaned[numeric_feature_cols])
                
                # Применяем ко всем данным символа
                # Train
                train_valid_mask = train_mask & train_data[numeric_feature_cols].notna().all(axis=1)
                if train_valid_mask.sum() > 0:
                    train_to_scale = train_data.loc[train_valid_mask, numeric_feature_cols].copy()
                    # Применяем ту же очистку
                    for col in numeric_feature_cols:
                        # ИСПРАВЛЕНО: Конвертация в числовой тип
                        train_to_scale[col] = pd.to_numeric(train_to_scale[col], errors='coerce')
                        
                        if train_to_scale[col].notna().sum() == 0:
                            continue
                            
                        q01 = train_cleaned[col].quantile(0.01) if col in train_cleaned.columns else train_to_scale[col].quantile(0.01)
                        q99 = train_cleaned[col].quantile(0.99) if col in train_cleaned.columns else train_to_scale[col].quantile(0.99)
                        train_to_scale[col] = train_to_scale[col].clip(lower=q01, upper=q99)
                        train_to_scale[col] = train_to_scale[col].replace([np.inf, -np.inf], [q99, q01])
                        train_to_scale[col] = train_to_scale[col].fillna(train_to_scale[col].median())
                    
                    train_data.loc[train_valid_mask, numeric_feature_cols] = self.scalers[symbol].transform(train_to_scale)
                
                # Val 
                val_valid_mask = val_mask & val_data[numeric_feature_cols].notna().all(axis=1)
                if val_valid_mask.sum() > 0:
                    val_to_scale = val_data.loc[val_valid_mask, numeric_feature_cols].copy()
                    # Применяем ту же очистку используя статистики из train
                    for col in numeric_feature_cols:
                        # ИСПРАВЛЕНО: Конвертация в числовой тип
                        val_to_scale[col] = pd.to_numeric(val_to_scale[col], errors='coerce')
                        
                        if val_to_scale[col].notna().sum() == 0:
                            continue
                            
                        q01 = train_cleaned[col].quantile(0.01) if col in train_cleaned.columns else val_to_scale[col].quantile(0.01)
                        q99 = train_cleaned[col].quantile(0.99) if col in train_cleaned.columns else val_to_scale[col].quantile(0.99)
                        val_to_scale[col] = val_to_scale[col].clip(lower=q01, upper=q99)
                        val_to_scale[col] = val_to_scale[col].replace([np.inf, -np.inf], [q99, q01])
                        val_to_scale[col] = val_to_scale[col].fillna(val_to_scale[col].median())
                    
                    val_data.loc[val_valid_mask, numeric_feature_cols] = self.scalers[symbol].transform(val_to_scale)
                
                # Test
                test_valid_mask = test_mask & test_data[numeric_feature_cols].notna().all(axis=1)
                if test_valid_mask.sum() > 0:
                    test_to_scale = test_data.loc[test_valid_mask, numeric_feature_cols].copy()
                    # Применяем ту же очистку используя статистики из train
                    for col in numeric_feature_cols:
                        # ИСПРАВЛЕНО: Конвертация в числовой тип
                        test_to_scale[col] = pd.to_numeric(test_to_scale[col], errors='coerce')
                        
                        if test_to_scale[col].notna().sum() == 0:
                            continue
                            
                        q01 = train_cleaned[col].quantile(0.01) if col in train_cleaned.columns else test_to_scale[col].quantile(0.01)
                        q99 = train_cleaned[col].quantile(0.99) if col in train_cleaned.columns else test_to_scale[col].quantile(0.99)
                        test_to_scale[col] = test_to_scale[col].clip(lower=q01, upper=q99)
                        test_to_scale[col] = test_to_scale[col].replace([np.inf, -np.inf], [q99, q01])
                        test_to_scale[col] = test_to_scale[col].fillna(test_to_scale[col].median())
                    
                    test_data.loc[test_valid_mask, numeric_feature_cols] = self.scalers[symbol].transform(test_to_scale)
        
        # КРИТИЧНО: Удаляем строки с NaN в future переменных
        # NaN появляются в последних N строках каждого символа из-за shift(-N)
        future_cols = [col for col in train_data.columns if col.startswith('future_')]
        if future_cols:
            if not self.disable_progress:
                self.logger.info("🧑 Удаление строк с NaN в целевых переменных...")
            
            # Подсчет до удаления
            train_before = len(train_data)
            val_before = len(val_data)
            test_before = len(test_data)
            
            # Удаляем строки с NaN в любой из future колонок
            train_data = train_data.dropna(subset=future_cols)
            val_data = val_data.dropna(subset=future_cols)
            test_data = test_data.dropna(subset=future_cols)
            
            if not self.disable_progress:
                self.logger.info(f"  Удалено строк: Train={train_before - len(train_data)}, "
                               f"Val={val_before - len(val_data)}, Test={test_before - len(test_data)}")
        
        # Проверка на оставшиеся NaN
        nan_check = {
            'train': train_data.isna().sum().sum(),
            'val': val_data.isna().sum().sum(),
            'test': test_data.isna().sum().sum()
        }
        
        for split, nan_count in nan_check.items():
            if nan_count > 0:
                if not self.disable_progress:
                    self.logger.warning(f"⚠️  Осталось {nan_count} NaN в {split} данных")
        
        # Финальная статистика
        if not self.disable_progress:
            self.logger.info(f"✅ Размеры данных без data leakage:")
            self.logger.info(f"   - Train: {len(train_data)} записей")
            self.logger.info(f"   - Val: {len(val_data)} записей") 
            self.logger.info(f"   - Test: {len(test_data)} записей")
            self.logger.info(f"   - Признаков: {len(feature_cols)}")
            
            self.logger.end_stage("feature_engineering_no_leakage", 
                                train_size=len(train_data),
                                val_size=len(val_data),
                                test_size=len(test_data))
        
        return train_data, val_data, test_data
    
    def _add_enhanced_features(self, df: pd.DataFrame, all_symbols_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Добавление расширенных признаков для улучшения direction prediction
        
        Args:
            df: DataFrame с базовыми признаками
            all_symbols_data: словарь с данными всех символов для cross-asset features
            
        Returns:
            DataFrame с enhanced features
        """
        try:
            from data.enhanced_features import EnhancedFeatureEngineer
        except ImportError:
            self.logger.warning("⚠️ Модуль enhanced_features не найден, пропускаем enhanced features")
            return df
        
        self.logger.info("🚀 Добавление enhanced features для direction prediction...")
        
        enhanced_engineer = EnhancedFeatureEngineer()
        enhanced_dfs = []
        
        # Обрабатываем каждый символ
        for symbol in tqdm(df['symbol'].unique(), desc="Enhanced features", disable=self.disable_progress):
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Применяем enhanced features
            enhanced_data = enhanced_engineer.create_enhanced_features(
                symbol_data,
                all_symbols_data if len(all_symbols_data) > 1 else None
            )
            
            enhanced_dfs.append(enhanced_data)
        
        # Объединяем результаты
        result_df = pd.concat(enhanced_dfs, ignore_index=True)
        
        # Логируем статистику новых признаков
        original_cols = set(df.columns)
        new_cols = set(result_df.columns) - original_cols
        
        if new_cols:
            self.logger.info(f"✅ Добавлено {len(new_cols)} enhanced features")
            
            # Категоризация новых признаков
            categories = {
                'market_regime': [col for col in new_cols if 'regime' in col or 'wyckoff' in col],
                'microstructure': [col for col in new_cols if any(x in col for x in ['ofi', 'tick', 'imbalance'])],
                'cross_asset': [col for col in new_cols if any(x in col for x in ['btc_', 'sector_', 'beta_'])],
                'sentiment': [col for col in new_cols if any(x in col for x in ['fear_greed', 'panic', 'euphoria'])]
            }
            
            for category, cols in categories.items():
                if cols:
                    self.logger.info(f"  - {category}: {len(cols)} признаков")
        
        return result_df
