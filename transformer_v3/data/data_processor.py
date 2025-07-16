"""
Единый упрощенный процессор данных для Transformer v3
Объединяет функциональность preprocessor и feature_engineer
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import pywt
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


class DataProcessor:
    """Упрощенный процессор данных с фильтрацией шума"""
    
    # Минимальный набор технических индикаторов (30 ключевых)
    CORE_INDICATORS = [
        # Основные трендовые (8)
        'rsi_val', 'macd_val', 'macd_signal', 'macd_hist',
        'adx_val', 'adx_plus_di', 'adx_minus_di', 'sar',
        
        # Волатильность (6)
        'atr_val', 'bb_upper', 'bb_lower', 'bb_basis',
        'bb_position', 'atr_norm',
        
        # Объемные (4)
        'obv', 'cmf', 'volume_ratio', 'mfi',
        
        # Осцилляторы (6)
        'stoch_k', 'stoch_d', 'cci_val', 'williams_r',
        'roc', 'rsi_dist_from_mid',
        
        # Ценовые изменения (6)
        'price_change_1', 'price_change_4', 'price_change_16',
        'volatility_4', 'volatility_16', 'log_return'
    ]
    
    # Простые инженерные признаки (5)
    ENGINEERED_FEATURES = [
        'rsi_oversold',    # RSI < 30
        'rsi_overbought',  # RSI > 70
        'macd_bullish',    # MACD > Signal
        'strong_trend',    # ADX > 25
        'high_volume'      # Volume ratio > 2
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = RobustScaler(quantile_range=(5, 95))  # Более агрессивное клиппирование
        self.pca = None
        self.feature_columns = None
        self.variance_threshold = 0.01  # Минимальная дисперсия признака
        
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Основной метод обработки данных
        """
        logger.info("🔧 Упрощенная обработка данных...")
        
        # 1. Извлечение базовых признаков
        df = self._extract_core_features(df)
        
        # 2. Создание минимальных инженерных признаков
        df = self._create_simple_features(df)
        
        # 3. One-hot encoding символов (только топ-10)
        df = self._encode_symbols(df)
        
        # 4. Фильтрация шума с помощью wavelets
        df = self._denoise_features(df)
        
        # 5. Удаление признаков с низкой дисперсией
        df = self._remove_low_variance_features(df)
        
        logger.info(f"✅ Финальное количество признаков: {len(self.feature_columns)}")
        
        return df
    
    def _extract_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение только ключевых технических индикаторов"""
        logger.info("📊 Извлечение основных технических индикаторов...")
        
        processed_data = []
        
        for symbol in tqdm(df['symbol'].unique(), desc="Обработка символов"):
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
            
            features_list = []
            
            for idx, row in symbol_df.iterrows():
                feature_dict = {
                    'symbol': symbol,
                    'timestamp': row['timestamp'],
                    'buy_expected_return': row.get('buy_expected_return', 0),
                    'sell_expected_return': row.get('sell_expected_return', 0)
                }
                
                # Извлекаем только основные индикаторы
                indicators = row.get('technical_indicators', {})
                
                # Базовые OHLCV для расчета log_return
                feature_dict['close'] = float(row.get('close', 0))
                feature_dict['volume'] = float(row.get('volume', 0))
                
                for indicator in self.CORE_INDICATORS:
                    if indicator == 'log_return':
                        # Рассчитываем log return
                        if idx > 0:
                            prev_close = symbol_df.iloc[idx-1]['close']
                            if prev_close > 0:
                                feature_dict['log_return'] = np.log(row['close'] / prev_close)
                            else:
                                feature_dict['log_return'] = 0.0
                        else:
                            feature_dict['log_return'] = 0.0
                    else:
                        value = indicators.get(indicator, 0.0)
                        if value is None or pd.isna(value):
                            value = 0.0
                        feature_dict[indicator] = float(value)
                
                features_list.append(feature_dict)
            
            symbol_features_df = pd.DataFrame(features_list)
            processed_data.append(symbol_features_df)
        
        result_df = pd.concat(processed_data, ignore_index=True)
        
        # Удаляем временные колонки
        result_df = result_df.drop(['close', 'volume'], axis=1, errors='ignore')
        
        return result_df
    
    def _create_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание минимального набора инженерных признаков"""
        logger.info("🔨 Создание простых инженерных признаков...")
        
        # RSI признаки
        if 'rsi_val' in df.columns:
            df['rsi_oversold'] = (df['rsi_val'] < 30).astype(float)
            df['rsi_overbought'] = (df['rsi_val'] > 70).astype(float)
        
        # MACD признак
        if 'macd_val' in df.columns and 'macd_signal' in df.columns:
            df['macd_bullish'] = (df['macd_val'] > df['macd_signal']).astype(float)
        
        # ADX признак
        if 'adx_val' in df.columns:
            df['strong_trend'] = (df['adx_val'] > 25).astype(float)
        
        # Volume признак
        if 'volume_ratio' in df.columns:
            df['high_volume'] = (df['volume_ratio'] > 2.0).astype(float)
        
        return df
    
    def _encode_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encoding только для топ-10 символов по объему"""
        logger.info("🏷️ Кодирование символов...")
        
        # Определяем топ-10 символов по количеству записей
        top_symbols = df['symbol'].value_counts().head(10).index.tolist()
        
        for symbol in top_symbols:
            col_name = f"is_{symbol.replace('USDT', '').lower()}"
            df[col_name] = (df['symbol'] == symbol).astype(float)
        
        # Удаляем колонку symbol
        df = df.drop('symbol', axis=1, errors='ignore')
        
        return df
    
    def _denoise_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение wavelet denoising для снижения шума"""
        logger.info("🔇 Применение wavelet denoising...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns 
                          if col not in ['timestamp', 'buy_expected_return', 'sell_expected_return']]
        
        for col in numeric_columns:
            if df[col].std() > 0:  # Только для непостоянных признаков
                # Применяем wavelet denoising
                coeffs = pywt.wavedec(df[col].fillna(0).values, 'db4', level=3)
                
                # Мягкое пороговое значение для удаления шума
                threshold = 0.1 * np.std(coeffs[-1])
                coeffs = list(coeffs)
                for i in range(1, len(coeffs)):
                    coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
                
                # Восстанавливаем сигнал
                denoised = pywt.waverec(coeffs, 'db4')
                
                # Обрезаем до нужной длины
                if len(denoised) > len(df):
                    denoised = denoised[:len(df)]
                elif len(denoised) < len(df):
                    denoised = np.pad(denoised, (0, len(df) - len(denoised)), mode='edge')
                
                df[col] = denoised
        
        return df
    
    def _remove_low_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление признаков с низкой дисперсией"""
        logger.info("🗑️ Удаление признаков с низкой дисперсией...")
        
        # Сохраняем важные колонки
        important_cols = ['timestamp', 'buy_expected_return', 'sell_expected_return']
        feature_cols = [col for col in df.columns if col not in important_cols]
        
        # Применяем фильтр дисперсии
        selector = VarianceThreshold(threshold=self.variance_threshold)
        features_filtered = selector.fit_transform(df[feature_cols])
        
        # Получаем имена оставшихся признаков
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) 
                           if selector.variances_[i] >= self.variance_threshold]
        
        # Создаем новый DataFrame
        result_df = pd.DataFrame(features_filtered, columns=selected_features)
        
        # Добавляем обратно важные колонки
        for col in important_cols:
            if col in df.columns:
                result_df[col] = df[col].values
        
        # Сохраняем список признаков
        self.feature_columns = [col for col in result_df.columns 
                               if col not in important_cols]
        
        logger.info(f"   Удалено {len(feature_cols) - len(selected_features)} признаков с низкой дисперсией")
        
        return result_df
    
    def normalize_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                      X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Нормализация данных с агрессивным клиппированием выбросов
        """
        logger.info("📊 Нормализация данных с клиппированием выбросов...")
        
        # Обучаем scaler только на train
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Агрессивное клиппирование выбросов
        clip_value = 3.0
        X_train_scaled = np.clip(X_train_scaled, -clip_value, clip_value)
        X_val_scaled = np.clip(X_val_scaled, -clip_value, clip_value)
        X_test_scaled = np.clip(X_test_scaled, -clip_value, clip_value)
        
        # Опционально: PCA для дальнейшего снижения размерности
        if len(self.feature_columns) > 40:
            logger.info(f"   Применение PCA: {len(self.feature_columns)} -> 40 признаков")
            self.pca = PCA(n_components=40, random_state=42)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_val_scaled = self.pca.transform(X_val_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Временное разделение данных с защитой от утечки
        """
        logger.info("✂️ Разделение данных по времени...")
        
        # Сортируем по времени
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Определяем границы
        n_samples = len(df)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)
        
        # Добавляем зазор для предотвращения утечки
        gap = 100  # 100 свечей = 25 часов
        
        # Разделяем данные
        train_df = df.iloc[:train_end - gap]
        val_df = df.iloc[train_end:val_end - gap]
        test_df = df.iloc[val_end:]
        
        logger.info(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        
        # Извлекаем признаки и целевые переменные
        feature_cols = self.feature_columns
        
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        X_test = test_df[feature_cols]
        
        y_buy_train = train_df['buy_expected_return'].values
        y_buy_val = val_df['buy_expected_return'].values
        y_buy_test = test_df['buy_expected_return'].values
        
        y_sell_train = train_df['sell_expected_return'].values
        y_sell_val = val_df['sell_expected_return'].values
        y_sell_test = test_df['sell_expected_return'].values
        
        # Нормализация
        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)
        
        return {
            'buy': {
                'X_train': X_train, 'y_train': y_buy_train,
                'X_val': X_val, 'y_val': y_buy_val,
                'X_test': X_test, 'y_test': y_buy_test
            },
            'sell': {
                'X_train': X_train, 'y_train': y_sell_train,
                'X_val': X_val, 'y_val': y_sell_val,
                'X_test': X_test, 'y_test': y_sell_test
            }
        }