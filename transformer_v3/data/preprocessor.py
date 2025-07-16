"""
Препроцессор данных для Transformer v3
Извлечение признаков из JSON и подготовка данных
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import Config, TECHNICAL_INDICATORS, ENGINEERED_FEATURES, VALIDATION_PARAMS

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.all_features = TECHNICAL_INDICATORS + ENGINEERED_FEATURES
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков из JSON и создание инженерных признаков
        
        Args:
            df: DataFrame с сырыми данными
            
        Returns:
            DataFrame с извлеченными признаками
        """
        logger.info("🔧 Извлечение признаков из technical_indicators...")
        
        # Проверка expected returns
        self._check_expected_returns(df)
        
        # Группируем по символам для правильной обработки
        grouped_data = []
        
        for symbol in tqdm(df["symbol"].unique(), desc="Обработка символов"):
            symbol_df = df[df["symbol"] == symbol].copy()
            symbol_df = symbol_df.sort_values("timestamp").reset_index(drop=True)
            
            # Извлекаем признаки для каждой строки
            features_list = []
            
            for idx, row in symbol_df.iterrows():
                feature_dict = {
                    'symbol': symbol,
                    'timestamp': row['timestamp'],
                    'datetime': row['datetime'],
                    'buy_expected_return': row['buy_expected_return'],
                    'sell_expected_return': row['sell_expected_return']
                }
                
                # Извлекаем технические индикаторы
                indicators = row["technical_indicators"]
                for indicator in TECHNICAL_INDICATORS:
                    value = indicators.get(indicator, 0.0)
                    if value is None or pd.isna(value):
                        value = 0.0
                    feature_dict[indicator] = float(value)
                
                # Создаем инженерные признаки
                feature_dict.update(self._create_engineered_features(indicators))
                
                features_list.append(feature_dict)
            
            symbol_features_df = pd.DataFrame(features_list)
            grouped_data.append(symbol_features_df)
        
        # Объединяем все данные
        result_df = pd.concat(grouped_data, ignore_index=True)
        
        logger.info(f"✅ Извлечено {len(self.all_features)} признаков для {len(result_df)} записей")
        
        # Сохраняем список признаков
        self.feature_columns = self.all_features
        
        return result_df
    
    def _create_engineered_features(self, indicators: Dict) -> Dict:
        """Создание инженерных признаков"""
        features = {}
        
        # RSI признаки
        rsi = indicators.get("rsi_val", 50.0)
        features["rsi_oversold"] = 1.0 if rsi is not None and rsi < 30 else 0.0
        features["rsi_overbought"] = 1.0 if rsi is not None and rsi > 70 else 0.0
        
        # MACD признаки
        macd = indicators.get("macd_val", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)
        features["macd_bullish"] = 1.0 if macd is not None and macd_signal is not None and macd > macd_signal else 0.0
        
        # Bollinger Bands признаки
        bb_position = indicators.get("bb_position", 0.5)
        features["bb_near_lower"] = 1.0 if bb_position is not None and bb_position < 0.2 else 0.0
        features["bb_near_upper"] = 1.0 if bb_position is not None and bb_position > 0.8 else 0.0
        
        # ADX тренд
        adx = indicators.get("adx_val", 0.0)
        features["strong_trend"] = 1.0 if adx is not None and adx > 25 else 0.0
        
        # Объем
        volume_ratio = indicators.get("volume_ratio", 1.0)
        features["high_volume"] = 1.0 if volume_ratio is not None and volume_ratio > 2.0 else 0.0
        
        return features
    
    def _check_expected_returns(self, df: pd.DataFrame):
        """Проверка диапазона expected returns"""
        buy_returns = df['buy_expected_return'].values
        sell_returns = df['sell_expected_return'].values
        
        buy_outliers = np.sum((buy_returns < -1.1) | (buy_returns > 5.8))
        sell_outliers = np.sum((sell_returns < -1.1) | (sell_returns > 5.8))
        
        if buy_outliers > 0 or sell_outliers > 0:
            logger.warning(f"⚠️ Найдены значения expected_return вне ожидаемого диапазона [-1.1%, +5.8%]:")
            logger.warning(f"   BUY outliers: {buy_outliers} ({buy_outliers/len(df)*100:.2f}%)")
            logger.warning(f"   SELL outliers: {sell_outliers} ({sell_outliers/len(df)*100:.2f}%)")
    
    def split_data_temporal(self, 
                          df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Временное разделение данных с учетом gap для предотвращения утечки
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            Словарь с разделенными данными
        """
        logger.info("📊 Временное разделение данных...")
        
        # Параметры из конфигурации
        train_ratio = VALIDATION_PARAMS['train_ratio']
        val_ratio = VALIDATION_PARAMS['val_ratio']
        test_ratio = VALIDATION_PARAMS['test_ratio']
        gap = VALIDATION_PARAMS['gap_periods']
        
        # Группируем по символам
        grouped_splits = {
            "train": [],
            "val": [],
            "test": []
        }
        
        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
            n = len(symbol_df)
            
            if n < 1000:  # Минимум данных для символа
                logger.warning(f"⚠️ Символ {symbol} имеет только {n} записей, пропускаем")
                continue
            
            # Временное разделение
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            # Разделяем с учетом gap
            splits = {
                "train": symbol_df[:train_end - gap],
                "val": symbol_df[train_end + gap:val_end - gap],
                "test": symbol_df[val_end + gap:]
            }
            
            # Добавляем в общие списки
            for split_name, split_df in splits.items():
                if len(split_df) > 0:
                    grouped_splits[split_name].append(split_df)
        
        # Объединяем все splits
        final_splits = {}
        for split_name, dfs in grouped_splits.items():
            if dfs:
                final_splits[split_name] = pd.concat(dfs, ignore_index=True)
                logger.info(f"   {split_name}: {len(final_splits[split_name]):,} записей")
            else:
                logger.warning(f"⚠️ Нет данных для {split_name}")
                final_splits[split_name] = pd.DataFrame()
        
        return final_splits
    
    def normalize_features(self, 
                         train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Нормализация признаков
        
        Args:
            train_df: Обучающие данные
            val_df: Валидационные данные
            test_df: Тестовые данные
            
        Returns:
            Нормализованные данные
        """
        logger.info("📏 Нормализация признаков...")
        
        # Fit scaler только на train данных
        train_features = train_df[self.feature_columns]
        self.scaler.fit(train_features)
        
        # Transform все данные
        train_normalized = train_df.copy()
        val_normalized = val_df.copy()
        test_normalized = test_df.copy()
        
        train_normalized[self.feature_columns] = self.scaler.transform(train_features)
        
        if len(val_df) > 0:
            val_normalized[self.feature_columns] = self.scaler.transform(val_df[self.feature_columns])
            
        if len(test_df) > 0:
            test_normalized[self.feature_columns] = self.scaler.transform(test_df[self.feature_columns])
        
        # Проверка статистики expected returns
        for col in ['buy_expected_return', 'sell_expected_return']:
            if col in train_normalized.columns:
                logger.info(f"📊 {col} статистика: mean={train_normalized[col].mean():.4f}, "
                           f"std={train_normalized[col].std():.4f}, "
                           f"min={train_normalized[col].min():.4f}, "
                           f"max={train_normalized[col].max():.4f}")
                
                # Масштабируем expected returns если они слишком большие
                if train_normalized[col].abs().max() > 10:
                    logger.warning(f"⚠️ {col} имеет большие значения, масштабируем на 100")
                    train_normalized[col] = train_normalized[col] / 100
                    if len(val_df) > 0:
                        val_normalized[col] = val_normalized[col] / 100
                    if len(test_df) > 0:
                        test_normalized[col] = test_normalized[col] / 100
        
        logger.info("✅ Нормализация завершена")
        
        return train_normalized, val_normalized, test_normalized
    
    def convert_to_binary_labels(self, returns: pd.Series, threshold: float = 0.3) -> pd.Series:
        """
        Преобразование expected returns в бинарные метки
        
        Args:
            returns: Series с expected returns
            threshold: Порог для классификации (%)
            
        Returns:
            Бинарные метки
        """
        return (returns > threshold).astype(np.float32)
    
    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Получение статистики по признакам"""
        stats = []
        
        for col in self.feature_columns:
            if col in df.columns:
                col_stats = {
                    'feature': col,
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'missing': df[col].isna().sum(),
                    'zeros': (df[col] == 0).sum()
                }
                stats.append(col_stats)
        
        return pd.DataFrame(stats)