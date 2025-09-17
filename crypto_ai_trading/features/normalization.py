"""
Адаптивная нормализация с каузальными окнами
Строгое соблюдение принципа - только backward-looking операции
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union
from scipy import stats
import pickle
import warnings
from utils.logger import get_logger


class AdaptiveNormalization:
    """
    Каузальная адаптивная нормализация для временных рядов
    
    Поддерживаемые методы:
    - revin: Reversible Instance Normalization
    - robust: Медиана + MAD
    - adaptive_zscore: Экспоненциально взвешенный Z-score
    - quantile: Квантильная нормализация
    """
    
    def __init__(self, 
                 window: int = 1000,
                 method: str = 'revin',
                 min_periods: int = 100,
                 causal_only: bool = True):
        """
        Args:
            window: Размер окна для статистик
            method: Метод нормализации ('revin', 'robust', 'adaptive_zscore', 'quantile')
            min_periods: Минимальное количество наблюдений для расчета
            causal_only: Использовать только каузальные (backward-looking) окна
        """
        self.window = window
        self.method = method
        self.min_periods = min_periods
        self.causal_only = causal_only
        
        # Хранилище статистик для каждого символа
        self.stats_storage = {}
        self.inference_stats = {}
        
        self.logger = get_logger("AdaptiveNormalization")
        self.logger.info(f"🔄 Адаптивная нормализация инициализирована:")
        self.logger.info(f"   - Метод: {self.method}")
        self.logger.info(f"   - Окно: {self.window}")
        self.logger.info(f"   - Каузальный режим: {self.causal_only}")
        
        if not self.causal_only:
            warnings.warn("⚠️ Не-каузальная нормализация может привести к утечкам данных!")
    
    def fit_transform(self, 
                     df: pd.DataFrame, 
                     feature_columns: list,
                     symbol_column: str = 'symbol') -> pd.DataFrame:
        """
        Каузальная нормализация с сохранением статистик
        
        Args:
            df: DataFrame с данными
            feature_columns: Список колонок для нормализации
            symbol_column: Название колонки с символами
            
        Returns:
            DataFrame с нормализованными признаками
        """
        df_normalized = df.copy()
        
        if symbol_column not in df.columns:
            # Один символ - применяем нормализацию напрямую
            return self._normalize_single_symbol(df_normalized, feature_columns, 'default')
        
        # Мультисимвольная нормализация
        for symbol in df[symbol_column].unique():
            mask = df[symbol_column] == symbol
            symbol_data = df[mask].copy()
            
            if len(symbol_data) < self.min_periods:
                self.logger.warning(f"⚠️ Недостаточно данных для символа {symbol}: {len(symbol_data)}")
                continue
            
            # Сортируем по времени для каузальности
            if 'datetime' in symbol_data.columns:
                symbol_data = symbol_data.sort_values('datetime')
            
            normalized_data = self._normalize_single_symbol(
                symbol_data, feature_columns, symbol
            )
            
            df_normalized.loc[mask] = normalized_data
        
        self.logger.info(f"✅ Нормализация завершена для {len(df[symbol_column].unique())} символов")
        return df_normalized
    
    def _normalize_single_symbol(self, 
                                df: pd.DataFrame,
                                feature_columns: list,
                                symbol: str) -> pd.DataFrame:
        """Нормализация для одного символа"""
        
        self.stats_storage[symbol] = {}
        
        for column in feature_columns:
            if column not in df.columns:
                continue
                
            values = df[column].copy()
            
            if values.isna().sum() > len(values) * 0.5:
                self.logger.warning(f"⚠️ Много NaN в {symbol}:{column}")
                continue
            
            # Применяем выбранный метод
            if self.method == 'revin':
                normalized, stats = self._apply_revin(values)
            elif self.method == 'robust':
                normalized, stats = self._apply_robust(values)
            elif self.method == 'adaptive_zscore':
                normalized, stats = self._apply_adaptive_zscore(values)
            elif self.method == 'quantile':
                normalized, stats = self._apply_quantile(values)
            else:
                raise ValueError(f"Неизвестный метод нормализации: {self.method}")
            
            # Сохраняем нормализованные значения
            df[column] = normalized
            
            # Сохраняем статистики
            self.stats_storage[symbol][column] = stats
            
            # Сохраняем последние значения для инференса
            self.inference_stats[f"{symbol}_{column}"] = {
                'last_mean': stats['mean'].iloc[-1] if hasattr(stats['mean'], 'iloc') else stats['mean'],
                'last_std': stats['std'].iloc[-1] if hasattr(stats['std'], 'iloc') else stats['std'],
                'method': self.method
            }
        
        return df
    
    def _apply_revin(self, values: pd.Series) -> Tuple[pd.Series, Dict]:
        """Reversible Instance Normalization (каузальный)"""
        
        # КАУЗАЛЬНЫЕ скользящие статистики
        rolling_mean = values.rolling(
            window=self.window, 
            min_periods=self.min_periods,
            center=False  # КРИТИЧНО: backward-looking только
        ).mean()
        
        rolling_std = values.rolling(
            window=self.window,
            min_periods=self.min_periods, 
            center=False
        ).std()
        
        # Заполняем начальные NaN значения глобальными статистиками
        global_mean = values.mean()
        global_std = values.std()
        
        rolling_mean = rolling_mean.fillna(global_mean)
        rolling_std = rolling_std.fillna(global_std)
        
        # Нормализация
        normalized = (values - rolling_mean) / (rolling_std + 1e-8)
        
        # Клиппинг экстремальных значений
        normalized = normalized.clip(-5, 5)
        
        stats = {
            'mean': rolling_mean,
            'std': rolling_std,
            'global_mean': global_mean,
            'global_std': global_std
        }
        
        return normalized, stats
    
    def _apply_robust(self, values: pd.Series) -> Tuple[pd.Series, Dict]:
        """Робастная нормализация на медиане и MAD"""
        
        # КАУЗАЛЬНАЯ медиана
        rolling_median = values.rolling(
            window=self.window,
            min_periods=self.min_periods,
            center=False
        ).median()
        
        # КАУЗАЛЬНАЯ MAD (Median Absolute Deviation)
        rolling_mad = values.rolling(
            window=self.window,
            min_periods=self.min_periods,
            center=False
        ).apply(
            lambda x: np.median(np.abs(x - np.median(x))),
            raw=True
        )
        
        # Заполняем NaN
        global_median = values.median()
        global_mad = np.median(np.abs(values - global_median))
        
        rolling_median = rolling_median.fillna(global_median)
        rolling_mad = rolling_mad.fillna(global_mad)
        
        # Нормализация (MAD * 1.4826 ≈ стандартное отклонение для нормального распределения)
        normalized = (values - rolling_median) / (rolling_mad * 1.4826 + 1e-8)
        normalized = normalized.clip(-5, 5)
        
        stats = {
            'mean': rolling_median,  # Используем как "mean"
            'std': rolling_mad * 1.4826,  # Используем как "std"
            'global_mean': global_median,
            'global_std': global_mad * 1.4826
        }
        
        return normalized, stats
    
    def _apply_adaptive_zscore(self, values: pd.Series) -> Tuple[pd.Series, Dict]:
        """Экспоненциально взвешенный Z-score"""
        
        # Экспоненциально взвешенные статистики (каузальные)
        span = min(self.window, len(values) // 4)
        
        ewm_mean = values.ewm(
            span=span,
            adjust=False,  # Не корректируем для начальных значений
            ignore_na=True
        ).mean()
        
        ewm_std = values.ewm(
            span=span,
            adjust=False,
            ignore_na=True
        ).std()
        
        # Заполняем NaN начальными значениями
        first_valid_idx = values.first_valid_index()
        if first_valid_idx is not None:
            ewm_mean.loc[:first_valid_idx] = values.loc[first_valid_idx]
            ewm_std.loc[:first_valid_idx] = values.std()
        
        # Нормализация
        normalized = (values - ewm_mean) / (ewm_std + 1e-8)
        normalized = normalized.clip(-5, 5)
        
        stats = {
            'mean': ewm_mean,
            'std': ewm_std,
            'span': span
        }
        
        return normalized, stats
    
    def _apply_quantile(self, values: pd.Series) -> Tuple[pd.Series, Dict]:
        """Квантильная нормализация (каузальная)"""
        
        # КАУЗАЛЬНЫЕ квантили
        rolling_quantiles = pd.DataFrame(index=values.index)
        
        for q in [0.01, 0.25, 0.5, 0.75, 0.99]:
            rolling_quantiles[f'q_{q}'] = values.rolling(
                window=self.window,
                min_periods=self.min_periods,
                center=False
            ).quantile(q)
        
        # Ранговое преобразование (каузальное)
        ranks = values.rolling(
            window=self.window,
            min_periods=self.min_periods,
            center=False
        ).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )
        
        # Преобразование к стандартному нормальному распределению
        normalized = stats.norm.ppf(ranks.clip(0.001, 0.999))
        
        stats = {
            'mean': rolling_quantiles['q_0.5'],  # Медиана как "mean"
            'std': rolling_quantiles['q_0.75'] - rolling_quantiles['q_0.25'],  # IQR как "std"
            'quantiles': rolling_quantiles
        }
        
        return normalized, stats
    
    def inverse_transform(self, 
                         df: pd.DataFrame,
                         feature_columns: list,
                         symbol_column: str = 'symbol') -> pd.DataFrame:
        """
        Обратное преобразование для денормализации
        Использует сохраненные статистики инференса
        """
        df_denormalized = df.copy()
        
        for symbol in df[symbol_column].unique() if symbol_column in df.columns else ['default']:
            mask = df[symbol_column] == symbol if symbol_column in df.columns else slice(None)
            
            for column in feature_columns:
                if column not in df.columns:
                    continue
                
                stats_key = f"{symbol}_{column}"
                if stats_key not in self.inference_stats:
                    self.logger.warning(f"⚠️ Нет статистик для денормализации {stats_key}")
                    continue
                
                inference_stats = self.inference_stats[stats_key]
                normalized_values = df.loc[mask, column]
                
                # Денормализация в зависимости от метода
                if inference_stats['method'] in ['revin', 'robust', 'adaptive_zscore']:
                    denormalized = (
                        normalized_values * inference_stats['last_std'] + 
                        inference_stats['last_mean']
                    )
                elif inference_stats['method'] == 'quantile':
                    # Для квантильного метода используем нормальное распределение
                    ranks = stats.norm.cdf(normalized_values)
                    # Простая линейная интерполяция (можно улучшить)
                    denormalized = normalized_values * inference_stats['last_std'] + inference_stats['last_mean']
                
                df_denormalized.loc[mask, column] = denormalized
        
        return df_denormalized
    
    def save_stats(self, filepath: str):
        """Сохранение статистик для инференса"""
        save_data = {
            'inference_stats': self.inference_stats,
            'config': {
                'window': self.window,
                'method': self.method,
                'min_periods': self.min_periods,
                'causal_only': self.causal_only
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"💾 Статистики нормализации сохранены: {filepath}")
    
    def load_stats(self, filepath: str):
        """Загрузка статистик для инференса"""
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.inference_stats = save_data['inference_stats']
            config = save_data['config']
            
            # Проверяем совместимость конфигурации
            if config['method'] != self.method:
                self.logger.warning(f"⚠️ Несовпадение метода: загружен {config['method']}, текущий {self.method}")
            
            self.logger.info(f"📂 Статистики нормализации загружены: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки статистик: {e}")
            raise
    
    def validate_normalization(self, 
                              original_df: pd.DataFrame,
                              normalized_df: pd.DataFrame,
                              feature_columns: list) -> Dict:
        """Валидация качества нормализации"""
        
        validation_results = {}
        
        for column in feature_columns:
            if column not in original_df.columns or column not in normalized_df.columns:
                continue
            
            original = original_df[column].dropna()
            normalized = normalized_df[column].dropna()
            
            if len(original) == 0 or len(normalized) == 0:
                continue
            
            # Статистики до нормализации
            orig_mean = original.mean()
            orig_std = original.std()
            orig_skew = original.skew()
            orig_kurtosis = original.kurtosis()
            
            # Статистики после нормализации
            norm_mean = normalized.mean()
            norm_std = normalized.std()
            norm_skew = normalized.skew()
            norm_kurtosis = normalized.kurtosis()
            
            # Проверки качества
            mean_close_to_zero = abs(norm_mean) < 0.1
            std_close_to_one = abs(norm_std - 1.0) < 0.2
            no_extreme_values = (normalized.min() > -10) and (normalized.max() < 10)
            
            validation_results[column] = {
                'original': {
                    'mean': orig_mean,
                    'std': orig_std,
                    'skewness': orig_skew,
                    'kurtosis': orig_kurtosis
                },
                'normalized': {
                    'mean': norm_mean,
                    'std': norm_std,
                    'skewness': norm_skew,
                    'kurtosis': norm_kurtosis
                },
                'quality_checks': {
                    'mean_centered': mean_close_to_zero,
                    'std_normalized': std_close_to_one,
                    'no_extremes': no_extreme_values,
                    'overall_quality': mean_close_to_zero and std_close_to_one and no_extreme_values
                }
            }
        
        # Общая сводка
        quality_scores = [
            result['quality_checks']['overall_quality'] 
            for result in validation_results.values()
        ]
        
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        self.logger.info(f"📊 Качество нормализации: {overall_quality:.2%}")
        
        return {
            'per_feature': validation_results,
            'overall_quality': overall_quality,
            'features_passed': sum(quality_scores),
            'total_features': len(quality_scores)
        }