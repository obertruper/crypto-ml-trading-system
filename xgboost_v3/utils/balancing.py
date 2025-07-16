"""
Стратегии балансировки классов
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight

from config import Config, FEATURE_CONFIG

logger = logging.getLogger(__name__)


class BalanceStrategy:
    """Класс для балансировки несбалансированных классов"""
    
    def __init__(self, config: Config):
        self.config = config
        self.binary_features = FEATURE_CONFIG['binary_thresholds'].keys()
        
    def balance_data(self, X: pd.DataFrame, y: pd.Series, 
                    model_name: str = "") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Балансировка данных согласно выбранной стратегии
        
        Args:
            X: Признаки
            y: Метки
            model_name: Название модели для логирования
            
        Returns:
            Сбалансированные X, y
        """
        logger.info(f"\n🔄 Балансировка классов с помощью {self.config.training.balance_method.upper()}...")
        
        # Статистика до балансировки
        self._log_class_distribution(y, "До балансировки")
        
        if self.config.training.balance_method == "none":
            return X, y
        elif self.config.training.balance_method == "smote":
            X_balanced, y_balanced = self._apply_smote(X, y)
        elif self.config.training.balance_method == "adasyn":
            X_balanced, y_balanced = self._apply_adasyn(X, y)
        elif self.config.training.balance_method == "random_oversample":
            X_balanced, y_balanced = self._apply_random_oversampler(X, y)
        elif self.config.training.balance_method == "smote_tomek":
            X_balanced, y_balanced = self._apply_smote_tomek(X, y)
        else:
            logger.warning(f"Неизвестный метод балансировки: {self.config.training.balance_method}")
            return X, y
            
        # Статистика после балансировки
        self._log_class_distribution(y_balanced, "После балансировки")
        
        # Исправляем бинарные признаки после SMOTE
        if self.config.training.balance_method in ["smote", "adasyn", "smote_tomek"]:
            X_balanced = self._fix_binary_features_after_smote(X_balanced, model_name)
            
        return X_balanced, y_balanced
        
    def _apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Применение SMOTE для балансировки"""
        logger.info("🔄 Применение SMOTE для балансировки классов...")
        
        # Разделяем признаки для правильной работы SMOTE
        continuous_features, binary_features = self._separate_features(X)
        
        # Применяем SMOTE только к непрерывным признакам
        smote = SMOTE(
            sampling_strategy='auto',
            k_neighbors=min(self.config.training.smote_k_neighbors, (y == 1).sum() - 1),
            random_state=42
        )
        
        try:
            if continuous_features:
                X_continuous = X[continuous_features]
                X_continuous_resampled, y_resampled = smote.fit_resample(X_continuous, y)
                
                # Восстанавливаем бинарные признаки
                # Для новых синтетических примеров используем случайные значения из существующих
                n_synthetic = len(y_resampled) - len(y)
                
                if binary_features and n_synthetic > 0:
                    # Берем случайные индексы из оригинальных данных класса 1
                    minority_indices = y[y == 1].index
                    random_indices = np.random.choice(minority_indices, n_synthetic, replace=True)
                    
                    # Создаем DataFrame для всех данных
                    X_resampled = pd.DataFrame(X_continuous_resampled, columns=continuous_features)
                    
                    # Добавляем бинарные признаки
                    for feature in binary_features:
                        original_values = X.loc[y.index, feature].values
                        synthetic_values = X.loc[random_indices, feature].values
                        X_resampled[feature] = np.concatenate([original_values, synthetic_values])
                else:
                    X_resampled = pd.DataFrame(X_continuous_resampled, columns=continuous_features)
            else:
                # Если нет непрерывных признаков, используем RandomOverSampler
                logger.warning("Нет непрерывных признаков для SMOTE, используем RandomOverSampler")
                return self._apply_random_oversampler(X, y)
                
        except Exception as e:
            logger.error(f"Ошибка при применении SMOTE: {e}")
            logger.warning("Возвращаем исходные данные")
            return X, y
            
        logger.info(f"   Добавлено синтетических примеров: {len(y_resampled) - len(y)}")
        
        return X_resampled, y_resampled
        
    def _apply_adasyn(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Применение ADASYN для балансировки"""
        logger.info("🔄 Применение ADASYN для адаптивной балансировки...")
        
        try:
            adasyn = ADASYN(
                sampling_strategy='auto',
                n_neighbors=min(5, (y == 1).sum() - 1),
                random_state=42
            )
            
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            
        except Exception as e:
            logger.error(f"Ошибка при применении ADASYN: {e}")
            return self._apply_smote(X, y)
            
        return X_resampled, y_resampled
        
    def _apply_random_oversampler(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Применение случайной передискретизации"""
        logger.info("🔄 Применение Random Oversampling...")
        
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        
        return X_resampled, y_resampled
        
    def _apply_smote_tomek(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Применение комбинации SMOTE и Tomek links"""
        logger.info("🔄 Применение SMOTE + Tomek links...")
        
        try:
            smt = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smt.fit_resample(X, y)
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            
        except Exception as e:
            logger.error(f"Ошибка при применении SMOTE-Tomek: {e}")
            return self._apply_smote(X, y)
            
        return X_resampled, y_resampled
        
    def _separate_features(self, X: pd.DataFrame) -> Tuple[list, list]:
        """Разделение признаков на непрерывные и бинарные"""
        binary_features = []
        continuous_features = []
        
        for col in X.columns:
            # Проверяем, является ли признак бинарным
            unique_values = X[col].unique()
            if len(unique_values) <= 2 and set(unique_values).issubset({0, 1}):
                binary_features.append(col)
            else:
                continuous_features.append(col)
                
        return continuous_features, binary_features
        
    def _fix_binary_features_after_smote(self, X: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Исправление бинарных признаков после SMOTE"""
        # Ограничиваем значения технических индикаторов
        X = self._clip_technical_indicators(X)
        
        # Пересоздаем бинарные признаки
        X = self._recreate_binary_features(X, model_name)
        
        return X
        
    def _clip_technical_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ограничение значений технических индикаторов"""
        logger.info("✂️ Ограничение значений технических индикаторов...")
        
        from config.features_config import TECHNICAL_INDICATORS_BOUNDS
        
        for indicator, (min_val, max_val) in TECHNICAL_INDICATORS_BOUNDS.items():
            if indicator in X.columns:
                original_outliers = ((X[indicator] < min_val) | (X[indicator] > max_val)).sum()
                if original_outliers > 0:
                    X[indicator] = X[indicator].clip(min_val, max_val)
                    logger.info(f"   📊 {indicator}: ограничено {original_outliers} значений в диапазон [{min_val}, {max_val}]")
                    
        logger.info("   ✂️ Ограничены значения для индикаторов после SMOTE")
        
        return X
        
    def _recreate_binary_features(self, X: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Пересоздание бинарных признаков на основе актуальных значений"""
        logger.info("🔄 Пересоздание бинарных признаков после SMOTE...")
        
        thresholds = FEATURE_CONFIG['binary_thresholds']
        recreated_features = []
        
        # RSI условия
        if 'rsi_val' in X.columns and 'rsi_oversold' in X.columns:
            X['rsi_oversold'] = (X['rsi_val'] < thresholds['rsi_oversold']).astype(int)
            recreated_features.append('rsi_oversold')
            
        if 'rsi_val' in X.columns and 'rsi_overbought' in X.columns:
            X['rsi_overbought'] = (X['rsi_val'] > thresholds['rsi_overbought']).astype(int)
            recreated_features.append('rsi_overbought')
            
        # MACD условие
        if 'macd_hist' in X.columns and 'macd_bullish' in X.columns:
            X['macd_bullish'] = (X['macd_hist'] > thresholds['macd_bullish']).astype(int)
            recreated_features.append('macd_bullish')
            
        # Сильный тренд
        if 'adx_val' in X.columns and 'strong_trend' in X.columns:
            X['strong_trend'] = (X['adx_val'] > thresholds['strong_trend']).astype(int)
            recreated_features.append('strong_trend')
            
        # Всплеск объема
        if 'volume_ratio' in X.columns and 'volume_spike' in X.columns:
            X['volume_spike'] = (X['volume_ratio'] > thresholds['volume_spike']).astype(int)
            recreated_features.append('volume_spike')
            
        # Позиция в Bollinger Bands
        if 'bb_position' in X.columns:
            if 'bb_near_lower' in X.columns:
                X['bb_near_lower'] = (X['bb_position'] < thresholds['bb_near_lower']).astype(int)
                recreated_features.append('bb_near_lower')
                
            if 'bb_near_upper' in X.columns:
                X['bb_near_upper'] = (X['bb_position'] > thresholds['bb_near_upper']).astype(int)
                recreated_features.append('bb_near_upper')
                
        if recreated_features:
            logger.info(f"   ♻️ Пересозданы бинарные признаки после SMOTE для {model_name}: {recreated_features}")
            
            # Проверка статистики
            stats = []
            for feature in recreated_features[:4]:  # Первые 4 для примера
                if feature in X.columns:
                    pct = X[feature].mean() * 100
                    stats.append(f"{feature}: {pct:.1f}%")
                    
            if stats:
                logger.info(f"   📊 Статистика бинарных признаков после SMOTE: {', '.join(stats)}")
                
        # Валидация бинарных признаков
        self._validate_binary_features(X)
        
        return X
        
    def _validate_binary_features(self, X: pd.DataFrame):
        """Проверка корректности бинарных признаков"""
        problems = []
        
        for col in X.columns:
            if col in self.binary_features or col.startswith(('is_', 'has_')):
                unique_vals = X[col].unique()
                
                # Проверка на константность
                if len(unique_vals) == 1:
                    problems.append(f"{col}: константный признак (всегда {unique_vals[0]})")
                    
                # Проверка на бинарность
                elif not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    problems.append(f"{col}: небинарные значения {unique_vals}")
                    
        if problems:
            logger.warning("⚠️ Обнаружены проблемы с бинарными признаками:")
            for problem in problems:
                logger.warning(f"   - {problem}")
                
    def _log_class_distribution(self, y: pd.Series, stage: str):
        """Логирование распределения классов"""
        class_counts = y.value_counts().sort_index()
        total = len(y)
        
        logger.info(f"   {stage}:")
        for class_val, count in class_counts.items():
            percentage = count / total * 100
            logger.info(f"      Класс {class_val}: {count:,} ({percentage:.1f}%)")
            
    def calculate_class_weights(self, y: pd.Series) -> dict:
        """Расчет весов классов для взвешенной функции потерь"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        class_weight_dict = dict(zip(classes, weights))
        
        logger.info("📊 Веса классов:")
        for class_val, weight in class_weight_dict.items():
            logger.info(f"   Класс {class_val}: {weight:.3f}")
            
        return class_weight_dict