"""
Улучшенный балансировщик данных для XGBoost v3.0
"""

import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
import logging
from typing import Tuple, Optional, Dict

from config import Config

logger = logging.getLogger(__name__)


class DataBalancer:
    """Класс для балансировки несбалансированных данных"""
    
    def __init__(self, config: Config):
        self.config = config
        self.method = config.training.balance_method
        
    def balance_data(self, X: pd.DataFrame, y: pd.Series, 
                    is_classification: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Основной метод балансировки данных
        """
        if not is_classification or self.method == "none":
            return X, y
            
        logger.info(f"🔄 Балансировка данных методом: {self.method}")
        
        # Убеждаемся, что индексы синхронизированы
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Статистика до балансировки
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"   До балансировки: {class_dist}")
        
        # Проверяем минимальное количество примеров
        min_class_count = min(counts)
        if min_class_count < 6:
            logger.warning(f"⚠️ Слишком мало примеров в меньшем классе ({min_class_count}), пропускаем балансировку")
            return X, y
        
        # Выбор метода балансировки
        if self.method == "smote":
            X_balanced, y_balanced = self._apply_smote(X, y)
        elif self.method == "adasyn":
            X_balanced, y_balanced = self._apply_adasyn(X, y)
        elif self.method == "class_weight":
            # Для class_weight просто возвращаем веса, не меняя данные
            weights = self._calculate_class_weights(y)
            logger.info(f"   Веса классов: {weights}")
            return X, y
        else:
            raise ValueError(f"Неизвестный метод балансировки: {self.method}")
            
        # Статистика после балансировки
        unique, counts = np.unique(y_balanced, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"   После балансировки: {class_dist}")
            
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
        
    def _apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Применение SMOTE для балансировки"""
        try:
            # Проверяем минимальное количество сэмплов в меньшем классе
            unique, counts = np.unique(y, return_counts=True)
            min_samples = min(counts)
            
            # Адаптируем k_neighbors под размер данных
            k_neighbors = min(self.config.training.smote_k_neighbors, min_samples - 1)
            
            if k_neighbors < 1:
                logger.warning("⚠️ Недостаточно данных для SMOTE, используем RandomOverSampler")
                ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
                return ros.fit_resample(X, y)
                
            # Идентифицируем бинарные признаки
            binary_cols = self._identify_binary_columns(X)
            continuous_cols = [col for col in X.columns if col not in binary_cols]
            
            if len(continuous_cols) < 2:
                logger.warning("⚠️ Недостаточно непрерывных признаков для SMOTE, используем RandomOverSampler")
                ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
                return ros.fit_resample(X, y)
            
            # Применяем SMOTE только к непрерывным признакам
            smote = SMOTE(
                sampling_strategy='auto',
                k_neighbors=k_neighbors,
                random_state=42
            )
            
            if binary_cols:
                # Применяем SMOTE только к непрерывным признакам
                X_continuous = X[continuous_cols]
                X_cont_resampled, y_resampled = smote.fit_resample(X_continuous, y)
                
                # Количество новых синтетических примеров
                n_synthetic = len(y_resampled) - len(y)
                
                if n_synthetic > 0:
                    # Для бинарных признаков используем случайную выборку из minority класса
                    minority_mask = y == 1
                    minority_binary = X.loc[minority_mask, binary_cols]
                    
                    # Случайно выбираем значения бинарных признаков из minority класса
                    synthetic_indices = np.random.choice(
                        len(minority_binary), 
                        size=n_synthetic, 
                        replace=True
                    )
                    synthetic_binary = minority_binary.iloc[synthetic_indices].values
                    
                    # Объединяем результаты
                    X_resampled = np.column_stack([
                        X_cont_resampled,
                        np.vstack([X[binary_cols].values, synthetic_binary])
                    ])
                    
                    # Восстанавливаем порядок колонок
                    col_order = continuous_cols + binary_cols
                    X_resampled_df = pd.DataFrame(X_resampled, columns=col_order)
                    X_resampled = X_resampled_df[X.columns].values
                else:
                    X_resampled = X_cont_resampled
            else:
                # Если нет бинарных признаков, применяем SMOTE ко всем
                X_resampled, y_resampled = smote.fit_resample(X, y)
            
            logger.info(f"   ✅ SMOTE выполнен успешно (k_neighbors={k_neighbors})")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"❌ Ошибка при применении SMOTE: {e}")
            logger.warning("   Используем RandomOverSampler как fallback")
            ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
            return ros.fit_resample(X, y)
            
    def _apply_adasyn(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Применение ADASYN для балансировки с оптимизацией для серверов"""
        try:
            # Проверяем минимальное количество сэмплов
            unique, counts = np.unique(y, return_counts=True)
            min_samples = min(counts)
            
            # Определяем оптимальные параметры для сервера
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            
            if cpu_count > 64:
                # Для мощных серверов используем больше соседей и параллелизм
                n_neighbors = min(15, min_samples - 1, len(y) // 1000)
                n_jobs = min(32, cpu_count // 4)  # Используем 1/4 ядер
                logger.info(f"   🚀 Мощный сервер: n_neighbors={n_neighbors}, n_jobs={n_jobs}")
            else:
                # Адаптируем n_neighbors для обычных систем
                n_neighbors = min(5, min_samples - 1)
                n_jobs = 1
            
            if n_neighbors < 1:
                logger.warning("⚠️ Недостаточно данных для ADASYN, используем RandomOverSampler")
                ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
                return ros.fit_resample(X, y)
                
            adasyn = ADASYN(
                sampling_strategy='auto',
                n_neighbors=n_neighbors,
                random_state=42,
                n_jobs=n_jobs
            )
            
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            logger.info(f"   ✅ ADASYN выполнен успешно (n_neighbors={n_neighbors})")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"❌ Ошибка при применении ADASYN: {e}")
            logger.warning("   Используем RandomOverSampler как fallback")
            ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
            return ros.fit_resample(X, y)
            
    def _identify_binary_columns(self, X: pd.DataFrame) -> list:
        """Идентификация бинарных колонок"""
        binary_cols = []
        
        for col in X.columns:
            unique_vals = X[col].dropna().unique()
            # Считаем бинарными только колонки со значениями 0 и 1
            if len(unique_vals) <= 2 and all(val in [0, 1, 0.0, 1.0] for val in unique_vals):
                binary_cols.append(col)
            # Также проверяем колонки с паттернами бинарных имен
            elif any(pattern in col.lower() for pattern in ['is_', '_oversold', '_overbought', 
                                                            'bullish', 'bearish', 'spike',
                                                            'strong_trend', 'bb_near_']):
                # Исправляем значения
                X[col] = (X[col] != 0).astype(int)
                binary_cols.append(col)
                
        if binary_cols:
            logger.info(f"   📊 Обнаружено {len(binary_cols)} бинарных признаков")
            
        return binary_cols
            
    def _calculate_class_weights(self, y: pd.Series) -> dict:
        """Расчет весов классов для балансировки"""
        weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        
        return dict(zip(np.unique(y), weights))
        
    def get_scale_pos_weight(self, y: pd.Series) -> Optional[float]:
        """
        Расчет scale_pos_weight для XGBoost
        
        Используется для бинарной классификации
        """
        if len(np.unique(y)) != 2:
            return None
            
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        
        if pos_count == 0:
            return None
            
        scale_pos_weight = neg_count / pos_count
        
        logger.info(f"   scale_pos_weight = {scale_pos_weight:.2f} (neg: {neg_count}, pos: {pos_count})")
        
        return scale_pos_weight