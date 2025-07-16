"""
Препроцессинг данных для моделей
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger


class DataPreprocessor:
    """Класс для препроцессинга временных рядов"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: конфигурация препроцессинга
        """
        self.config = config
        self.logger = get_logger("DataPreprocessor")
        
        # Параметры
        self.scaling_method = config.get('preprocessing', {}).get('scaling_method', 'robust')
        self.handle_missing = config.get('preprocessing', {}).get('handle_missing', 'interpolate')
        self.remove_outliers = config.get('preprocessing', {}).get('remove_outliers', True)
        self.outlier_threshold = config.get('preprocessing', {}).get('outlier_threshold', 3)
        
        # Скейлеры для разных типов признаков
        self.scalers = {}
        self.feature_groups = {}
        
        # Статистика
        self.preprocessing_stats = {}
        
    def fit(self, data: pd.DataFrame, exclude_targets: bool = True) -> 'DataPreprocessor':
        """
        Обучение препроцессора на данных (ТОЛЬКО на обучающей выборке!)
        
        Args:
            data: DataFrame с данными для обучения (ТОЛЬКО train!)
            exclude_targets: исключить целевые переменные из нормализации
            
        Returns:
            self
        """
        self.logger.info("Обучение препроцессора на TRAIN данных...")
        
        # ВАЖНО: Проверяем что это train данные
        if hasattr(data, 'attrs') and data.attrs.get('dataset_type') != 'train':
            self.logger.warning("⚠️ ВНИМАНИЕ: fit() должен вызываться ТОЛЬКО на train данных!")
        
        # Определение групп признаков
        self._identify_feature_groups(data, exclude_targets=exclude_targets)
        
        # Обучение скейлеров для каждой группы
        for group_name, features in self.feature_groups.items():
            if features:
                group_data = data[features].select_dtypes(include=[np.number])
                
                if len(group_data.columns) > 0:
                    # Выбор скейлера в зависимости от типа данных
                    if group_name == 'price':
                        scaler = RobustScaler()  # Устойчив к выбросам
                    elif group_name == 'volume':
                        scaler = MinMaxScaler()  # Нормализация в [0, 1]
                    elif group_name == 'returns':
                        scaler = StandardScaler()  # Стандартизация
                    elif group_name == 'indicators':
                        # Для индикаторов с известными диапазонами не нормализуем
                        # RSI, Stochastic и т.д. уже в диапазоне [0, 100]
                        continue
                    elif group_name == 'fixed_range':
                        # Не нормализуем индикаторы с фиксированными диапазонами
                        # RSI [0, 100], Stochastic [0, 100], ADX [0, 100], 
                        # CCI [-200, 200], Williams %R [-100, 0], signal_strength [0, 1]
                        continue
                    else:
                        scaler = self._create_scaler()
                    
                    # Обучение скейлера ТОЛЬКО на train данных
                    clean_data = self._handle_missing_values(group_data)
                    scaler.fit(clean_data)
                    self.scalers[group_name] = scaler
                    
                    # Сохранение статистики
                    self.preprocessing_stats[group_name] = {
                        'mean': clean_data.mean().to_dict(),
                        'std': clean_data.std().to_dict(),
                        'min': clean_data.min().to_dict(),
                        'max': clean_data.max().to_dict()
                    }
        
        self.logger.info(f"Препроцессор обучен на {len(self.feature_groups)} группах признаков")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Трансформация данных
        
        Args:
            data: DataFrame для трансформации
            
        Returns:
            Трансформированный DataFrame
        """
        self.logger.info(f"Трансформация {len(data)} записей...")
        
        data_transformed = data.copy()
        
        # Обработка пропущенных значений
        data_transformed = self._handle_missing_values(data_transformed)
        
        # Удаление выбросов если нужно
        if self.remove_outliers:
            data_transformed = self._remove_outliers(data_transformed)
        
        # Применение скейлеров по группам
        for group_name, features in self.feature_groups.items():
            if group_name in self.scalers and features:
                valid_features = [f for f in features if f in data_transformed.columns]
                if valid_features:
                    numeric_features = data_transformed[valid_features].select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_features:
                        data_transformed[numeric_features] = self.scalers[group_name].transform(
                            data_transformed[numeric_features]
                        )
        
        # Дополнительная обработка
        data_transformed = self._additional_preprocessing(data_transformed)
        
        return data_transformed
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обучение и трансформация одновременно
        
        Args:
            data: DataFrame для обучения и трансформации
            
        Returns:
            Трансформированный DataFrame
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: pd.DataFrame, feature_group: str = 'price') -> pd.DataFrame:
        """
        Обратная трансформация данных
        
        Args:
            data: трансформированный DataFrame
            feature_group: группа признаков для обратной трансформации
            
        Returns:
            DataFrame с исходным масштабом
        """
        if feature_group not in self.scalers:
            self.logger.warning(f"Скейлер для группы {feature_group} не найден")
            return data
        
        data_inverse = data.copy()
        features = self.feature_groups.get(feature_group, [])
        valid_features = [f for f in features if f in data_inverse.columns]
        
        if valid_features:
            data_inverse[valid_features] = self.scalers[feature_group].inverse_transform(
                data_inverse[valid_features]
            )
        
        return data_inverse
    
    def _identify_feature_groups(self, data: pd.DataFrame, exclude_targets: bool = True):
        """Определение групп признаков"""
        columns = data.columns.tolist()
        
        # Исключаем целевые переменные из нормализации (v4.0 - 20 целевых)
        if exclude_targets:
            target_keywords = ['future_', 'direction_', 'will_reach_', 'max_drawdown_', 'max_rally_',
                             'volatility_1h', 'volatility_4h', 'volatility_12h']
            # Удалено: 'best_action', 'signal_strength', 'risk_reward', 'optimal_hold' - больше не целевые
            columns = [col for col in columns 
                      if not any(keyword in col for keyword in target_keywords)]
        
        # Группы по типам данных
        self.feature_groups = {
            'price': [col for col in columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'price'])],
            'volume': [col for col in columns if 'volume' in col.lower() or 'turnover' in col.lower()],
            'returns': [col for col in columns if 'return' in col.lower() or 'change' in col.lower()],
            'indicators': [col for col in columns if any(x in col.lower() for x in ['rsi', 'macd', 'ema', 'sma', 'bb_'])],
            'volatility': [col for col in columns if any(x in col.lower() for x in ['atr', 'volatility', 'std'])],
            'microstructure': [col for col in columns if any(x in col.lower() for x in ['spread', 'imbalance', 'pressure'])],
            'temporal': [col for col in columns if any(x in col.lower() for x in ['hour', 'day', 'week', 'month'])],
            'fixed_range': [col for col in columns if any(x in col.lower() for x in ['rsi', 'stoch_k', 'stoch_d', 'adx', 'cci', 'williams_r', 'signal_strength'])]
        }
        
        # Удаление пустых групп
        self.feature_groups = {k: v for k, v in self.feature_groups.items() if v}
        
        # Все остальные числовые признаки
        identified_features = set(sum(self.feature_groups.values(), []))
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        other_features = [col for col in numeric_columns if col not in identified_features 
                         and col not in ['id', 'timestamp']]
        
        if other_features:
            self.feature_groups['other'] = other_features
    
    def _create_scaler(self):
        """Создание скейлера по конфигурации"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений"""
        if self.handle_missing == 'drop':
            return data.dropna()
        
        elif self.handle_missing == 'interpolate':
            # Интерполяция для временных рядов
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].interpolate(method='linear', limit_direction='both')
            
        elif self.handle_missing == 'forward_fill':
            data = data.fillna(method='ffill').fillna(method='bfill')
            
        elif self.handle_missing == 'impute':
            # KNN импутация для сложных случаев
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                imputer = KNNImputer(n_neighbors=5)
                data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        
        # Заполнение оставшихся пропусков
        data = data.fillna(0)
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Удаление выбросов"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['id', 'timestamp'] or 'target' in col:
                continue
                
            # Z-score метод
            if self.outlier_threshold > 0:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < self.outlier_threshold]
        
        return data
    
    def _additional_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Дополнительная обработка данных"""
        
        # Клиппинг экстремальных значений
        for col in data.select_dtypes(include=[np.number]).columns:
            if 'return' in col.lower():
                # Ограничение доходностей разумными значениями
                data[col] = data[col].clip(-0.5, 0.5)
            elif any(x in col.lower() for x in ['rsi', 'percent', 'prob']):
                # Ограничение процентных значений
                data[col] = data[col].clip(0, 100)
        
        # Проверка на бесконечные значения
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return data
    
    def get_feature_importance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Оценка важности признаков"""
        importance_scores = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        target_columns = [col for col in numeric_columns if 'target' in col or 'future' in col]
        feature_columns = [col for col in numeric_columns if col not in target_columns + ['id', 'timestamp']]
        
        if target_columns and feature_columns:
            # Корреляция с целевыми переменными
            for target in target_columns[:1]:  # Берем первую целевую переменную
                for feature in feature_columns:
                    correlation = data[feature].corr(data[target])
                    importance_scores.append({
                        'feature': feature,
                        'correlation': abs(correlation),
                        'signed_correlation': correlation
                    })
        
        importance_df = pd.DataFrame(importance_scores)
        if not importance_df.empty:
            importance_df = importance_df.sort_values('correlation', ascending=False)
        
        return importance_df


class SequencePreprocessor:
    """Препроцессор для последовательностей"""
    
    def __init__(self, sequence_length: int, stride: int = 1):
        """
        Args:
            sequence_length: длина последовательности
            stride: шаг между последовательностями
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.logger = get_logger("SequencePreprocessor")
    
    def create_sequences(self, 
                        data: pd.DataFrame,
                        target_column: Optional[str] = None,
                        group_column: Optional[str] = 'symbol') -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание последовательностей из данных
        
        Args:
            data: DataFrame с временными рядами
            target_column: целевая переменная
            group_column: столбец для группировки (например, symbol)
            
        Returns:
            Кортеж (sequences, targets)
        """
        sequences = []
        targets = []
        
        if group_column and group_column in data.columns:
            # Создание последовательностей по группам
            for group_name, group_data in data.groupby(group_column):
                group_sequences, group_targets = self._create_sequences_from_group(
                    group_data, target_column
                )
                sequences.extend(group_sequences)
                targets.extend(group_targets)
        else:
            # Создание последовательностей из всех данных
            sequences, targets = self._create_sequences_from_group(data, target_column)
        
        return np.array(sequences), np.array(targets)
    
    def _create_sequences_from_group(self,
                                   data: pd.DataFrame,
                                   target_column: Optional[str] = None) -> Tuple[List, List]:
        """Создание последовательностей из одной группы"""
        sequences = []
        targets = []
        
        # Выбор признаков
        feature_columns = [col for col in data.columns 
                          if col not in ['id', 'symbol', 'datetime', 'timestamp']]
        
        if target_column and target_column in data.columns:
            feature_columns = [col for col in feature_columns if col != target_column]
            target_data = data[target_column].values
        else:
            target_data = None
        
        feature_data = data[feature_columns].values
        
        # Создание окон
        for i in range(0, len(data) - self.sequence_length + 1, self.stride):
            sequence = feature_data[i:i + self.sequence_length]
            sequences.append(sequence)
            
            if target_data is not None:
                # Целевое значение - следующее после последовательности
                if i + self.sequence_length < len(target_data):
                    targets.append(target_data[i + self.sequence_length])
        
        return sequences, targets


def create_train_val_test_split(data: pd.DataFrame,
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15,
                              test_ratio: float = 0.15,
                              time_column: str = 'datetime',
                              gap_days: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделение данных на train/val/test с учетом времени и временным gap
    
    Args:
        data: исходные данные
        train_ratio: доля обучающей выборки
        val_ratio: доля валидационной выборки
        test_ratio: доля тестовой выборки
        time_column: столбец с временем
        gap_days: количество дней gap между выборками (для предотвращения утечек)
        
    Returns:
        Кортеж (train_data, val_data, test_data)
    """
    logger = get_logger("DataSplit")
    
    # Сортировка по времени
    data_sorted = data.sort_values(time_column)
    
    # Преобразуем в datetime если нужно
    if not pd.api.types.is_datetime64_any_dtype(data_sorted[time_column]):
        data_sorted[time_column] = pd.to_datetime(data_sorted[time_column])
    
    # Получаем уникальные даты
    unique_dates = data_sorted[time_column].dt.date.unique()
    n_dates = len(unique_dates)
    
    # Рассчитываем индексы с учетом gap
    train_end_idx = int(n_dates * train_ratio)
    val_start_idx = min(train_end_idx + gap_days, n_dates - 1)
    val_end_idx = min(val_start_idx + int(n_dates * val_ratio), n_dates - 1)
    test_start_idx = min(val_end_idx + gap_days, n_dates - 1)
    
    # Получаем даты для разделения
    train_end_date = unique_dates[train_end_idx]
    val_start_date = unique_dates[val_start_idx]
    val_end_date = unique_dates[val_end_idx]
    test_start_date = unique_dates[test_start_idx]
    
    # Разделяем данные
    train_data = data_sorted[data_sorted[time_column].dt.date <= train_end_date].copy()
    val_data = data_sorted[(data_sorted[time_column].dt.date >= val_start_date) & 
                          (data_sorted[time_column].dt.date <= val_end_date)].copy()
    test_data = data_sorted[data_sorted[time_column].dt.date >= test_start_date].copy()
    
    # Добавляем атрибуты для отслеживания типа датасета
    train_data.attrs['dataset_type'] = 'train'
    val_data.attrs['dataset_type'] = 'val'
    test_data.attrs['dataset_type'] = 'test'
    
    logger.info(f"📊 Разделение данных с временным gap ({gap_days} дней):")
    logger.info(f"   - Train: {len(train_data):,} записей (до {train_end_date})")
    logger.info(f"   - Val: {len(val_data):,} записей ({val_start_date} - {val_end_date})")
    logger.info(f"   - Test: {len(test_data):,} записей (от {test_start_date})")
    
    return train_data, val_data, test_data