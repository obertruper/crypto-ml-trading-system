"""
Создание временных последовательностей для Transformer v3
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm

from config import Config, SEQUENCE_PARAMS

logger = logging.getLogger(__name__)


class SequenceCreator:
    """Класс для создания временных последовательностей"""
    
    def __init__(self, config: Config):
        self.config = config
        self.sequence_length = config.model.sequence_length
        self.stride = config.training.sequence_stride
        
    def create_sequences(self, 
                        df: pd.DataFrame,
                        feature_columns: List[str],
                        target_column: str,
                        augment: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Создание последовательностей для временных рядов
        
        Args:
            df: DataFrame с данными
            feature_columns: Список колонок признаков
            target_column: Колонка с целевыми значениями
            augment: Применять ли аугментацию данных
            
        Returns:
            Tuple (sequences, targets, symbols)
        """
        logger.info(f"🔄 Создание последовательностей длиной {self.sequence_length}...")
        
        sequences = []
        targets = []
        seq_symbols = []
        
        # Группируем по символам
        for symbol in tqdm(df['symbol'].unique(), desc="Создание последовательностей"):
            symbol_df = df[df['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
            
            if len(symbol_df) < self.sequence_length + 1:
                logger.warning(f"⚠️ Недостаточно данных для {symbol}: {len(symbol_df)} записей")
                continue
            
            # Извлекаем данные
            X_symbol = symbol_df[feature_columns].values
            y_symbol = symbol_df[target_column].values
            
            # Проверяем и масштабируем expected returns если нужно
            y_mean = np.mean(y_symbol)
            y_std = np.std(y_symbol)
            if abs(y_mean) > 10 or y_std > 10:
                logger.warning(f"⚠️ {symbol} - {target_column}: mean={y_mean:.2f}, std={y_std:.2f} - масштабируем!")
                y_symbol = y_symbol / 100.0
            
            # Создаем последовательности с заданным шагом
            for i in range(0, len(X_symbol) - self.sequence_length, self.stride):
                seq = X_symbol[i:i + self.sequence_length]
                target = y_symbol[i + self.sequence_length]
                
                sequences.append(seq)
                targets.append(target)
                seq_symbols.append(symbol)
                
                # Аугментация данных
                if augment and self.config.training.use_augmentation:
                    aug_sequences = self._augment_sequence(seq)
                    for aug_seq in aug_sequences:
                        sequences.append(aug_seq)
                        targets.append(target)
                        seq_symbols.append(symbol)
        
        # Конвертируем в numpy arrays
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        seq_symbols = np.array(seq_symbols)
        
        # Проверка и замена NaN
        nan_sequences = np.isnan(sequences).sum()
        nan_targets = np.isnan(targets).sum()
        
        if nan_sequences > 0 or nan_targets > 0:
            logger.warning(f"⚠️ Найдено NaN: sequences={nan_sequences}, targets={nan_targets}")
            sequences = np.nan_to_num(sequences, nan=0.0)
            targets = np.nan_to_num(targets, nan=0.0)
        
        logger.info(f"✅ Создано {len(sequences):,} последовательностей")
        logger.info(f"   Форма данных: {sequences.shape}")
        logger.info(f"   Уникальных символов: {len(np.unique(seq_symbols))}")
        logger.info(f"   Целевая переменная - mean: {np.mean(targets):.4f}, std: {np.std(targets):.4f}")
        logger.info(f"   Целевая переменная - min: {np.min(targets):.4f}, max: {np.max(targets):.4f}")
        
        return sequences, targets, seq_symbols
    
    def create_sequences_for_splits(self,
                                   train_df: pd.DataFrame,
                                   val_df: pd.DataFrame,
                                   test_df: pd.DataFrame,
                                   feature_columns: List[str],
                                   target_type: str = 'buy') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Создание последовательностей для всех splits
        
        Args:
            train_df: Обучающие данные
            val_df: Валидационные данные  
            test_df: Тестовые данные
            feature_columns: Список признаков
            target_type: 'buy' или 'sell'
            
        Returns:
            Словарь с последовательностями для каждого split
        """
        target_column = f"{target_type}_expected_return"
        
        result = {}
        
        # Train split с аугментацией
        if len(train_df) > 0:
            X_train, y_train, symbols_train = self.create_sequences(
                train_df, feature_columns, target_column, augment=True
            )
            result['train'] = {
                'X': X_train,
                'y': y_train,
                'symbols': symbols_train
            }
        
        # Val split без аугментации
        if len(val_df) > 0:
            X_val, y_val, symbols_val = self.create_sequences(
                val_df, feature_columns, target_column, augment=False
            )
            result['val'] = {
                'X': X_val,
                'y': y_val,
                'symbols': symbols_val
            }
        
        # Test split без аугментации
        if len(test_df) > 0:
            X_test, y_test, symbols_test = self.create_sequences(
                test_df, feature_columns, target_column, augment=False
            )
            result['test'] = {
                'X': X_test,
                'y': y_test,
                'symbols': symbols_test
            }
        
        return result
    
    def _augment_sequence(self, sequence: np.ndarray) -> List[np.ndarray]:
        """
        Аугментация последовательности
        
        Args:
            sequence: Исходная последовательность
            
        Returns:
            Список аугментированных последовательностей
        """
        augmented = []
        
        if not self.config.training.use_augmentation:
            return augmented
        
        # 1. Добавление шума
        noise_level = self.config.training.noise_level
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, sequence.shape)
            noisy_seq = sequence + noise
            augmented.append(noisy_seq)
        
        # 2. Временной сдвиг (небольшой)
        if len(sequence) > 10:
            # Сдвиг на 1-2 шага
            for shift in [1, 2]:
                shifted_seq = np.roll(sequence, shift, axis=0)
                # Заполняем начало средними значениями
                shifted_seq[:shift] = sequence[:shift].mean(axis=0)
                augmented.append(shifted_seq)
        
        return augmented
    
    def create_sliding_windows(self,
                             df: pd.DataFrame,
                             feature_columns: List[str],
                             window_size: int = None) -> np.ndarray:
        """
        Создание скользящих окон для предсказания
        
        Args:
            df: DataFrame с данными
            feature_columns: Список признаков
            window_size: Размер окна (если None - используется sequence_length)
            
        Returns:
            Массив скользящих окон
        """
        if window_size is None:
            window_size = self.sequence_length
            
        features = df[feature_columns].values
        
        if len(features) < window_size:
            # Паддинг если данных недостаточно
            pad_size = window_size - len(features)
            padding = np.repeat(features[0:1], pad_size, axis=0)
            features = np.vstack([padding, features])
        
        # Создаем окно
        window = features[-window_size:]
        
        return window.reshape(1, window_size, -1)
    
    def get_sequence_statistics(self, sequences: np.ndarray, targets: np.ndarray) -> Dict:
        """Получение статистики по последовательностям"""
        stats = {
            'n_sequences': len(sequences),
            'sequence_length': sequences.shape[1] if sequences.ndim > 1 else 0,
            'n_features': sequences.shape[2] if sequences.ndim > 2 else 0,
            'target_stats': {
                'mean': np.mean(targets),
                'std': np.std(targets),
                'min': np.min(targets),
                'max': np.max(targets),
                'positive_ratio': np.mean(targets > 0)
            }
        }
        
        # Статистика по классам для классификации
        if self.config.training.task_type == 'classification_binary':
            unique, counts = np.unique(targets, return_counts=True)
            stats['class_distribution'] = dict(zip(unique, counts))
        
        return stats