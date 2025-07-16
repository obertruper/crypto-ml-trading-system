"""
Масштабирование целевой переменной для улучшения обучения
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Optional
import joblib
import os


class TargetScaler:
    """Класс для масштабирования целевых переменных с учетом выбросов"""
    
    def __init__(self, method='robust', clip_quantiles=(0.01, 0.99)):
        """
        Args:
            method: 'standard' или 'robust'
            clip_quantiles: квантили для обрезки выбросов
        """
        self.method = method
        self.clip_quantiles = clip_quantiles
        self.scaler = None
        self.clip_values = None
        self.is_fitted = False
        
    def fit(self, y: np.ndarray) -> 'TargetScaler':
        """Обучение масштабировщика на обучающих данных"""
        # Flatten если многомерный
        y_flat = y.flatten()
        
        # Удаляем NaN для расчета статистик
        y_clean = y_flat[~np.isnan(y_flat)]
        
        # Вычисляем квантили для обрезки
        if self.clip_quantiles:
            self.clip_values = {
                'lower': np.quantile(y_clean, self.clip_quantiles[0]),
                'upper': np.quantile(y_clean, self.clip_quantiles[1])
            }
            # Обрезаем выбросы
            y_clean = np.clip(y_clean, self.clip_values['lower'], self.clip_values['upper'])
        
        # Создаем и обучаем scaler
        if self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        self.scaler.fit(y_clean.reshape(-1, 1))
        self.is_fitted = True
        
        # Выводим статистику
        print(f"Целевая переменная до масштабирования:")
        print(f"  Mean: {np.mean(y_clean):.4f}")
        print(f"  Std: {np.std(y_clean):.4f}")
        print(f"  Min: {np.min(y_clean):.4f}")
        print(f"  Max: {np.max(y_clean):.4f}")
        
        if self.clip_values:
            print(f"  Обрезка по квантилям: [{self.clip_values['lower']:.4f}, {self.clip_values['upper']:.4f}]")
            
        return self
        
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Масштабирование целевой переменной"""
        if not self.is_fitted:
            raise ValueError("Scaler должен быть обучен перед использованием")
            
        original_shape = y.shape
        y_flat = y.flatten()
        
        # Обрабатываем NaN
        nan_mask = np.isnan(y_flat)
        y_clean = y_flat.copy()
        
        # Обрезаем выбросы если нужно
        if self.clip_values:
            y_clean[~nan_mask] = np.clip(
                y_clean[~nan_mask], 
                self.clip_values['lower'], 
                self.clip_values['upper']
            )
        
        # Масштабируем
        y_scaled = np.full_like(y_flat, np.nan)
        if np.any(~nan_mask):
            y_scaled[~nan_mask] = self.scaler.transform(
                y_clean[~nan_mask].reshape(-1, 1)
            ).flatten()
            
        return y_scaled.reshape(original_shape)
        
    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Обратное преобразование"""
        if not self.is_fitted:
            raise ValueError("Scaler должен быть обучен перед использованием")
            
        original_shape = y_scaled.shape
        y_flat = y_scaled.flatten()
        
        # Обрабатываем NaN
        nan_mask = np.isnan(y_flat)
        y_original = np.full_like(y_flat, np.nan)
        
        if np.any(~nan_mask):
            y_original[~nan_mask] = self.scaler.inverse_transform(
                y_flat[~nan_mask].reshape(-1, 1)
            ).flatten()
            
        return y_original.reshape(original_shape)
        
    def save(self, path: str):
        """Сохранение масштабировщика"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'clip_values': self.clip_values,
            'method': self.method,
            'is_fitted': self.is_fitted
        }, path)
        
    def load(self, path: str):
        """Загрузка масштабировщика"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.clip_values = data['clip_values']
        self.method = data['method']
        self.is_fitted = data['is_fitted']
        return self


def scale_targets_in_dataset(train_data: pd.DataFrame, 
                           val_data: pd.DataFrame,
                           test_data: pd.DataFrame,
                           target_col: str,
                           scaler_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TargetScaler]:
    """
    Масштабирует целевую переменную в датасетах
    
    Returns:
        train_data, val_data, test_data с масштабированными целевыми, scaler
    """
    # Создаем копии чтобы не изменять оригиналы
    train_scaled = train_data.copy()
    val_scaled = val_data.copy()
    test_scaled = test_data.copy()
    
    # Создаем и обучаем scaler
    scaler = TargetScaler(method='robust', clip_quantiles=(0.01, 0.99))
    
    # Обучаем только на train данных
    train_targets = train_scaled[target_col].values
    scaler.fit(train_targets)
    
    # Применяем ко всем датасетам
    train_scaled[f'{target_col}_scaled'] = scaler.transform(train_targets)
    val_scaled[f'{target_col}_scaled'] = scaler.transform(val_scaled[target_col].values)
    test_scaled[f'{target_col}_scaled'] = scaler.transform(test_scaled[target_col].values)
    
    # Сохраняем если указан путь
    if scaler_path:
        scaler.save(scaler_path)
        print(f"Scaler сохранен в {scaler_path}")
        
    # Статистика после масштабирования
    print(f"\nЦелевая переменная после масштабирования:")
    print(f"  Train - mean: {train_scaled[f'{target_col}_scaled'].mean():.4f}, std: {train_scaled[f'{target_col}_scaled'].std():.4f}")
    print(f"  Val - mean: {val_scaled[f'{target_col}_scaled'].mean():.4f}, std: {val_scaled[f'{target_col}_scaled'].std():.4f}")
    print(f"  Test - mean: {test_scaled[f'{target_col}_scaled'].mean():.4f}, std: {test_scaled[f'{target_col}_scaled'].std():.4f}")
    
    return train_scaled, val_scaled, test_scaled, scaler