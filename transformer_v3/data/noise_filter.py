"""
Модуль для продвинутой фильтрации шума в финансовых данных
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pywt
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NoiseFilter:
    """Класс для фильтрации шума в финансовых временных рядах"""
    
    def __init__(self, method: str = 'wavelet'):
        """
        Args:
            method: Метод фильтрации ('wavelet', 'savgol', 'gaussian', 'median')
        """
        self.method = method
        
    def filter_series(self, data: np.ndarray, preserve_edges: bool = True) -> np.ndarray:
        """
        Применение фильтрации к временному ряду
        
        Args:
            data: Временной ряд
            preserve_edges: Сохранять ли резкие изменения (важно для финансовых данных)
            
        Returns:
            Отфильтрованный ряд
        """
        if self.method == 'wavelet':
            return self._wavelet_denoise(data, preserve_edges)
        elif self.method == 'savgol':
            return self._savgol_filter(data)
        elif self.method == 'gaussian':
            return self._gaussian_filter(data)
        elif self.method == 'median':
            return self._median_filter(data)
        else:
            return data
    
    def _wavelet_denoise(self, data: np.ndarray, preserve_edges: bool = True) -> np.ndarray:
        """Wavelet denoising с адаптивным порогом"""
        # Используем db4 wavelet (хорош для финансовых данных)
        wavelet = 'db4'
        level = min(4, int(np.log2(len(data))))
        
        # Декомпозиция
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # Адаптивный порог на основе MAD (Median Absolute Deviation)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        if preserve_edges:
            # Мягкий порог для сохранения важных движений
            threshold = sigma * np.sqrt(2 * np.log(len(data))) * 0.5
        else:
            # Более агрессивный порог
            threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        # Применяем пороговую обработку
        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = pywt.threshold(coeffs_thresh[i], threshold, mode='soft')
        
        # Восстановление
        return pywt.waverec(coeffs_thresh, wavelet)[:len(data)]
    
    def _savgol_filter(self, data: np.ndarray) -> np.ndarray:
        """Savitzky-Golay фильтр для сглаживания с сохранением формы"""
        window_length = min(21, len(data) // 4)
        if window_length % 2 == 0:
            window_length += 1
        
        polyorder = min(3, window_length - 1)
        
        return signal.savgol_filter(data, window_length, polyorder)
    
    def _gaussian_filter(self, data: np.ndarray) -> np.ndarray:
        """Гауссов фильтр для плавного сглаживания"""
        sigma = len(data) * 0.01  # 1% от длины
        return gaussian_filter1d(data, sigma=sigma)
    
    def _median_filter(self, data: np.ndarray) -> np.ndarray:
        """Медианный фильтр для удаления выбросов"""
        kernel_size = min(5, len(data) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return signal.medfilt(data, kernel_size=kernel_size)
    
    def adaptive_filter(self, data: np.ndarray, volatility: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Адаптивная фильтрация на основе локальной волатильности
        
        Args:
            data: Временной ряд
            volatility: Локальная волатильность (если None, рассчитывается автоматически)
            
        Returns:
            Адаптивно отфильтрованный ряд
        """
        if volatility is None:
            # Рассчитываем локальную волатильность
            returns = np.diff(np.log(np.abs(data) + 1e-10))
            volatility = pd.Series(returns).rolling(20, min_periods=1).std().values
            volatility = np.concatenate([[volatility[0]], volatility])
        
        # Нормализуем волатильность
        volatility = volatility / np.nanmean(volatility)
        
        # Применяем разную степень фильтрации в зависимости от волатильности
        filtered_data = np.zeros_like(data)
        
        for i in range(len(data)):
            # Определяем размер окна на основе волатильности
            if volatility[i] > 1.5:  # Высокая волатильность
                # Минимальная фильтрация
                window = max(3, int(5 / volatility[i]))
            else:  # Низкая волатильность
                # Более сильная фильтрация
                window = min(21, int(10 / max(0.5, volatility[i])))
            
            # Применяем локальное сглаживание
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            filtered_data[i] = np.median(data[start:end])
        
        return filtered_data
    
    def ensemble_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Ансамблевая фильтрация - комбинация нескольких методов
        """
        # Применяем разные методы
        wavelet_filtered = self._wavelet_denoise(data, preserve_edges=True)
        savgol_filtered = self._savgol_filter(data)
        median_filtered = self._median_filter(data)
        
        # Взвешенное среднее (больший вес wavelet)
        weights = [0.5, 0.3, 0.2]
        ensemble = (weights[0] * wavelet_filtered + 
                   weights[1] * savgol_filtered + 
                   weights[2] * median_filtered)
        
        return ensemble
    
    def detect_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Обнаружение выбросов на основе Z-score
        
        Returns:
            Маска выбросов (True = выброс)
        """
        # Используем MAD для робастной оценки
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / (mad + 1e-10)
        
        return np.abs(modified_z_scores) > threshold
    
    def clean_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Очистка выбросов с интерполяцией
        """
        outliers = self.detect_outliers(data, threshold)
        cleaned_data = data.copy()
        
        if np.any(outliers):
            # Интерполируем выбросы
            indices = np.arange(len(data))
            cleaned_data[outliers] = np.interp(
                indices[outliers],
                indices[~outliers],
                data[~outliers]
            )
        
        return cleaned_data