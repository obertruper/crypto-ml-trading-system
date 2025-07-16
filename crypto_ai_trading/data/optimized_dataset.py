"""
Оптимизированный Dataset для максимальной производительности GPU
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path

from utils.logger import get_logger

class OptimizedTimeSeriesDataset(Dataset):
    """Оптимизированный Dataset с предварительным вычислением квантилей"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 context_window: int = 168,
                 prediction_window: int = 4,
                 feature_cols: List[str] = None,
                 target_cols: List[str] = None,
                 stride: int = 1,
                 normalize: bool = True,
                 scaler_path: Optional[str] = None,
                 fit_scaler: bool = False,
                 precompute_quantiles: bool = True):
        
        self.logger = get_logger("OptimizedDataset")
        self.data = data.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.target_window = prediction_window
        self.stride = stride
        
        # Определение признаков
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns 
                               if col not in ['id', 'symbol', 'datetime', 'timestamp', 'sector']
                               and not col.startswith(('target_', 'future_', 'optimal_'))]
        else:
            self.feature_cols = feature_cols
            
        if target_cols is None:
            self.target_cols = [col for col in data.columns 
                              if col.startswith(('target_', 'future_return_', 'long_tp', 'short_tp', 
                                               'long_sl', 'short_sl', 'long_optimal', 'short_optimal',
                                               'best_direction', 'signal_strength'))]
        else:
            self.target_cols = target_cols
        
        # Создание индексов
        self._create_indices()
        
        # Настройка нормализации
        self.normalize = normalize
        self.scaler = None
        self.volume_based_cols = []
        self.price_based_cols = []
        self.ratio_cols = []
        
        # ОПТИМИЗАЦИЯ: Конвертация данных в numpy заранее
        self.logger.info("📊 Конвертация данных в numpy arrays...")
        self.feature_data = self.data[self.feature_cols].values.astype(np.float32)
        self.target_data = self.data[self.target_cols].values.astype(np.float32) if self.target_cols else None
        
        # ОПТИМИЗАЦИЯ: Предвычисление квантилей
        self.quantiles = {}
        if normalize and precompute_quantiles:
            self.logger.info("🚀 Предвычисление квантилей для всего датасета...")
            self._precompute_quantiles()
        
        if self.normalize:
            self._setup_normalization(scaler_path, fit_scaler)
        
        # Кэш для часто используемых данных
        self.cache = {}
        self.cache_size = 1000  # Кэшировать последние 1000 элементов
        
        self.logger.info(f"✅ OptimizedDataset создан: {len(self)} примеров")
    
    def _precompute_quantiles(self):
        """Предвычисление квантилей для всех признаков"""
        # Сначала определяем типы колонок
        self._identify_column_types()
        
        for i, col in enumerate(self.feature_cols):
            if col not in self.ratio_cols:  # Не вычисляем для ratio колонок
                col_data = self.data[col].values
                # Применяем log трансформацию если нужно
                if col in self.volume_based_cols:
                    col_data = np.log1p(np.clip(col_data, 0, None))
                
                # Вычисляем квантили один раз для всего датасета
                self.quantiles[col] = {
                    'q99': np.percentile(col_data, 99),
                    'q01': np.percentile(col_data, 1)
                }
    
    def _identify_column_types(self):
        """Определение типов колонок для специальной обработки"""
        volume_keywords = ['volume', 'quote_volume', 'buy_volume', 'sell_volume', 'taker_volume']
        price_keywords = ['price', 'high', 'low', 'open', 'close', 'vwap']
        ratio_keywords = ['ratio', 'pct', 'percent', 'position', '_norm', 'consensus']
        
        for col in self.feature_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in volume_keywords):
                self.volume_based_cols.append(col)
            elif any(keyword in col_lower for keyword in price_keywords):
                self.price_based_cols.append(col)
            elif any(keyword in col_lower for keyword in ratio_keywords):
                self.ratio_cols.append(col)
    
    def _create_indices(self):
        """Создание индексов для валидных окон"""
        self.indices = []
        
        for symbol in self.data['symbol'].unique():
            symbol_data = self.data[self.data['symbol'] == symbol]
            symbol_indices = symbol_data.index.tolist()
            
            for i in range(0, len(symbol_indices) - self.context_window - self.prediction_window + 1, self.stride):
                window_indices = symbol_indices[i:i + self.context_window + self.prediction_window]
                
                if all(window_indices[j+1] - window_indices[j] == 1 for j in range(len(window_indices)-1)):
                    self.indices.append({
                        'symbol': symbol,
                        'start_idx': window_indices[0],
                        'context_end_idx': window_indices[self.context_window - 1],
                        'target_end_idx': window_indices[-1]
                    })
    
    def _setup_normalization(self, scaler_path, fit_scaler):
        """Настройка нормализации"""
        from sklearn.preprocessing import RobustScaler
        
        # Определение типов колонок если еще не определены
        if not self.volume_based_cols and not self.price_based_cols and not self.ratio_cols:
            self._identify_column_types()
        
        if scaler_path and Path(scaler_path).exists() and not fit_scaler:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = RobustScaler()
            if fit_scaler:
                # Подготовка данных для fit
                fit_data = self.feature_data.copy()
                
                # Log трансформация для объемных колонок
                for i, col in enumerate(self.feature_cols):
                    if col in self.volume_based_cols:
                        fit_data[:, i] = np.log1p(np.clip(fit_data[:, i], 0, None))
                
                # Fit scaler
                self.scaler.fit(fit_data)
                
                if scaler_path:
                    Path(scaler_path).parent.mkdir(exist_ok=True)
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Оптимизированное получение элемента"""
        # Проверка кэша
        if idx in self.cache:
            return self.cache[idx]
        
        index_info = self.indices[idx]
        
        # Получение данных через numpy slicing (быстрее чем pandas)
        start_idx = index_info['start_idx']
        end_idx = index_info['context_end_idx'] + 1
        
        # Быстрое извлечение данных
        feature_values = self.feature_data[start_idx:end_idx].copy()
        
        # Применение нормализации
        if self.normalize and self.scaler is not None:
            # Log трансформация для объемных колонок
            for i, col in enumerate(self.feature_cols):
                if col in self.volume_based_cols:
                    feature_values[:, i] = np.log1p(np.clip(feature_values[:, i], 0, None))
            
            # Использование предвычисленных квантилей
            if self.quantiles:
                for i, col in enumerate(self.feature_cols):
                    if col in self.quantiles:
                        q_data = self.quantiles[col]
                        feature_values[:, i] = np.clip(feature_values[:, i], 
                                                      q_data['q01'], q_data['q99'])
            
            # Применение scaler
            feature_values = self.scaler.transform(feature_values)
            feature_values = np.clip(feature_values, -10, 10)
        
        # Конвертация в тензор
        X = torch.from_numpy(feature_values).float()
        
        # Обработка целевых переменных
        if self.target_data is not None:
            target_idx = index_info['context_end_idx']
            y_values = self.target_data[target_idx].copy()
            
            # Специальная обработка для best_direction
            best_dir_idx = None
            for i, col in enumerate(self.target_cols):
                if col == 'best_direction':
                    best_dir_idx = i
                    # Предполагаем, что значения уже числовые (0, 1, 2)
                    break
            
            y = torch.from_numpy(y_values).float().unsqueeze(0)
        else:
            y = torch.empty(0)
        
        # Информация (минимальная для скорости)
        info = {
            'symbol': index_info['symbol'],
            'idx': idx
        }
        
        # Кэширование
        if len(self.cache) >= self.cache_size:
            # Удаляем старые элементы
            old_keys = list(self.cache.keys())[:100]
            for k in old_keys:
                del self.cache[k]
        
        self.cache[idx] = (X, y, info)
        
        return X, y, info


def create_optimized_dataloaders(train_data, val_data, test_data, config, logger):
    """Создание оптимизированных DataLoader'ов"""
    from torch.utils.data import DataLoader
    
    batch_size = config['model']['batch_size']
    num_workers = 4  # Оптимально для загрузки данных
    
    # Создание датасетов
    train_dataset = OptimizedTimeSeriesDataset(
        train_data,
        context_window=config['model']['context_window'],
        feature_cols=config.get('feature_cols'),
        target_cols=config.get('target_cols'),
        normalize=True,
        fit_scaler=True,
        scaler_path='cache/scaler.pkl'
    )
    
    val_dataset = OptimizedTimeSeriesDataset(
        val_data,
        context_window=config['model']['context_window'],
        feature_cols=train_dataset.feature_cols,
        target_cols=train_dataset.target_cols,
        normalize=True,
        scaler_path='cache/scaler.pkl'
    )
    
    test_dataset = OptimizedTimeSeriesDataset(
        test_data,
        context_window=config['model']['context_window'],
        feature_cols=train_dataset.feature_cols,
        target_cols=train_dataset.target_cols,
        normalize=True,
        scaler_path='cache/scaler.pkl'
    )
    
    # Создание DataLoader'ов с оптимизациями
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Важно для GPU!
        persistent_workers=True,  # Переиспользование воркеров
        prefetch_factor=2,  # Предзагрузка
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    logger.info(f"✅ Оптимизированные DataLoader'ы созданы:")
    logger.info(f"   - num_workers: {num_workers}")
    logger.info(f"   - pin_memory: True")
    logger.info(f"   - prefetch_factor: 2")
    logger.info(f"   - persistent_workers: True")
    
    return train_loader, val_loader, test_loader