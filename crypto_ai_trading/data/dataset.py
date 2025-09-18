"""
PyTorch Dataset классы для загрузки и обработки временных рядов
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler
import pickle
from tqdm import tqdm

from utils.logger import get_logger
from data.constants import TRADING_TARGET_VARIABLES, SERVICE_COLUMNS, get_feature_columns

class TimeSeriesDataset(Dataset):
    """Dataset для временных рядов криптовалют"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 context_window: int = 168,
                 prediction_window: int = 4,
                 feature_cols: List[str] = None,
                 target_cols: List[str] = None,
                 stride: int = 1,
                 normalize: bool = True,
                 scaler_path: Optional[str] = None,
                 fit_scaler: bool = False):
        """
        Args:
            data: DataFrame с данными
            context_window: размер входного окна
            prediction_window: размер окна предсказания
            feature_cols: список признаков
            target_cols: список целевых переменных
            stride: шаг между окнами
        """
        self.logger = get_logger("TimeSeriesDataset")
        self.data = data.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.target_window = prediction_window  # Добавляем target_window для совместимости
        self.stride = stride
        
        # Категориальные переменные v4.0 и их маппинг
        # ИСПРАВЛЕНО: Правильный маппинг для торговли
        # LONG=0 (покупка), SHORT=1 (продажа), FLAT=2 (боковик)
        self.categorical_targets = {
            'direction_15m': {'LONG': 0, 'SHORT': 1, 'FLAT': 2, 'UP': 0, 'DOWN': 1, 0: 0, 1: 1, 2: 2},
            'direction_1h':  {'LONG': 0, 'SHORT': 1, 'FLAT': 2, 'UP': 0, 'DOWN': 1, 0: 0, 1: 1, 2: 2},
            'direction_4h':  {'LONG': 0, 'SHORT': 1, 'FLAT': 2, 'UP': 0, 'DOWN': 1, 0: 0, 1: 1, 2: 2},
            'direction_12h': {'LONG': 0, 'SHORT': 1, 'FLAT': 2, 'UP': 0, 'DOWN': 1, 0: 0, 1: 1, 2: 2}
        }
        
        # Определение признаков и целевых переменных
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns 
                               if col not in ['id', 'symbol', 'datetime', 'timestamp', 'sector']
                               and not col.startswith(('target_', 'future_', 'optimal_'))
                               and all(kw not in col for kw in ('expected_value', 'optimal_entry', '_tp', '_sl'))]
        else:
            self.feature_cols = feature_cols

        # Колонки, которые нужно исключить из нормализации
        # 1) Категориальные индексы (symbol_id и т.п.)
        self.skip_scale_cols = [col for col in self.feature_cols if col in ('symbol_id',)]
        # 2) Технические индикаторы с фиксированными или специфическими диапазонами
        technical_indicators = [
            'rsi', 'stoch_k', 'stoch_d', 'adx', 'adx_pos', 'adx_neg',
            'rsi_oversold', 'rsi_overbought', 'toxicity', 'psar_trend',
            'cci', 'williams_r', 'roc', 'momentum', 'kama', 'trix',
            'ppo', 'macd', 'macd_signal', 'macd_diff', 'bb_position'
        ]
        for col in self.feature_cols:
            cl = col.lower()
            if any(tok in cl for tok in technical_indicators):
                if col not in self.skip_scale_cols:
                    self.skip_scale_cols.append(col)
        # Индексы колонок, которые будут нормализоваться
        self.scale_indices = [i for i, col in enumerate(self.feature_cols) if col not in self.skip_scale_cols]
            
        if target_cols is None:
            # Обновленный список целевых переменных для торговой модели v4.0 (20 переменных)
            self.target_cols = [col for col in data.columns 
                              if col.startswith(('future_return_', 'direction_', 
                                               'long_will_reach_', 'short_will_reach_',
                                               'max_drawdown_', 'max_rally_'))]
        else:
            self.target_cols = target_cols
        
        # Создание индексов для эффективного доступа
        self._create_indices()
        
        # Нормализация данных
        self.normalize = normalize
        self.scaler = None
        self.volume_based_cols = []
        self.price_based_cols = []
        self.ratio_cols = []
        
        if self.normalize:
            self._setup_normalization(scaler_path, fit_scaler)

        self.column_clip_bounds: Dict[str, Tuple[float, float]] = {}
        if self.normalize:
            self._compute_column_clip_bounds()
        
        self.logger.info(f"Dataset создан: {len(self)} примеров, "
                        f"{len(self.feature_cols)} признаков, "
                        f"{len(self.target_cols)} целевых переменных")
    
    def _create_indices(self):
        """Создание индексов для валидных окон"""
        self.indices = []
        
        # Получаем уникальные символы
        symbols = self.data['symbol'].unique()
        self.logger.info(f"🔄 Создание индексов для {len(symbols)} символов...")
        
        # Группировка по символам с прогресс-баром
        for symbol in tqdm(symbols, desc="Создание окон", leave=False):
            symbol_data = self.data[self.data['symbol'] == symbol]
            symbol_indices = symbol_data.index.tolist()
            
            # Создание окон с учетом stride
            window_count = 0
            for i in range(0, len(symbol_indices) - self.context_window - self.prediction_window + 1, self.stride):
                # Проверка непрерывности временного ряда (оптимизированная)
                start_idx = i
                end_idx = i + self.context_window + self.prediction_window
                
                # Быстрая проверка последовательности через разность первого и последнего индекса
                if symbol_indices[end_idx - 1] - symbol_indices[start_idx] == end_idx - start_idx - 1:
                    self.indices.append({
                        'symbol': symbol,
                        'start_idx': symbol_indices[start_idx],
                        'context_end_idx': symbol_indices[start_idx + self.context_window - 1],
                        'target_end_idx': symbol_indices[end_idx - 1]
                    })
                    window_count += 1
            
            if window_count > 0:
                self.logger.debug(f"  {symbol}: создано {window_count} окон")
        
        self.logger.info(f"✅ Создано {len(self.indices)} окон с stride={self.stride}")
    
    def _setup_normalization(self, scaler_path: Optional[str] = None, fit_scaler: bool = False):
        """Настройка нормализации данных"""
        self.logger.info("🔧 Настройка нормализации данных...")
        
        # Определяем типы колонок для разной нормализации
        self.volume_based_cols = [col for col in self.feature_cols if any(
            pattern in col.lower() for pattern in ['volume', 'turnover', 'obv', 'liquidity', 'cmf', 'mfi']
        )]
        
        self.price_based_cols = [col for col in self.feature_cols if any(
            pattern in col.lower() for pattern in ['price', 'vwap', 'high', 'low', 'open', 'close']
        )]
        
        self.ratio_cols = [col for col in self.feature_cols if any(
            pattern in col.lower() for pattern in ['ratio', 'rsi', 'stoch', 'bb_', 'pct', 'toxicity']
        )]
        
        # Загрузка или создание scaler
        if scaler_path and Path(scaler_path).exists() and not fit_scaler:
            self.logger.info(f"📥 Загрузка scaler из {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            # Проверяем совместимость по числу признаков
            try:
                expected = getattr(self.scaler, 'n_features_in_', None)
                scale_cols = [c for c in self.feature_cols if c not in self.skip_scale_cols]
                current = len(scale_cols)
                if expected is not None and expected != current:
                    self.logger.warning(
                        f"⚠️ Несовпадение признаков скейлера: ожидалось {expected}, текущих {current}. "
                        f"Отключаю нормализацию для этого датасета, чтобы избежать ошибок."
                    )
                    self.scaler = None
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось проверить совместимость скейлера: {e}")
        else:
            self.logger.info("🔨 Создание нового scaler...")
            self.scaler = RobustScaler(quantile_range=(5, 95))

            if fit_scaler:
                # Фитим scaler на тренировочных данных
                self.logger.info("📊 Обучение scaler на данных...")
                
                # Подготовка данных для обучения scaler
                # Обучаем только на подмножестве колонок, требующих нормализации
                scale_cols = [c for c in self.feature_cols if c not in self.skip_scale_cols]
                scaler_data = self.data[scale_cols].copy()
                
                # Применяем log-трансформацию к объемным колонкам
                for col in self.volume_based_cols:
                    if col in scaler_data.columns:
                        # ИСПРАВЛЕНО: Конвертация в числовой тип перед log-трансформацией
                        scaler_data[col] = pd.to_numeric(scaler_data[col], errors='coerce')
                        # Log трансформация с защитой от отрицательных значений
                        scaler_data[col] = np.log1p(np.clip(scaler_data[col], 0, None))
                
                # Клиппинг экстремальных значений перед обучением scaler
                for col in scaler_data.columns:
                    if col not in self.ratio_cols:  # Не клиппим ratio колонки
                        # ИСПРАВЛЕНО: Конвертация в числовой тип перед квантилями
                        scaler_data[col] = pd.to_numeric(scaler_data[col], errors='coerce')
                        
                        # Пропускаем колонки с только NaN значениями
                        if scaler_data[col].notna().sum() == 0:
                            self.logger.warning(f"Колонка '{col}' содержит только NaN значения, пропускаем")
                            continue
                            
                        q99 = scaler_data[col].quantile(0.99)
                        q01 = scaler_data[col].quantile(0.01)
                        scaler_data[col] = np.clip(scaler_data[col], q01, q99)
                
                # Обучаем scaler
                if len(scale_cols) > 0:
                    self.scaler.fit(scaler_data.values)
                else:
                    self.logger.warning("Нет колонок для нормализации (только skip-колонки)")
                
                # Сохраняем scaler если указан путь
                if scaler_path:
                    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    self.logger.info(f"💾 Scaler сохранен в {scaler_path}")
        
        self.logger.info(f"✅ Нормализация настроена: {len(self.volume_based_cols)} объемных, "
                        f"{len(self.price_based_cols)} ценовых, {len(self.ratio_cols)} ratio колонок")

    def _compute_column_clip_bounds(self):
        """Предварительный расчет квантилей для клиппинга признаков"""
        self.logger.info("📐 Предварительный расчет квантилей для клиппинга признаков")

        for col in self.feature_cols:
            if col in self.ratio_cols or col in self.skip_scale_cols:
                continue

            series = pd.to_numeric(self.data[col], errors='coerce')
            if col in self.volume_based_cols:
                series = np.log1p(np.clip(series, 0, None))

            values = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)

            if values.size < 100:
                continue

            q01 = np.nanpercentile(values, 1)
            q99 = np.nanpercentile(values, 99)

            if np.isfinite(q01) and np.isfinite(q99):
                self.column_clip_bounds[col] = (q01, q99)

        self.logger.info(f"✅ Предрасчитаны квантили для {len(self.column_clip_bounds)} колонок")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Получение одного примера"""
        index_info = self.indices[idx]
        
        # Извлечение контекстных данных
        context_start = index_info['start_idx']
        context_end = index_info['context_end_idx'] + 1
        context_data = self.data.iloc[context_start:context_end]
        
        # ИСПРАВЛЕНО: Целевые переменные берем из последней строки контекста
        # а не из будущих строк!
        
        # Преобразование в тензоры с обработкой object типов
        feature_data = context_data[self.feature_cols]
        
        # ПОЛНАЯ версия: надёжная конвертация в числовые типы
        feature_data = feature_data.copy()

        # Быстрый путь: если все признаки уже числовые — избегаем постколоночных преобразований
        if all(pd.api.types.is_numeric_dtype(dt) for dt in feature_data.dtypes):
            feature_values = feature_data.to_numpy(dtype=np.float32, copy=False)
        else:
            # Преобразуем только необходимые колонки
            for col in feature_data.columns:
                try:
                    if pd.api.types.is_object_dtype(feature_data[col]) or pd.api.types.is_categorical_dtype(feature_data[col]):
                        if pd.api.types.is_categorical_dtype(feature_data[col]):
                            feature_data[col] = feature_data[col].cat.codes.astype(np.float32)
                        else:
                            feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0.0).astype(np.float32)
                    else:
                        feature_data[col] = feature_data[col].astype(np.float32)
                except Exception:
                    feature_data[col] = np.zeros(len(feature_data), dtype=np.float32)

        # Получаем массив и финальная очистка от inf/nan
        feature_values = feature_data.values
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Применяем нормализацию если включена
        if self.normalize and self.scaler is not None and len(self.scale_indices) > 0:
            # Создаем копию для нормализации
            norm_values = feature_values.copy()

            # Применяем log-трансформацию к объемным колонкам
            for i, col in enumerate(self.feature_cols):
                if col in self.volume_based_cols and col not in self.skip_scale_cols:
                    # Log трансформация с защитой от отрицательных значений
                    norm_values[:, i] = np.log1p(np.clip(norm_values[:, i], 0, None))
            
            # Клиппинг экстремальных значений
            for i, col in enumerate(self.feature_cols):
                if col not in self.ratio_cols and col not in self.skip_scale_cols:
                    bounds = self.column_clip_bounds.get(col)
                    if bounds:
                        q01, q99 = bounds
                        norm_values[:, i] = np.clip(norm_values[:, i], q01, q99)
            
            # Применяем RobustScaler
            try:
                # Масштабируем только подмножество колонок
                cols = self.scale_indices
                sub = norm_values[:, cols]
                sub_scaled = self.scaler.transform(sub)
                norm_values[:, cols] = sub_scaled
            except Exception as e:
                self.logger.warning(f"Ошибка при нормализации: {e}")
                # Fallback к исходным значениям
                norm_values = feature_values
            
            # Финальный клиппинг после нормализации
            norm_values = np.clip(norm_values, -10, 10)
            
            X = torch.FloatTensor(norm_values)
        else:
            X = torch.FloatTensor(feature_values)
        
        # Обработка целевых переменных
        if len(self.target_cols) > 0:
            # Берем целевые значения из ПОСЛЕДНЕЙ строки контекста
            # Это значения future_return_X которые уже содержат будущую доходность
            y_data = context_data.iloc[-1][self.target_cols]
            
            # Конвертируем категориальные переменные в числовые
            y_values = []
            for col in self.target_cols:
                value = y_data[col]
                
                # Обработка категориальной переменной best_direction
                if col == 'best_direction':
                    if pd.api.types.is_categorical_dtype(value) or isinstance(value, str):
                        # Конвертируем в числовой код: LONG=0, SHORT=1, NEUTRAL=2
                        if value == 'LONG':
                            y_values.append(0)
                        elif value == 'SHORT':
                            y_values.append(1)
                        elif value == 'NEUTRAL':
                            y_values.append(2)
                        else:
                            y_values.append(2)  # fallback к NEUTRAL
                    else:
                        y_values.append(float(value))
                else:
                    # Проверяем, является ли это категориальной переменной v4.0
                    if col in self.categorical_targets:
                        # Преобразуем категориальное значение в числовое
                        mapping = self.categorical_targets[col]
                        # Пробуем разные форматы значения
                        if value in mapping:
                            numeric_value = mapping[value]
                        elif str(value) in mapping:
                            numeric_value = mapping[str(value)]
                        elif isinstance(value, (int, float)):
                            # Если это уже число, используем его напрямую
                            numeric_value = int(value)
                        else:
                            # Fallback к FLAT только если действительно не можем определить
                            self.logger.warning(f"Неизвестное значение '{value}' для {col}, используем FLAT=2")
                            numeric_value = 2
                        y_values.append(float(numeric_value))
                    else:
                        # Обычные числовые переменные
                        try:
                            y_values.append(float(value))
                        except (ValueError, TypeError):
                            # Если не удается преобразовать, используем 0
                            self.logger.warning(f"Не удалось преобразовать значение '{value}' для колонки '{col}', используем 0")
                            y_values.append(0.0)
            
            y_values = np.array(y_values, dtype=np.float32)
            
            # Если модель ожидает несколько временных шагов, дублируем значение
            if self.target_window > 1:
                y = torch.FloatTensor([y_values] * self.target_window)
            else:
                y = torch.FloatTensor([y_values])
        else:
            y = torch.FloatTensor([])
        
        # Данные для анализа (оставляем для совместимости)
        target_start = index_info['context_end_idx'] + 1
        target_end = index_info['target_end_idx'] + 1
        target_data = self.data.iloc[target_start:target_end]
        
        # Дополнительная информация
        info = {
            'symbol': index_info['symbol'],
            'context_start_time': str(context_data.iloc[0]['datetime']),
            'context_end_time': str(context_data.iloc[-1]['datetime']),
            'target_start_time': str(target_data.iloc[0]['datetime']) if len(target_data) > 0 else None,
            'target_end_time': str(target_data.iloc[-1]['datetime']) if len(target_data) > 0 else None
        }
        
        return X, y, info


class TradingDataset(TimeSeriesDataset):
    """Специализированный Dataset для торговых сигналов"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 config: Dict = None,
                 context_window: int = 168,
                 prediction_window: int = 4,
                 feature_cols: List[str] = None,
                 target_cols: List[str] = None,
                 include_price_data: bool = True,
                 **kwargs):
        
        # КРИТИЧНОЕ ИСПРАВЛЕНИЕ: Правильная работа с торговыми целевыми переменными
        if target_cols is not None:
            # Если target_cols переданы явно, используем их
            trading_targets = target_cols
            print(f"✅ Используется {len(trading_targets)} целевых переменных из параметров")
        elif target_cols is None and config is not None:
            # Получаем конфигурацию модели
            model_config = config.get('model', {})
            task_type = model_config.get('task_type', 'regression')
            
            if task_type == 'trading':
                # Для торговой модели используем специальные целевые переменные
                trading_target_variables = model_config.get('target_variables', [])
                
                if trading_target_variables:
                    # Проверяем какие целевые переменные есть в данных
                    available_trading_targets = [var for var in trading_target_variables if var in data.columns]
                    
                    if available_trading_targets:
                        trading_targets = available_trading_targets
                        print(f"✅ Используется торговая модель с {len(trading_targets)} целевыми переменными")
                        print(f"   Доступные цели: {trading_targets}")
                    else:
                        print("❌ Торговые целевые переменные из конфига не найдены в данных!")
                        # Fallback к автоматическому поиску торговых целей
                        auto_trading_targets = [col for col in data.columns 
                                              if col.startswith(('long_tp', 'short_tp', 'long_sl', 'short_sl', 'best_direction'))]
                        
                        if auto_trading_targets:
                            trading_targets = auto_trading_targets[:11]  # Ограничиваем основными
                            print(f"🔧 Автоматически найдено {len(trading_targets)} торговых целей")
                        else:
                            raise ValueError("❌ Торговые целевые переменные не найдены!")
                else:
                    # Автоматический поиск торговых целевых переменных
                    auto_trading_targets = [col for col in data.columns 
                                          if col.startswith(('long_tp', 'short_tp', 'long_sl', 'short_sl', 'best_direction'))]
                    
                    if auto_trading_targets:
                        # Приоритетные торговые цели
                        priority_targets = [
                            'long_tp1_reached', 'long_tp2_reached', 'long_tp3_reached', 'long_sl_reached',
                            'short_tp1_reached', 'short_tp2_reached', 'short_tp3_reached', 'short_sl_reached',
                            'best_direction'
                        ]
                        trading_targets = [t for t in priority_targets if t in data.columns]
                        
                        if len(trading_targets) < 5:  # Минимум для торговой модели
                            trading_targets = auto_trading_targets[:11]  # Берем первые 11
                        
                        print(f"🔧 Автоматически выбрано {len(trading_targets)} торговых целей")
                    else:
                        raise ValueError("❌ Торговые целевые переменные не найдены в данных!")
            else:
                # Для регрессии/классификации используем стандартную логику
                target_variable = model_config.get('target_variable', 'future_return_4')
                
                # ИСПРАВЛЕНИЕ: Проверяем что целевая переменная существует в данных
                if target_variable in data.columns:
                    trading_targets = [target_variable]
                    print(f"✅ Используется целевая переменная: {target_variable} (тип: {task_type})")
                else:
                    # Fallback к доступным целевым переменным
                    available_targets = [col for col in data.columns 
                                   if col.startswith(('target_', 'future_return_'))]
                    
                    if available_targets:
                        # Приоритет: future_return_4 > future_return_3 > future_return_2 > future_return_1
                        preferred_targets = ['future_return_4', 'future_return_3', 'future_return_2', 'future_return_1']
                        trading_targets = None
                        for pref_target in preferred_targets:
                            if pref_target in available_targets:
                                trading_targets = [pref_target]
                                break
                        
                        if trading_targets is None:
                            trading_targets = [available_targets[0]]  # Берем первую доступную
                        
                        print(f"⚠️  Целевая переменная {target_variable} не найдена. Используется: {trading_targets[0]}")
                    else:
                        # КРИТИЧНОЕ ИСПРАВЛЕНИЕ: НЕ создаем переменные заново при использовании кэша!
                        # Проверяем есть ли ANY торговые целевые переменные в кэшированных данных
                        any_trading_targets = [col for col in data.columns 
                                             if any(pattern in col for pattern in ['_hit', '_reached', '_tp', '_sl', 'best_direction', 'future_return'])]
                        
                        if any_trading_targets:
                            # Используем существующие торговые переменные из кэша
                            trading_targets = any_trading_targets[:36]  # Ограничиваем до 36 для модели
                            print(f"✅ Найдено {len(trading_targets)} торговых переменных в кэшированных данных")
                            print(f"   Используем существующие цели из кэша: {trading_targets[:5]}...")
                        else:
                            raise ValueError("❌ Кэшированные данные не содержат торговых целевых переменных! Пересоздайте кэш: python prepare_trading_data.py --force-recreate")
                    
        elif target_cols is None:
            # Автоматически находим целевые переменные (старое поведение)
            available_targets = [col for col in data.columns 
                               if col.startswith(('target_', 'future_return_'))]
            
            if available_targets:
                # ИСПРАВЛЕНО: Используем только future_return_4 для совместимости с pred_len=4
                if 'future_return_4' in available_targets:
                    trading_targets = ['future_return_4']
                else:
                    # Fallback к любому future_return
                    future_returns = [col for col in available_targets if 'future_return_' in col]
                    trading_targets = future_returns[:1] if future_returns else [available_targets[0]]
            else:
                raise ValueError("❌ Целевые переменные не найдены в данных и config не предоставлен!")
        else:
            trading_targets = target_cols
        
        # ИСПРАВЛЕНИЕ: Проверка размерности целей с конфигурацией модели
        if config is not None:
            model_config = config.get('model', {})
            task_type = model_config.get('task_type', 'regression')
            expected_output_size = model_config.get('output_size', 1)
            
            # Для торговой модели НЕ обрезаем целевые переменные
            if task_type != 'trading' and len(trading_targets) != expected_output_size:
                print(f"⚠️  Несоответствие размерностей: target_cols={len(trading_targets)}, output_size={expected_output_size}")
                
                # Автоматическое исправление: берем только нужное количество целей
                if len(trading_targets) > expected_output_size:
                    trading_targets = trading_targets[:expected_output_size]
                    print(f"🔧 Исправлено: используются первые {expected_output_size} целей: {trading_targets}")
            elif task_type == 'trading':
                print(f"✅ Торговая модель: используется {len(trading_targets)} целевых переменных")
        
        # ВАЖНО: НЕ пересоздаем данные если они уже есть в кэше!
        # Проверяем все ли нужные целевые переменные присутствуют
        missing_targets = [target for target in trading_targets if target not in data.columns]
        
        if missing_targets:
            print(f"⚠️  Отсутствуют целевые переменные: {missing_targets}")
            print("❌ Пересоздайте кэш с правильными целевыми переменными:")
            print("   python prepare_trading_data.py --force-recreate")
            raise ValueError(f"Отсутствуют целевые переменные в кэшированных данных: {missing_targets}")
        else:
            print(f"✅ Все {len(trading_targets)} целевых переменных найдены в кэшированных данных")
        
        # Добавляем target_window если он передан в kwargs
        if 'target_window' in kwargs:
            target_window = kwargs.pop('target_window')
        else:
            target_window = prediction_window
            
        super().__init__(
            data=data,
            context_window=context_window,
            prediction_window=target_window,
            feature_cols=feature_cols,
            target_cols=trading_targets,
            **kwargs
        )
        
        self.include_price_data = include_price_data
        
        # Индексы ценовых данных
        if include_price_data:
            self.price_cols = ['open', 'high', 'low', 'close', 'volume']
    
    def __getitem__(self, idx):
        """Получение примера с торговой информацией"""
        X, y, info = super().__getitem__(idx)
        
        index_info = self.indices[idx]
        
        # Добавляем ценовые данные если нужно
        if self.include_price_data:
            context_start = index_info['start_idx']
            context_end = index_info['context_end_idx'] + 1
            price_data = self.data.iloc[context_start:context_end][self.price_cols]
            info['price_data'] = torch.FloatTensor(price_data.values)
        
        # Добавляем последнюю цену для расчета уровней
        last_idx = index_info['context_end_idx']
        info['last_close'] = self.data.iloc[last_idx]['close']
        info['atr'] = self.data.iloc[last_idx].get('atr', info['last_close'] * 0.02)
        
        return X, y, info


class MultiSymbolDataset(Dataset):
    """Dataset для одновременной работы с несколькими символами"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 symbols: List[str],
                 context_window: int = 168,
                 prediction_window: int = 4,
                 feature_cols: List[str] = None,
                 synchronize: bool = True):
        """
        Args:
            data: DataFrame со всеми данными
            symbols: список символов для батча
            context_window: размер входного окна
            prediction_window: размер окна предсказания
            feature_cols: список признаков
            synchronize: синхронизировать временные метки между символами
        """
        self.logger = get_logger("MultiSymbolDataset")
        self.symbols = symbols
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.synchronize = synchronize
        
        # Фильтрация данных по символам
        self.data = data[data['symbol'].isin(symbols)].copy()
        
        # Определение признаков
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns 
                               if col not in ['id', 'symbol', 'datetime', 'timestamp', 'sector']
                               and not col.startswith(('target_', 'future_', 'optimal_'))
                               and all(kw not in col for kw in ('expected_value', 'optimal_entry', '_tp', '_sl'))]
        else:
            self.feature_cols = feature_cols
        
        # Создание синхронизированных индексов
        if synchronize:
            self._create_synchronized_indices()
        else:
            self._create_independent_indices()
        
        self.logger.info(f"MultiSymbolDataset создан: {len(self)} батчей, "
                        f"{len(symbols)} символов")
    
    def _create_synchronized_indices(self):
        """Создание синхронизированных по времени индексов"""
        self.indices = []
        
        # Находим общие временные метки
        common_times = None
        for symbol in self.symbols:
            symbol_times = set(self.data[self.data['symbol'] == symbol]['datetime'].values)
            if common_times is None:
                common_times = symbol_times
            else:
                common_times = common_times.intersection(symbol_times)
        
        common_times = sorted(list(common_times))
        
        # Создаем окна на основе общих временных меток
        window_size = self.context_window + self.prediction_window
        for i in range(len(common_times) - window_size + 1):
            window_times = common_times[i:i + window_size]
            self.indices.append({
                'start_time': window_times[0],
                'context_end_time': window_times[self.context_window - 1],
                'end_time': window_times[-1]
            })
    
    def _create_independent_indices(self):
        """Создание независимых индексов для каждого символа"""
        # Используем TimeSeriesDataset для каждого символа
        self.symbol_datasets = {}
        
        for symbol in self.symbols:
            symbol_data = self.data[self.data['symbol'] == symbol]
            self.symbol_datasets[symbol] = TimeSeriesDataset(
                data=symbol_data,
                context_window=self.context_window,
                prediction_window=self.prediction_window,
                feature_cols=self.feature_cols
            )
        
        # Индексы - это минимальное количество примеров среди всех символов
        min_length = min(len(ds) for ds in self.symbol_datasets.values())
        self.indices = list(range(min_length))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Получение батча данных для всех символов"""
        batch_X = []
        batch_y = []
        batch_info = []
        
        if self.synchronize:
            # Синхронизированная выборка
            index_info = self.indices[idx]
            
            for symbol in self.symbols:
                # Получаем данные для временного окна
                symbol_data = self.data[
                    (self.data['symbol'] == symbol) &
                    (self.data['datetime'] >= index_info['start_time']) &
                    (self.data['datetime'] <= index_info['end_time'])
                ].sort_values('datetime')
                
                if len(symbol_data) == self.context_window + self.prediction_window:
                    # Разделение на контекст и цель
                    context_data = symbol_data.iloc[:self.context_window]
                    target_data = symbol_data.iloc[self.context_window:]
                    
                    X = torch.FloatTensor(context_data[self.feature_cols].values)
                    y = torch.FloatTensor(target_data[self.feature_cols].values)
                    
                    batch_X.append(X)
                    batch_y.append(y)
                    batch_info.append({
                        'symbol': symbol,
                        'start_time': index_info['start_time'],
                        'end_time': index_info['end_time']
                    })
        else:
            # Независимая выборка
            for symbol in self.symbols:
                X, y, info = self.symbol_datasets[symbol][idx]
                batch_X.append(X)
                batch_y.append(y)
                batch_info.append(info)
        
        # Стекирование в батч
        batch_X = torch.stack(batch_X)
        batch_y = torch.stack(batch_y) if len(batch_y[0]) > 0 else torch.tensor([])
        
        return batch_X, batch_y, batch_info


def create_data_loaders(train_data: pd.DataFrame,
                       val_data: pd.DataFrame,
                       test_data: pd.DataFrame,
                       config: Dict,
                       feature_cols: List[str] = None,
                       target_cols: List[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создание DataLoader'ов для обучения"""
    
    logger = get_logger("DataLoaders")
    
    batch_size = config['model']['batch_size']
    context_window = config['model']['context_window']
    pred_window = config['model']['pred_len']
    num_workers = config['performance']['num_workers']
    persistent_workers = config['performance'].get('persistent_workers', True) if num_workers > 0 else False
    prefetch_factor = config['performance'].get('prefetch_factor', 2)
    
    logger.info("📦 Создание унифицированных датасетов...")
    
    # Проверка валидности данных
    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise ValueError("❌ Один из датасетов пуст!")
    
    # Логирование структуры данных
    logger.info(f"📊 Исходные данные:")
    logger.info(f"   - Train: {len(train_data):,} записей, {len(train_data.columns)} колонок")
    logger.info(f"   - Val: {len(val_data):,} записей, {len(val_data.columns)} колонок") 
    logger.info(f"   - Test: {len(test_data):,} записей, {len(test_data.columns)} колонок")
    
    if feature_cols:
        logger.info(f"   - Признаков: {len(feature_cols)}")
    if target_cols:
        logger.info(f"   - Целевых переменных: {len(target_cols)}")
    
    # Получаем параметры нормализации из конфига
    normalize = config.get('data', {}).get('normalize', True)
    scaler_path = config.get('data', {}).get('scaler_path', 'models_saved/data_scaler.pkl')
    
    # Получаем параметры DataLoader из конфига
    pin_memory = config['performance'].get('dataloader_pin_memory', True)
    drop_last = config['performance'].get('dataloader_drop_last', True)
    
    # Получаем параметры stride из конфига
    train_stride = config.get('data', {}).get('train_stride', 8)  # Увеличен для ускорения
    val_stride = config.get('data', {}).get('val_stride', 16)    # Еще больше для валидации
    
    # Проверка наличия scaler для решения о необходимости обучения
    from pathlib import Path
    scaler_exists = Path(scaler_path).exists()
    if scaler_exists:
        logger.info(f"✅ Найден существующий scaler: {scaler_path}")
    else:
        logger.info(f"⚠️ Scaler не найден, будет создан новый: {scaler_path}")
    
    # Создание датасетов с проверкой ошибок
    try:
        train_dataset = TradingDataset(
            data=train_data,
            config=config,
            context_window=context_window,
            prediction_window=pred_window,
            feature_cols=feature_cols,
            target_cols=target_cols,
            stride=train_stride,  # Используем stride из конфига для ускорения
            normalize=normalize,
            scaler_path=scaler_path,
            fit_scaler=not scaler_exists  # Обучаем scaler только если его нет
        )
        
        val_dataset = TradingDataset(
            data=val_data,
            config=config,
            context_window=context_window,
            prediction_window=pred_window,
            feature_cols=feature_cols,
            target_cols=target_cols,
            stride=val_stride,  # Используем stride из конфига
            normalize=normalize,
            scaler_path=scaler_path,
            fit_scaler=False  # Используем уже обученный scaler
        )
        
        test_dataset = TradingDataset(
            data=test_data,
            config=config,
            context_window=context_window,
            prediction_window=pred_window,
            feature_cols=feature_cols,
            target_cols=target_cols,
            stride=4,
            normalize=normalize,
            scaler_path=scaler_path,
            fit_scaler=False  # Используем уже обученный scaler
        )
    except Exception as e:
        logger.error(f"❌ Ошибка при создании датасетов: {e}")
        raise
    
    logger.info(f"Размеры датасетов после создания окон:")
    logger.info(f"   - Train: {len(train_dataset):,} окон (из {len(train_data):,} записей)")
    logger.info(f"   - Val: {len(val_dataset):,} окон (из {len(val_data):,} записей, stride={val_dataset.stride})")
    logger.info(f"   - Test: {len(test_dataset):,} окон (из {len(test_data):,} записей, stride={test_dataset.stride})")
    
    # Создание DataLoader'ов с оптимизацией для GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,  # Также включаем для валидации для стабильности
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Для теста оставляем все данные
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader


def collate_trading_batch(batch):
    """Кастомная функция для объединения батчей с дополнительной информацией"""
    X_list, y_list, info_list = zip(*batch)
    
    # Стекирование тензоров
    X = torch.stack(X_list)
    y = torch.stack(y_list) if len(y_list[0]) > 0 else None
    
    # Объединение информации
    batch_info = {
        'symbols': [info['symbol'] for info in info_list],
        'last_closes': torch.tensor([info['last_close'] for info in info_list]),
        'atrs': torch.tensor([info['atr'] for info in info_list])
    }
    
    if 'price_data' in info_list[0]:
        batch_info['price_data'] = torch.stack([info['price_data'] for info in info_list])
    
    return X, y, batch_info
