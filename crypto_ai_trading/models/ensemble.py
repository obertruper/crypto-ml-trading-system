"""
Ансамблирование моделей для улучшения предсказаний
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Callable
import numpy as np
from pathlib import Path
import pickle
from abc import ABC, abstractmethod

from utils.logger import get_logger
from .patchtst import PatchTSTForPrediction

class BaseEnsemble(nn.Module, ABC):
    """Базовый класс для ансамблей"""
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.logger = get_logger("Ensemble")
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Прямой проход через ансамбль"""
        pass
    
    def load_models(self, model_paths: List[str], model_class: type, config: Dict):
        """Загрузка моделей из checkpoint'ов"""
        loaded_models = []
        
        for path in model_paths:
            checkpoint = torch.load(path)
            model = model_class(**config)
            model.load_state_dict(checkpoint['model_state_dict'])
            loaded_models.append(model)
            
        self.models = nn.ModuleList(loaded_models)
        self.logger.info(f"Загружено {len(loaded_models)} моделей в ансамбль")


class VotingEnsemble(BaseEnsemble):
    """Ансамбль с голосованием"""
    
    def __init__(self, 
                 models: List[nn.Module],
                 weights: Optional[List[float]] = None,
                 voting: str = 'soft'):
        """
        Args:
            models: список моделей
            weights: веса для взвешенного голосования
            voting: 'soft' или 'hard' голосование
        """
        super().__init__(models)
        self.voting = voting
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Нормализация весов
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Усреднение предсказаний всех моделей"""
        all_predictions = []
        
        # Получаем предсказания от каждой модели
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                all_predictions.append(pred)
        
        # Усреднение предсказаний
        if self.voting == 'soft':
            # Взвешенное среднее для всех выходов
            ensemble_pred = {}
            
            for key in all_predictions[0].keys():
                weighted_preds = []
                for i, pred in enumerate(all_predictions):
                    weighted_preds.append(self.weights[i] * pred[key])
                
                ensemble_pred[key] = torch.stack(weighted_preds).sum(dim=0)
        else:
            # Hard voting для классификации
            ensemble_pred = self._hard_voting(all_predictions)
        
        return ensemble_pred
    
    def _hard_voting(self, predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Жесткое голосование для классификационных выходов"""
        ensemble_pred = {}
        
        # Для вероятностей TP/SL используем голосование по порогу
        tp_votes = []
        for pred in predictions:
            tp_probs = pred['tp_probs']
            tp_votes.append((tp_probs > 0.5).float())
        
        # Большинство голосов
        tp_votes_stacked = torch.stack(tp_votes)
        ensemble_pred['tp_probs'] = tp_votes_stacked.mean(dim=0)
        
        # Для остальных выходов используем среднее
        for key in ['price_pred', 'sl_prob', 'volatility']:
            if key in predictions[0]:
                values = [pred[key] for pred in predictions]
                ensemble_pred[key] = torch.stack(values).mean(dim=0)
        
        return ensemble_pred


class StackingEnsemble(BaseEnsemble):
    """Стекинг ансамбль с мета-моделью"""
    
    def __init__(self,
                 base_models: List[nn.Module],
                 meta_model: nn.Module,
                 use_original_features: bool = True):
        """
        Args:
            base_models: базовые модели
            meta_model: мета-модель для комбинирования
            use_original_features: использовать исходные признаки в мета-модели
        """
        super().__init__(base_models)
        self.meta_model = meta_model
        self.use_original_features = use_original_features
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Двухуровневое предсказание"""
        # Уровень 1: предсказания базовых моделей
        base_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                # Преобразуем предсказания в вектор признаков
                features = self._extract_features(pred)
                base_predictions.append(features)
        
        # Объединяем предсказания базовых моделей
        stacked_features = torch.cat(base_predictions, dim=-1)
        
        # Добавляем исходные признаки если нужно
        if self.use_original_features:
            # Flatten исходные признаки
            original_flat = x.view(x.size(0), -1)
            stacked_features = torch.cat([stacked_features, original_flat], dim=-1)
        
        # Уровень 2: мета-модель
        ensemble_pred = self.meta_model(stacked_features)
        
        return ensemble_pred
    
    def _extract_features(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Извлечение признаков из предсказаний для мета-модели"""
        features = []
        
        # Преобразуем все предсказания в плоский вектор
        for key, value in predictions.items():
            if value.dim() > 2:
                value = value.view(value.size(0), -1)
            features.append(value)
        
        return torch.cat(features, dim=-1)


class BaggingEnsemble(BaseEnsemble):
    """Bootstrap Aggregating ансамбль"""
    
    def __init__(self,
                 model_class: type,
                 n_models: int,
                 model_config: Dict,
                 bootstrap_ratio: float = 0.8):
        """
        Args:
            model_class: класс модели для создания
            n_models: количество моделей в ансамбле
            model_config: конфигурация для моделей
            bootstrap_ratio: доля данных для каждой модели
        """
        # Создаем модели
        models = [model_class(**model_config) for _ in range(n_models)]
        super().__init__(models)
        
        self.bootstrap_ratio = bootstrap_ratio
        self.model_class = model_class
        self.model_config = model_config
        
    def train_ensemble(self, train_data, val_data, training_function: Callable):
        """Обучение всех моделей ансамбля на bootstrap выборках"""
        
        n_samples = len(train_data)
        bootstrap_size = int(n_samples * self.bootstrap_ratio)
        
        for i, model in enumerate(self.models):
            self.logger.info(f"Обучение модели {i+1}/{len(self.models)}")
            
            # Bootstrap выборка
            indices = np.random.choice(n_samples, bootstrap_size, replace=True)
            bootstrap_train = train_data[indices]
            
            # Обучение модели
            training_function(model, bootstrap_train, val_data)
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Усреднение предсказаний"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Простое усреднение
        ensemble_pred = {}
        for key in predictions[0].keys():
            values = [pred[key] for pred in predictions]
            ensemble_pred[key] = torch.stack(values).mean(dim=0)
            
            # Добавляем стандартное отклонение как меру неопределенности
            ensemble_pred[f'{key}_std'] = torch.stack(values).std(dim=0)
        
        return ensemble_pred


class DynamicEnsemble(BaseEnsemble):
    """Динамический ансамбль с адаптивными весами"""
    
    def __init__(self,
                 models: List[nn.Module],
                 performance_window: int = 100,
                 update_frequency: int = 10):
        """
        Args:
            models: список моделей
            performance_window: окно для оценки производительности
            update_frequency: частота обновления весов
        """
        super().__init__(models)
        
        self.performance_window = performance_window
        self.update_frequency = update_frequency
        
        # Инициализация весов и метрик
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        self.performance_history = {i: [] for i in range(len(models))}
        self.update_counter = 0
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Взвешенное предсказание с динамическими весами"""
        predictions = []
        
        # Нормализация весов через softmax
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Взвешенное усреднение
        ensemble_pred = {}
        for key in predictions[0].keys():
            weighted_preds = []
            for i, pred in enumerate(predictions):
                weighted_preds.append(normalized_weights[i] * pred[key])
            
            ensemble_pred[key] = torch.stack(weighted_preds).sum(dim=0)
        
        return ensemble_pred
    
    def update_weights(self, performances: List[float]):
        """Обновление весов на основе производительности"""
        self.update_counter += 1
        
        # Сохраняем историю производительности
        for i, perf in enumerate(performances):
            self.performance_history[i].append(perf)
            # Ограничиваем размер истории
            if len(self.performance_history[i]) > self.performance_window:
                self.performance_history[i].pop(0)
        
        # Обновляем веса если пришло время
        if self.update_counter % self.update_frequency == 0:
            # Средняя производительность за окно
            avg_performances = []
            for i in range(len(self.models)):
                if self.performance_history[i]:
                    avg_perf = np.mean(self.performance_history[i])
                else:
                    avg_perf = 0.5  # Начальное значение
                avg_performances.append(avg_perf)
            
            # Обновляем веса пропорционально производительности
            new_weights = torch.tensor(avg_performances, dtype=torch.float32)
            self.weights.data = new_weights / new_weights.sum()
            
            self.logger.info(f"Обновлены веса ансамбля: {self.weights.data.tolist()}")


class TemporalEnsemble(BaseEnsemble):
    """Временной ансамбль для разных горизонтов прогнозирования"""
    
    def __init__(self,
                 short_term_models: List[nn.Module],
                 medium_term_models: List[nn.Module],
                 long_term_models: List[nn.Module],
                 horizons: Dict[str, int]):
        """
        Args:
            short_term_models: модели для краткосрочных прогнозов
            medium_term_models: модели для среднесрочных прогнозов
            long_term_models: модели для долгосрочных прогнозов
            horizons: границы горизонтов в шагах времени
        """
        all_models = short_term_models + medium_term_models + long_term_models
        super().__init__(all_models)
        
        self.short_term_models = nn.ModuleList(short_term_models)
        self.medium_term_models = nn.ModuleList(medium_term_models)
        self.long_term_models = nn.ModuleList(long_term_models)
        self.horizons = horizons
        
    def forward(self, x: torch.Tensor, target_horizon: int) -> Dict[str, torch.Tensor]:
        """Выбор моделей на основе горизонта прогнозирования"""
        
        # Определяем какие модели использовать
        if target_horizon <= self.horizons['short']:
            models_to_use = self.short_term_models
        elif target_horizon <= self.horizons['medium']:
            models_to_use = self.medium_term_models
        else:
            models_to_use = self.long_term_models
        
        # Получаем предсказания от выбранных моделей
        predictions = []
        for model in models_to_use:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Усреднение
        ensemble_pred = {}
        for key in predictions[0].keys():
            values = [pred[key] for pred in predictions]
            ensemble_pred[key] = torch.stack(values).mean(dim=0)
        
        return ensemble_pred


def create_ensemble(ensemble_type: str, 
                   models: List[nn.Module],
                   config: Dict) -> BaseEnsemble:
    """Фабрика для создания ансамблей"""
    
    ensemble_types = {
        'voting': VotingEnsemble,
        'stacking': StackingEnsemble,
        'bagging': BaggingEnsemble,
        'dynamic': DynamicEnsemble,
        'temporal': TemporalEnsemble
    }
    
    if ensemble_type not in ensemble_types:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    # Специальная обработка для разных типов
    if ensemble_type == 'voting':
        return VotingEnsemble(
            models=models,
            weights=config.get('weights'),
            voting=config.get('voting_method', 'soft')
        )
    elif ensemble_type == 'stacking':
        # Нужна мета-модель
        meta_model = config.get('meta_model')
        if meta_model is None:
            raise ValueError("Stacking ensemble requires meta_model in config")
        return StackingEnsemble(
            base_models=models,
            meta_model=meta_model,
            use_original_features=config.get('use_original_features', True)
        )
    elif ensemble_type == 'dynamic':
        return DynamicEnsemble(
            models=models,
            performance_window=config.get('performance_window', 100),
            update_frequency=config.get('update_frequency', 10)
        )
    else:
        return ensemble_types[ensemble_type](models=models, **config)