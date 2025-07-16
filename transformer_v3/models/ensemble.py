"""
Ансамблирование TFT моделей
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import logging
from typing import List, Dict, Optional
from pathlib import Path

from config import Config
from models.tft_trainer import TFTTrainer

logger = logging.getLogger(__name__)


class TFTEnsemble:
    """Ансамбль из нескольких TFT моделей"""
    
    def __init__(self, config: Config, base_name: str = "tft"):
        self.config = config
        self.base_name = base_name
        self.models = []
        self.trainers = []
        self.weights = None
        
    def train_ensemble(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      n_models: int = 3,
                      feature_columns: List[str] = None) -> List[keras.Model]:
        """
        Обучение ансамбля моделей
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            X_val: Валидационные данные
            y_val: Валидационные метки
            n_models: Количество моделей в ансамбле
            feature_columns: Названия признаков
            
        Returns:
            Список обученных моделей
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Обучение ансамбля из {n_models} моделей")
        logger.info(f"{'='*60}")
        
        self.models = []
        self.trainers = []
        val_scores = []
        
        for i in range(n_models):
            logger.info(f"\n📊 Модель {i+1}/{n_models}")
            
            # Создаем trainer с уникальным именем
            model_name = f"{self.base_name}_model_{i+1}"
            trainer = TFTTrainer(self.config, model_name=model_name)
            
            # Добавляем случайность для разнообразия
            # 1. Bootstrap sampling
            if i > 0:  # Первая модель на полных данных
                indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_train_boot = X_train[indices]
                y_train_boot = y_train[indices]
            else:
                X_train_boot = X_train
                y_train_boot = y_train
            
            # 2. Разные инициализации (через seed)
            tf.random.set_seed(42 + i)
            np.random.seed(42 + i)
            
            # Обучаем модель
            model = trainer.train(
                X_train_boot, y_train_boot,
                X_val, y_val,
                feature_columns=feature_columns
            )
            
            # Оцениваем на валидации
            val_metrics = trainer.evaluate(X_val, y_val, f"Validation (Model {i+1})")
            
            # Сохраняем score для взвешивания
            if self.config.training.task_type == 'regression':
                val_score = -val_metrics['mae']  # Минимизируем MAE
            else:
                val_score = val_metrics['accuracy']  # Максимизируем accuracy
                
            val_scores.append(val_score)
            
            self.models.append(model)
            self.trainers.append(trainer)
        
        # Вычисляем веса для взвешенного усреднения
        self._calculate_weights(val_scores)
        
        logger.info(f"\n✅ Ансамбль обучен. Веса моделей: {self.weights}")
        
        return self.models
    
    def _calculate_weights(self, scores: List[float]):
        """Вычисление весов моделей на основе их производительности"""
        scores = np.array(scores)
        
        # Нормализуем scores в положительный диапазон
        min_score = scores.min()
        if min_score < 0:
            scores = scores - min_score
        
        # Softmax для получения весов
        exp_scores = np.exp(scores)
        self.weights = exp_scores / exp_scores.sum()
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """
        Предсказание ансамбля
        
        Args:
            X: Данные для предсказания
            return_proba: Вернуть вероятности
            
        Returns:
            Усредненные предсказания
        """
        if not self.models:
            raise ValueError("Ансамбль не обучен")
        
        predictions = []
        
        for i, model in enumerate(self.models):
            pred = model.predict(X, verbose=0)
            predictions.append(pred.flatten())
        
        # Взвешенное усреднение
        predictions = np.array(predictions)
        
        if self.weights is not None:
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        # Применяем порог для классификации
        if self.config.training.task_type == "classification_binary" and not return_proba:
            threshold = self.config.training.classification_threshold / 100
            ensemble_pred = (ensemble_pred > threshold).astype(int)
        
        return ensemble_pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, dataset_name: str = "Test") -> Dict[str, float]:
        """Оценка ансамбля"""
        predictions = self.predict(X, return_proba=True)
        
        # Используем метрики от первого trainer
        if self.trainers:
            metrics = self.trainers[0].metrics_calculator.calculate_metrics(
                y, predictions, 
                task_type=self.config.training.task_type
            )
        else:
            metrics = {}
        
        logger.info(f"\n📊 Метрики ансамбля на {dataset_name}:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {metric_name}: {value:.4f}")
        
        return metrics
    
    def save_ensemble(self, save_dir: Path):
        """Сохранение всех моделей ансамбля"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n💾 Сохранение ансамбля в {save_dir}")
        
        # Сохраняем каждую модель
        for i, trainer in enumerate(self.trainers):
            model_dir = save_dir / f"model_{i+1}"
            trainer.save_model(model_dir)
        
        # Сохраняем метаданные ансамбля
        ensemble_meta = {
            'n_models': len(self.models),
            'weights': self.weights.tolist() if self.weights is not None else None,
            'base_name': self.base_name,
            'task_type': self.config.training.task_type
        }
        
        import json
        with open(save_dir / "ensemble_metadata.json", 'w') as f:
            json.dump(ensemble_meta, f, indent=2)
        
        logger.info("✅ Ансамбль сохранен")
    
    def load_ensemble(self, load_dir: Path):
        """Загрузка ансамбля"""
        load_dir = Path(load_dir)
        
        # Загружаем метаданные
        import json
        with open(load_dir / "ensemble_metadata.json", 'r') as f:
            meta = json.load(f)
        
        self.weights = np.array(meta['weights']) if meta['weights'] else None
        n_models = meta['n_models']
        
        # Загружаем модели
        self.models = []
        self.trainers = []
        
        for i in range(n_models):
            model_dir = load_dir / f"model_{i+1}"
            model_path = model_dir / f"{self.base_name}_model_{i+1}.h5"
            
            trainer = TFTTrainer(self.config, model_name=f"{self.base_name}_model_{i+1}")
            trainer.load_model(model_path)
            
            self.models.append(trainer.model)
            self.trainers.append(trainer)
        
        logger.info(f"✅ Загружен ансамбль из {n_models} моделей")