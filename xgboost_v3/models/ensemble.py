"""
Ансамблевые методы для XGBoost моделей
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score
import joblib

from models.xgboost_trainer import XGBoostTrainer
from config import Config
from config.constants import ENSEMBLE_PARAMS, EPSILON_STD

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Класс для создания ансамбля XGBoost моделей"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = []
        self.weights = None
        self.model_metrics = []
        
    def train_ensemble(self, 
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_val: pd.DataFrame,
                      y_val: pd.Series,
                      n_models: Optional[int] = None) -> List[XGBoostTrainer]:
        """
        Обучение ансамбля моделей
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
            X_val: Валидационные признаки  
            y_val: Валидационные метки
            n_models: Количество моделей в ансамбле
            
        Returns:
            Список обученных моделей
        """
        if n_models is None:
            n_models = self.config.training.ensemble_size
            
        logger.info(f"\n🎯 Обучение ансамбля из {n_models} моделей")
        
        for i in range(n_models):
            logger.info(f"\n📌 Модель {i+1}/{n_models}")
            
            # Создаем модель с разными параметрами
            model_params = self._get_diverse_params(i)
            
            # Создаем подвыборку данных для разнообразия
            X_train_sub, y_train_sub = self._create_subsample(X_train, y_train, seed=i)
            
            # Обучаем модель
            trainer = XGBoostTrainer(
                config=self.config,
                model_name=f"{self.config.training.task_type}_model_{i}"
            )
            
            trainer.train(
                X_train_sub, y_train_sub,
                X_val, y_val,
                model_params=model_params
            )
            
            # Оцениваем модель
            metrics = trainer.evaluate(X_val, y_val, f"Validation (Model {i+1})")
            
            self.models.append(trainer)
            self.model_metrics.append(metrics)
            
        # Вычисляем веса моделей
        self._calculate_weights(X_val, y_val)
        
        logger.info(f"\n✅ Ансамбль из {n_models} моделей обучен")
        self._log_ensemble_performance()
        
        return self.models
        
    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Предсказание ансамбля
        
        Args:
            X: Признаки для предсказания
            return_proba: Вернуть вероятности
            
        Returns:
            Усредненные предсказания ансамбля
        """
        if not self.models:
            raise ValueError("Ансамбль не обучен")
            
        predictions = []
        
        for model in self.models:
            pred = model.predict(X, return_proba=True)
            predictions.append(pred)
            
        # Объединяем предсказания
        predictions = np.array(predictions)
        
        if self.config.training.ensemble_method == "weighted":
            # Взвешенное среднее
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        elif self.config.training.ensemble_method == "voting":
            # Голосование для классификации
            if self.config.training.task_type != "regression":
                # Мажоритарное голосование
                threshold = ENSEMBLE_PARAMS['voting_threshold']
                binary_preds = (predictions > threshold).astype(int)
                ensemble_pred = np.mean(binary_preds, axis=0)
                if not return_proba:
                    ensemble_pred = (ensemble_pred > threshold).astype(int)
            else:
                ensemble_pred = np.mean(predictions, axis=0)
        else:
            # Простое среднее
            ensemble_pred = np.mean(predictions, axis=0)
            
        if not return_proba and self.config.training.task_type == "classification_binary":
            ensemble_pred = (ensemble_pred > ENSEMBLE_PARAMS['voting_threshold']).astype(int)
            
        return ensemble_pred
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> Dict[str, float]:
        """Оценка ансамбля"""
        predictions = self.predict(X, return_proba=True)
        
        # Используем metrics calculator первой модели
        if self.models:
            metrics = self.models[0].metrics_calculator.calculate_classification_metrics(y, predictions)
            
            logger.info(f"\n📊 Метрики ансамбля на {dataset_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"   {metric_name}: {value:.4f}")
                    
            return metrics
            
        return {}
        
    def save_ensemble(self, save_dir: str):
        """Сохранение всех моделей ансамбля"""
        logger.info(f"\n💾 Сохранение ансамбля...")
        
        for i, model in enumerate(self.models):
            model.save_model(save_dir)
            
        # Конвертируем numpy типы в Python типы для JSON сериализации
        def convert_to_native_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(i) for i in obj]
            else:
                return obj
                
        # Сохраняем метаданные ансамбля
        ensemble_metadata = {
            'n_models': len(self.models),
            'weights': self.weights.tolist() if self.weights is not None else None,
            'model_metrics': convert_to_native_types(self.model_metrics),
            'ensemble_method': self.config.training.ensemble_method
        }
        
        import json
        from pathlib import Path
        
        metadata_path = Path(save_dir) / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2, default=str)
            
        logger.info(f"✅ Ансамбль сохранен в {save_dir}")
        
    def _get_diverse_params(self, model_idx: int) -> Dict:
        """Получение разнообразных параметров для моделей ансамбля"""
        base_params = self.config.model.to_dict()
        
        # Варьируем параметры для разнообразия
        variations = ENSEMBLE_PARAMS['model_variations']
        
        # Применяем вариацию
        variation = variations[model_idx % len(variations)]
        base_params.update(variation)
        
        # Разные random_state для разнообразия
        base_params['random_state'] = 42 + model_idx
        
        return base_params
        
    def _create_subsample(self, X: pd.DataFrame, y: pd.Series, 
                         subsample_ratio: float = None, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """Создание подвыборки для обучения"""
        if subsample_ratio is None:
            subsample_ratio = ENSEMBLE_PARAMS['subsample_ratio']
            
        np.random.seed(seed)
        
        n_samples = int(len(X) * subsample_ratio)
        # Bootstrap sampling если включено
        replace = ENSEMBLE_PARAMS.get('bootstrap', True)
        indices = np.random.choice(len(X), n_samples, replace=replace)
        
        return X.iloc[indices], y.iloc[indices]
        
    def _calculate_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Вычисление весов моделей на основе их производительности"""
        if self.config.training.ensemble_method != "weighted":
            return
            
        logger.info("\n📊 Вычисление весов моделей...")
        
        scores = []
        
        for i, model in enumerate(self.models):
            pred = model.predict(X_val, return_proba=True)
            
            if self.config.training.task_type == "regression":
                # Для регрессии используем отрицательный MAE
                from sklearn.metrics import mean_absolute_error
                score = -mean_absolute_error(y_val, pred)
            else:
                # Для классификации используем ROC-AUC
                try:
                    score = roc_auc_score(y_val, pred)
                except:
                    score = 0.5
                    
            scores.append(score)
            logger.info(f"   Модель {i+1}: score = {score:.4f}")
            
        # Нормализуем веса
        scores = np.array(scores)
        
        # Проверяем разброс scores
        norm_params = ENSEMBLE_PARAMS['score_normalization']
        smoothing_params = ENSEMBLE_PARAMS['weight_smoothing']
        
        if scores.std() < norm_params['similarity_threshold']:  # Если модели слишком похожи
            logger.warning("⚠️ Модели показывают очень близкие результаты, используем равные веса")
            self.weights = np.ones(len(scores)) / len(scores)
        else:
            # Используем softmax для более сбалансированного взвешивания
            # Масштабируем scores для избежания экстремальных весов
            scores_normalized = (scores - scores.mean()) / (scores.std() + EPSILON_STD)
            # Ограничиваем диапазон для стабильности
            scores_normalized = np.clip(scores_normalized, 
                                       norm_params['clip_min'], 
                                       norm_params['clip_max'])
            # Применяем softmax
            exp_scores = np.exp(scores_normalized)
            self.weights = exp_scores / exp_scores.sum()
            
            # Проверяем на экстремальные веса
            if self.weights.max() > smoothing_params['extreme_weight_threshold']:
                logger.warning("⚠️ Обнаружены экстремальные веса, применяем сглаживание")
                # Сглаживаем веса
                uniform_weights = np.ones(len(scores)) / len(scores)
                self.weights = (smoothing_params['model_weight'] * self.weights + 
                               smoothing_params['uniform_weight'] * uniform_weights)
        
        logger.info(f"   Веса: {self.weights}")
        
    def _log_ensemble_performance(self):
        """Логирование производительности ансамбля"""
        logger.info("\n📊 Сводка по моделям ансамбля:")
        
        # Собираем ключевые метрики
        summary = []
        for i, metrics in enumerate(self.model_metrics):
            if self.config.training.task_type == "regression":
                key_metric = metrics.get('mae', np.inf)
                metric_name = "MAE"
            else:
                key_metric = metrics.get('roc_auc', 0)
                metric_name = "ROC-AUC"
                
            summary.append({
                'Model': f"Model {i+1}",
                metric_name: key_metric,
                'Weight': self.weights[i] if self.weights is not None else 1/len(self.models)
            })
            
        summary_df = pd.DataFrame(summary)
        logger.info(f"\n{summary_df.to_string(index=False)}")
        
        # Средняя производительность
        avg_metric = summary_df[metric_name].mean()
        logger.info(f"\nСредний {metric_name}: {avg_metric:.4f}")