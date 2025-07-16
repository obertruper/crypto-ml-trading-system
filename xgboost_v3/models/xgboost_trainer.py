"""
Основной класс для обучения XGBoost моделей
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Union, List
from pathlib import Path
import joblib
import json

from config import Config
from config.constants import PREDICTION_PARAMS
from utils.metrics import MetricsCalculator
from models.data_balancer import DataBalancer

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """Класс для обучения XGBoost моделей"""
    
    def __init__(self, config: Config, model_name: str = "xgboost_model"):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.metrics_calculator = MetricsCalculator(config)
        self.data_balancer = DataBalancer(config)
        self.feature_importance = None
        self.optimal_threshold = None  # Сохраняем оптимальный порог
        self.training_history = {
            'train': {'loss': [], 'metric': []},
            'val': {'loss': [], 'metric': []}
        }
        
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             X_val: pd.DataFrame,
             y_val: pd.Series,
             model_params: Optional[Dict] = None) -> xgb.Booster:
        """
        Обучение модели XGBoost
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
            X_val: Валидационные признаки
            y_val: Валидационные метки
            model_params: Параметры модели (если None, используются из конфига)
            
        Returns:
            Обученная модель XGBoost
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Обучение модели: {self.model_name}")
        logger.info(f"{'='*60}")
        
        # Балансировка классов если нужно
        if self.config.training.task_type != "regression" and self.config.training.balance_method != "none":
            X_train, y_train = self.data_balancer.balance_data(X_train, y_train, is_classification=True)
            
        # Параметры модели
        if model_params is None:
            model_params = self.config.model.to_dict()
            
        # Автоматическое вычисление scale_pos_weight для несбалансированных классов
        if self.config.training.task_type == "classification_binary" and model_params.get('scale_pos_weight') is None:
            scale_pos_weight = self._calculate_scale_pos_weight(y_train)
            model_params['scale_pos_weight'] = scale_pos_weight
            
        # Настройка GPU если доступен
        if self._check_gpu_available():
            model_params['tree_method'] = 'gpu_hist'
            model_params['predictor'] = 'gpu_predictor'
            logger.info("🖥️ Используется GPU для обучения")
        else:
            model_params['tree_method'] = 'hist'
            model_params['predictor'] = 'cpu_predictor'
            
        # Создание DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_val.columns))
        
        # Callback для сохранения истории
        evals_result = {}
        
        # Обучение модели
        logger.info("🚀 Начало обучения...")
        logger.info(f"   Размер обучающей выборки: {X_train.shape}")
        logger.info(f"   Размер валидационной выборки: {X_val.shape}")
        
        self.model = xgb.train(
            params=model_params,
            dtrain=dtrain,
            num_boost_round=self.config.model.n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            verbose_eval=100,
            evals_result=evals_result
        )
        
        # Сохраняем историю обучения
        self._save_training_history(evals_result)
        
        # Вычисляем feature importance
        self._calculate_feature_importance()
        
        # Оценка на валидации
        val_metrics = self.evaluate(X_val, y_val, dataset_name="Validation")
        
        # Дополнительная диагностика для классификации
        if self.config.training.task_type == "classification_binary":
            val_proba = self.predict(X_val, return_proba=True)
            logger.info(f"\n📊 Диагностика предсказаний на валидации:")
            logger.info(f"   Распределение вероятностей: min={val_proba.min():.3f}, max={val_proba.max():.3f}, mean={val_proba.mean():.3f}")
            logger.info(f"   Квантили: 25%={np.percentile(val_proba, 25):.3f}, 50%={np.percentile(val_proba, 50):.3f}, 75%={np.percentile(val_proba, 75):.3f}")
            
            # Проверяем распределение с разными порогами
            for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
                n_pos = (val_proba > thr).sum()
                logger.info(f"   Порог {thr:.1f}: {n_pos} положительных ({n_pos/len(val_proba)*100:.1f}%)")
        
        logger.info(f"\n✅ Обучение завершено. Best iteration: {self.model.best_iteration}")
        
        return self.model
        
    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Предсказание модели
        
        Args:
            X: Признаки для предсказания
            return_proba: Вернуть вероятности (для классификации)
            
        Returns:
            Предсказания модели
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
            
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        predictions = self.model.predict(dmatrix)
        
        if self.config.training.task_type == "classification_binary" and not return_proba:
            # Применяем порог вероятности для бинарной классификации
            # Используем оптимальный порог если он был найден, иначе - дефолтный
            threshold = self.optimal_threshold if self.optimal_threshold is not None else PREDICTION_PARAMS['probability_threshold']
            
            # Логирование для диагностики (до преобразования)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"📊 Распределение вероятностей при предсказании:")
                logger.info(f"   Min: {predictions.min():.3f}, Max: {predictions.max():.3f}, Mean: {predictions.mean():.3f}")
                n_positive = (predictions > threshold).sum()
                logger.info(f"   Порог: {threshold:.3f} {'(оптимальный)' if self.optimal_threshold is not None else '(дефолтный)'}")
                logger.info(f"   Предсказано класс 1: {n_positive} из {len(predictions)} ({n_positive/len(predictions)*100:.1f}%)")
            
            predictions = (predictions > threshold).astype(int)
            
        return predictions
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> Dict[str, float]:
        """
        Оценка модели на данных
        
        Args:
            X: Признаки
            y: Истинные метки
            dataset_name: Название датасета для логирования
            
        Returns:
            Словарь с метриками
        """
        predictions = self.predict(X, return_proba=True)
        
        if self.config.training.task_type == "regression":
            metrics = self.metrics_calculator.calculate_regression_metrics(y, predictions)
        else:
            # Для классификации используем вероятности и оптимальный порог
            metrics = self.metrics_calculator.calculate_classification_metrics(y, predictions)
            
            # Сохраняем оптимальный порог если это валидация
            if dataset_name == "Validation" and 'threshold' in metrics:
                self.optimal_threshold = metrics['threshold']
                logger.info(f"💡 Сохранен оптимальный порог: {self.optimal_threshold:.4f}")
            
        # Логирование результатов
        logger.info(f"\n📊 Метрики на {dataset_name}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"   {metric_name}: {value:.4f}")
                
        return metrics
        
    def save_model(self, save_dir: Path):
        """Сохранение модели и метаданных"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем модель
        model_path = save_dir / f"{self.model_name}.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"   ✅ {model_path.name}")
        
        # Сохраняем метаданные
        metadata = {
            'model_name': self.model_name,
            'task_type': self.config.training.task_type,
            'best_iteration': self.model.best_iteration if self.model else None,
            'optimal_threshold': self.optimal_threshold,  # Сохраняем оптимальный порог
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__
            }
        }
        
        # Конвертируем numpy типы в обычные Python типы для JSON
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(v) for v in obj]
            return obj
        
        metadata = convert_to_json_serializable(metadata)
        
        metadata_path = save_dir / f"{self.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def load_model(self, model_path: Path):
        """Загрузка модели"""
        self.model = joblib.load(model_path)
        logger.info(f"✅ Модель загружена из {model_path}")
        
        # Пытаемся загрузить метаданные
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.optimal_threshold = metadata.get('optimal_threshold')
                if self.optimal_threshold:
                    logger.info(f"   💡 Загружен оптимальный порог: {self.optimal_threshold:.4f}")
        
    def _calculate_scale_pos_weight(self, y_train: pd.Series) -> float:
        """Расчет scale_pos_weight для балансировки классов"""
        n_positive = (y_train == 1).sum()
        n_negative = (y_train == 0).sum()
        
        if n_positive == 0:
            logger.warning("⚠️ Нет положительных примеров в обучающей выборке!")
            return 1.0
            
        scale_pos_weight = n_negative / n_positive
        
        # Адаптивное ограничение максимального значения
        max_scale = min(3.0, n_negative / n_positive * 0.7)  # Ограничиваем до 3 и берем 70% от реального соотношения
        if scale_pos_weight > max_scale:
            logger.warning(f"⚠️ Очень высокий scale_pos_weight: {scale_pos_weight:.2f}, ограничиваем до {max_scale:.2f}")
            scale_pos_weight = max_scale
        
        logger.info(f"📊 Распределение классов:")
        logger.info(f"   Класс 0 (не входить): {n_negative:,} ({n_negative/(n_negative+n_positive)*100:.1f}%)")
        logger.info(f"   Класс 1 (входить): {n_positive:,} ({n_positive/(n_negative+n_positive)*100:.1f}%)")
        logger.info(f"   scale_pos_weight = {scale_pos_weight:.2f}")
        
        return scale_pos_weight
        
    def _check_gpu_available(self) -> bool:
        """Проверка доступности GPU"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except:
            return False
            
    def _save_training_history(self, evals_result: Dict):
        """Сохранение истории обучения"""
        # XGBoost возвращает историю в формате {dataset: {metric: [values]}}
        for dataset in evals_result:
            for metric in evals_result[dataset]:
                if dataset == 'train':
                    self.training_history['train']['metric'] = evals_result[dataset][metric]
                elif dataset == 'val':
                    self.training_history['val']['metric'] = evals_result[dataset][metric]
                    
    def _calculate_feature_importance(self):
        """Вычисление важности признаков"""
        if self.model is None:
            return
            
        try:
            # Пробуем разные методы получения важности
            importance = self.model.get_score(importance_type='gain')
            
            if not importance:
                # Если gain не работает, пробуем weight
                importance = self.model.get_score(importance_type='weight')
                
            if not importance:
                # Если и weight не работает, пробуем cover
                importance = self.model.get_score(importance_type='cover')
                
            if not importance:
                # Если ничего не работает, используем get_fscore
                importance = self.model.get_fscore()
                
            if not importance:
                logger.warning("⚠️ Не удалось получить feature importance из модели")
                self.feature_importance = {}
                return
                
            # Сортируем по важности
            self.feature_importance = dict(sorted(importance.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True))
            
            # Логируем топ-10 признаков
            logger.info("\n📊 Топ-10 важных признаков:")
            for i, (feature, score) in enumerate(list(self.feature_importance.items())[:10]):
                logger.info(f"   {i+1}. {feature}: {score:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Ошибка при расчете feature importance: {e}")
            self.feature_importance = {}
            
    def get_feature_importance(self) -> pd.DataFrame:
        """Получить feature importance как DataFrame"""
        if self.feature_importance is None:
            return pd.DataFrame()
            
        df = pd.DataFrame(list(self.feature_importance.items()), 
                         columns=['feature', 'importance'])
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df