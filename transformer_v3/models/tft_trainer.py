"""
Trainer для Temporal Fusion Transformer
Аналог XGBoostTrainer из xgboost_v3
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
import pickle
from datetime import datetime

from config import Config
from models.tft_model import TemporalFusionTransformer
from utils.metrics import MetricsCalculator
from utils.visualization import VisualizationCallback

logger = logging.getLogger(__name__)


class TFTTrainer:
    """Класс для обучения TFT моделей"""
    
    def __init__(self, config: Config, model_name: str = "tft_model"):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.metrics_calculator = MetricsCalculator(config)
        self.training_history = None
        self.scaler = None
        self.feature_columns = None
        
        # Настройка GPU
        self._setup_gpu()
        
    def _setup_gpu(self):
        """Настройка GPU для TensorFlow"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Настройка multi-GPU стратегии
                if self.config.model.use_multi_gpu and len(gpus) > 1:
                    # Определяем количество GPU для использования
                    num_gpus = self.config.model.num_gpus if self.config.model.num_gpus > 0 else len(gpus)
                    num_gpus = min(num_gpus, len(gpus))
                    
                    # Создаем MirroredStrategy для multi-GPU
                    self.strategy = tf.distribute.MirroredStrategy(
                        devices=[f"/gpu:{i}" for i in range(num_gpus)]
                    )
                    logger.info(f"🚀 Multi-GPU стратегия активирована: {num_gpus} GPU")
                    logger.info(f"   Эффективный batch size: {self.config.training.batch_size * num_gpus}")
                else:
                    # Single GPU стратегия
                    self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
                    logger.info(f"🖥️ Используется single GPU: {gpus[0].name}")
                
                # Включаем динамическое выделение памяти для всех GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                # Mixed precision для ускорения
                if self.config.model.use_mixed_precision:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info("⚡ Mixed precision включен для ускорения в 2x")
                else:
                    # Используем float32 для стабильности
                    policy = tf.keras.mixed_precision.Policy('float32')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info("📊 Используется float32 (можно включить mixed precision для ускорения)")
                
                # Оптимизация для TF
                tf.config.optimizer.set_jit(True)  # XLA compilation для ускорения
                tf.config.run_functions_eagerly(False)
                    
            except RuntimeError as e:
                logger.error(f"❌ Ошибка настройки GPU: {e}")
                self.strategy = tf.distribute.get_strategy()  # Default strategy
        else:
            logger.info("💻 GPU не найден, используется CPU")
            self.strategy = tf.distribute.get_strategy()
    
    def create_model(self, num_features: int) -> TemporalFusionTransformer:
        """
        Создание модели TFT
        
        Args:
            num_features: Количество признаков
            
        Returns:
            Модель TFT
        """
        logger.info(f"🏗️ Создание модели {self.model_name}...")
        
        # Создаем модель в рамках стратегии распределения
        with self.strategy.scope():
            model = TemporalFusionTransformer(
                config=self.config,
                num_features=num_features,
                name=self.model_name
            )
            
            # Компиляция модели
            self._compile_model(model)
            
            # Инициализация весов
            dummy_input = tf.zeros((1, self.config.model.sequence_length, num_features))
            _ = model(dummy_input)
        
        logger.info(f"✅ Создана модель с {model.count_params():,} параметрами")
        
        return model
    
    def _compile_model(self, model: TemporalFusionTransformer):
        """Компиляция модели с оптимизатором и метриками"""
        # Оптимизатор
        optimizer = self._create_optimizer()
        
        # Loss функция
        if self.config.training.task_type == 'regression':
            if self.config.training.loss_function == 'huber':
                # Huber loss более устойчива к выбросам
                loss = keras.losses.Huber(
                    delta=self.config.training.huber_delta,
                    reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                )
            elif self.config.training.loss_function == 'mse':
                loss = keras.losses.MeanSquaredError(
                    reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                )
            else:  # mae
                loss = keras.losses.MeanAbsoluteError(
                    reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                )
                
            metrics = [
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse')
            ]
        else:  # classification_binary
            loss = keras.losses.BinaryCrossentropy(
                label_smoothing=self.config.training.label_smoothing
            )
            metrics = [
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def _create_optimizer(self):
        """Создание оптимизатора"""
        # Скорректированный learning rate для multi-GPU
        lr = self.config.training.learning_rate
        if hasattr(self, 'strategy') and self.strategy.num_replicas_in_sync > 1:
            # Увеличиваем LR пропорционально количеству GPU
            lr = lr * self.strategy.num_replicas_in_sync
            logger.info(f"📈 Learning rate скорректирован для {self.strategy.num_replicas_in_sync} GPU: {lr}")
        
        if self.config.training.optimizer == 'adam':
            optimizer = keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=self.config.training.beta_1,
                beta_2=self.config.training.beta_2,
                epsilon=self.config.training.epsilon,
                clipnorm=1.0  # Gradient clipping для стабильности
            )
        elif self.config.training.optimizer == 'adamw':
            optimizer = keras.optimizers.AdamW(
                learning_rate=lr,
                beta_1=self.config.training.beta_1,
                beta_2=self.config.training.beta_2,
                epsilon=self.config.training.epsilon,
                weight_decay=0.01,
                clipnorm=1.0
            )
        else:  # sgd
            optimizer = keras.optimizers.SGD(
                learning_rate=lr,
                momentum=0.9,
                nesterov=True,
                clipnorm=1.0
            )
        
        # Оборачиваем в mixed precision optimizer если нужно
        if self.config.model.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            logger.info("⚡ Оптимизатор обернут для mixed precision")
            
        return optimizer
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             feature_columns: List[str] = None) -> keras.Model:
        """
        Обучение модели
        
        Args:
            X_train: Обучающие последовательности [batch, seq_len, features]
            y_train: Обучающие метки
            X_val: Валидационные последовательности
            y_val: Валидационные метки
            feature_columns: Названия признаков
            
        Returns:
            Обученная модель
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Обучение модели: {self.model_name}")
        logger.info(f"{'='*60}")
        
        # Проверка данных на NaN
        logger.info("🔍 Проверка данных на NaN...")
        train_nan_count = np.isnan(X_train).sum() + np.isnan(y_train).sum()
        val_nan_count = np.isnan(X_val).sum() + np.isnan(y_val).sum()
        
        if train_nan_count > 0:
            logger.warning(f"⚠️ Найдено {train_nan_count} NaN значений в обучающих данных!")
            # Заменяем NaN на 0
            X_train = np.nan_to_num(X_train, nan=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0)
            
        if val_nan_count > 0:
            logger.warning(f"⚠️ Найдено {val_nan_count} NaN значений в валидационных данных!")
            X_val = np.nan_to_num(X_val, nan=0.0)
            y_val = np.nan_to_num(y_val, nan=0.0)
            
        # Проверка статистики данных
        logger.info(f"📊 Статистика y_train: mean={np.mean(y_train):.4f}, std={np.std(y_train):.4f}, "
                   f"min={np.min(y_train):.4f}, max={np.max(y_train):.4f}")
        logger.info(f"📊 Статистика X_train: mean={np.mean(X_train):.4f}, std={np.std(X_train):.4f}")
        
        # Сохраняем колонки
        self.feature_columns = feature_columns
        
        # Создаем модель
        num_features = X_train.shape[2]
        self.model = self.create_model(num_features)
        
        # Callbacks
        callbacks = self._create_callbacks()
        
        # Преобразуем в tf.data.Dataset для эффективности
        train_dataset = self._create_dataset(X_train, y_train, shuffle=True)
        val_dataset = self._create_dataset(X_val, y_val, shuffle=False)
        
        # Обучение
        logger.info("🏋️ Начало обучения...")
        logger.info(f"   Размер обучающей выборки: {X_train.shape}")
        logger.info(f"   Размер валидационной выборки: {X_val.shape}")
        
        self.training_history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.training.epochs,
            callbacks=callbacks,
            verbose=1 if self.config.training.verbose else 0
        )
        
        logger.info(f"✅ Обучение завершено")
        
        return self.model
    
    def _create_dataset(self, X: np.ndarray, y: np.ndarray, shuffle: bool = False) -> tf.data.Dataset:
        """Создание tf.data.Dataset с эффективным использованием памяти"""
        # Конвертируем в float32 заранее для экономии памяти
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # Заменяем NaN на 0
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        # Для больших датасетов используем генератор вместо from_tensor_slices
        if len(X) > 100000:  # Если больше 100k примеров
            logger.info(f"📊 Большой датасет ({len(X)} записей), используем генератор")
            
            def generator():
                for i in range(len(X)):
                    yield X[i], y[i]
            
            dataset = tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.config.model.sequence_length, X.shape[2]), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.float32)
                )
            )
        else:
            # Для маленьких датасетов используем обычный способ
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        # Опции для оптимизации производительности
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.experimental_optimization.parallel_batch = True
        options.experimental_threading.private_threadpool_size = 8
        options.experimental_threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options)
        
        if shuffle:
            # Большой buffer для лучшего перемешивания
            buffer_size = min(50000, len(X))
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        
        # Батчинг с учетом gradient accumulation и multi-GPU
        actual_batch_size = self.config.training.batch_size
        
        # Для multi-GPU: каждая GPU получает batch_size данных
        # Глобальный batch = batch_size * num_gpus
        if hasattr(self, 'strategy') and self.strategy.num_replicas_in_sync > 1:
            # Это batch PER REPLICA (на каждую GPU)
            actual_batch_size = self.config.training.batch_size
            global_batch_size = actual_batch_size * self.strategy.num_replicas_in_sync
            logger.info(f"🚀 Multi-GPU батчинг: {actual_batch_size} per GPU, глобальный batch: {global_batch_size}")
        elif hasattr(self.config.training, 'gradient_accumulation_steps'):
            # Делим batch size на количество шагов накопления
            actual_batch_size = self.config.training.batch_size // self.config.training.gradient_accumulation_steps
            logger.info(f"📦 Используется gradient accumulation: {self.config.training.gradient_accumulation_steps} шагов")
            logger.info(f"   Batch per step: {actual_batch_size}, эффективный batch: {self.config.training.batch_size}")
        
        dataset = dataset.batch(actual_batch_size, drop_remainder=True)
        
        # Prefetch с автоматической настройкой
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Cache в памяти для небольших датасетов
        if len(X) < 500000:  # Кешируем если меньше 500k примеров
            dataset = dataset.cache()
            logger.info("💾 Датасет закеширован в памяти")
        
        return dataset
    
    def _create_callbacks(self) -> List[keras.callbacks.Callback]:
        """Создание callbacks для обучения"""
        callbacks = []
        
        # Early Stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=self.config.training.early_stopping_monitor,
                patience=self.config.training.early_stopping_patience,
                mode=self.config.training.early_stopping_mode,
                restore_best_weights=self.config.training.restore_best_weights,
                verbose=1
            )
        )
        
        # Reduce LR on Plateau
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=self.config.training.early_stopping_monitor,
                factor=self.config.training.reduce_lr_factor,
                patience=self.config.training.reduce_lr_patience,
                min_lr=self.config.training.reduce_lr_min,
                verbose=1
            )
        )
        
        # Model Checkpoint
        if self.config.training.save_models:
            checkpoint_path = Path(self.config.training.model_dir) / f"{self.model_name}_best.h5"
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path),
                    monitor=self.config.training.checkpoint_monitor,
                    save_best_only=self.config.training.save_best_only,
                    save_weights_only=False,
                    verbose=1
                )
            )
        
        # TensorBoard
        if self.config.training.tensorboard:
            log_dir = self.config.get_log_dir() / "tensorboard" / self.model_name
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=str(log_dir),
                    histogram_freq=1,
                    write_graph=True,
                    update_freq='epoch'
                )
            )
        
        # Visualization Callback
        if self.config.training.save_plots:
            callbacks.append(
                VisualizationCallback(
                    log_dir=self.config.get_log_dir(),
                    model_name=self.model_name,
                    update_freq=self.config.training.plot_update_freq,
                    task=self.config.training.task_type
                )
            )
        
        return callbacks
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """
        Предсказание модели
        
        Args:
            X: Последовательности для предсказания
            return_proba: Вернуть вероятности (для классификации)
            
        Returns:
            Предсказания
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        predictions = self.model.predict(X, verbose=0)
        predictions = predictions.flatten()
        
        if self.config.training.task_type == "classification_binary" and not return_proba:
            # Применяем порог
            threshold = self.config.training.classification_threshold / 100
            predictions = (predictions > threshold).astype(int)
        
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, dataset_name: str = "Test") -> Dict[str, float]:
        """
        Оценка модели
        
        Args:
            X: Признаки
            y: Истинные метки
            dataset_name: Название датасета
            
        Returns:
            Словарь с метриками
        """
        predictions = self.predict(X, return_proba=True)
        
        if self.config.training.task_type == "regression":
            metrics = self.metrics_calculator.calculate_regression_metrics(y, predictions)
        else:
            metrics = self.metrics_calculator.calculate_classification_metrics(y, predictions)
        
        # Логирование
        logger.info(f"\n📊 Метрики на {dataset_name}:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {metric_name}: {value:.4f}")
        
        return metrics
    
    def save_model(self, save_dir: Path):
        """Сохранение модели и метаданных"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем модель
        model_path = save_dir / f"{self.model_name}.h5"
        self.model.save(model_path)
        logger.info(f"   ✅ Модель сохранена: {model_path}")
        
        # Сохраняем веса отдельно (для надежности)
        weights_path = str(save_dir / f"{self.model_name}.weights.h5")
        self.model.save_weights(weights_path)
        
        # Сохраняем метаданные
        metadata = {
            'model_name': self.model_name,
            'task_type': self.config.training.task_type,
            'created_at': datetime.now().isoformat(),
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__
            },
            'feature_columns': self.feature_columns,
            'training_history': self.training_history.history if self.training_history else None
        }
        
        metadata_path = save_dir / f"{self.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Сохраняем scaler если есть
        if self.scaler:
            scaler_path = save_dir / f"{self.model_name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def load_model(self, model_path: Path):
        """Загрузка модели"""
        self.model = keras.models.load_model(model_path, compile=False)
        self._compile_model(self.model)
        logger.info(f"✅ Модель загружена из {model_path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Получение важности признаков из VSN"""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.get_feature_importance()
        if importance is None or self.feature_columns is None:
            return pd.DataFrame()
        
        # Усредняем по батчам если нужно
        if importance.ndim > 1:
            importance = importance.mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_columns[:len(importance)],
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df