#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Продвинутое обучение модели с дополнительными параметрами и оптимизациями
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import pickle
import time
from datetime import datetime
import psycopg2
import logging
import yaml
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# Настройка расширенного логирования
log_dir = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(log_dir, exist_ok=True)

# Предварительная настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка для использования GPU если доступен
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logger.info(f"🖥️ GPU доступен: {physical_devices[0].name}")
else:
    logger.info("⚠️ GPU не найден, используется CPU")

# Файловый логгер
file_handler = logging.FileHandler(f'{log_dir}/training.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Консольный логгер
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Настройка основного логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class DetailedLoggingCallback(keras.callbacks.Callback):
    """Кастомный callback для детального логирования"""
    
    def __init__(self, log_dir, model_name):
        super().__init__()
        self.log_dir = log_dir
        self.model_name = model_name
        self.epoch_start_time = None
        
        # Создаем файл для детальных метрик
        self.metrics_file = open(f'{log_dir}/{model_name}_metrics.csv', 'w')
        self.metrics_file.write('epoch,loss,val_loss,accuracy,val_accuracy,precision,val_precision,recall,val_recall,auc,val_auc,lr,time\n')
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Эпоха {epoch + 1} началась для {self.model_name}")
        logger.info(f"   Learning rate: {self.model.optimizer.learning_rate.numpy():.6f}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        
        # Детальное логирование метрик
        logger.info(f"\n📊 Результаты эпохи {epoch + 1} ({self.model_name}):")
        logger.info(f"   ⏱️  Время: {epoch_time:.2f} сек")
        logger.info(f"   📉 Loss: {logs.get('loss', 0):.4f} (train) | {logs.get('val_loss', 0):.4f} (val)")
        logger.info(f"   🎯 Accuracy: {logs.get('accuracy', 0):.4f} (train) | {logs.get('val_accuracy', 0):.4f} (val)")
        logger.info(f"   🎯 Precision: {logs.get('precision', 0):.4f} (train) | {logs.get('val_precision', 0):.4f} (val)")
        logger.info(f"   🎯 Recall: {logs.get('recall', 0):.4f} (train) | {logs.get('val_recall', 0):.4f} (val)")
        logger.info(f"   📈 AUC: {logs.get('auc', 0):.4f} (train) | {logs.get('val_auc', 0):.4f} (val)")
        
        # Сохраняем в CSV
        self.metrics_file.write(f"{epoch + 1},{logs.get('loss', 0):.4f},{logs.get('val_loss', 0):.4f},"
                               f"{logs.get('accuracy', 0):.4f},{logs.get('val_accuracy', 0):.4f},"
                               f"{logs.get('precision', 0):.4f},{logs.get('val_precision', 0):.4f},"
                               f"{logs.get('recall', 0):.4f},{logs.get('val_recall', 0):.4f},"
                               f"{logs.get('auc', 0):.4f},{logs.get('val_auc', 0):.4f},"
                               f"{self.model.optimizer.learning_rate.numpy():.6f},{epoch_time:.2f}\n")
        self.metrics_file.flush()
        
    def on_train_batch_begin(self, batch, logs=None):
        if batch % 100 == 0:
            logger.debug(f"   Батч {batch} начался...")
            
    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0 and logs:
            logger.debug(f"   Батч {batch}: loss={logs.get('loss', 0):.4f}")
            
    def on_train_end(self, logs=None):
        self.metrics_file.close()
        logger.info(f"\n✅ Обучение {self.model_name} завершено!")
        logger.info(f"📊 Метрики сохранены в {self.log_dir}/{self.model_name}_metrics.csv")


class AdvancedTrainer:
    """Продвинутый класс для обучения с дополнительными возможностями"""
    
    def __init__(self, config_path='config.yaml'):
        # Загружаем конфигурацию
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Настройки БД
        self.db_config = self.config['database'].copy()
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
        
        # Параметры модели (расширенные)
        self.sequence_length = self.config['model']['sequence_length']
        self.prediction_horizon = self.config['model']['prediction_horizon']
        self.batch_size = self.config['model']['batch_size']
        self.epochs = 100  # Увеличиваем количество эпох
        self.learning_rate = 0.001  # Начальный learning rate
        
        # Дополнительные параметры
        self.use_attention = True
        self.use_residual = True
        self.dropout_rate = 0.2
        self.l2_regularization = 1e-4
        self.gradient_clip = 1.0
        
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_columns = None
        self.log_dir = log_dir  # Используем глобальный log_dir
        
    def connect_db(self):
        """Подключение к БД"""
        self.conn = psycopg2.connect(**self.db_config)
        logger.info("✅ Подключен к PostgreSQL")
        
    def create_focal_loss(self, alpha=0.9, gamma=2.0):
        """
        Focal Loss для борьбы с дисбалансом классов
        alpha=0.9 даёт больший вес положительным примерам (minority class)
        gamma=2.0 фокусирует модель на сложных примерах
        """
        def focal_loss(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            # Вычисляем focal loss
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_factor = tf.ones_like(y_true) * alpha
            alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
            cross_entropy = -tf.math.log(p_t)
            weight = alpha_t * tf.pow((1 - p_t), gamma)
            loss = weight * cross_entropy
            
            return tf.reduce_mean(loss)
        
        return focal_loss
    
    def create_advanced_model(self, input_shape, name):
        """Создание продвинутой архитектуры модели"""
        
        inputs = layers.Input(shape=input_shape, name='input')
        
        # Feature extraction с residual connections
        x = layers.LSTM(256, return_sequences=True, 
                       kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
                       recurrent_regularizer=keras.regularizers.l2(self.l2_regularization))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Residual block 1
        residual = x
        x = layers.LSTM(128, return_sequences=True,
                       kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
                       recurrent_regularizer=keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        
        if self.use_residual:
            # Приводим размерности для residual connection
            residual = layers.Dense(128)(residual)
            x = layers.Add()([x, residual])
        
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Attention mechanism
        if self.use_attention:
            attention = layers.MultiHeadAttention(
                num_heads=4, 
                key_dim=16,
                dropout=self.dropout_rate
            )(x, x)
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)
        
        # Global features
        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPooling1D()(x)
        
        # Combine features
        combined = layers.Concatenate()([avg_pool, max_pool])
        
        # Deep layers with batch norm
        x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(self.l2_regularization))(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate * 1.5)(x)
        
        x = layers.Dense(64, kernel_regularizer=keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(32, kernel_regularizer=keras.regularizers.l2(self.l2_regularization))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate * 0.5)(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        
        # Compile с advanced оптимизатором
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=self.gradient_clip
        )
        
        # Используем focal loss для лучшей работы с дисбалансом
        model.compile(
            optimizer=optimizer,
            loss=self.create_focal_loss(alpha=0.9, gamma=2.0),
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных с дополнительной обработкой"""
        logger.info("📊 Загрузка данных из PostgreSQL...")
        logger.info("🔍 Выполняется запрос к БД для получения фьючерсных данных...")
        
        start_time = time.time()
        
        # Загружаем данные фьючерсов
        query = """
        SELECT 
            p.symbol, p.timestamp, p.datetime,
            p.technical_indicators,
            p.buy_profit_target, p.buy_loss_target,
            p.sell_profit_target, p.sell_loss_target
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
        ORDER BY p.symbol, p.timestamp
        """
        
        df = pd.read_sql_query(query, self.conn)
        load_time = time.time() - start_time
        
        logger.info(f"✅ Загружено {len(df)} записей за {load_time:.2f} сек")
        logger.info(f"📊 Уникальных символов: {df['symbol'].nunique()}")
        logger.info(f"📅 Период данных: {df['datetime'].min()} - {df['datetime'].max()}")
        
        if len(df) == 0:
            return None
        
        # Извлекаем признаки
        if len(df) > 0:
            sample_indicators = df.iloc[0]['technical_indicators']
            self.feature_columns = list(sample_indicators.keys())
            
            features = []
            for _, row in df.iterrows():
                indicators = row['technical_indicators']
                feature_values = []
                for col in self.feature_columns:
                    val = indicators.get(col, 0)
                    if val is None or np.isnan(val) or np.isinf(val):
                        val = 0
                    feature_values.append(float(val))
                features.append(feature_values)
            
            X = np.array(features)
            
            # Дополнительная очистка данных
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Нормализация с сохранением статистики
            X_scaled = self.scaler.fit_transform(X)
            
            # Сохраняем статистику нормализации
            scaler_stats = {
                'mean': self.scaler.center_.tolist() if hasattr(self.scaler, 'center_') else None,
                'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'features': self.feature_columns
            }
            
            with open('trained_model/scaler_stats.json', 'w') as f:
                json.dump(scaler_stats, f, indent=2)
            
            return {
                'X': X_scaled,
                'y_buy_profit': df['buy_profit_target'].values,
                'y_buy_loss': df['buy_loss_target'].values,
                'y_sell_profit': df['sell_profit_target'].values,
                'y_sell_loss': df['sell_loss_target'].values,
                'symbols': df['symbol'].values,
                'timestamps': df['timestamp'].values
            }
        
        return None
    
    def create_sequences_advanced(self, X, y, symbols):
        """Создание последовательностей с учетом символов"""
        sequences = []
        labels = []
        
        unique_symbols = np.unique(symbols)
        
        for symbol in unique_symbols:
            symbol_mask = symbols == symbol
            symbol_indices = np.where(symbol_mask)[0]
            
            if len(symbol_indices) < self.sequence_length:
                continue
            
            # Создаем последовательности для этого символа
            X_symbol = X[symbol_indices]
            y_symbol = y[symbol_indices]
            
            for i in range(len(X_symbol) - self.sequence_length):
                sequences.append(X_symbol[i:i + self.sequence_length])
                labels.append(y_symbol[i + self.sequence_length])
        
        return np.array(sequences), np.array(labels)
    
    def balance_dataset(self, X, y, strategy='oversample'):
        """Балансировка датасета через оверсэмплинг"""
        from sklearn.utils import resample
        
        # Разделяем классы
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        
        logger.info(f"   Исходное распределение: {len(pos_indices)} pos / {len(neg_indices)} neg")
        
        if strategy == 'oversample':
            # Оверсэмплинг положительных примеров
            n_samples = len(neg_indices)
            pos_indices_resampled = resample(pos_indices, 
                                            replace=True, 
                                            n_samples=n_samples,
                                            random_state=42)
            
            # Объединяем индексы
            all_indices = np.concatenate([neg_indices, pos_indices_resampled])
            np.random.shuffle(all_indices)
            
            X_balanced = X[all_indices]
            y_balanced = y[all_indices]
            
            logger.info(f"   После балансировки: {sum(y_balanced)} pos / {len(y_balanced) - sum(y_balanced)} neg")
            
            return X_balanced, y_balanced
        else:
            return X, y
    
    def train_with_callbacks(self, model, X_train, y_train, X_val, y_val, model_name):
        """Обучение с продвинутыми callbacks"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ: {model_name}")
        logger.info(f"{'='*70}")
        
        # Статистика данных
        logger.info(f"\n📊 Статистика данных для обучения:")
        logger.info(f"   Размер обучающей выборки: {len(X_train)}")
        logger.info(f"   Размер валидационной выборки: {len(X_val)}")
        logger.info(f"   Форма входных данных: {X_train.shape}")
        logger.info(f"   Положительных примеров (train): {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
        logger.info(f"   Положительных примеров (val): {sum(y_val)} ({sum(y_val)/len(y_val)*100:.1f}%)")
        
        # Балансировка обучающей выборки
        logger.info(f"\n⚖️ Балансировка обучающей выборки:")
        X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train, strategy='oversample')
        
        # Вычисляем веса классов (теперь они должны быть более сбалансированы)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_balanced),
            y=y_train_balanced
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        logger.info(f"\n⚖️ Веса классов после балансировки:")
        logger.info(f"   Вес класса 0 (негативный): {class_weight_dict.get(0, 1.0):.2f}")
        logger.info(f"   Вес класса 1 (позитивный): {class_weight_dict.get(1, 1.0):.2f}")
        
        # Callbacks
        callbacks = [
            # Наш детальный логгер
            DetailedLoggingCallback(self.log_dir, model_name),
            # Early stopping с восстановлением лучших весов (следим за recall)
            keras.callbacks.EarlyStopping(
                monitor='val_recall',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Уменьшение learning rate при застое
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_recall',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1,
                mode='max'
            ),
            
            # Сохранение лучшей модели (по recall)
            keras.callbacks.ModelCheckpoint(
                filepath=f'trained_model/{model_name}_best.h5',
                monitor='val_recall',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard для визуализации
            keras.callbacks.TensorBoard(
                log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            
            # Кастомный callback для логирования
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: logger.info(
                    f"Epoch {epoch + 1}: loss={logs['loss']:.4f}, "
                    f"val_loss={logs['val_loss']:.4f}, "
                    f"val_auc={logs['val_auc']:.4f}"
                )
            )
        ]
        
        # Обучение с балансированными данными
        history = model.fit(
            X_train_balanced, y_train_balanced,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Поиск оптимального порога для максимизации F1-score"""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Находим порог с максимальным F1
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_f1 = f1_scores[optimal_idx]
        
        logger.info(f"   🎯 Оптимальный порог: {optimal_threshold:.3f} (F1={optimal_f1:.3f})")
        
        return optimal_threshold
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Расширенная оценка модели"""
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Находим оптимальный порог
        optimal_threshold = self.find_optimal_threshold(y_test, y_pred)
        
        # Предсказания с оптимальным порогом
        y_pred_binary = (y_pred > optimal_threshold).astype(int).flatten()
        
        # Метрики
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test, y_pred_binary, zero_division=0),
            'f1': f1_score(y_test, y_pred_binary, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0,
            'optimal_threshold': optimal_threshold
        }
        
        # Дополнительные метрики
        tp = np.sum((y_test == 1) & (y_pred_binary == 1))
        tn = np.sum((y_test == 0) & (y_pred_binary == 0))
        fp = np.sum((y_test == 0) & (y_pred_binary == 1))
        fn = np.sum((y_test == 1) & (y_pred_binary == 0))
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Специфичность
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        logger.info(f"\n📊 Метрики {model_name}:")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1 Score: {metrics['f1']:.4f}")
        logger.info(f"   AUC: {metrics['auc']:.4f}")
        logger.info(f"   Specificity: {metrics['specificity']:.4f}")
        logger.info(f"   Оптимальный порог: {metrics['optimal_threshold']:.3f}")
        
        return metrics
    
    def train(self):
        """Основной процесс обучения"""
        logger.info("\n" + "="*80)
        logger.info("🚀 НАЧАЛО ПРОЦЕССА ОБУЧЕНИЯ ML МОДЕЛИ ДЛЯ КРИПТОТРЕЙДИНГА")
        logger.info("="*80)
        logger.info(f"📅 Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📁 Директория логов: {self.log_dir}")
        
        try:
            # Создаем директории
            os.makedirs('trained_model', exist_ok=True)
            os.makedirs('logs', exist_ok=True)
            os.makedirs('plots', exist_ok=True)
            
            logger.info("\n📊 КОНФИГУРАЦИЯ:")
            logger.info(f"   Sequence length: {self.sequence_length}")
            logger.info(f"   Batch size: {self.batch_size}")
            logger.info(f"   Epochs: {self.epochs}")
            logger.info(f"   Learning rate: {self.learning_rate}")
            logger.info(f"   Use attention: {self.use_attention}")
            logger.info(f"   Use residual: {self.use_residual}")
            
            # Подключаемся к БД
            logger.info("\n🔌 Подключение к PostgreSQL...")
            self.connect_db()
            logger.info("✅ Подключение установлено")
            
            # Загружаем данные
            data = self.load_and_prepare_data()
            if not data:
                logger.error("❌ Нет данных для обучения")
                return
            
            # Список моделей для обучения
            model_configs = [
                ('buy_profit_model', data['y_buy_profit']),
                ('sell_profit_model', data['y_sell_profit'])
            ]
            
            results = {}
            
            for model_name, y_target in model_configs:
                logger.info(f"\n{'='*60}")
                logger.info(f"Обучение {model_name}")
                logger.info(f"{'='*60}")
                
                # Создаем последовательности
                logger.info(f"\n🔄 Создание последовательностей для {model_name}...")
                seq_start_time = time.time()
                
                X_seq, y_seq = self.create_sequences_advanced(
                    data['X'], y_target, data['symbols']
                )
                
                seq_time = time.time() - seq_start_time
                
                if len(X_seq) == 0:
                    logger.warning(f"⚠️ Недостаточно данных для {model_name}")
                    continue
                
                logger.info(f"✅ Создано {len(X_seq)} последовательностей за {seq_time:.2f} сек")
                logger.info(f"   Форма данных: {X_seq.shape}")
                logger.info(f"   Положительных примеров: {sum(y_seq)} ({sum(y_seq)/len(y_seq)*100:.1f}%)")
                logger.info(f"   Отрицательных примеров: {len(y_seq) - sum(y_seq)} ({(len(y_seq) - sum(y_seq))/len(y_seq)*100:.1f}%)")
                
                # Разделение на train/val/test
                n_samples = len(X_seq)
                train_end = int(n_samples * 0.7)
                val_end = int(n_samples * 0.85)
                
                X_train = X_seq[:train_end]
                y_train = y_seq[:train_end]
                X_val = X_seq[train_end:val_end]
                y_val = y_seq[train_end:val_end]
                X_test = X_seq[val_end:]
                y_test = y_seq[val_end:]
                
                # Создаем модель
                model = self.create_advanced_model(X_seq.shape[1:], model_name)
                logger.info(f"✅ Создана модель с {model.count_params():,} параметрами")
                
                # Обучаем
                history = self.train_with_callbacks(
                    model, X_train, y_train, X_val, y_val, model_name
                )
                
                # Оцениваем
                test_metrics = self.evaluate_model(model, X_test, y_test, model_name)
                
                # Сохраняем результаты
                results[model_name] = {
                    'history': history.history,
                    'metrics': test_metrics,
                    'model': model
                }
                
                self.models[model_name] = model
            
            # Сохраняем все модели и метаданные
            logger.info("\n💾 Сохранение моделей...")
            self.save_all_models(results)
            
            # Итоговая статистика
            logger.info("\n" + "="*80)
            logger.info("📊 ИТОГОВАЯ СТАТИСТИКА ОБУЧЕНИЯ")
            logger.info("="*80)
            
            for model_name, result in results.items():
                metrics = result['metrics']
                logger.info(f"\n🤖 {model_name}:")
                logger.info(f"   Test Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"   Test Precision: {metrics['precision']:.4f}")
                logger.info(f"   Test Recall: {metrics['recall']:.4f}")
                logger.info(f"   Test F1: {metrics['f1']:.4f}")
                logger.info(f"   Test AUC: {metrics['auc']:.4f}")
                logger.info(f"   Эпох обучено: {len(result['history']['loss'])}")
            
            # Сохраняем итоговый отчет
            report_path = f'{self.log_dir}/training_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"ОТЧЕТ ОБ ОБУЧЕНИИ\n")
                f.write(f"{'='*50}\n")
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Конфигурация:\n")
                f.write(f"  - Sequence length: {self.sequence_length}\n")
                f.write(f"  - Batch size: {self.batch_size}\n")
                f.write(f"  - Epochs: {self.epochs}\n")
                f.write(f"  - Learning rate: {self.learning_rate}\n")
                f.write(f"\nРезультаты:\n")
                for model_name, result in results.items():
                    f.write(f"\n{model_name}:\n")
                    for metric, value in result['metrics'].items():
                        f.write(f"  - {metric}: {value:.4f}\n")
            
            logger.info(f"\n📄 Отчет сохранен: {report_path}")
            logger.info(f"📁 Все логи в директории: {self.log_dir}")
            
            logger.info("\n" + "="*80)
            logger.info("✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            logger.error(f"Трейсбек:", exc_info=True)
            
            # Сохраняем ошибку в файл
            error_path = f'{self.log_dir}/error.log'
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Ошибка: {e}\n")
                import traceback
                f.write(traceback.format_exc())
            
            raise
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()
                logger.info("📤 Соединение с БД закрыто")
    
    def save_all_models(self, results):
        """Сохранение всех моделей и метаданных"""
        save_start_time = time.time()
        
        logger.info("\n📦 Сохранение результатов обучения...")
        
        # Сохраняем модели
        for name, model in self.models.items():
            model_path = f'trained_model/{name}_advanced.h5'
            model.save(model_path)
            logger.info(f"   ✅ Модель сохранена: {model_path} ({os.path.getsize(model_path) / 1024 / 1024:.2f} MB)")
        
        # Сохраняем scaler
        with open('trained_model/scaler_advanced.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Сохраняем метаданные
        metadata = {
            'type': 'advanced_lstm_attention',
            'sequence_length': self.sequence_length,
            'features': self.feature_columns,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual,
            'created_at': datetime.now().isoformat(),
            'results': {
                name: {
                    'metrics': result['metrics'],
                    'epochs_trained': len(result['history']['loss']),
                    'optimal_threshold': result['metrics'].get('optimal_threshold', 0.5)
                }
                for name, result in results.items()
            }
        }
        
        with open('trained_model/metadata_advanced.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Создаем визуализацию
        self.create_training_visualizations(results)
        
        save_time = time.time() - save_start_time
        logger.info(f"\n✅ Все результаты сохранены за {save_time:.2f} сек")
        logger.info(f"   📁 Модели: trained_model/")
        logger.info(f"   📊 Графики: plots/")
        logger.info(f"   📝 Логи: {self.log_dir}/")
    
    def create_training_visualizations(self, results):
        """Создание графиков процесса обучения"""
        import matplotlib.pyplot as plt
        
        logger.info("\n📈 Создание визуализаций...")
        
        # Создаем общий график для всех моделей
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Процесс обучения моделей для криптотрейдинга', fontsize=16)
        
        colors = ['blue', 'green', 'red', 'orange']
        
        for idx, (model_name, result) in enumerate(results.items()):
            history = result['history']
            color = colors[idx % len(colors)]
            
            # Loss график
            axes[0, 0].plot(history['loss'], label=f'{model_name} (train)', 
                          color=color, alpha=0.8)
            axes[0, 0].plot(history['val_loss'], label=f'{model_name} (val)', 
                          color=color, linestyle='--', alpha=0.8)
            
            # Accuracy график
            axes[0, 1].plot(history['accuracy'], label=f'{model_name} (train)', 
                          color=color, alpha=0.8)
            axes[0, 1].plot(history['val_accuracy'], label=f'{model_name} (val)', 
                          color=color, linestyle='--', alpha=0.8)
            
            # AUC график
            axes[1, 0].plot(history['auc'], label=f'{model_name} (train)', 
                          color=color, alpha=0.8)
            axes[1, 0].plot(history['val_auc'], label=f'{model_name} (val)', 
                          color=color, linestyle='--', alpha=0.8)
            
            # Финальные метрики
            metrics = result['metrics']
            axes[1, 1].bar(idx * 5 + np.arange(4), 
                         [metrics['accuracy'], metrics['precision'], 
                          metrics['recall'], metrics['f1']],
                         label=model_name, color=color, alpha=0.8)
        
        # Настройка графиков
        axes[0, 0].set_title('Loss по эпохам')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(loc='upper right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Accuracy по эпохам')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend(loc='lower right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('AUC по эпохам')
        axes[1, 0].set_xlabel('Эпоха')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend(loc='lower right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Финальные метрики (test)')
        axes[1, 1].set_ylabel('Значение')
        axes[1, 1].set_xticks(np.arange(len(results)) * 5 + 1.5)
        axes[1, 1].set_xticklabels(['Acc', 'Prec', 'Rec', 'F1'] * len(results), 
                                   rotation=45, fontsize=8)
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Сохраняем
        plot_path = f'plots/training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   ✅ График сохранен: {plot_path}")


if __name__ == "__main__":
    trainer = AdvancedTrainer()
    trainer.train()