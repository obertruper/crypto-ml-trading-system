#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Продвинутая модель на основе Transformer для предсказания криптовалютных движений
Использует современные архитектуры: Temporal Fusion Transformer, PatchTST, и Informer
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import matplotlib
matplotlib.use('Agg')  # Для сохранения графиков без GUI

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# Настройка логирования
log_dir = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f'{log_dir}/plots', exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка GPU
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
logger.addHandler(file_handler)


class TransformerBlock(layers.Layer):
    """Блок трансформера с multi-head attention и позиционным кодированием"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(layers.Layer):
    """Позиционное кодирование для временных рядов"""
    
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # Применяем sin к четным индексам
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Применяем cos к нечетным индексам
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class PatchTST(layers.Layer):
    """PatchTST - разбиение временных рядов на патчи для трансформера"""
    
    def __init__(self, patch_len=16, stride=8):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        num_features = tf.shape(x)[2]
        
        # Создаем патчи
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        patches = []
        
        for i in range(num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, start_idx:end_idx, :]
            patches.append(patch)
        
        # Stack патчи
        patches = tf.stack(patches, axis=1)
        # Reshape для трансформера
        patches = tf.reshape(patches, [batch_size, num_patches, self.patch_len * num_features])
        
        return patches


class TemporalFusionTransformer(keras.Model):
    """Temporal Fusion Transformer для финансовых временных рядов"""
    
    def __init__(self, num_features, sequence_length, 
                 d_model=128, num_heads=8, num_transformer_blocks=4,
                 mlp_units=[256, 128], dropout=0.2):
        super(TemporalFusionTransformer, self).__init__()
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Variable Selection Network
        self.vsn_dense = layers.Dense(d_model, activation='gelu')
        self.vsn_grn = self.gated_residual_network(d_model, dropout)
        
        # Static covariate encoder
        self.static_encoder = keras.Sequential([
            layers.Dense(d_model, activation='gelu'),
            layers.Dropout(dropout),
            self.gated_residual_network(d_model, dropout)
        ])
        
        # LSTM encoder для локальной обработки
        self.lstm_encoder = layers.LSTM(d_model, return_sequences=True)
        
        # Позиционное кодирование
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        
        # Transformer блоки
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, mlp_units[0], dropout)
            for _ in range(num_transformer_blocks)
        ]
        
        # Interpretable Multi-Head Attention
        self.interpretable_attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            dropout=dropout
        )
        
        # Output layers
        self.output_dense = keras.Sequential([
            layers.Dense(mlp_units[0], activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(mlp_units[1], activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(1)  # Регрессия
        ])
        
    def gated_residual_network(self, units, dropout_rate):
        """Gated Residual Network для TFT"""
        return keras.Sequential([
            layers.Dense(units, activation=None),
            layers.ELU(),
            layers.Dense(units, activation=None),
            layers.Dropout(dropout_rate),
            layers.Dense(units, activation='sigmoid'),
            layers.Multiply()
        ])
    
    def call(self, inputs, training=False):
        # Variable selection
        selected = self.vsn_dense(inputs)
        selected = self.vsn_grn(selected)
        
        # LSTM encoding
        lstm_out = self.lstm_encoder(selected)
        
        # Позиционное кодирование
        x = self.pos_encoding(lstm_out)
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        # Interpretable attention
        attn_output = self.interpretable_attention(x, x, training=training)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(attn_output)
        
        # Output
        return self.output_dense(x)


class VisualizationCallback(keras.callbacks.Callback):
    """Callback для визуализации процесса обучения в реальном времени"""
    
    def __init__(self, log_dir, model_name, update_freq=5):
        super().__init__()
        self.log_dir = log_dir
        self.model_name = model_name
        self.update_freq = update_freq
        self.epoch_count = 0
        
        # История метрик
        self.history = {
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': [],
            'lr': []
        }
        
        # Создаем фигуру для графиков
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f'Training Progress: {model_name}', fontsize=16)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        
        # Сохраняем метрики
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['mae'].append(logs.get('mae', 0))
        self.history['val_mae'].append(logs.get('val_mae', 0))
        self.history['lr'].append(self.model.optimizer.learning_rate.numpy())
        
        # Обновляем графики каждые N эпох
        if self.epoch_count % self.update_freq == 0:
            self.update_plots()
    
    def update_plots(self):
        """Обновление графиков"""
        epochs = range(1, len(self.history['loss']) + 1)
        
        # График 1: Loss
        self.axes[0, 0].clear()
        self.axes[0, 0].plot(epochs, self.history['loss'], 'b-', label='Train Loss')
        self.axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        self.axes[0, 0].set_title('Model Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True)
        
        # График 2: MAE
        self.axes[0, 1].clear()
        self.axes[0, 1].plot(epochs, self.history['mae'], 'b-', label='Train MAE')
        self.axes[0, 1].plot(epochs, self.history['val_mae'], 'r-', label='Val MAE')
        self.axes[0, 1].set_title('Mean Absolute Error')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('MAE')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True)
        
        # График 3: Learning Rate
        self.axes[1, 0].clear()
        self.axes[1, 0].plot(epochs, self.history['lr'], 'g-')
        self.axes[1, 0].set_title('Learning Rate')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('LR')
        self.axes[1, 0].grid(True)
        self.axes[1, 0].set_yscale('log')
        
        # График 4: Статистика
        self.axes[1, 1].clear()
        self.axes[1, 1].axis('off')
        
        # Текущие метрики
        current_stats = f"""
        Current Epoch: {len(self.history['loss'])}
        
        Train Loss: {self.history['loss'][-1]:.4f}
        Val Loss: {self.history['val_loss'][-1]:.4f}
        
        Train MAE: {self.history['mae'][-1]:.4f}%
        Val MAE: {self.history['val_mae'][-1]:.4f}%
        
        Learning Rate: {self.history['lr'][-1]:.6f}
        
        Best Val Loss: {min(self.history['val_loss']):.4f}
        Best Epoch: {self.history['val_loss'].index(min(self.history['val_loss'])) + 1}
        """
        
        self.axes[1, 1].text(0.1, 0.5, current_stats, fontsize=12, 
                             verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig(f'{self.log_dir}/plots/training_progress_{self.model_name}.png', dpi=100)
        
        # Также сохраняем последний график для быстрого просмотра
        plt.savefig(f'{self.log_dir}/plots/latest_progress.png', dpi=100)
        
        logger.info(f"📊 График обновлен: {self.log_dir}/plots/latest_progress.png")


class TransformerTrainer:
    """Основной класс для обучения Transformer моделей"""
    
    def __init__(self, config_path='config.yaml'):
        # Загружаем конфигурацию
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Настройки БД
        self.db_config = self.config['database'].copy()
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
        
        # Параметры модели
        self.sequence_length = self.config['model']['sequence_length']
        self.batch_size = 32  # Меньше для трансформеров
        self.epochs = 150
        self.learning_rate = 0.0001
        
        # Архитектура
        self.d_model = 128
        self.num_heads = 8
        self.num_transformer_blocks = 4
        self.dropout_rate = 0.2
        
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_columns = None
        self.log_dir = log_dir
        
    def connect_db(self):
        """Подключение к БД"""
        self.conn = psycopg2.connect(**self.db_config)
        logger.info("✅ Подключен к PostgreSQL")
    
    def create_model(self, num_features, name):
        """Создание TFT модели"""
        model = TemporalFusionTransformer(
            num_features=num_features,
            sequence_length=self.sequence_length,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_transformer_blocks=self.num_transformer_blocks,
            mlp_units=[256, 128],
            dropout=self.dropout_rate
        )
        
        # Compile
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.Huber(delta=1.0),
            metrics=[
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse')
            ]
        )
        
        return model
    
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных"""
        logger.info("📊 Загрузка данных из PostgreSQL...")
        
        query = """
        SELECT 
            p.symbol, p.timestamp, p.datetime,
            p.technical_indicators
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
          AND p.technical_indicators->>'buy_expected_return' IS NOT NULL
        ORDER BY p.symbol, p.timestamp
        """
        
        df = pd.read_sql_query(query, self.conn)
        logger.info(f"✅ Загружено {len(df)} записей")
        
        if len(df) == 0:
            logger.warning("⚠️ Нет данных с новыми метками!")
            return None
        
        # Извлекаем признаки и метки
        features = []
        buy_targets = []
        sell_targets = []
        
        for _, row in df.iterrows():
            indicators = row['technical_indicators']
            
            # Извлекаем признаки
            feature_values = []
            for key, value in indicators.items():
                if key not in ['buy_expected_return', 'sell_expected_return']:
                    val = float(value) if value is not None else 0.0
                    feature_values.append(val)
            
            features.append(feature_values)
            
            # Целевые значения
            buy_targets.append(float(indicators.get('buy_expected_return', 0.0)))
            sell_targets.append(float(indicators.get('sell_expected_return', 0.0)))
        
        X = np.array(features)
        X_scaled = self.scaler.fit_transform(X)
        
        # Сохраняем список признаков
        sample_indicators = df.iloc[0]['technical_indicators']
        self.feature_columns = [k for k in sample_indicators.keys() 
                               if k not in ['buy_expected_return', 'sell_expected_return']]
        
        return {
            'X': X_scaled,
            'y_buy': np.array(buy_targets),
            'y_sell': np.array(sell_targets),
            'symbols': df['symbol'].values,
            'timestamps': df['timestamp'].values
        }
    
    def create_sequences(self, X, y, symbols):
        """Создание последовательностей для трансформера"""
        sequences = []
        targets = []
        
        unique_symbols = np.unique(symbols)
        
        for symbol in unique_symbols:
            symbol_mask = symbols == symbol
            symbol_indices = np.where(symbol_mask)[0]
            
            if len(symbol_indices) < self.sequence_length:
                continue
            
            X_symbol = X[symbol_indices]
            y_symbol = y[symbol_indices]
            
            for i in range(len(X_symbol) - self.sequence_length):
                sequences.append(X_symbol[i:i + self.sequence_length])
                targets.append(y_symbol[i + self.sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def train_with_visualization(self, model, X_train, y_train, X_val, y_val, model_name):
        """Обучение с визуализацией"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 НАЧАЛО ОБУЧЕНИЯ TRANSFORMER: {model_name}")
        logger.info(f"{'='*70}")
        
        # Callbacks
        callbacks = [
            # Визуализация
            VisualizationCallback(self.log_dir, model_name, update_freq=5),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce LR
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=f'trained_model/{model_name}_transformer_best.h5',
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir=f'{self.log_dir}/tensorboard/{model_name}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                f'{self.log_dir}/{model_name}_history.csv',
                separator=',',
                append=False
            )
        ]
        
        # Обучение
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Оценка модели с визуализацией результатов"""
        
        # Предсказания
        y_pred = model.predict(X_test).flatten()
        
        # Метрики
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Процент правильного направления
        direction_accuracy = np.mean((y_pred > 0) == (y_test > 0))
        
        # Визуализация результатов
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Evaluation: {model_name}', fontsize=16)
        
        # График 1: Scatter plot
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0, 0].set_xlabel('True Values (%)')
        axes[0, 0].set_ylabel('Predictions (%)')
        axes[0, 0].set_title(f'Predictions vs True Values (R² = {r2:.3f})')
        axes[0, 0].grid(True)
        
        # График 2: Распределение ошибок
        errors = y_pred - y_test
        axes[0, 1].hist(errors, bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Prediction Error (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Error Distribution (MAE = {mae:.3f}%)')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].grid(True)
        
        # График 3: Временной ряд предсказаний
        sample_size = min(500, len(y_test))
        axes[1, 0].plot(y_test[:sample_size], 'b-', label='True', alpha=0.7)
        axes[1, 0].plot(y_pred[:sample_size], 'r-', label='Predicted', alpha=0.7)
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].set_title('Sample Predictions Timeline')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # График 4: Метрики
        axes[1, 1].axis('off')
        metrics_text = f"""
        Model Performance Metrics:
        
        MAE:  {mae:.4f}%
        RMSE: {rmse:.4f}%
        R²:   {r2:.4f}
        
        Direction Accuracy: {direction_accuracy:.2%}
        
        Average Prediction: {np.mean(y_pred):.3f}%
        Std Prediction:     {np.std(y_pred):.3f}%
        
        Average True:       {np.mean(y_test):.3f}%
        Std True:          {np.std(y_test):.3f}%
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                       verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/plots/{model_name}_evaluation.png', dpi=150)
        plt.close()
        
        logger.info(f"\n📊 Метрики {model_name}:")
        logger.info(f"   MAE: {mae:.4f}%")
        logger.info(f"   RMSE: {rmse:.4f}%")
        logger.info(f"   R²: {r2:.4f}")
        logger.info(f"   Direction Accuracy: {direction_accuracy:.2%}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
    
    def train(self):
        """Основной процесс обучения"""
        logger.info("\n" + "="*80)
        logger.info("🚀 НАЧАЛО ОБУЧЕНИЯ TRANSFORMER МОДЕЛИ")
        logger.info("="*80)
        
        try:
            # Создаем директории
            os.makedirs('trained_model', exist_ok=True)
            os.makedirs('logs', exist_ok=True)
            os.makedirs('plots', exist_ok=True)
            
            # Подключаемся к БД
            self.connect_db()
            
            # Загружаем данные
            data = self.load_and_prepare_data()
            if not data:
                logger.error("❌ Нет данных для обучения")
                return
            
            # Создаем пустую входную последовательность для инициализации
            dummy_input = tf.zeros((1, self.sequence_length, len(self.feature_columns)))
            
            # Модели для обучения
            model_configs = [
                ('buy_transformer', data['y_buy']),
                ('sell_transformer', data['y_sell'])
            ]
            
            results = {}
            
            for model_name, y_target in model_configs:
                logger.info(f"\n{'='*60}")
                logger.info(f"Обучение {model_name}")
                logger.info(f"{'='*60}")
                
                # Создаем последовательности
                X_seq, y_seq = self.create_sequences(
                    data['X'], y_target, data['symbols']
                )
                
                if len(X_seq) == 0:
                    logger.warning(f"⚠️ Недостаточно данных для {model_name}")
                    continue
                
                # Разделение данных
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
                model = self.create_model(X_seq.shape[2], model_name)
                
                # Инициализируем модель
                _ = model(dummy_input)
                
                logger.info(f"✅ Создана модель с {model.count_params():,} параметрами")
                
                # Обучаем с визуализацией
                history = self.train_with_visualization(
                    model, X_train, y_train, X_val, y_val, model_name
                )
                
                # Оцениваем
                test_metrics = self.evaluate_model(
                    model, X_test, y_test, model_name
                )
                
                # Сохраняем результаты
                results[model_name] = {
                    'history': history.history,
                    'metrics': test_metrics,
                    'model': model
                }
                
                self.models[model_name] = model
                
                # Создаем итоговый отчет
                self.create_final_report(model_name, history, test_metrics)
            
            # Сохраняем все модели
            self.save_models(results)
            
            logger.info("\n✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            logger.info(f"📊 Графики сохранены в: {self.log_dir}/plots/")
            logger.info(f"📝 Логи сохранены в: {self.log_dir}/")
            
        except Exception as e:
            logger.error(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            logger.error(f"Трейсбек:", exc_info=True)
            raise
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()
                logger.info("📤 Соединение с БД закрыто")
    
    def create_final_report(self, model_name, history, metrics):
        """Создание итогового отчета по модели"""
        
        report = f"""
{'='*80}
ОТЧЕТ ПО ОБУЧЕНИЮ МОДЕЛИ: {model_name}
{'='*80}

Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

АРХИТЕКТУРА МОДЕЛИ:
- Тип: Temporal Fusion Transformer (TFT)
- Sequence Length: {self.sequence_length}
- Model Dimension: {self.d_model}
- Number of Heads: {self.num_heads}
- Transformer Blocks: {self.num_transformer_blocks}
- Dropout Rate: {self.dropout_rate}

ПАРАМЕТРЫ ОБУЧЕНИЯ:
- Batch Size: {self.batch_size}
- Initial Learning Rate: {self.learning_rate}
- Epochs: {len(history.history['loss'])}
- Optimizer: Adam

ФИНАЛЬНЫЕ МЕТРИКИ:
- MAE: {metrics['mae']:.4f}%
- RMSE: {metrics['rmse']:.4f}%
- R²: {metrics['r2']:.4f}
- Direction Accuracy: {metrics['direction_accuracy']:.2%}

ЛУЧШИЕ РЕЗУЛЬТАТЫ:
- Best Val Loss: {min(history.history['val_loss']):.4f}
- Best Epoch: {history.history['val_loss'].index(min(history.history['val_loss'])) + 1}

{'='*80}
"""
        
        with open(f'{self.log_dir}/{model_name}_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"📝 Отчет сохранен: {self.log_dir}/{model_name}_report.txt")
    
    def save_models(self, results):
        """Сохранение моделей и метаданных"""
        
        # Сохраняем модели
        for name, model in self.models.items():
            model_path = f'trained_model/{name}.h5'
            model.save_weights(model_path)
            logger.info(f"   ✅ Модель сохранена: {model_path}")
        
        # Сохраняем scaler
        with open('trained_model/scaler_transformer.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Сохраняем метаданные
        metadata = {
            'type': 'temporal_fusion_transformer',
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_transformer_blocks': self.num_transformer_blocks,
            'features': self.feature_columns,
            'created_at': datetime.now().isoformat(),
            'results': {
                name: result['metrics']
                for name, result in results.items()
            }
        }
        
        with open('trained_model/metadata_transformer.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("💾 Метаданные сохранены")


if __name__ == "__main__":
    trainer = TransformerTrainer()
    trainer.train()