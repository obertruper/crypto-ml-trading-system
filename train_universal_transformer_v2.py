#!/usr/bin/env python3
"""
Enhanced Temporal Fusion Transformer для криптотрейдинга v2.1
Улучшения v2.0:
- Корреляция с Bitcoin и market features
- Focal Loss для лучшей работы с дисбалансом
- Multi-scale convolutions
- Attention pooling
- Gradient accumulation
- Time-based features
- Ensemble support

Улучшения v2.1:
- OHLC features (нормализованные отношения, свечные паттерны)
- Symbol embeddings (one-hot для топ монет, категории)
- Относительные метрики к EMA/VWAP
- Layer Normalization для OHLC стабильности
- Оптимизация порога классификации
- Post-processing фильтры
"""

import os
import sys
import time
import json
import pickle
import yaml
import warnings
import argparse
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy import stats
import logging

# Настройка логирования
log_dir = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f"{log_dir}/plots", exist_ok=True)
os.makedirs(f"{log_dir}/tensorboard", exist_ok=True)
os.makedirs("trained_model", exist_ok=True)  # Создаем директорию для моделей

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database_utils import PostgreSQLManager

# Подавляем предупреждения
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Настройка GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"🖥️ GPU доступен: {gpus[0].name}")
        
        # Временно отключаем mixed precision для стабильности
        # TODO: включить обратно после отладки nan проблемы
        # policy = tf.keras.mixed_precision.Policy('mixed_float16')
        # tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("⚠️ Mixed precision временно отключен для стабильности")
    except RuntimeError as e:
        logger.error(f"Ошибка настройки GPU: {e}")


class MemoryCleanupCallback(keras.callbacks.Callback):
    """Callback для очистки памяти между эпохами"""
    
    def on_epoch_end(self, epoch, logs=None):
        # Очищаем кэш и собираем мусор
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Логирование использования памяти GPU (если доступно)
        if tf.config.list_physical_devices('GPU'):
            try:
                # Пытаемся получить информацию о памяти
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                if 'current' in gpu_memory:
                    used_mb = gpu_memory['current'] / 1024 / 1024
                    logger.info(f"💾 GPU память после эпохи {epoch+1}: {used_mb:.0f} MB")
            except:
                pass  # Игнорируем ошибки если функция недоступна


class VisualizationCallback(keras.callbacks.Callback):
    """Callback для визуализации процесса обучения"""
    
    def __init__(self, log_dir, model_name, update_freq=5, task='regression'):
        super().__init__()
        self.log_dir = log_dir
        self.model_name = model_name
        self.update_freq = update_freq
        self.epoch_count = 0
        self.task = task
        
        self.history = {
            'loss': [],
            'val_loss': [],
            'lr': []
        }
        
        if task == 'regression':
            self.history.update({
                'mae': [],
                'val_mae': [],
                'rmse': [],
                'val_rmse': []
            })
        else:  # classification
            self.history.update({
                'accuracy': [],
                'val_accuracy': [],
                'precision': [],
                'val_precision': [],
                'recall': [],
                'val_recall': []
            })
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f'Training Progress: {model_name}', fontsize=16)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        
        # Сохраняем общие метрики
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        
        # Безопасное получение learning rate
        try:
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, '__call__'):  # Если это schedule
                lr_value = lr(self.model.optimizer.iterations).numpy()
            else:
                lr_value = lr.numpy()
        except:
            lr_value = logs.get('lr', 0.0001)  # Fallback значение
        self.history['lr'].append(lr_value)
        
        # Сохраняем специфичные метрики
        if self.task == 'regression':
            self.history['mae'].append(logs.get('mae', 0))
            self.history['val_mae'].append(logs.get('val_mae', 0))
            self.history['rmse'].append(logs.get('rmse', 0))
            self.history['val_rmse'].append(logs.get('val_rmse', 0))
        else:  # classification
            self.history['accuracy'].append(logs.get('accuracy', 0))
            self.history['val_accuracy'].append(logs.get('val_accuracy', 0))
            self.history['precision'].append(logs.get('precision', 0))
            self.history['val_precision'].append(logs.get('val_precision', 0))
            self.history['recall'].append(logs.get('recall', 0))
            self.history['val_recall'].append(logs.get('val_recall', 0))
        
        # Сохраняем метрики в CSV
        metrics_df = pd.DataFrame(self.history)
        metrics_df.to_csv(f'{self.log_dir}/{self.model_name}_metrics.csv', index=False)
        
        # Обновляем графики
        if self.epoch_count % self.update_freq == 0:
            self.update_plots()
            
    def update_plots(self):
        """Обновление графиков"""
        epochs = range(1, len(self.history['loss']) + 1)
        
        # График 1: Loss
        self.axes[0, 0].clear()
        self.axes[0, 0].plot(epochs, self.history['loss'], 'b-', label='Train Loss', linewidth=2)
        self.axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        self.axes[0, 0].set_title('Model Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Основная метрика
        self.axes[0, 1].clear()
        if self.task == 'regression':
            self.axes[0, 1].plot(epochs, self.history['mae'], 'b-', label='Train MAE', linewidth=2)
            self.axes[0, 1].plot(epochs, self.history['val_mae'], 'r-', label='Val MAE', linewidth=2)
            self.axes[0, 1].set_title('Mean Absolute Error (%)')
            self.axes[0, 1].set_ylabel('MAE (%)')
        else:  # classification
            self.axes[0, 1].plot(epochs, self.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
            self.axes[0, 1].plot(epochs, self.history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
            self.axes[0, 1].set_title('Model Accuracy')
            self.axes[0, 1].set_ylabel('Accuracy')
            self.axes[0, 1].set_ylim(0, 1)
        
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Learning Rate
        self.axes[1, 0].clear()
        self.axes[1, 0].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        self.axes[1, 0].set_title('Learning Rate Schedule')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 0].set_yscale('log')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Статистика
        self.axes[1, 1].clear()
        self.axes[1, 1].axis('off')
        
        if self.task == 'regression':
            current_stats = f"""
Current Epoch: {len(self.history['loss'])}

Loss:
  Train:  {self.history['loss'][-1]:.4f}
  Val:    {self.history['val_loss'][-1]:.4f}
  
MAE (%):
  Train:  {self.history['mae'][-1]:.3f}%
  Val:    {self.history['val_mae'][-1]:.3f}%

RMSE (%):
  Train:  {self.history['rmse'][-1]:.3f}%
  Val:    {self.history['val_rmse'][-1]:.3f}%

Best Results:
  Val Loss: {min(self.history['val_loss']):.4f}
  Val MAE:  {min(self.history['val_mae']):.3f}%
  Epoch:    {self.history['val_loss'].index(min(self.history['val_loss'])) + 1}

Learning Rate: {self.history['lr'][-1]:.2e}
"""
        else:  # classification
            current_stats = f"""
Current Epoch: {len(self.history['loss'])}

Loss:
  Train:  {self.history['loss'][-1]:.4f}
  Val:    {self.history['val_loss'][-1]:.4f}
  
Accuracy:
  Train:  {self.history['accuracy'][-1]:.3f}
  Val:    {self.history['val_accuracy'][-1]:.3f}

Precision:
  Train:  {self.history['precision'][-1]:.3f}
  Val:    {self.history['val_precision'][-1]:.3f}
  
Recall:
  Train:  {self.history['recall'][-1]:.3f}
  Val:    {self.history['val_recall'][-1]:.3f}

Best Results:
  Val Loss: {min(self.history['val_loss']):.4f}
  Val Acc:  {max(self.history['val_accuracy']) if self.history['val_accuracy'] else 0:.3f}
  Epoch:    {self.history['val_accuracy'].index(max(self.history['val_accuracy'])) + 1 if self.history['val_accuracy'] else 0}

Learning Rate: {self.history['lr'][-1]:.2e}
"""
        
        self.axes[1, 1].text(0.1, 0.5, current_stats, fontsize=12, 
                            verticalalignment='center', family='monospace',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/plots/training_progress.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{self.log_dir}/plots/epoch_{self.epoch_count:03d}.png', dpi=100)
        
        logger.info(f"📊 График обновлен: эпоха {self.epoch_count}")


class FocalLoss(keras.losses.Loss):
    """Focal Loss для работы с дисбалансом классов"""
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        # Ensure float32 computation for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Label smoothing
        y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Clip predictions to prevent log(0) - более безопасные границы
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss with numerical stability
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Добавляем epsilon чтобы избежать log(0)
        p_t = tf.clip_by_value(p_t, epsilon, 1.0)
        
        # Focal loss computation
        focal_weight = tf.pow((1 - p_t), self.gamma)
        focal_loss = -alpha_t * focal_weight * tf.math.log(p_t + epsilon)
        
        # Проверка на nan/inf
        focal_loss = tf.where(tf.math.is_finite(focal_loss), focal_loss, 0.0)
        
        return tf.reduce_mean(focal_loss)


class MultiScaleConv1D(layers.Layer):
    """Multi-scale 1D Convolutions для захвата паттернов разных масштабов"""
    def __init__(self, filters, kernel_sizes=[3, 5, 7], **kwargs):
        super().__init__(**kwargs)
        self.convs = [
            layers.Conv1D(filters // len(kernel_sizes), kernel_size, padding='same', activation='relu')
            for kernel_size in kernel_sizes
        ]
        self.batch_norm = layers.BatchNormalization()
        
    def call(self, inputs):
        outputs = [conv(inputs) for conv in self.convs]
        concatenated = layers.concatenate(outputs, axis=-1)
        return self.batch_norm(concatenated)


class AttentionPooling(layers.Layer):
    """Attention-based pooling вместо GlobalAveragePooling"""
    def __init__(self, hidden_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.hidden_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.hidden_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            shape=(self.hidden_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, inputs):
        # Attention scores
        hidden = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        scores = tf.matmul(hidden, tf.expand_dims(self.u, -1))
        scores = tf.squeeze(scores, -1)
        
        # Softmax weights
        weights = tf.nn.softmax(scores, axis=1)
        weights = tf.expand_dims(weights, -1)
        
        # Weighted sum
        weighted_input = inputs * weights
        return tf.reduce_sum(weighted_input, axis=1)


class GatedResidualNetwork(layers.Layer):
    """Enhanced GRN с dropout и layer norm"""
    def __init__(self, hidden_dim, output_dim=None, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(self.hidden_dim, activation='elu')
        self.dense2 = layers.Dense(self.output_dim)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = layers.LayerNormalization()
        
        if self.output_dim != input_shape[-1]:
            self.skip_proj = layers.Dense(self.output_dim)
        else:
            self.skip_proj = None
            
        self.gate = layers.Dense(self.output_dim, activation='sigmoid')
        
    def call(self, inputs, training=False):
        hidden = self.dense1(inputs)
        hidden = self.dropout(hidden, training=training)
        hidden = self.dense2(hidden)
        hidden = self.dropout(hidden, training=training)
        
        if self.skip_proj is not None:
            residual = self.skip_proj(inputs)
        else:
            residual = inputs
            
        gate_values = self.gate(hidden)
        output = gate_values * hidden + (1 - gate_values) * residual
        return self.layer_norm(output)


class PositionalEncoding(layers.Layer):
    """Позиционное кодирование с learnable parameters"""
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
    def build(self, input_shape):
        # Standard positional encoding
        position = tf.expand_dims(tf.range(0, self.sequence_length, dtype=tf.float32), 1)
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * 
                          -(tf.math.log(10000.0) / self.d_model))
        
        pe = tf.zeros((self.sequence_length, self.d_model))
        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)
        
        # Правильное чередование sin и cos
        # pe_sin и pe_cos имеют форму (sequence_length, d_model//2)
        # Создаем полный pe тензор
        pe_list = []
        for i in range(self.d_model):
            if i % 2 == 0:
                # Четные позиции - sin
                pe_list.append(pe_sin[:, i // 2:i // 2 + 1])
            else:
                # Нечетные позиции - cos
                pe_list.append(pe_cos[:, i // 2:i // 2 + 1])
        
        pe = tf.concat(pe_list, axis=1)
        
        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(1, self.sequence_length, self.d_model),
            initializer=tf.constant_initializer(pe.numpy()),
            trainable=True  # Делаем обучаемым
        )
        
    def call(self, inputs):
        return inputs + self.pe


class EnhancedTransformerBlock(layers.Layer):
    """Улучшенный Transformer блок с pre-norm и gate mechanism"""
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model,
            dropout=dropout
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.gate = layers.Dense(d_model, activation='sigmoid')
        
    def call(self, inputs, training=False):
        # Pre-norm
        normed = self.layernorm1(inputs)
        attn_output = self.attention(normed, normed, training=training)
        gate1 = self.gate(attn_output)
        out1 = inputs + gate1 * attn_output
        
        # Feed-forward with pre-norm
        normed2 = self.layernorm2(out1)
        ffn_output = self.ffn(normed2, training=training)
        gate2 = self.gate(ffn_output)
        return out1 + gate2 * ffn_output


class EnhancedTemporalFusionTransformer(keras.Model):
    """Enhanced TFT с улучшениями v2.1"""
    def __init__(self, num_features, sequence_length, d_model=256, num_heads=8, 
                 num_transformer_blocks=6, mlp_units=[512, 256], dropout=0.3, 
                 task='regression', use_multi_scale=True):
        super().__init__()
        
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.task = task
        self.use_multi_scale = use_multi_scale
        
        # Input normalization для стабильности OHLC features
        self.input_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Variable selection network
        self.vsn_dense = layers.Dense(d_model, activation='relu')
        self.vsn_grn = GatedResidualNetwork(d_model, num_features, dropout)
        
        # Multi-scale convolutions (опционально)
        if use_multi_scale:
            self.multi_scale_conv = MultiScaleConv1D(d_model)
            self.projection_layer = layers.Dense(d_model)  # Проекция после конкатенации
        
        # LSTM encoder
        self.lstm_encoder = layers.Bidirectional(
            layers.LSTM(d_model // 2, return_sequences=True, dropout=dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        
        # Enhanced Transformer blocks
        self.transformer_blocks = [
            EnhancedTransformerBlock(d_model, num_heads, mlp_units[0], dropout)
            for _ in range(num_transformer_blocks)
        ]
        
        # Attention pooling вместо GlobalAveragePooling
        self.attention_pooling = AttentionPooling(d_model)
        
        # Output layers
        if task == 'regression':
            self.output_dense = keras.Sequential([
                layers.Dense(mlp_units[0], activation='gelu'),
                layers.Dropout(dropout),
                layers.BatchNormalization(),
                layers.Dense(mlp_units[1], activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(1)  # dtype='float32' убрано, так как mixed precision отключен
            ])
        else:  # classification
            self.output_dense = keras.Sequential([
                layers.Dense(mlp_units[0], activation='gelu'),
                layers.Dropout(dropout),
                layers.BatchNormalization(),
                layers.Dense(mlp_units[1], activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(1, activation='sigmoid')  # dtype='float32' убрано, так как mixed precision отключен
            ])
    
    def call(self, inputs, training=False):
        # Input normalization
        normalized_inputs = self.input_norm(inputs)
        
        # Variable selection
        selected = self.vsn_dense(normalized_inputs)
        selected = self.vsn_grn(selected, training=training)
        
        # Multi-scale convolutions
        if self.use_multi_scale:
            conv_features = self.multi_scale_conv(selected)
            selected = layers.concatenate([selected, conv_features], axis=-1)
            selected = self.projection_layer(selected)  # Project back to d_model
        
        # LSTM encoding
        lstm_out = self.lstm_encoder(selected, training=training)
        
        # Positional encoding
        x = self.pos_encoding(lstm_out)
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        # Attention pooling
        pooled = self.attention_pooling(x)
        
        # Output
        return self.output_dense(pooled, training=training)


class MarketFeatureExtractor:
    """Извлечение рыночных признаков включая корреляцию с BTC"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.btc_data = None
        self.market_data = {}
        
    def load_market_data(self):
        """Загрузка данных BTC и топ монет для корреляций"""
        logger.info("📊 Загрузка рыночных данных для корреляций...")
        
        # Загружаем BTC
        query_btc = """
        SELECT timestamp, close, volume,
               (high - low) / close as volatility
        FROM raw_market_data
        WHERE symbol = 'BTCUSDT' 
          AND market_type = 'futures'
        ORDER BY timestamp
        """
        self.btc_data = self.db.fetch_dataframe(query_btc).set_index('timestamp')
        
        # Загружаем топ монеты для cross-correlations
        top_symbols = ['ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
        for symbol in top_symbols:
            query = f"""
            SELECT timestamp, close
            FROM raw_market_data
            WHERE symbol = '{symbol}'
              AND market_type = 'futures'
            ORDER BY timestamp
            """
            df = self.db.fetch_dataframe(query)
            if not df.empty:
                self.market_data[symbol] = df.set_index('timestamp')['close']
        
        logger.info(f"✅ Загружены данные BTC: {len(self.btc_data)} записей")
        logger.info(f"✅ Загружены корреляции с: {list(self.market_data.keys())}")
    
    def calculate_features(self, df):
        """Расчет дополнительных рыночных признаков включая OHLC features"""
        logger.info("🔧 Расчет рыночных признаков и OHLC features...")
        
        # Добавляем временные признаки
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['datetime']).dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Циклическое кодирование времени
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Группируем по символам для расчета корреляций и OHLC features
        enhanced_dfs = []
        for symbol, symbol_df in df.groupby('symbol'):
            symbol_df = symbol_df.copy()
            
            # BTC correlation features
            if symbol != 'BTCUSDT' and self.btc_data is not None:
                # Выравниваем по timestamp
                merged = symbol_df.merge(
                    self.btc_data[['close', 'volatility']], 
                    left_on='timestamp', 
                    right_index=True,
                    how='left',
                    suffixes=('', '_btc')
                )
                
                # Корреляция с BTC (rolling)
                if len(merged) > 20:
                    symbol_df['btc_correlation_20'] = merged['close'].rolling(20).corr(merged['close_btc'])
                    symbol_df['btc_correlation_60'] = merged['close'].rolling(60).corr(merged['close_btc'])
                else:
                    symbol_df['btc_correlation_20'] = 0
                    symbol_df['btc_correlation_60'] = 0
                
                # BTC price changes
                symbol_df['btc_return_1h'] = merged['close_btc'].pct_change(4)
                symbol_df['btc_return_4h'] = merged['close_btc'].pct_change(16)
                # Рассчитываем волатильность BTC
                btc_returns = merged['close_btc'].pct_change()
                symbol_df['btc_volatility'] = btc_returns.rolling(20).std()
                
                # Relative strength to BTC
                symbol_df['relative_strength_btc'] = (
                    symbol_df['close'].pct_change(20) / 
                    merged['close_btc'].pct_change(20).replace(0, 1)
                )
            else:
                # Для BTC или если нет данных
                symbol_df['btc_correlation_20'] = 1 if symbol == 'BTCUSDT' else 0
                symbol_df['btc_correlation_60'] = 1 if symbol == 'BTCUSDT' else 0
                symbol_df['btc_return_1h'] = 0
                symbol_df['btc_return_4h'] = 0
                symbol_df['btc_volatility'] = 0
                symbol_df['relative_strength_btc'] = 1
            
            # OHLC Features - критически важные для понимания движения цены
            # Нормализованные отношения (относительно close)
            symbol_df['open_ratio'] = (symbol_df['open'] / symbol_df['close'] - 1)
            symbol_df['high_ratio'] = (symbol_df['high'] / symbol_df['close'] - 1)
            symbol_df['low_ratio'] = (symbol_df['low'] / symbol_df['close'] - 1)
            symbol_df['hl_spread'] = (symbol_df['high'] - symbol_df['low']) / symbol_df['close']
            
            # Свечные паттерны
            symbol_df['body_size'] = np.abs(symbol_df['close'] - symbol_df['open']) / symbol_df['close']
            symbol_df['upper_shadow'] = (symbol_df['high'] - symbol_df[['open', 'close']].max(axis=1)) / symbol_df['close']
            symbol_df['lower_shadow'] = (symbol_df[['open', 'close']].min(axis=1) - symbol_df['low']) / symbol_df['close']
            symbol_df['is_bullish'] = (symbol_df['close'] > symbol_df['open']).astype(int)
            
            # Логарифмические returns для стабильности
            symbol_df['log_return'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1)).fillna(0)
            symbol_df['log_volume'] = np.log(symbol_df['volume'] + 1)
            
            # Относительные метрики к скользящим средним
            if 'technical_indicators' in symbol_df.columns:
                # Извлекаем EMA значения из technical_indicators
                ema_15_values = []
                ema_50_values = []
                vwap_values = []
                
                for idx, row in symbol_df.iterrows():
                    indicators = row['technical_indicators']
                    ema_15_values.append(indicators.get('ema_15', row['close']))
                    ema_50_values.append(indicators.get('ema_50', row['close']))
                    vwap_values.append(indicators.get('vwap_val', row['close']))
                
                symbol_df['price_to_ema15'] = symbol_df['close'] / pd.Series(ema_15_values, index=symbol_df.index) - 1
                symbol_df['price_to_ema50'] = symbol_df['close'] / pd.Series(ema_50_values, index=symbol_df.index) - 1
                symbol_df['price_to_vwap'] = symbol_df['close'] / pd.Series(vwap_values, index=symbol_df.index) - 1
            else:
                # Добавляем дефолтные значения если technical_indicators отсутствует
                symbol_df['price_to_ema15'] = 0.0
                symbol_df['price_to_ema50'] = 0.0
                symbol_df['price_to_vwap'] = 0.0
            
            # Symbol embeddings
            # Топ 10 монет по капитализации
            top_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
                          'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT']
            
            for top_symbol in top_symbols:
                symbol_df[f'is_{top_symbol.lower().replace("usdt", "")}'] = int(symbol == top_symbol)
            
            # Категории монет
            major_coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            meme_coins = ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', '1000PEPEUSDT', 'FLOKIUSDT', 'WIFUSDT']
            defi_coins = ['AAVEUSDT', 'UNIUSDT', 'CAKEUSDT', 'DYDXUSDT']
            
            symbol_df['is_major'] = int(symbol in major_coins)
            symbol_df['is_meme'] = int(symbol in meme_coins)
            symbol_df['is_defi'] = int(symbol in defi_coins)
            symbol_df['is_alt'] = int(not (symbol in major_coins or symbol in meme_coins or symbol in defi_coins))
            
            # Market regime (volatility-based)
            if 'volatility_16' in symbol_df.columns:
                vol_percentile = symbol_df['volatility_16'].rolling(100).rank(pct=True)
                symbol_df['market_regime_low_vol'] = (vol_percentile < 0.33).astype(int)
                symbol_df['market_regime_med_vol'] = ((vol_percentile >= 0.33) & (vol_percentile < 0.67)).astype(int)
                symbol_df['market_regime_high_vol'] = (vol_percentile >= 0.67).astype(int)
            else:
                symbol_df['market_regime_low_vol'] = 0
                symbol_df['market_regime_med_vol'] = 1
                symbol_df['market_regime_high_vol'] = 0
            
            # Умная обработка NaN значений
            numeric_columns = symbol_df.select_dtypes(include=[np.number]).columns
            
            # Для временных рядов используем forward fill, затем backward fill
            for col in numeric_columns:
                if 'correlation' in col or 'return' in col or 'ratio' in col:
                    # Для корреляций и returns используем forward fill
                    symbol_df[col] = symbol_df[col].ffill().bfill()
                else:
                    # Для остальных - заполняем нулями
                    symbol_df[col] = symbol_df[col].fillna(0)
            
            enhanced_dfs.append(symbol_df)
        
        result_df = pd.concat(enhanced_dfs, ignore_index=True)
        logger.info("✅ Рыночные признаки рассчитаны")
        return result_df


class EnhancedTransformerTrainer:
    """Enhanced Trainer с gradient accumulation и ensemble support"""
    
    def __init__(self, db_manager: PostgreSQLManager, config_path='config.yaml', 
                 task='regression', ensemble_size=1, test_mode=False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.db = db_manager
        self.task = task
        self.ensemble_size = ensemble_size
        self.test_mode = test_mode
        self.sequence_length = self.config['model']['sequence_length']
        
        # Параметры для тестового режима
        if self.test_mode:
            self.batch_size = 8  # Еще меньше для теста
            self.gradient_accumulation_steps = 2  # Быстрее
            self.epochs = 3  # Только 3 эпохи
            self.test_symbols = ['BTCUSDT', 'ETHUSDT']  # Только 2 символа
            logger.info("⚡ Тестовый режим: batch_size=8, epochs=3, symbols=2")
        else:
            self.batch_size = 8  # Уменьшено для RTX 4090 (24GB)
            self.gradient_accumulation_steps = 4  # Эффективный batch_size = 32
            self.epochs = 100
        
        # Enhanced архитектура (оптимизировано для RTX 4090)
        self.d_model = 128  # Уменьшено с 256
        self.num_heads = 8
        self.num_transformer_blocks = 4  # Уменьшено с 6
        self.ff_dim = 256  # Уменьшено с 512
        self.dropout_rate = 0.3  # Увеличено
        
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_columns = None
        self.log_dir = log_dir
        
        # Market feature extractor
        self.market_extractor = MarketFeatureExtractor(db_manager)
        
        # Расширенный список индикаторов
        self.TECHNICAL_INDICATORS = [
            # Трендовые индикаторы
            'ema_15', 'adx_val', 'adx_plus_di', 'adx_minus_di',
            'macd_val', 'macd_signal', 'macd_hist', 'sar',
            'ichimoku_conv', 'ichimoku_base', 'aroon_up', 'aroon_down',
            
            # Осцилляторы
            'rsi_val', 'stoch_k', 'stoch_d', 'cci_val', 'roc_val',
            'williams_r', 'awesome_osc', 'ultimate_osc',
            
            # Волатильность
            'atr_val', 'bb_position', 'bb_width', 'donchian_position',
            'keltner_position', 'ulcer_index', 'mass_index',
            
            # Объемные индикаторы
            'obv_val', 'obv_signal', 'cmf_val', 'force_index',
            'eom_val', 'vpt_val', 'nvi_val', 'vwap_val',
            
            # Дополнительные
            'ema_50', 'ema_200', 'trix_val', 'trix_signal',
            'vortex_pos', 'vortex_neg', 'vortex_ratio',
            'price_change_1', 'price_change_4', 'price_change_16',
            'volatility_4', 'volatility_16', 'volume_ratio'
        ]
        
        # Дополнительные market features
        self.MARKET_FEATURES = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
            'btc_correlation_20', 'btc_correlation_60',
            'btc_return_1h', 'btc_return_4h', 'btc_volatility',
            'relative_strength_btc',
            'market_regime_low_vol', 'market_regime_med_vol', 'market_regime_high_vol'
        ]
        
        # OHLC features
        self.OHLC_FEATURES = [
            'open_ratio', 'high_ratio', 'low_ratio', 'hl_spread',
            'body_size', 'upper_shadow', 'lower_shadow', 'is_bullish',
            'log_return', 'log_volume',
            'price_to_ema15', 'price_to_ema50', 'price_to_vwap'
        ]
        
        # Symbol embeddings
        self.SYMBOL_FEATURES = [
            'is_btc', 'is_eth', 'is_bnb', 'is_sol', 'is_xrp',
            'is_ada', 'is_doge', 'is_avax', 'is_dot', 'is_matic',
            'is_major', 'is_meme', 'is_defi', 'is_alt',
            'market_regime_low_vol', 'market_regime_med_vol', 'market_regime_high_vol'
        ]
        
        self.feature_importance = {}
        
    def convert_to_binary_labels(self, returns, threshold=0.3):
        """Преобразование expected returns в бинарные метки"""
        return (returns > threshold).astype(np.float32)
    
    def create_model_with_warmup(self, input_shape, name):
        """Создание модели с warmup learning rate"""
        model = EnhancedTemporalFusionTransformer(
            num_features=input_shape[1],
            sequence_length=input_shape[0],
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_transformer_blocks=self.num_transformer_blocks,
            mlp_units=[self.ff_dim, self.ff_dim//2],
            dropout=self.dropout_rate,
            task=self.task,
            use_multi_scale=True
        )
        
        # Warmup schedule - уменьшаем для стабильности с mixed precision
        initial_learning_rate = 0.00001  # Уменьшено в 10 раз
        target_learning_rate = 0.0001   # Уменьшено в 3 раза
        warmup_steps = 2000  # Увеличиваем warmup период
        
        lr_schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate,
            decay_steps=warmup_steps,
            end_learning_rate=target_learning_rate,
            power=1.0
        )
        
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Оборачиваем оптимизатор для mixed precision
        # TODO: раскомментировать когда включим mixed precision обратно
        # if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        #     optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # Компиляция
        if self.task == 'regression':
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.Huber(delta=1.0),
                metrics=[
                    keras.metrics.MeanAbsoluteError(name='mae'),
                    keras.metrics.RootMeanSquaredError(name='rmse')
                ]
            )
        else:  # classification
            model.compile(
                optimizer=optimizer,
                loss=FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05),
                metrics=[
                    keras.metrics.BinaryAccuracy(name='accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')
                ]
            )
        
        # Инициализация весов
        dummy_input = tf.zeros((1, input_shape[0], input_shape[1]))
        _ = model(dummy_input)
        
        logger.info(f"✅ Создана enhanced модель {name} с {model.count_params():,} параметрами")
        logger.info(f"   Эффективный batch size: {self.batch_size * self.gradient_accumulation_steps}")
        
        return model
    
    def augment_data(self, X, y, augmentation_factor=0.1):
        """Аугментация данных с noise injection и mixup (оптимизированная версия)"""
        # Выбираем только 50% данных для аугментации для экономии памяти
        sample_size = len(X) // 2
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]
        y_sample = y[sample_indices]
        
        augmented_X = []
        augmented_y = []
        
        # Только Mixup (без noise injection для экономии памяти)
        indices = np.random.permutation(sample_size)
        lambda_mix = np.random.beta(0.2, 0.2, size=(sample_size, 1, 1))
        X_mixed = lambda_mix * X_sample + (1 - lambda_mix) * X_sample[indices]
        
        if self.task == 'regression':
            lambda_y = lambda_mix.squeeze()
            y_mixed = lambda_y * y_sample + (1 - lambda_y) * y_sample[indices]
        else:
            # Для классификации используем тот же lambda
            lambda_y = lambda_mix.squeeze()
            y_mixed = (lambda_y * y_sample + (1 - lambda_y) * y_sample[indices] > 0.5).astype(np.float32)
        
        augmented_X.append(X_mixed)
        augmented_y.append(y_mixed)
        
        # Объединяем (оригинал + mixup)
        X_augmented = np.concatenate([X] + augmented_X, axis=0)
        y_augmented = np.concatenate([y] + augmented_y, axis=0)
        
        # Перемешиваем
        shuffle_indices = np.random.permutation(len(X_augmented))
        
        return X_augmented[shuffle_indices], y_augmented[shuffle_indices]
    
    def load_data(self):
        """Загрузка данных с market features"""
        logger.info("📊 Загрузка данных из PostgreSQL...")
        start_time = time.time()
        
        # Сначала получаем список символов
        symbols_query = """
        SELECT DISTINCT p.symbol 
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
        ORDER BY p.symbol
        """
        
        symbols_df = self.db.fetch_dataframe(symbols_query)
        symbols = symbols_df['symbol'].tolist()
        
        # В тестовом режиме ограничиваем символы
        if self.test_mode:
            symbols = [s for s in symbols if s in self.test_symbols]
            logger.info(f"⚡ Тестовый режим: загружаем только {symbols}")
        
        logger.info(f"📋 Найдено {len(symbols)} символов для загрузки")
        
        # Загружаем данные по символам батчами
        all_data = []
        batch_size = 5  # Загружаем по 5 символов за раз
        
        from tqdm import tqdm
        for i in tqdm(range(0, len(symbols), batch_size), desc="Загрузка батчей"):
            batch_symbols = symbols[i:i+batch_size]
            symbols_str = "', '".join(batch_symbols)
            
            batch_query = f"""
            SELECT 
                p.symbol, p.timestamp, p.datetime,
                p.technical_indicators,
                p.buy_expected_return,
                p.sell_expected_return,
                p.is_long_entry,
                p.is_short_entry,
                p.open, p.high, p.low, p.close, p.volume
            FROM processed_market_data p
            JOIN raw_market_data r ON p.raw_data_id = r.id
            WHERE p.technical_indicators IS NOT NULL
              AND r.market_type = 'futures'
              AND p.symbol IN ('{symbols_str}')
              {"AND p.timestamp > EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - INTERVAL '10 days'))::bigint" if self.test_mode else ""}
            ORDER BY p.symbol, p.timestamp
            """
            
            try:
                batch_df = self.db.fetch_dataframe(batch_query)
                all_data.append(batch_df)
                logger.info(f"   ✅ Загружено {len(batch_df):,} записей для {batch_symbols}")
            except Exception as e:
                logger.warning(f"   ⚠️ Ошибка при загрузке {batch_symbols}: {e}")
                # Пробуем загрузить по одному символу
                for symbol in batch_symbols:
                    try:
                        single_query = f"""
                        SELECT 
                            p.symbol, p.timestamp, p.datetime,
                            p.technical_indicators,
                            p.buy_expected_return,
                            p.sell_expected_return,
                            p.is_long_entry,
                            p.is_short_entry,
                            p.open, p.high, p.low, p.close, p.volume
                        FROM processed_market_data p
                        JOIN raw_market_data r ON p.raw_data_id = r.id
                        WHERE p.technical_indicators IS NOT NULL
                          AND r.market_type = 'futures'
                          AND p.symbol = '{symbol}'
                        ORDER BY p.timestamp
                        """
                        single_df = self.db.fetch_dataframe(single_query)
                        all_data.append(single_df)
                        logger.info(f"   ✅ Загружено {len(single_df):,} записей для {symbol}")
                    except Exception as e2:
                        logger.error(f"   ❌ Не удалось загрузить {symbol}: {e2}")
        
        # Объединяем все данные
        if not all_data:
            raise ValueError("Не удалось загрузить данные!")
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Загружаем market data
        self.market_extractor.load_market_data()
        
        # Добавляем market features
        df = self.market_extractor.calculate_features(df)
        
        load_time = time.time() - start_time
        logger.info(f"✅ Загружено {len(df)} записей за {load_time:.2f} секунд ({load_time/60:.1f} минут)")
        
        if len(df) == 0:
            raise ValueError("Нет данных для обучения!")
        
        # Статистика
        symbol_counts = df['symbol'].value_counts()
        logger.info("📊 Распределение по символам:")
        for symbol, count in symbol_counts.items():
            logger.info(f"   {symbol}: {count:,} записей")
        
        return df
    
    def prepare_features_and_targets(self, df):
        """Подготовка признаков с market features"""
        logger.info("🔧 Подготовка признаков и целевых значений...")
        start_time = time.time()
        
        all_features = self.TECHNICAL_INDICATORS + self.MARKET_FEATURES + self.OHLC_FEATURES + self.SYMBOL_FEATURES
        logger.info(f"📊 Используется {len(all_features)} признаков")
        logger.info(f"   - Технические индикаторы: {len(self.TECHNICAL_INDICATORS)}")
        logger.info(f"   - Market features: {len(self.MARKET_FEATURES)}")
        logger.info(f"   - OHLC features: {len(self.OHLC_FEATURES)}")
        logger.info(f"   - Symbol features: {len(self.SYMBOL_FEATURES)}")
        
        # Временное разделение данных
        train_end = int(len(df) * 0.7)
        val_end = int(len(df) * 0.85)
        
        df['split'] = 'test'
        df.loc[df.index < train_end, 'split'] = 'train'
        df.loc[(df.index >= train_end) & (df.index < val_end), 'split'] = 'val'
        
        # Группируем по split
        grouped_data = {
            'train': {'X': [], 'y_buy': [], 'y_sell': [], 'symbols': []},
            'val': {'X': [], 'y_buy': [], 'y_sell': [], 'symbols': []},
            'test': {'X': [], 'y_buy': [], 'y_sell': [], 'symbols': []}
        }
        
        # Обработка по символам
        from tqdm import tqdm
        for symbol, symbol_df in tqdm(df.groupby('symbol'), desc="Обработка символов"):
            for split_name, split_df in symbol_df.groupby('split'):
                X_split = []
                
                for _, row in split_df.iterrows():
                    feature_values = []
                    
                    # Технические индикаторы
                    indicators = row['technical_indicators']
                    for indicator in self.TECHNICAL_INDICATORS:
                        if indicator in ['volatility_4', 'volatility_16']:
                            # Эти значения могут быть в основных колонках
                            value = row.get(indicator, indicators.get(indicator, 0.0))
                        else:
                            value = indicators.get(indicator, 0.0)
                        
                        if value is None or pd.isna(value):
                            value = 0.0
                        feature_values.append(float(value))
                    
                    # Market features
                    for feature in self.MARKET_FEATURES:
                        value = row.get(feature, 0.0)
                        if value is None or pd.isna(value):
                            value = 0.0
                        feature_values.append(float(value))
                    
                    # OHLC features
                    for feature in self.OHLC_FEATURES:
                        value = row.get(feature, 0.0)
                        if value is None or pd.isna(value):
                            value = 0.0
                        feature_values.append(float(value))
                    
                    # Symbol features
                    for feature in self.SYMBOL_FEATURES:
                        value = row.get(feature, 0.0)
                        if value is None or pd.isna(value):
                            value = 0.0
                        feature_values.append(float(value))
                    
                    # Дополнительные инженерные признаки
                    # RSI extremes
                    rsi = indicators.get("rsi_val", 50.0)
                    feature_values.append(1.0 if rsi is not None and rsi < 30 else 0.0)
                    feature_values.append(1.0 if rsi is not None and rsi > 70 else 0.0)
                    
                    # MACD signal
                    macd = indicators.get("macd_val", 0.0)
                    macd_signal = indicators.get("macd_signal", 0.0)
                    feature_values.append(1.0 if macd is not None and macd_signal is not None and macd > macd_signal else 0.0)
                    
                    # Bollinger bands position
                    bb_position = indicators.get("bb_position", 0.5)
                    feature_values.append(1.0 if bb_position is not None and bb_position < 0.2 else 0.0)
                    feature_values.append(1.0 if bb_position is not None and bb_position > 0.8 else 0.0)
                    
                    # Trend strength
                    adx = indicators.get("adx_val", 0.0)
                    feature_values.append(1.0 if adx is not None and adx > 25 else 0.0)
                    
                    # Volume spike
                    volume_ratio = indicators.get("volume_ratio", 1.0)
                    feature_values.append(1.0 if volume_ratio is not None and volume_ratio > 2.0 else 0.0)
                    
                    X_split.append(feature_values)
                
                # Целевые значения
                y_buy = split_df["buy_expected_return"].values.astype(float)
                y_sell = split_df["sell_expected_return"].values.astype(float)
                
                # Добавляем в соответствующий split
                grouped_data[split_name]["X"].extend(X_split)
                grouped_data[split_name]["y_buy"].extend(y_buy)
                grouped_data[split_name]["y_sell"].extend(y_sell)
                grouped_data[split_name]["symbols"].extend([symbol] * len(X_split))
        
        prep_time = time.time() - start_time
        logger.info(f"✅ Подготовка завершена за {prep_time:.2f} секунд ({prep_time/60:.1f} минут)")
        
        return grouped_data
    
    def create_sequences(self, X, y, symbols, stride=3):
        """Создание последовательностей с меньшим stride для большего количества данных"""
        logger.info(f"🔄 Создание последовательностей длиной {self.sequence_length}...")
        start_time = time.time()
        
        sequences = []
        targets = []
        seq_symbols = []
        
        unique_symbols = np.unique(symbols)
        
        for symbol in unique_symbols:
            symbol_mask = symbols == symbol
            symbol_indices = np.where(symbol_mask)[0]
            
            if len(symbol_indices) < self.sequence_length + 1:
                continue
            
            X_symbol = X[symbol_indices]
            y_symbol = y[symbol_indices]
            
            # Создаем последовательности с меньшим stride
            for i in range(0, len(X_symbol) - self.sequence_length, stride):
                sequences.append(X_symbol[i:i + self.sequence_length])
                targets.append(y_symbol[i + self.sequence_length])
                seq_symbols.append(symbol)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        seq_time = time.time() - start_time
        logger.info(f"✅ Создано {len(sequences)} последовательностей за {seq_time:.2f} секунд")
        logger.info(f"   Форма данных: {sequences.shape}")
        
        return sequences, targets, np.array(seq_symbols)
    
    def train_model_with_gradient_accumulation(self, model, X_train, y_train, X_val, y_val, model_name):
        """Обучение с gradient accumulation для большего эффективного batch size"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 НАЧАЛО ОБУЧЕНИЯ: {model_name}")
        logger.info(f"{'='*70}")
        
        # Custom training loop для gradient accumulation
        @tf.function
        def train_step(inputs, targets, accumulation_steps):
            accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
            
            for i in range(accumulation_steps):
                start_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size
                
                batch_inputs = inputs[start_idx:end_idx]
                batch_targets = targets[start_idx:end_idx]
                
                with tf.GradientTape() as tape:
                    predictions = model(batch_inputs, training=True)
                    loss = model.compiled_loss(batch_targets, predictions)
                    loss = loss / accumulation_steps
                
                gradients = tape.gradient(loss, model.trainable_variables)
                accumulated_gradients = [
                    acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)
                ]
            
            model.optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
            return loss * accumulation_steps
        
        # Callbacks
        callbacks = [
            VisualizationCallback(self.log_dir, model_name, update_freq=5, task=self.task),
            
            MemoryCleanupCallback(),  # Добавляем очистку памяти
            
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Убрано LearningRateScheduler - используется CosineDecay в оптимизаторе
            
            keras.callbacks.ModelCheckpoint(
                filepath=f'trained_model/{model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                save_weights_only=False  # Сохраняем всю модель
            ),
            
            keras.callbacks.TensorBoard(
                log_dir=f'{self.log_dir}/tensorboard/{model_name}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        # Обучение с аугментацией
        if self.task == 'classification_binary':
            # Аугментация данных
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
            
            # Подсчет весов классов
            unique, counts = np.unique(y_train_aug, return_counts=True)
            class_weight = {0: counts[1] / counts[0], 1: 1.0} if len(unique) == 2 else None
            
            if class_weight:
                logger.info(f"📊 Баланс классов после аугментации - 0: {counts[0]:,}, 1: {counts[1]:,}")
                logger.info(f"⚖️ Веса классов - 0: {class_weight[0]:.2f}, 1: {class_weight[1]:.2f}")
            
            history = model.fit(
                X_train_aug, y_train_aug,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size * self.gradient_accumulation_steps,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
        else:
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
            
            history = model.fit(
                X_train_aug, y_train_aug,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size * self.gradient_accumulation_steps,
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def evaluate_model_with_uncertainty(self, model, X_test, y_test, model_name, n_samples=7):
        """Оценка модели с uncertainty estimation через MC Dropout"""
        logger.info(f"\n📊 Оценка модели {model_name} с uncertainty estimation...")
        
        # Батчированная обработка для экономии памяти
        batch_size = 1000  # Меньший размер батча для оценки
        n_samples_test = len(X_test)
        predictions = []
        
        # Multiple forward passes с dropout
        for sample_idx in range(n_samples):
            batch_predictions = []
            
            # Обрабатываем по батчам
            for i in range(0, n_samples_test, batch_size):
                batch_X = X_test[i:min(i + batch_size, n_samples_test)]
                batch_pred = model(batch_X, training=True)  # training=True для активации dropout
                batch_predictions.append(batch_pred.numpy())
            
            # Объединяем предсказания батчей
            predictions.append(np.concatenate(batch_predictions, axis=0))
        
        predictions = np.array(predictions)
        
        # Среднее и стандартное отклонение предсказаний
        y_pred_mean = predictions.mean(axis=0).flatten()
        y_pred_std = predictions.std(axis=0).flatten()
        
        if self.task == 'regression':
            # Метрики регрессии
            mae = mean_absolute_error(y_test, y_pred_mean)
            mse = mean_squared_error(y_test, y_pred_mean)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_mean)
            direction_accuracy = np.mean((y_pred_mean > 0) == (y_test > 0))
            
            # Анализ uncertainty
            high_confidence_mask = y_pred_std < np.percentile(y_pred_std, 30)
            high_conf_mae = mean_absolute_error(y_test[high_confidence_mask], y_pred_mean[high_confidence_mask])
            
            logger.info(f"\n📈 Результаты {model_name}:")
            logger.info(f"   MAE: {mae:.4f}")
            logger.info(f"   RMSE: {rmse:.4f}")
            logger.info(f"   R²: {r2:.4f}")
            logger.info(f"   Direction Accuracy: {direction_accuracy:.2%}")
            logger.info(f"   High Confidence MAE: {high_conf_mae:.4f}")
            logger.info(f"   Mean Uncertainty: {y_pred_std.mean():.4f}")
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'high_conf_mae': high_conf_mae,
                'mean_uncertainty': y_pred_std.mean()
            }
        else:  # classification
            # Используем оптимизированный порог вместо 0.5
            optimal_threshold = getattr(self, f'optimal_threshold_{model_name}', 0.5)
            y_pred_binary = (y_pred_mean > optimal_threshold).astype(int)
            
            # Метрики классификации
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary, zero_division=0)
            recall = recall_score(y_test, y_pred_binary, zero_division=0)
            f1 = f1_score(y_test, y_pred_binary, zero_division=0)
            
            # Фильтрация по confidence
            confidence = 1 - y_pred_std
            high_conf_mask = confidence > 0.7
            
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = accuracy_score(
                    y_test[high_conf_mask], 
                    y_pred_binary[high_conf_mask]
                )
                high_conf_precision = precision_score(
                    y_test[high_conf_mask], 
                    y_pred_binary[high_conf_mask],
                    zero_division=0
                )
            else:
                high_conf_accuracy = 0
                high_conf_precision = 0
            
            logger.info(f"\n📈 Результаты {model_name}:")
            logger.info(f"   Accuracy: {accuracy:.2%}")
            logger.info(f"   Precision: {precision:.2%}")
            logger.info(f"   Recall: {recall:.2%}")
            logger.info(f"   F1-Score: {f1:.2%}")
            logger.info(f"   High Confidence Samples: {high_conf_mask.sum()} ({high_conf_mask.mean():.1%})")
            logger.info(f"   High Confidence Accuracy: {high_conf_accuracy:.2%}")
            logger.info(f"   High Confidence Precision: {high_conf_precision:.2%}")
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'high_conf_accuracy': high_conf_accuracy,
                'high_conf_precision': high_conf_precision,
                'high_conf_samples': high_conf_mask.sum()
            }
        
        return metrics
    
    def optimize_threshold(self, model, X_val, y_val, model_name):
        """Оптимизация порога для бинарной классификации"""
        logger.info(f"🎯 Оптимизация порога для {model_name}...")
        
        # Получаем предсказания
        y_pred_proba = model.predict(X_val, verbose=0).flatten()
        
        # Пробуем разные пороги
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"   Лучший порог: {best_threshold:.2f} (F1: {best_f1:.3f})")
        return best_threshold
    
    def post_process_predictions(self, predictions, timestamps, symbols, min_interval=4):
        """Post-processing предсказаний с временными ограничениями"""
        processed_predictions = predictions.copy()
        
        # Группируем по символам
        for symbol in np.unique(symbols):
            symbol_mask = symbols == symbol
            symbol_preds = processed_predictions[symbol_mask]
            symbol_times = timestamps[symbol_mask]
            
            # Убираем повторные сигналы в течение min_interval свечей
            last_signal_time = -np.inf
            for i in range(len(symbol_preds)):
                if symbol_preds[i] == 1:
                    if i - last_signal_time < min_interval:
                        processed_predictions[symbol_mask][i] = 0
                    else:
                        last_signal_time = i
        
        return processed_predictions
    
    def train_ensemble(self):
        """Обучение ансамбля моделей"""
        logger.info("="*80)
        logger.info(f"🚀 НАЧАЛО ОБУЧЕНИЯ ENHANCED TRANSFORMER ENSEMBLE (v2.1)")
        logger.info(f"📊 Режим: {self.task}")
        logger.info(f"🎯 Размер ансамбля: {self.ensemble_size}")
        logger.info("="*80)
        
        try:
            # Создаем директории
            os.makedirs('trained_model', exist_ok=True)
            
            # Загрузка и подготовка данных
            df = self.load_data()
            grouped_data = self.prepare_features_and_targets(df)
            
            # Конвертируем в numpy arrays
            X_train = np.array(grouped_data['train']['X'])
            X_val = np.array(grouped_data['val']['X'])
            X_test = np.array(grouped_data['test']['X'])
            
            all_buy = grouped_data['train']['y_buy'] + grouped_data['val']['y_buy'] + grouped_data['test']['y_buy']
            all_sell = grouped_data['train']['y_sell'] + grouped_data['val']['y_sell'] + grouped_data['test']['y_sell']
            
            # Нормализация
            logger.info("🔄 Нормализация признаков...")
            X_train = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            
            # Статистика
            if self.task == 'regression':
                logger.info("\n📊 Статистика expected returns:")
                logger.info(f"   Buy - Среднее: {np.mean(all_buy):.3f}%")
                logger.info(f"   Buy - Std: {np.std(all_buy):.3f}%")
                logger.info(f"   Sell - Среднее: {np.mean(all_sell):.3f}%")
                logger.info(f"   Sell - Std: {np.std(all_sell):.3f}%")
            else:
                buy_binary = self.convert_to_binary_labels(np.array(all_buy), threshold=0.3)
                sell_binary = self.convert_to_binary_labels(np.array(all_sell), threshold=0.3)
                
                logger.info("\n📊 Статистика бинарных меток (порог > 0.3%):")
                logger.info(f"   Buy - Класс 1: {np.mean(buy_binary):.1%}")
                logger.info(f"   Sell - Класс 1: {np.mean(sell_binary):.1%}")
            
            # Обучение ансамбля
            results = {}
            ensemble_models = {'buy': [], 'sell': []}
            
            for model_type in ['buy', 'sell']:
                logger.info(f"\n{'='*60}")
                logger.info(f"Подготовка ансамбля моделей: {model_type}")
                logger.info(f"{'='*60}")
                
                for ensemble_idx in range(self.ensemble_size):
                    model_name = f"{model_type}_enhanced_v2.1_{ensemble_idx}"
                    
                    # Создание последовательностей с разным stride для разнообразия
                    stride = 3 + ensemble_idx  # 3, 4, 5...
                    
                    if model_type == 'buy':
                        y_data = {'train': grouped_data['train']['y_buy'],
                                 'val': grouped_data['val']['y_buy'],
                                 'test': grouped_data['test']['y_buy']}
                    else:
                        y_data = {'train': grouped_data['train']['y_sell'],
                                 'val': grouped_data['val']['y_sell'],
                                 'test': grouped_data['test']['y_sell']}
                    
                    # Преобразуем в бинарные метки для классификации
                    if self.task == 'classification_binary':
                        for split in y_data:
                            y_data[split] = self.convert_to_binary_labels(np.array(y_data[split]))
                    
                    # Создаем последовательности
                    X_train_seq, y_train_seq, _ = self.create_sequences(
                        X_train, np.array(y_data['train']), 
                        np.array(grouped_data['train']['symbols']), stride=stride
                    )
                    X_val_seq, y_val_seq, _ = self.create_sequences(
                        X_val, np.array(y_data['val']),
                        np.array(grouped_data['val']['symbols']), stride=stride
                    )
                    X_test_seq, y_test_seq, _ = self.create_sequences(
                        X_test, np.array(y_data['test']),
                        np.array(grouped_data['test']['symbols']), stride=stride
                    )
                    
                    # Создание и обучение модели
                    model = self.create_model_with_warmup((self.sequence_length, X_train.shape[1]), model_name)
                    
                    # Обучение с gradient accumulation
                    history = self.train_model_with_gradient_accumulation(
                        model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, model_name
                    )
                    
                    # Оптимизация порога для классификации
                    if self.task == 'classification_binary':
                        optimal_threshold = self.optimize_threshold(model, X_val_seq, y_val_seq, model_name)
                        setattr(self, f'optimal_threshold_{model_name}', optimal_threshold)
                    
                    # Оценка с uncertainty
                    metrics = self.evaluate_model_with_uncertainty(
                        model, X_test_seq, y_test_seq, model_name
                    )
                    
                    results[model_name] = metrics
                    ensemble_models[model_type].append(model)
                    self.models[model_name] = model
            
            # Сохранение результатов
            self.save_enhanced_results(results, ensemble_models)
            
            logger.info("\n✅ ОБУЧЕНИЕ ENHANCED ENSEMBLE УСПЕШНО ЗАВЕРШЕНО!")
            logger.info(f"📊 Результаты сохранены в: {self.log_dir}")
            
        except Exception as e:
            logger.error(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            logger.error("Трейсбек:", exc_info=True)
            raise
    
    def convert_numpy_types(self, obj):
        """Конвертация numpy типов для JSON сериализации"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_numpy_types(item) for item in obj]
        return obj
    
    def save_enhanced_results(self, results, ensemble_models):
        """Сохранение результатов enhanced модели"""
        logger.info("\n💾 Сохранение enhanced моделей и результатов...")
        
        # Сохраняем модели
        for name, model in self.models.items():
            model_path = f'trained_model/{name}.h5'
            model.save(model_path)
            logger.info(f"   ✅ Модель сохранена: {model_path}")
        
        # Сохраняем scaler
        with open('trained_model/scaler_v2.1.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Сохраняем конфигурацию enhanced features
        feature_config = {
            'technical_indicators': self.TECHNICAL_INDICATORS,
            'market_features': self.MARKET_FEATURES,
            'ohlc_features': self.OHLC_FEATURES,
            'symbol_features': self.SYMBOL_FEATURES,
            'engineered_features': [
                'rsi_oversold', 'rsi_overbought', 'macd_bullish',
                'bb_near_lower', 'bb_near_upper', 'strong_trend', 'volume_spike'
            ],
            'total_features': len(self.TECHNICAL_INDICATORS) + len(self.MARKET_FEATURES) + 
                            len(self.OHLC_FEATURES) + len(self.SYMBOL_FEATURES) + 7
        }
        
        with open('trained_model/feature_config_v2.1.json', 'w') as f:
            json.dump(feature_config, f, indent=2)
        
        # Сохраняем метаданные
        metadata = {
            'model_version': '2.1',
            'type': 'enhanced_temporal_fusion_transformer',
            'task_type': self.task,
            'ensemble_size': self.ensemble_size,
            'total_features': len(self.TECHNICAL_INDICATORS) + len(self.MARKET_FEATURES) + len(self.OHLC_FEATURES) + len(self.SYMBOL_FEATURES) + 7,  # +7 for engineered
            'architecture': {
                'sequence_length': self.sequence_length,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_transformer_blocks': self.num_transformer_blocks,
                'ff_dim': self.ff_dim,
                'dropout_rate': self.dropout_rate,
                'use_multi_scale': True,
                'use_attention_pooling': True
            },
            'training': {
                'batch_size': self.batch_size,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'epochs': self.epochs,
                'loss_type': 'focal_loss' if self.task == 'classification_binary' else 'huber_loss',
                'augmentation': True
            },
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Конвертируем numpy типы перед сохранением
        metadata = self.convert_numpy_types(metadata)
        
        with open('trained_model/metadata_v2.1.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Создаем финальный отчет
        with open(f'{self.log_dir}/final_report_v2.1.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED TEMPORAL FUSION TRANSFORMER v2.1 - ФИНАЛЬНЫЙ ОТЧЕТ\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Задача: {self.task}\n")
            f.write(f"Размер ансамбля: {self.ensemble_size}\n\n")
            
            f.write("АРХИТЕКТУРА:\n")
            f.write(f"- Transformer blocks: {self.num_transformer_blocks}\n")
            f.write(f"- Model dimension: {self.d_model}\n")
            f.write(f"- Attention heads: {self.num_heads}\n")
            f.write(f"- Feed-forward dim: {self.ff_dim}\n")
            f.write(f"- Dropout rate: {self.dropout_rate}\n")
            f.write(f"- Multi-scale conv: Да\n")
            f.write(f"- Attention pooling: Да\n\n")
            
            f.write("УЛУЧШЕНИЯ v2.0:\n")
            f.write("- Market features (BTC correlation, time features)\n")
            f.write("- Focal Loss для классификации\n")
            f.write("- Gradient accumulation\n")
            f.write("- Data augmentation (noise, mixup)\n")
            f.write("- Uncertainty estimation\n")
            f.write("- Enhanced architecture\n\n")
            
            f.write("УЛУЧШЕНИЯ v2.1:\n")
            f.write("- OHLC features (13 признаков): нормализованные отношения, свечные паттерны\n")
            f.write("- Symbol embeddings (14 признаков): one-hot для топ-10 монет, категории\n")
            f.write("- Относительные метрики к EMA15/50 и VWAP\n")
            f.write("- Layer Normalization для стабильности OHLC\n")
            f.write("- Оптимизация порога классификации\n")
            f.write("- Post-processing фильтры временных интервалов\n")
            f.write(f"- Всего признаков: {len(self.TECHNICAL_INDICATORS) + len(self.MARKET_FEATURES) + len(self.OHLC_FEATURES) + len(self.SYMBOL_FEATURES) + 7}\n\n")
            
            f.write("РЕЗУЛЬТАТЫ:\n")
            for model_name, metrics in results.items():
                f.write(f"\n{model_name}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  - {metric_name}: {value:.4f}\n")
                    else:
                        f.write(f"  - {metric_name}: {value}\n")
        
        logger.info("✅ Все результаты сохранены!")


def main():
    """Основная функция запуска enhanced обучения"""
    parser = argparse.ArgumentParser(description='Enhanced Temporal Fusion Transformer v2.1')
    parser.add_argument('--task', type=str, default='classification_binary',
                       choices=['regression', 'classification_binary'],
                       help='Тип задачи: regression или classification_binary')
    parser.add_argument('--ensemble_size', type=int, default=3,
                       help='Количество моделей в ансамбле (default: 3)')
    parser.add_argument('--test_mode', action='store_true',
                       help='Тестовый режим: 2 символа, 3 эпохи, последние 10 дней')
    args = parser.parse_args()
    
    logger.info("🚀 Запуск Enhanced Temporal Fusion Transformer v2.1")
    logger.info(f"📊 Режим: {args.task}")
    logger.info(f"🎯 Размер ансамбля: {args.ensemble_size}")
    if args.test_mode:
        logger.info("⚡ ТЕСТОВЫЙ РЕЖИМ: ограниченные данные и эпохи")
    
    # Подключение к БД
    db_config = {
        'dbname': 'crypto_trading',
        'user': 'ruslan',
        'password': 'ruslan',
        'host': 'localhost',
        'port': 5555
    }
    
    db_manager = PostgreSQLManager(**db_config)
    
    # Создание и запуск trainer
    trainer = EnhancedTransformerTrainer(
        db_manager, 
        task=args.task,
        ensemble_size=args.ensemble_size,
        test_mode=args.test_mode
    )
    trainer.train_ensemble()


if __name__ == "__main__":
    main()