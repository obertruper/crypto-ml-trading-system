#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer модель для предсказания ожидаемой доходности в криптотрейдинге
- Temporal Fusion Transformer (TFT) архитектура
- Поддержка регрессии и бинарной классификации
- Продвинутая визуализация и логирование
- Обучение двух моделей: buy и sell предикторы
"""

import numpy as np
import pandas as pd
import tensorflow as tf
# Настройка памяти GPUgpus = tf.config.experimental.list_physical_devices("GPU")if gpus:    try:        tf.config.experimental.set_memory_growth(gpus[0], True)    except RuntimeError as e:        print(e)
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import pickle
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values, Json
import logging
import yaml
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# Настройка логирования
log_dir = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f'{log_dir}/plots', exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Файловый логгер
file_handler = logging.FileHandler(f'{log_dir}/training.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Настройка GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logger.info(f"🖥️ GPU доступен: {physical_devices[0].name}")
else:
    logger.info("⚠️ GPU не найден, используется CPU")


class PostgreSQLManager:
    """Менеджер для работы с PostgreSQL"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config.copy()
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
        self.connection = None

    def connect(self):
        """Создает подключение к БД"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = True
            logger.info("✅ Подключение к PostgreSQL установлено")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
            raise

    def disconnect(self):
        """Закрывает подключение к БД"""
        if self.connection:
            self.connection.close()
            logger.info("📤 Подключение к PostgreSQL закрыто")

    def fetch_dataframe(self, query: str, params=None) -> pd.DataFrame:
        """Выполняет запрос и возвращает результат как DataFrame"""
        try:
            return pd.read_sql_query(query, self.connection, params=params)
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса DataFrame: {e}")
            raise


class TransformerBlock(layers.Layer):
    """Блок трансформера с multi-head attention"""
    
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
        
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class GatedResidualNetwork(layers.Layer):
    """Gated Residual Network для TFT"""
    
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation='elu')
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gate_dense = layers.Dense(units, activation='sigmoid')
        self.layernorm = layers.LayerNormalization()
        
    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        gate = self.gate_dense(inputs)
        x = gate * x
        x = x + inputs
        return self.layernorm(x)


class TemporalFusionTransformer(keras.Model):
    """Temporal Fusion Transformer для задачи регрессии и классификации"""
    
    def __init__(self, num_features, sequence_length, 
                 d_model=128, num_heads=8, num_transformer_blocks=4,
                 mlp_units=[256, 128], dropout=0.2, task='regression'):
        super(TemporalFusionTransformer, self).__init__()
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Variable Selection Network
        self.vsn_dense = layers.Dense(d_model, activation='gelu')
        self.vsn_grn = GatedResidualNetwork(d_model, dropout)
        
        # Static covariate encoder
        self.static_encoder = keras.Sequential([
            layers.Dense(d_model, activation='gelu'),
            layers.Dropout(dropout),
            GatedResidualNetwork(d_model, dropout)
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
        
        # Output layer для регрессии
        # Выходной слой в зависимости от задачи
        self.task = task
        if task == 'regression':
            self.output_dense = keras.Sequential([
                layers.Dense(mlp_units[0], activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(mlp_units[1], activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(1)  # Линейный выход для регрессии
            ])
        else:  # classification_binary
            self.output_dense = keras.Sequential([
                layers.Dense(mlp_units[0], activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(mlp_units[1], activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(1, activation='sigmoid')  # Sigmoid для бинарной классификации
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
        self.history['lr'].append(self.model.optimizer.learning_rate.numpy())
        
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


class UniversalTransformerTrainer:
    """Тренер для Transformer моделей регрессии и классификации"""
    
    def __init__(self, db_manager: PostgreSQLManager, config_path='config.yaml', task='regression'):
        # Загружаем конфигурацию
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.db = db_manager
        self.task = task  # 'regression' или 'classification_binary'
        self.sequence_length = self.config['model']['sequence_length']
        self.batch_size = 32
        self.epochs = 100
        
        # Параметры архитектуры
        self.d_model = 128
        self.num_heads = 8
        self.num_transformer_blocks = 4
        self.ff_dim = 256
        self.dropout_rate = 0.2
        
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_columns = None
        self.log_dir = log_dir
        
        # Полный список технических индикаторов (49 штук)
        self.TECHNICAL_INDICATORS = [
            # Трендовые индикаторы
            'ema_15', 'adx_val', 'adx_plus_di', 'adx_minus_di',
            'macd_val', 'macd_signal', 'macd_hist', 'sar',
            'ichimoku_conv', 'ichimoku_base', 'aroon_up', 'aroon_down',
            'dpo',
            
            # Осцилляторы
            'rsi_val', 'stoch_k', 'stoch_d', 'cci_val', 'williams_r',
            'roc', 'ult_osc', 'mfi',
            
            # Волатильность
            'atr_val', 'bb_upper', 'bb_lower', 'bb_basis',
            'donchian_upper', 'donchian_lower',
            
            # Объемные индикаторы
            'obv', 'cmf', 'volume_sma', 'volume_ratio',
            
            # Vortex индикаторы
            'vortex_vip', 'vortex_vin',
            
            # Производные индикаторы
            'macd_signal_ratio', 'adx_diff', 'bb_position',
            'rsi_dist_from_mid', 'stoch_diff', 'vortex_ratio',
            'ichimoku_diff', 'atr_norm',
            
            # Временные признаки
            'hour', 'day_of_week', 'is_weekend',
            
            # Ценовые паттерны
            'price_change_1', 'price_change_4', 'price_change_16',
            'volatility_4', 'volatility_16'
        ]
        
        self.feature_importance = {}
        
    def convert_to_binary_labels(self, returns, threshold=0.3):
        """Преобразование expected returns в бинарные метки (порог > 0.3%)"""
        return (returns > threshold).astype(np.float32)
    
    def create_model(self, input_shape, name):
        """Создание модели для регрессии или классификации"""
        
        model = TemporalFusionTransformer(
            num_features=input_shape[1],
            sequence_length=input_shape[0],
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_transformer_blocks=self.num_transformer_blocks,
            mlp_units=[self.ff_dim, self.ff_dim//2],
            dropout=self.dropout_rate,
            task=self.task
        )
        
        # Оптимизатор
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0002,  # Увеличен для лучшей сходимости с порогом 0.3%
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Компиляция в зависимости от задачи
        if self.task == 'regression':
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.Huber(delta=1.0),
                metrics=[
                    keras.metrics.MeanAbsoluteError(name='mae'),
                    keras.metrics.RootMeanSquaredError(name='rmse')
                ]
            )
        else:  # classification_binary
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                metrics=[
                    keras.metrics.BinaryAccuracy(name='accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')
                ]
            )
        
        # Инициализация весов
        dummy_input = tf.zeros((1, input_shape[0], input_shape[1]))
        _ = model(dummy_input)
        
        logger.info(f"✅ Создана модель {name} с {model.count_params():,} параметрами")
        
        return model
    
    def load_data(self):
        """Загрузка данных из PostgreSQL"""
        logger.info("📊 Загрузка данных из PostgreSQL...")
        start_time = time.time()
        
        # Загружаем данные с expected returns из колонок БД
        query = """
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
        ORDER BY p.symbol, p.timestamp
        """
        
        df = self.db.fetch_dataframe(query)
        load_time = time.time() - start_time
        logger.info(f"✅ Загружено {len(df)} записей за {load_time:.2f} секунд ({load_time/60:.1f} минут)")
        
        if len(df) == 0:
            raise ValueError("Нет данных для обучения!")
        
        # Статистика по символам
        symbol_counts = df['symbol'].value_counts()
        logger.info("📊 Распределение по символам:")
        for symbol, count in symbol_counts.items():
            logger.info(f"   {symbol}: {count:,} записей")
        
        return df
    
    def prepare_features_and_targets(self, df):
        """Подготовка признаков и целевых значений с правильным временным разделением"""
        logger.info("🔧 Подготовка признаков и целевых значений...")
        logger.info(f"📊 Используется {len(self.TECHNICAL_INDICATORS)} индикаторов")
        start_time = time.time()
        
        # Проверяем диапазон expected returns
        buy_returns = df['buy_expected_return'].values
        sell_returns = df['sell_expected_return'].values
        
        buy_outliers = np.sum((buy_returns < -1.1) | (buy_returns > 5.8))
        sell_outliers = np.sum((sell_returns < -1.1) | (sell_returns > 5.8))
        
        if buy_outliers > 0 or sell_outliers > 0:
            logger.warning(f"⚠️ Найдены значения expected_return вне ожидаемого диапазона [-1.1%, +5.8%]:")
            logger.warning(f"   BUY outliers: {buy_outliers} ({buy_outliers/len(df)*100:.2f}%)")
            logger.warning(f"   SELL outliers: {sell_outliers} ({sell_outliers/len(df)*100:.2f}%)")
        
        # Сортируем по времени
        df = df.sort_values(["symbol", "timestamp"])
        
        # Группируем по символам для правильного временного разделения
        grouped_data = {
            "train": {"X": [], "y_buy": [], "y_sell": [], "symbols": []},
            "val": {"X": [], "y_buy": [], "y_sell": [], "symbols": []},
            "test": {"X": [], "y_buy": [], "y_sell": [], "symbols": []},
        }
        
        for symbol in tqdm(df["symbol"].unique(), desc="Обработка символов"):
            symbol_df = df[df["symbol"] == symbol].reset_index(drop=True)
            n = len(symbol_df)
            
            # Временное разделение СЫРЫХ данных
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            gap = self.sequence_length  # Зазор для предотвращения утечки
            
            # Разделяем с учетом gap
            splits = {
                "train": symbol_df[:train_end - gap],
                "val": symbol_df[train_end + gap:val_end - gap],
                "test": symbol_df[val_end + gap:]
            }
            
            # Обрабатываем каждый split отдельно
            for split_name, split_df in splits.items():
                if len(split_df) < self.sequence_length + 1:
                    continue
                    
                # Извлекаем признаки
                X_split = []
                for _, row in split_df.iterrows():
                    feature_values = []
                    
                    # Технические индикаторы
                    indicators = row["technical_indicators"]
                    for indicator in self.TECHNICAL_INDICATORS:
                        value = indicators.get(indicator, 0.0)
                        if value is None or pd.isna(value):
                            value = 0.0
                        feature_values.append(float(value))
                    
                    # Инженерные признаки
                    rsi = indicators.get("rsi_val", 50.0)
                    feature_values.append(1.0 if rsi is not None and rsi < 30 else 0.0)
                    feature_values.append(1.0 if rsi is not None and rsi > 70 else 0.0)
                    
                    macd = indicators.get("macd_val", 0.0)
                    macd_signal = indicators.get("macd_signal_val", 0.0)
                    feature_values.append(1.0 if macd is not None and macd_signal is not None and macd > macd_signal else 0.0)
                    
                    bb_position = indicators.get("bb_position", 0.5)
                    feature_values.append(1.0 if bb_position is not None and bb_position < 0.2 else 0.0)
                    feature_values.append(1.0 if bb_position is not None and bb_position > 0.8 else 0.0)
                    
                    adx = indicators.get("adx_val", 0.0)
                    feature_values.append(1.0 if adx is not None and adx > 25 else 0.0)
                    
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
    def create_sequences(self, X, y, symbols, stride=5):
        """Создание последовательностей для временных рядов"""
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
                logger.warning(f"⚠️ Недостаточно данных для {symbol}: {len(symbol_indices)} записей")
                continue
            
            X_symbol = X[symbol_indices]
            y_symbol = y[symbol_indices]
            
            # Создаем последовательности
            for i in range(0, len(X_symbol) - self.sequence_length, stride):
                # Используем stride для уменьшения перекрытия
                sequences.append(X_symbol[i:i + self.sequence_length])
                targets.append(y_symbol[i + self.sequence_length])
                seq_symbols.append(symbol)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        seq_time = time.time() - start_time
        logger.info(f"✅ Создано {len(sequences)} последовательностей за {seq_time:.2f} секунд")
        logger.info(f"   Форма данных: {sequences.shape}")
        
        return sequences, targets, np.array(seq_symbols)
    
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """Обучение модели с визуализацией"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 НАЧАЛО ОБУЧЕНИЯ: {model_name}")
        logger.info(f"{'='*70}")
        
        # Callbacks
        callbacks = [
            # Визуализация
            VisualizationCallback(self.log_dir, model_name, update_freq=5, task=self.task),
            
            # Early stopping с увеличенным patience
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,  # Увеличено для лучшей сходимости
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce LR
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=f'trained_model/{model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir=f'{self.log_dir}/tensorboard/{model_name}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        # Обучение
        if self.task == 'classification_binary':
            # Подсчет весов классов для балансировки
            unique, counts = np.unique(y_train, return_counts=True)
            class_weight = {0: counts[1] / counts[0], 1: 1.0} if len(unique) == 2 else None
            
            if class_weight:
                logger.info(f"📊 Баланс классов - 0: {counts[0]:,}, 1: {counts[1]:,}")
                logger.info(f"⚖️ Веса классов - 0: {class_weight[0]:.2f}, 1: {class_weight[1]:.2f}")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
        else:  # regression
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
        """Оценка модели"""
        logger.info(f"\n📊 Оценка модели {model_name}...")
        
        # Предсказания
        y_pred = model.predict(X_test, verbose=0)
        y_pred = y_pred.flatten()
        
        if self.task == 'regression':
            # Метрики регрессии
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            direction_accuracy = np.mean((y_pred > 0) == (y_test > 0))
            
            # Визуализация
            self._plot_regression_results(y_test, y_pred, model_name, mae, rmse, r2, direction_accuracy)
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'mean_error': np.mean(y_pred - y_test),
                'std_error': np.std(y_pred - y_test)
            }
        else:  # classification_binary
            # Бинаризация предсказаний
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Метрики классификации
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary, zero_division=0)
            recall = recall_score(y_test, y_pred_binary, zero_division=0)
            f1 = f1_score(y_test, y_pred_binary, zero_division=0)
            auc = roc_auc_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred_binary)
            
            # Визуализация
            self._plot_classification_results(y_test, y_pred, y_pred_binary, model_name, 
                                            accuracy, precision, recall, f1, auc, cm)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm
            }
        
        # Логирование результатов
        logger.info(f"\n📊 Результаты для {model_name}:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {key}: {value:.4f}")
            elif key == 'confusion_matrix':
                logger.info(f"   Confusion Matrix:")
                logger.info(f"      TN: {value[0,0]:,}  FP: {value[0,1]:,}")
                logger.info(f"      FN: {value[1,0]:,}  TP: {value[1,1]:,}")
        
        return metrics
    
    
    def _plot_regression_results(self, y_test, y_pred, model_name, mae, rmse, r2, direction_accuracy):
        """Визуализация результатов регрессии"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Evaluation: {model_name}', fontsize=16)
        
        # График 1: Scatter plot
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Return (%)')
        axes[0, 0].set_ylabel('Predicted Return (%)')
        axes[0, 0].set_title(f'Predictions vs True Values (R² = {r2:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Распределение ошибок
        errors = y_pred - y_test
        axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Prediction Error (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Error Distribution (MAE = {mae:.3f}%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Временная визуализация
        sample_size = min(500, len(y_test))
        axes[1, 0].plot(y_test[:sample_size], 'b-', label='True', alpha=0.7, linewidth=1)
        axes[1, 0].plot(y_pred[:sample_size], 'r-', label='Predicted', alpha=0.7, linewidth=1)
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].set_title('Sample Predictions Timeline')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Статистика по диапазонам
        axes[1, 1].axis('off')
        
        # Анализ по диапазонам
        ranges = [(-np.inf, -2), (-2, -0.5), (-0.5, 0.5), (0.5, 2), (2, np.inf)]
        range_names = ['< -2%', '-2% to -0.5%', '-0.5% to 0.5%', '0.5% to 2%', '> 2%']
        range_stats = []
        
        for i, (low, high) in enumerate(ranges):
            mask = (y_test >= low) & (y_test < high)
            if np.any(mask):
                range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
                range_stats.append(f"{range_names[i]}: {np.sum(mask)} samples, MAE = {range_mae:.3f}%")
        
        stats_text = f"""
Performance Summary:
  MAE: {mae:.3f}%
  RMSE: {rmse:.3f}%
  R²: {r2:.3f}
  Direction Accuracy: {direction_accuracy:.1%}

Range Analysis:
  """ + '\n  '.join(range_stats)
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/plots/{model_name}_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    
    def _plot_classification_results(self, y_test, y_pred_proba, y_pred_binary, model_name, 
                                   accuracy, precision, recall, f1, auc, cm):
        """Визуализация результатов классификации"""
        from sklearn.metrics import roc_curve
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Evaluation: {model_name} (Binary Classification)', fontsize=16)
        
        # График 1: ROC кривая
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xticklabels(['Не входить', 'Входить'])
        axes[0, 1].set_yticklabels(['Не входить', 'Входить'])
        
        # График 3: Распределение вероятностей
        axes[1, 0].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, 
                       label='Класс 0 (Не входить)', density=True)
        axes[1, 0].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, 
                       label='Класс 1 (Входить)', density=True)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Probability Distribution by Class')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Метрики
        axes[1, 1].axis('off')
        stats_text = f"""
Performance Summary:
  Accuracy: {accuracy:.3f}
  Precision: {precision:.3f}
  Recall: {recall:.3f}
  F1-Score: {f1:.3f}
  ROC-AUC: {auc:.3f}
  
Confusion Matrix:
  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}
  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}
  
Positive Rate: {np.sum(y_test == 1) / len(y_test):.1%}
"""
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/plots/{model_name}_classification_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Основной процесс обучения"""
        logger.info("\n" + "="*80)
        if self.task == 'regression':
            logger.info("🚀 НАЧАЛО ОБУЧЕНИЯ TRANSFORMER МОДЕЛИ РЕГРЕССИИ")
            logger.info("📊 Предсказание expected returns для buy и sell позиций")
        else:
            logger.info("🚀 НАЧАЛО ОБУЧЕНИЯ TRANSFORMER МОДЕЛИ КЛАССИФИКАЦИИ")
            logger.info("📊 Бинарная классификация: входить/не входить (порог > 0.3%)")
        logger.info("="*80)
        
        try:
            # Создаем директории
            os.makedirs('trained_model', exist_ok=True)
            
            # Загружаем данные
            df = self.load_data()
            
            # Подготавливаем признаки
            grouped_data = self.prepare_features_and_targets(df)
            
            # Нормализация данных
            logger.info("🔄 Нормализация признаков...")
            
            # Объединяем все данные для fit scaler
            all_X = []
            for split in ["train", "val", "test"]:
                all_X.extend(grouped_data[split]["X"])
            
            self.scaler.fit(all_X)
            
            # Нормализуем каждый split
            for split in ["train", "val", "test"]:
                grouped_data[split]["X"] = self.scaler.transform(grouped_data[split]["X"]).tolist()
            
            # Статистика
            all_buy = []
            all_sell = []
            for split in ["train", "val", "test"]:
                all_buy.extend(grouped_data[split]["y_buy"])
                all_sell.extend(grouped_data[split]["y_sell"])
            
            # Статистика целевых значений
            if self.task == 'regression':
                logger.info("\n📊 Статистика buy_return:")
                logger.info(f"   Среднее: {np.mean(all_buy):.3f}%")
                logger.info(f"   Std: {np.std(all_buy):.3f}%")
                logger.info(f"   Min/Max: {np.min(all_buy):.3f}% / {np.max(all_buy):.3f}%")
            else:  # classification
                buy_binary = self.convert_to_binary_labels(np.array(all_buy), threshold=0.3)
                sell_binary = self.convert_to_binary_labels(np.array(all_sell), threshold=0.3)
                
                logger.info("\n📊 Статистика бинарных меток (порог > 0.3%):")
                logger.info(f"   Buy - Класс 0 (не входить): {np.sum(buy_binary == 0):,} ({np.mean(buy_binary == 0):.1%})")
                logger.info(f"   Buy - Класс 1 (входить): {np.sum(buy_binary == 1):,} ({np.mean(buy_binary == 1):.1%})")
                logger.info(f"   Sell - Класс 0 (не входить): {np.sum(sell_binary == 0):,} ({np.mean(sell_binary == 0):.1%})")
                logger.info(f"   Sell - Класс 1 (входить): {np.sum(sell_binary == 1):,} ({np.mean(sell_binary == 1):.1%})")
            
            # Проверка уникальности
            unique_buy = len(np.unique(all_buy))
            unique_sell = len(np.unique(all_sell))
            buy_uniqueness = unique_buy / len(all_buy) * 100
            sell_uniqueness = unique_sell / len(all_sell) * 100
            
            logger.info(f"   Уникальных значений: {unique_buy:,} ({buy_uniqueness:.1f}%)")
            
            logger.info("\n📊 Статистика sell_return:")
            logger.info(f"   Среднее: {np.mean(all_sell):.3f}%")
            logger.info(f"   Std: {np.std(all_sell):.3f}%")
            logger.info(f"   Min/Max: {np.min(all_sell):.3f}% / {np.max(all_sell):.3f}%")
            logger.info(f"   Уникальных значений: {unique_sell:,} ({sell_uniqueness:.1f}%)")
            
            # Предупреждение при низкой уникальности
            if buy_uniqueness < 10 or sell_uniqueness < 10:
                logger.warning("\n⚠️ ПРЕДУПРЕЖДЕНИЕ: Низкая уникальность целевых значений!")
                logger.warning("   Это нормально для данных с частичными закрытиями.")
                logger.warning(f"   Основные уровни: -1.1% (SL), 0.48%, 1.56%, 2.49%, 3.17%, 5.8% (TP)")
                
                # Анализ распределения
                buy_values, buy_counts = np.unique(all_buy, return_counts=True)
                sell_values, sell_counts = np.unique(all_sell, return_counts=True)
                
                logger.info("\n📊 Распределение expected returns:")
                logger.info("   Buy топ-5 значений:")
                for val, cnt in sorted(zip(buy_values, buy_counts), key=lambda x: -x[1])[:5]:
                    logger.info(f"     {val:.2f}%: {cnt} ({cnt/len(all_buy)*100:.1f}%)")
                
                # Остановить обучение только если слишком мало уникальных значений
                if buy_uniqueness < 1 or sell_uniqueness < 1:
                    logger.error("❌ Критически низкая уникальность! Обучение прервано.")
                    logger.error("   Все значения одинаковые.")
                    raise ValueError("Данные не подходят для обучения - все значения одинаковые")
            
            # Сохраняем статистику нормализации
            os.makedirs("trained_model", exist_ok=True)
            stats = {
                "mean": self.scaler.center_.tolist() if hasattr(self.scaler, "center_") else None,
                "scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
                "feature_names": self.TECHNICAL_INDICATORS + [
                    "rsi_oversold", "rsi_overbought", "macd_bullish",
                    "bb_near_lower", "bb_near_upper", "strong_trend", "high_volume"
                ]
            }
            
            with open("trained_model/scaler_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            # Конфигурации моделей
            if self.task == 'regression':
                model_configs = [
                    ('buy_return_predictor', 'buy'),
                    ('sell_return_predictor', 'sell')
                ]
            else:  # classification
                model_configs = [
                    ('buy_classifier', 'buy'),
                    ('sell_classifier', 'sell')
                ]
            
            results = {}
            
            for model_name, target_type in model_configs:
                logger.info(f"\n{'='*60}")
                logger.info(f"Подготовка модели: {model_name}")
                logger.info(f"{'='*60}")
                
                # Подготовка данных для конкретной модели
                all_X = []
                all_y = []
                all_symbols = []
                
                # Объединяем данные из всех splits для создания последовательностей
                for split in ["train", "val", "test"]:
                    X_split = np.array(grouped_data[split]["X"])
                    symbols_split = np.array(grouped_data[split]["symbols"])
                    
                    if target_type == 'buy':
                        y_split = np.array(grouped_data[split]["y_buy"])
                    else:  # 'sell'
                        y_split = np.array(grouped_data[split]["y_sell"])
                    
                    # Преобразуем в бинарные метки для классификации
                    if self.task == 'classification_binary':
                        y_split = self.convert_to_binary_labels(y_split, threshold=0.3)
                    
                    # Создаем последовательности для каждого split отдельно
                    X_seq, y_seq, seq_symbols = self.create_sequences(
                        X_split, y_split, symbols_split, stride=5
                    )
                    
                    if split == "train":
                        X_train = X_seq
                        y_train = y_seq
                        symbols_train = seq_symbols
                    elif split == "val":
                        X_val = X_seq
                        y_val = y_seq
                        symbols_val = seq_symbols
                    else:  # test
                        X_test = X_seq
                        y_test = y_seq
                        symbols_test = seq_symbols
                
                # Проверка наличия данных
                if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                    logger.warning(f"⚠️ Недостаточно данных для {model_name}")
                    continue
                
                # Проверяем уникальность целевых значений
                unique_train = len(np.unique(y_train)) / len(y_train) * 100
                logger.info(f"📊 Уникальность целевых значений: {unique_train:.1f}%")
                
                # Создаем модель
                model = self.create_model(
                    input_shape=(self.sequence_length, X_train.shape[2]),
                    name=model_name
                )
                
                # Обучаем
                history = self.train_model(
                    model, X_train, y_train, X_val, y_val, 
                    model_name
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
            
            # Сохраняем модели и метаданные
            self.save_models(results)
            
            # Создаем финальный отчет
            self.create_final_report(results)
            
            logger.info("\n✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            logger.info(f"📊 Результаты сохранены в: {self.log_dir}")
            
        except Exception as e:
            logger.error(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
            logger.error("Трейсбек:", exc_info=True)
            raise
    
    def save_models(self, results):
        """Сохранение моделей и метаданных"""
        logger.info("\n💾 Сохранение моделей...")
        
        # Сохраняем веса моделей
        for name, model in self.models.items():
            model_path = f'trained_model/{name}.h5'
            model.save(model_path)
            logger.info(f"   ✅ Модель сохранена: {model_path}")
        
        # Сохраняем scaler
        with open('trained_model/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info("   ✅ Scaler сохранен")
        
        # Сохраняем статистику scaler для каждого признака
        scaler_stats = {
            'center': self.scaler.center_.tolist() if hasattr(self.scaler, 'center_') else None,
            'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'feature_names': self.TECHNICAL_INDICATORS + [
                "rsi_oversold", "rsi_overbought", "macd_bullish",
                "bb_near_lower", "bb_near_upper", "strong_trend", "high_volume"
            ]
        }
        with open('trained_model/scaler_stats.json', 'w') as f:
            json.dump(scaler_stats, f, indent=2)
        logger.info("   ✅ Статистика scaler сохранена")
        
        # Сохраняем важность признаков
        with open('trained_model/feature_importance.json', 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        logger.info("   ✅ Важность признаков сохранена")
        
        # Сохраняем метаданные
        metadata = {
            'type': 'temporal_fusion_transformer',
            'task_type': 'regression',
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_transformer_blocks': self.num_transformer_blocks,
            'features': self.TECHNICAL_INDICATORS + [
                "rsi_oversold", "rsi_overbought", "macd_bullish",
                "bb_near_lower", "bb_near_upper", "strong_trend", "high_volume"
            ],
            'technical_indicators': self.TECHNICAL_INDICATORS,
            'created_at': datetime.now().isoformat(),
            'results': {
                name: result['metrics']
                for name, result in results.items()
            }
        }
        
        with open('trained_model/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("   ✅ Метаданные сохранены")
    
    def predict_live(self, model, current_indicators, symbol):
        """Предсказание для текущего момента с объяснением"""
        # Подготовка признаков
        feature_values = []
        for col in self.TECHNICAL_INDICATORS:
            val = current_indicators.get(col, 0.0)
            if val is None or np.isnan(val) or np.isinf(val):
                val = 0.0
            feature_values.append(float(val))
        
        # Добавляем инженерные признаки
        rsi = current_indicators.get('rsi_val', 50)
        feature_values.append(1.0 if rsi is not None and rsi < 30 else 0.0)
        feature_values.append(1.0 if rsi is not None and rsi > 70 else 0.0)
        
        macd = current_indicators.get('macd_val', 0)
        macd_signal = current_indicators.get('macd_signal', 0)
        feature_values.append(1.0 if macd is not None and macd_signal is not None and macd > macd_signal else 0.0)
        
        bb_position = current_indicators.get('bb_position', 0.5)
        feature_values.append(1.0 if bb_position is not None and bb_position < 0.2 else 0.0)
        feature_values.append(1.0 if bb_position is not None and bb_position > 0.8 else 0.0)
        
        adx = current_indicators.get('adx_val', 0)
        feature_values.append(1.0 if adx is not None and adx > 25 else 0.0)
        
        volume_ratio = current_indicators.get('volume_ratio', 1.0)
        feature_values.append(1.0 if volume_ratio is not None and volume_ratio > 2.0 else 0.0)
        
        # Нормализация
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Создаем последовательность (повторяем текущие данные)
        X_seq = np.repeat(X_scaled[np.newaxis, :, :], self.sequence_length, axis=1)
        
        # Предсказание
        prediction = model.predict(X_seq, verbose=0)[0, 0]
        
        # Анализ ключевых индикаторов
        key_indicators = self._get_key_indicators_for_prediction(current_indicators, prediction)
        
        # Уверенность в предсказании
        confidence = self._calculate_confidence(prediction)
        
        # Рекомендация
        if abs(prediction) < 0.5:
            action = 'HOLD'
            reason = 'Слабый сигнал'
        elif prediction > 1.5:
            action = 'STRONG_BUY' if 'buy' in model.name else 'STRONG_SELL'
            reason = 'Сильный сигнал на вход'
        elif prediction > 0.5:
            action = 'BUY' if 'buy' in model.name else 'SELL'
            reason = 'Умеренный сигнал'
        else:
            action = 'AVOID'
            reason = 'Высокий риск убытка'
        
        return {
            'symbol': symbol,
            'prediction': float(prediction),
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'key_indicators': key_indicators,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_key_indicators_for_prediction(self, indicators, prediction):
        """Определяет ключевые индикаторы, повлиявшие на предсказание"""
        key_factors = []
        
        # RSI анализ
        rsi = indicators.get('rsi_val', 50)
        if rsi is not None:
            if rsi < 30:
                key_factors.append(f'RSI перепродан ({rsi:.1f})')
            elif rsi > 70:
                key_factors.append(f'RSI перекуплен ({rsi:.1f})')
        
        # MACD анализ
        macd = indicators.get('macd_val', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_hist', 0)
        if macd is not None and macd_signal is not None and macd_hist is not None:
            if macd > macd_signal and macd_hist > 0:
                key_factors.append('MACD бычье пересечение')
            elif macd < macd_signal and macd_hist < 0:
                key_factors.append('MACD медвежье пересечение')
        
        # ADX тренд
        adx = indicators.get('adx_val', 0)
        if adx is not None and adx > 25:
            key_factors.append(f'Сильный тренд (ADX={adx:.1f})')
        
        # Объем
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio is not None:
            if volume_ratio > 2.0:
                key_factors.append(f'Высокий объем (x{volume_ratio:.1f})')
            elif volume_ratio < 0.5:
                key_factors.append(f'Низкий объем (x{volume_ratio:.1f})')
        
        # Bollinger Bands
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position is not None:
            if bb_position < 0.2:
                key_factors.append('Цена у нижней границы Bollinger')
            elif bb_position > 0.8:
                key_factors.append('Цена у верхней границы Bollinger')
        
        # Vortex
        vortex_ratio = indicators.get('vortex_ratio', 1.0)
        if vortex_ratio is not None:
            if vortex_ratio > 1.2:
                key_factors.append('Vortex бычий сигнал')
            elif vortex_ratio < 0.8:
                key_factors.append('Vortex медвежий сигнал')
        
        return key_factors[:5]  # Топ-5 факторов
    
    def _calculate_confidence(self, prediction):
        """Рассчитывает уверенность в предсказании"""
        # Чем дальше от 0, тем увереннее
        abs_pred = abs(prediction)
        
        if abs_pred < 0.5:
            return 'Низкая'
        elif abs_pred < 1.0:
            return 'Средняя'
        elif abs_pred < 2.0:
            return 'Высокая'
        else:
            return 'Очень высокая'
    
    def explain_prediction(self, model, indicators, prediction_result):
        """Подробное объяснение предсказания"""
        explanation = f"""
🎯 АНАЛИЗ СИГНАЛА
================

Символ: {prediction_result['symbol']}
Время: {prediction_result['timestamp']}
Предсказание: {prediction_result['prediction']:.2f}%
Действие: {prediction_result['action']}
Уверенность: {prediction_result['confidence']}

📊 КЛЮЧЕВЫЕ ФАКТОРЫ:
"""
        for factor in prediction_result['key_indicators']:
            explanation += f"• {factor}\n"
        
        explanation += f"""

📈 ИНТЕРПРЕТАЦИЯ:
{prediction_result['reason']}

💡 РЕКОМЕНДАЦИЯ:
"""
        
        if prediction_result['action'] in ['BUY', 'STRONG_BUY']:
            explanation += f"""Открыть LONG позицию
Stop Loss: -1.1% от входа
Take Profit: +1.2%, +2.4%, +3.5% (частичные закрытия)"""
        elif prediction_result['action'] in ['SELL', 'STRONG_SELL']:
            explanation += f"""Открыть SHORT позицию
Stop Loss: +1.1% от входа
Take Profit: -1.2%, -2.4%, -3.5% (частичные закрытия)"""
        else:
            explanation += "Воздержаться от входа в позицию"
        
        return explanation
    
    def create_final_report(self, results):
        """Создание итогового отчета"""
        report = f"""
{'='*80}
ИТОГОВЫЙ ОТЧЕТ ПО ОБУЧЕНИЮ
{'='*80}

Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Лог директория: {self.log_dir}
Тип задачи: {'РЕГРЕССИЯ' if self.task == 'regression' else 'БИНАРНАЯ КЛАССИФИКАЦИЯ'}

АРХИТЕКТУРА:
- Тип: Temporal Fusion Transformer (TFT)
- Sequence Length: {self.sequence_length}
- Model Dimension: {self.d_model}
- Number of Heads: {self.num_heads}
- Transformer Blocks: {self.num_transformer_blocks}
- Количество технических индикаторов: {len(self.TECHNICAL_INDICATORS)}
- Общее количество признаков: {len(self.TECHNICAL_INDICATORS) + 7}

РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:
"""
        
        for model_name, result in results.items():
            metrics = result['metrics']
            report += f"\n{model_name.upper()}:\n"
            
            if self.task == 'regression':
                report += f"""- MAE: {metrics['mae']:.4f}%
- RMSE: {metrics['rmse']:.4f}%
- R²: {metrics['r2']:.4f}
- Direction Accuracy: {metrics['direction_accuracy']:.2%}
- Mean Error: {metrics['mean_error']:.4f}%
- Std Error: {metrics['std_error']:.4f}%
"""
            else:  # classification
                report += f"""- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1']:.4f}
- ROC-AUC: {metrics['auc']:.4f}
- Confusion Matrix:
  - TN: {metrics['confusion_matrix'][0,0]:,}  FP: {metrics['confusion_matrix'][0,1]:,}
  - FN: {metrics['confusion_matrix'][1,0]:,}  TP: {metrics['confusion_matrix'][1,1]:,}
"""
        
        report += f"""
{'='*80}
ВИЗУАЛИЗАЦИЯ:
- Графики обучения: {self.log_dir}/plots/training_progress.png
- Графики оценки: {self.log_dir}/plots/*_evaluation.png
- Важность признаков: {self.log_dir}/plots/feature_importance.png

МОДЕЛИ СОХРАНЕНЫ:
- trained_model/*.h5
- trained_model/scaler.pkl
- trained_model/metadata.json

TENSORBOARD:
tensorboard --logdir {self.log_dir}/tensorboard/
{'='*80}
"""
        
        with open(f'{self.log_dir}/final_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"\n📝 Итоговый отчет сохранен: {self.log_dir}/final_report.txt")
        print(report)


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer тренер для криптотрейдинга')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Путь к конфигурационному файлу')
    parser.add_argument('--task', type=str, choices=['regression', 'classification_binary'],
                      default='regression', help='Тип задачи: regression или classification_binary')
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Инициализируем БД
    db_manager = PostgreSQLManager(config['database'])
    
    try:
        # Подключаемся к БД
        db_manager.connect()
        
        # Создаем и обучаем модель
        trainer = UniversalTransformerTrainer(
            db_manager, 
            config_path=args.config,
            task=args.task
        )
        trainer.train()
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()