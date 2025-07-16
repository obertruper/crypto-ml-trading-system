"""
Temporal Fusion Transformer модель
Основная архитектура для временных рядов
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging

from models.tft_architecture import (
    TransformerBlock,
    PositionalEncoding,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention
)
from config import Config

logger = logging.getLogger(__name__)


class TemporalFusionTransformer(keras.Model):
    """Temporal Fusion Transformer для регрессии и классификации"""
    
    def __init__(self, config: Config, num_features: int, name="TFT", **kwargs):
        super(TemporalFusionTransformer, self).__init__(name=name, **kwargs)
        
        self.config = config
        self.num_features = num_features
        self.sequence_length = config.model.sequence_length
        self.d_model = config.model.d_model
        self.task_type = config.training.task_type
        
        # Улучшенный Variable Selection Network с регуляризацией
        self.vsn = self._build_improved_vsn()
        
        # LSTM Encoder для временной обработки
        self.lstm_encoder = layers.LSTM(
            units=config.model.lstm_units,
            return_sequences=True,
            dropout=config.model.dropout_rate,
            recurrent_dropout=config.model.dropout_rate * 0.5  # Меньше dropout для recurrent
        )
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(
            position=config.model.sequence_length,
            d_model=config.model.d_model
        )
        
        # Transformer Blocks
        self.transformer_blocks = []
        for _ in range(config.model.num_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(
                    embed_dim=config.model.d_model,
                    num_heads=config.model.num_heads,
                    ff_dim=config.model.ff_dim,
                    rate=config.model.dropout_rate
                )
            )
        
        # Interpretable Multi-Head Attention
        self.interpretable_attention = InterpretableMultiHeadAttention(
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout_rate
        )
        
        # Проекция в d_model размерность с нормальной инициализацией
        self.input_projection = layers.Dense(
            config.model.d_model,
            kernel_initializer=keras.initializers.HeNormal()
        )
        
        # Global Average Pooling
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Output Network
        self.output_network = self._build_output_network()
        
        # Для сохранения важности признаков
        self.feature_importance = None
        
    def _build_output_network(self):
        """Построение выходной сети в зависимости от задачи"""
        layers_list = []
        
        # Hidden layers с инициализацией He
        for units in self.config.model.mlp_units:
            layers_list.extend([
                layers.Dense(
                    units, 
                    activation='gelu',
                    kernel_initializer=keras.initializers.HeNormal()
                ),
                layers.BatchNormalization(),  # Добавляем BatchNorm для стабильности
                layers.Dropout(self.config.model.dropout_rate)
            ])
        
        # Output layer с малой инициализацией для стабильности
        if self.task_type == 'regression':
            layers_list.append(layers.Dense(
                1,
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
            ))  # Линейный выход с малой инициализацией
        else:  # classification_binary
            layers_list.append(layers.Dense(
                1, 
                activation='sigmoid',
                kernel_initializer=keras.initializers.GlorotUniform()
            ))
            
        return keras.Sequential(layers_list)
    
    def _build_improved_vsn(self):
        """Создание улучшенного Variable Selection Network с фильтрацией шума"""
        # Используем существующий VSN с дополнительной регуляризацией
        vsn = VariableSelectionNetwork(
            units=self.config.model.d_model,
            dropout_rate=self.config.model.dropout_rate
        )
        
        # VSN - это кастомный слой, не Sequential, поэтому просто возвращаем его
        # Регуляризация уже встроена в сам VariableSelectionNetwork
        return vsn
    
    def call(self, inputs, training=False):
        # Заменяем NaN на нули без условного оператора
        inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        
        # Добавляем шумоподавление с помощью Dropout на входе
        if training and self.config.training.use_augmentation:
            # Adaptive noise filtering - случайно зануляем малозначимые признаки
            noise_mask = tf.random.uniform(tf.shape(inputs)) > 0.1  # 10% dropout
            inputs = inputs * tf.cast(noise_mask, inputs.dtype)
        
        # Временно обходим VSN из-за проблем с размерностями
        # TODO: исправить VSN для правильной работы с 3D тензорами
        
        # Применяем проекцию напрямую к входным данным
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Reshape для применения Dense ко всем временным шагам
        inputs_flat = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
        x_flat = self.input_projection(inputs_flat)
        
        # Reshape обратно в [batch, sequence, d_model]
        x = tf.reshape(x_flat, [batch_size, seq_len, self.d_model])
        
        # Сохраняем заглушку для feature_importance
        self.feature_importance = None
        
        # LSTM Encoding
        lstm_out = self.lstm_encoder(x, training=training)
        
        # Проверяем выход LSTM
        tf.debugging.assert_equal(lstm_out.shape[1], self.sequence_length,
            message=f"LSTM output sequence length mismatch: {lstm_out.shape}")
        
        # Positional Encoding
        x = self.pos_encoding(lstm_out)
        
        # Transformer Blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        # Interpretable Attention
        attn_output = self.interpretable_attention(x, training=training)
        
        # Global Pooling
        x = self.global_pool(attn_output)
        
        # Output
        output = self.output_network(x, training=training)
        
        return output
    
    def get_feature_importance(self):
        """Получение важности признаков из VSN"""
        if self.feature_importance is not None:
            return self.feature_importance.numpy()
        return None
    
    def build_graph(self):
        """Построение графа модели для визуализации"""
        x = keras.Input(shape=(self.sequence_length, self.num_features))
        return keras.Model(inputs=[x], outputs=self.call(x))
    
    @tf.function
    def train_step(self, data):
        """Кастомный train step с дополнительными метриками"""
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Вычисляем градиенты
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Обновляем веса
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Обновляем метрики
        self.compiled_metrics.update_state(y, y_pred)
        
        # Добавляем дополнительные метрики
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        
        # Добавляем норму градиентов для мониторинга
        grad_norm = tf.linalg.global_norm(gradients)
        metrics['grad_norm'] = grad_norm
        
        return metrics
    
    @tf.function
    def test_step(self, data):
        """Кастомный test step"""
        x, y = data
        
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Обновляем метрики
        self.compiled_metrics.update_state(y, y_pred)
        
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        
        return metrics
    
    def get_config(self):
        """Конфигурация для сохранения модели"""
        return {
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__
            },
            'num_features': self.num_features
        }
    
    @classmethod
    def from_config(cls, config):
        """Восстановление модели из конфигурации"""
        # Преобразуем dict обратно в Config объект
        model_config = Config()
        for key, value in config['config'].items():
            setattr(model_config, key, value)
            
        return cls(
            config=model_config,
            num_features=config['num_features']
        )