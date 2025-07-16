"""
Архитектурные компоненты Temporal Fusion Transformer
Адаптировано из train_universal_transformer.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging

from config.model_config import TFT_ARCHITECTURE

logger = logging.getLogger(__name__)


class TransformerBlock(layers.Layer):
    """Блок трансформера с multi-head attention"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim,
            dropout=rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config


class PositionalEncoding(layers.Layer):
    """Позиционное кодирование для временных рядов"""
    
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
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
        # Приводим к одному типу данных для совместимости с mixed precision
        pos_encoding = tf.cast(self.pos_encoding[:, :tf.shape(inputs)[1], :], inputs.dtype)
        return inputs + pos_encoding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model
        })
        return config


class GatedResidualNetwork(layers.Layer):
    """Gated Residual Network - ключевой компонент TFT"""
    
    def __init__(self, units, dropout_rate=0.1, use_residual=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        
        self.elu_dense = layers.Dense(units, activation='elu')
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gate_dense = layers.Dense(units, activation='sigmoid')
        self.layernorm = layers.LayerNormalization()
        
        # Projection layer для residual connection при несовпадении размерностей
        self.projection = None
        
    def build(self, input_shape):
        super().build(input_shape)
        # Если размерности не совпадают, создаем проекционный слой
        if self.use_residual and input_shape[-1] != self.units:
            self.projection = layers.Dense(self.units, use_bias=False)
        
    def call(self, inputs, training=False):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x, training=training)
        
        gate = self.gate_dense(inputs)
        x = gate * x
        
        # Residual connection с проекцией при необходимости
        if self.use_residual:
            if self.projection is not None:
                inputs_projected = self.projection(inputs)
            else:
                inputs_projected = inputs
            x = x + inputs_projected
        
        return self.layernorm(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate,
            "use_residual": self.use_residual
        })
        return config


class InterpretableMultiHeadAttention(layers.Layer):
    """Интерпретируемое multi-head attention для TFT"""
    
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        
        # Для интерпретируемости сохраняем веса внимания
        self.attention_weights = None
        
    def call(self, inputs, training=False):
        # Получаем выход и веса внимания
        attn_output = self.attention(
            inputs, inputs,
            training=training,
            return_attention_scores=False
        )
        
        return attn_output
    
    def get_attention_weights(self):
        """Получение весов внимания для интерпретации"""
        return self.attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate
        })
        return config


class VariableSelectionNetwork(layers.Layer):
    """Variable Selection Network для выбора важных признаков"""
    
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        # Fully connected для каждой переменной
        self.fc_layers = []
        self.grn = GatedResidualNetwork(units, dropout_rate)
        
    def build(self, input_shape):
        # Создаем слой для каждого признака
        n_features = input_shape[-1]
        for _ in range(n_features):
            self.fc_layers.append(
                layers.Dense(self.units, activation='gelu')
            )
        
        # Softmax для весов выбора
        self.softmax = layers.Dense(n_features, activation='softmax')
        
    def call(self, inputs, training=False):
        # Применяем FC к каждому признаку
        feature_outputs = []
        for i, fc in enumerate(self.fc_layers):
            feature_out = fc(inputs[..., i:i+1])
            feature_outputs.append(feature_out)
        
        # Объединяем
        concat_features = tf.concat(feature_outputs, axis=-1)
        
        # Проходим через GRN
        grn_output = self.grn(concat_features, training=training)
        
        # Получаем веса важности
        weights = self.softmax(grn_output)
        
        # Взвешиваем исходные признаки
        # weights имеет форму [batch, seq_len, n_features]
        # inputs имеет форму [batch, seq_len, n_features]
        weighted_features = inputs * weights
        
        return weighted_features, weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config


class StaticCovariateEncoder(layers.Layer):
    """Энкодер для статических ковариат (символы и т.д.)"""
    
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
        self.dense1 = layers.Dense(units, activation='gelu')
        self.dropout = layers.Dropout(dropout_rate)
        self.grn = GatedResidualNetwork(units, dropout_rate)
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.grn(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config