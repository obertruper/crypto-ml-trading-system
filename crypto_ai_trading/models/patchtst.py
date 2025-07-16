"""
Реализация PatchTST (Patch Time Series Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math

# Обработка импорта einops
try:
    from einops import rearrange, repeat
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    print("Warning: einops not available. Using manual tensor operations.")


# Функции для ручной перестановки размерностей (fallback для einops)
def manual_rearrange_b_n_c_p_to_bc_n_p(tensor):
    """Ручная реализация rearrange('b n c p -> (b c) n p')"""
    b, n, c, p = tensor.shape
    # (B, N, C, P) -> (B, C, N, P) -> (B*C, N, P)
    return tensor.permute(0, 2, 1, 3).contiguous().view(b * c, n, p)


def manual_rearrange_bc_t_to_b_t_c(tensor, n_vars):
    """Ручная реализация rearrange('(b c) t -> b t c', c=n_vars)"""
    bc, t = tensor.shape
    b = bc // n_vars
    c = n_vars
    # (B*C, T) -> (B, C, T) -> (B, T, C)
    return tensor.view(b, c, t).transpose(1, 2)


def manual_repeat_t_n(tensor, n):
    """Ручная реализация repeat('t -> t n', n=n)"""
    # tensor shape: (T,) -> (T, N)
    return tensor.unsqueeze(1).expand(-1, n)


# ===== КЛАССЫ УЛУЧШЕНИЙ ИЗ patchtst_improved.py =====

class FeatureAttention(nn.Module):
    """Механизм внимания для выбора важных признаков"""
    
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: (B, L, C)
        # Вычисляем веса внимания для каждого временного шага
        weights = self.attention(x)  # (B, L, C)
        # Применяем веса
        return x * weights


class ImprovedPatchEmbedding(nn.Module):
    """Улучшенное создание патчей с многомасштабным подходом"""
    
    def __init__(self, d_model: int, patch_len: int, stride: int, 
                 n_features: int, dropout: float = 0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Основная проекция патчей
        self.patch_proj = nn.Linear(patch_len, d_model)
        
        # Дополнительные проекции для разных масштабов
        self.multi_scale = nn.ModuleList([
            nn.Conv1d(n_features, d_model // 4, kernel_size=k, stride=1, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        
        # Создание основных патчей
        num_patches = (L - self.patch_len) // self.stride + 1
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (B, num_patches, C, patch_len)
        
        # Проекция патчей для каждого канала
        if EINOPS_AVAILABLE:
            patches = rearrange(patches, 'b n c p -> (b c) n p')
        else:
            patches = manual_rearrange_b_n_c_p_to_bc_n_p(patches)
            
        patch_embeds = self.patch_proj(patches)  # (B*C, num_patches, d_model)
        
        # Многомасштабные признаки
        x_conv = x.transpose(1, 2)  # (B, C, L)
        multi_scale_features = []
        for conv in self.multi_scale:
            feat = conv(x_conv)  # (B, d_model//4, L')
            feat = F.adaptive_avg_pool1d(feat, num_patches)  # (B, d_model//4, num_patches)
            multi_scale_features.append(feat)
        
        multi_scale_features = torch.cat(multi_scale_features, dim=1)  # (B, d_model, num_patches)
        multi_scale_features = multi_scale_features.transpose(1, 2)  # (B, num_patches, d_model)
        
        # Объединение признаков
        if EINOPS_AVAILABLE:
            patch_embeds = rearrange(patch_embeds, '(b c) n d -> b n (c d)', c=C)
        else:
            BC, N, D = patch_embeds.shape
            patch_embeds = patch_embeds.view(B, C, N, D).permute(0, 2, 1, 3).contiguous()
            patch_embeds = patch_embeds.view(B, N, C * D)
        
        # Уменьшаем размерность до d_model
        patch_embeds = self.fusion(patch_embeds)  # (B, num_patches, d_model)
        
        # Комбинируем с многомасштабными признаками
        combined = patch_embeds + multi_scale_features
        combined = self.norm(combined)
        combined = self.dropout(combined)
        
        return combined


# ===== КОНЕЦ КЛАССОВ УЛУЧШЕНИЙ =====


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):
    """Преобразование временного ряда в патчи"""
    
    def __init__(self, 
                 patch_len: int,
                 stride: int,
                 in_channels: int,
                 embed_dim: int,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        self.in_channels = in_channels
        
        self.proj = nn.Linear(patch_len, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, L, C = x.shape
        
        num_patches = (L - self.patch_len) // self.stride + 1
        
        # Создание патчей
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches shape: (B, num_patches, C, patch_len)
        
        # Перестановка размерностей
        if EINOPS_AVAILABLE:
            patches = rearrange(patches, 'b n c p -> (b c) n p')
        else:
            # Ручная перестановка: (B, num_patches, C, patch_len) -> (B*C, num_patches, patch_len)
            patches = manual_rearrange_b_n_c_p_to_bc_n_p(patches)
        
        patches = self.proj(patches)
        patches = self.norm(patches)
        
        return patches, num_patches

class FlattenHead(nn.Module):
    """Голова для прогнозирования"""
    
    def __init__(self,
                 n_vars: int,
                 nf: int,
                 target_window: int,
                 head_dropout: float = 0.0):
        super().__init__()
        
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        
        if EINOPS_AVAILABLE:
            x = rearrange(x, '(b n) t -> b t n', n=self.n_vars)
        else:
            # Ручная перестановка: (B*n_vars, target_window) -> (B, target_window, n_vars)
            x = manual_rearrange_bc_t_to_b_t_c(x, self.n_vars)
        
        return x


class PatchTSTForPrediction(nn.Module):
    """PatchTST модель для предсказания целевых переменных с поддержкой LONG/SHORT"""
    
    def __init__(self,
                 c_in: int,  # количество входных признаков
                 c_out: int,  # количество выходных переменных
                 context_window: int,
                 target_window: int,
                 patch_len: int = 16,
                 stride: int = 8,
                 n_layers: int = 3,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 256,
                 dropout: float = 0.0,
                 layer_dropout: float = 0.0,  # Stochastic depth
                 weight_noise: float = 0.0,  # Noise regularization
                 task_type: str = 'regression',  # 'regression' или 'classification'
                 **kwargs):
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.layer_dropout = layer_dropout
        self.weight_noise = weight_noise
        self.task_type = task_type
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Фиксированное вычисление num_patches с валидацией
        self.num_patches = self._calculate_num_patches(context_window, patch_len, stride)
        
        if self.num_patches <= 0:
            raise ValueError(
                f"Некорректные параметры патчей: context_window={context_window}, "
                f"patch_len={patch_len}, stride={stride} -> num_patches={self.num_patches}"
            )
        
        # Получаем параметры улучшений из конфигурации
        self.use_improvements = kwargs.get('use_improvements', False)
        self.feature_attention_enabled = kwargs.get('feature_attention', False)
        self.multi_scale_patches = kwargs.get('multi_scale_patches', False)
        
        # Feature attention (если включено)
        if self.use_improvements and self.feature_attention_enabled:
            self.feature_attention = FeatureAttention(c_in, d_model, dropout)
        else:
            self.feature_attention = None
            
        # Patch embedding (обычный или улучшенный)
        if self.use_improvements and self.multi_scale_patches:
            self.patch_embedding = ImprovedPatchEmbedding(
                d_model=d_model,
                patch_len=patch_len,
                stride=stride,
                n_features=c_in,
                dropout=dropout
            )
        else:
            self.patch_embedding = PatchEmbedding(
                patch_len=patch_len,
                stride=stride,
                in_channels=c_in,
                embed_dim=d_model,
                norm_layer=nn.LayerNorm
            )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection с поддержкой различных задач
        self.head_nf = d_model * self.num_patches
        
        if task_type == 'regression':
            # Для регрессии предсказываем continuous values
            self.output_projection = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(self.head_nf, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, c_out * target_window)
            )
        else:
            # Для классификации предсказываем вероятности
            self.output_projection = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(self.head_nf, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, c_out * target_window),
                nn.Sigmoid()  # Вероятности в диапазоне [0, 1]
            )
        
        # Инициализация весов
        self._init_weights()
    
    def _calculate_num_patches(self, context_window: int, patch_len: int, stride: int) -> int:
        """ИСПРАВЛЕННОЕ вычисление количества патчей с валидацией"""
        if context_window < patch_len:
            raise ValueError(f"context_window ({context_window}) должен быть >= patch_len ({patch_len})")
        
        return max(1, (context_window - patch_len) // stride + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N = x.shape
        
        # Проверка входных размерностей
        if L != self.context_window:
            raise ValueError(f"Expected context_window={self.context_window}, got L={L}")
        if N != self.c_in:
            raise ValueError(f"Expected c_in={self.c_in}, got N={N}")
        
        # Нормализация входных данных
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - x_mean) / x_std
        
        # Применяем feature attention если включено
        if self.feature_attention is not None:
            x_norm = self.feature_attention(x_norm)
        
        # Создание патчей
        if isinstance(self.patch_embedding, ImprovedPatchEmbedding):
            # Улучшенная версия возвращает только патчи
            x_patches = self.patch_embedding(x_norm)
            actual_num_patches = x_patches.size(1)
        else:
            # Обычная версия возвращает патчи и количество
            x_patches, actual_num_patches = self.patch_embedding(x_norm)
        # x_patches: (B*N, actual_num_patches, d_model)
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Строгая проверка без изменения архитектуры
        if actual_num_patches != self.num_patches:
            raise ValueError(
                f"Несоответствие количества патчей: ожидалось {self.num_patches}, "
                f"получено {actual_num_patches}. Проверьте параметры модели."
            )
        
        # Позиционное кодирование
        x_patches = self.pos_encoding(x_patches)
        
        # Transformer encoder
        x_encoded = self.transformer_encoder(x_patches)
        
        # Агрегация по всем признакам
        # x_encoded имеет размерность (B*N, num_patches, d_model)
        # Нужно reshape обратно и усреднить по признакам
        d_model = x_encoded.size(-1)
        x_encoded = x_encoded.view(B, N, self.num_patches, d_model)  # (B, N, num_patches, d_model)
        x_encoded = x_encoded.mean(dim=1)  # (B, num_patches, d_model) - усреднение по признакам
        
        # Прогнозирование
        output = self.output_projection(x_encoded)
        
        # Reshape to (B, target_window, c_out)
        output = output.view(B, self.target_window, self.c_out)
        
        return output
    
    def _init_weights(self):
        """Инициализация весов модели"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class PatchTSTForTrading(nn.Module):
    """Обертка для PatchTST с поддержкой торговых сигналов LONG/SHORT"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        
        # Базовые параметры
        self.n_features = model_config.get('input_size', 100)
        self.context_window = model_config.get('context_window', 168)
        
        # Создаем отдельные модели для LONG и SHORT
        base_params = {
            'c_in': self.n_features,
            'context_window': self.context_window,
            'target_window': model_config.get('pred_len', 4),
            'patch_len': model_config.get('patch_len', 16),
            'stride': model_config.get('stride', 8),
            'n_layers': model_config.get('e_layers', 3),
            'd_model': model_config.get('d_model', 128),
            'n_heads': model_config.get('n_heads', 8),
            'd_ff': model_config.get('d_ff', 512),
            'dropout': model_config.get('dropout', 0.1),
            'task_type': 'classification',
            'use_improvements': model_config.get('use_improvements', False),
            'feature_attention': model_config.get('feature_attention', False),
            'multi_scale_patches': model_config.get('multi_scale_patches', False)
        }
        
        # Модель для LONG: вероятности TP1, TP2, TP3, SL, оптимальное время входа
        self.long_model = PatchTSTForPrediction(
            c_out=5,  # TP1, TP2, TP3, SL, optimal_entry_time
            **base_params
        )
        
        # Модель для SHORT: вероятности TP1, TP2, TP3, SL, оптимальное время входа  
        self.short_model = PatchTSTForPrediction(
            c_out=5,  # TP1, TP2, TP3, SL, optimal_entry_time
            **base_params
        )
        
        # Модель для определения направления (LONG/SHORT/NEUTRAL)
        self.direction_model = PatchTSTForPrediction(
            c_out=3,  # LONG, SHORT, NEUTRAL
            **base_params
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Прямой проход с предсказанием всех необходимых значений"""
        
        # Предсказание направления
        direction_probs = self.direction_model(x)  # (B, 1, 3)
        direction_probs = direction_probs.squeeze(1)  # (B, 3)
        
        # Предсказания для LONG
        long_predictions = self.long_model(x)  # (B, 1, 5)
        long_predictions = long_predictions.squeeze(1)  # (B, 5)
        
        # Предсказания для SHORT
        short_predictions = self.short_model(x)  # (B, 1, 5)
        short_predictions = short_predictions.squeeze(1)  # (B, 5)
        
        return {
            'direction_probs': direction_probs,  # [long_prob, short_prob, neutral_prob]
            'long_tp_probs': long_predictions[:, :3],  # Вероятности TP1, TP2, TP3 для LONG
            'long_sl_prob': long_predictions[:, 3:4],  # Вероятность SL для LONG
            'long_entry_time': long_predictions[:, 4:5],  # Оптимальное время входа LONG
            'short_tp_probs': short_predictions[:, :3],  # Вероятности TP1, TP2, TP3 для SHORT
            'short_sl_prob': short_predictions[:, 3:4],  # Вероятность SL для SHORT
            'short_entry_time': short_predictions[:, 4:5],  # Оптимальное время входа SHORT
        }


def create_patchtst_model(config: Dict) -> PatchTSTForPrediction:
    """Создание модели из конфигурации с валидацией"""
    from utils.config_validator import ModelConfig
    
    # Валидация конфигурации модели
    if isinstance(config, dict):
        model_config = config.get('model', {})
        try:
            # Валидация через Pydantic
            validated_config = ModelConfig(**model_config)
            model_config = validated_config.dict()
        except Exception as e:
            raise ValueError(f"Неверная конфигурация модели: {e}")
    else:
        model_config = config['model']
    
    n_features = model_config.get('input_size', 100)
    n_targets = model_config.get('output_size', 1)  # количество целевых переменных
    
    # Определяем тип задачи
    task_type = model_config.get('task_type', 'regression')
    
    if task_type == 'trading':
        # Создаем специализированную модель для торговли
        model = PatchTSTForTrading(config)
    else:
        # Создаем базовую модель
        model = PatchTSTForPrediction(
            c_in=n_features,
            c_out=n_targets,
            context_window=model_config.get('context_window', 168),
            target_window=model_config.get('pred_len', 4),
            patch_len=model_config.get('patch_len', 16),
            stride=model_config.get('stride', 8),
            n_layers=model_config.get('e_layers', 3),
            d_model=model_config.get('d_model', 128),
            n_heads=model_config.get('n_heads', 8),
            d_ff=model_config.get('d_ff', 512),
            dropout=model_config.get('dropout', 0.1),
            task_type=task_type,
            # Параметры улучшений
            use_improvements=model_config.get('use_improvements', False),
            feature_attention=model_config.get('feature_attention', False),
            multi_scale_patches=model_config.get('multi_scale_patches', False)
        )
    
    return model