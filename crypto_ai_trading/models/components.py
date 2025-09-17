"""
Базовые компоненты для PatchTST архитектуры
Включает энкодер, позиционное кодирование и нормализацию
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    Нормализует данные по последней размерности (features)
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        
    def _get_statistics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вычисляет статистики по временной размерности"""
        # x: [batch_size, seq_len, num_features]
        mean = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, num_features]
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
        return mean, stdev
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, num_features]
            mode: 'norm' или 'denorm'
        """
        if mode == 'norm':
            self._mean, self._stdev = self._get_statistics(x)
            x = (x - self._mean) / self._stdev
            
            if self.affine:
                x = x * self.weight + self.bias
                
            return x
        
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias) / self.weight
                
            x = x * self._stdev + self._mean
            return x
        
        else:
            raise ValueError(f"Неизвестный mode: {mode}")


class AttentionLayer(nn.Module):
    """Multi-head самовнимание"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] (optional)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        residual = x
        
        # Multi-head attention
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(attn_output)
        output = self.dropout(output) + residual
        output = self.layer_norm(output)
        
        return output


class FeedForwardLayer(nn.Module):
    """Position-wise feed forward сеть"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Неподдерживаемая активация: {activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return self.layer_norm(x + residual)


class PatchTSTEncoderLayer(nn.Module):
    """Один слой трансформер энкодера"""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int, 
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        
        self.self_attn = AttentionLayer(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardLayer(d_model, d_ff, dropout, activation)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.self_attn(x, mask)
        x = self.feed_forward(x)
        return x


class PatchTSTEncoder(nn.Module):
    """Стек из encoder слоев"""
    
    def __init__(self, 
                 e_layers: int,
                 d_model: int,
                 n_heads: int, 
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 res_attention: bool = True):
        super().__init__()
        
        self.layers = nn.ModuleList([
            PatchTSTEncoderLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(e_layers)
        ])
        
        self.res_attention = res_attention
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            return_attention: Возвращать ли attention веса
        """
        attentions = [] if return_attention else None
        
        for layer in self.layers:
            x = layer(x)
            
        if return_attention:
            return x, attentions
        else:
            return x