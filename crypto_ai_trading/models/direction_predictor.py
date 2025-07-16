"""
Специализированная модель для предсказания направления движения цены
Фокус на максимизации directional accuracy для прибыльной торговли
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

from models.patchtst import PatchEmbedding, PositionalEncoding


class MultiScalePatchEmbedding(nn.Module):
    """Multi-scale патчи для захвата паттернов разных таймфреймов"""
    
    def __init__(self, n_features: int, d_model: int, scales: List[int] = [4, 16, 64]):
        super().__init__()
        self.scales = scales
        self.d_model = d_model
        
        # Вычисляем размеры для каждого масштаба
        base_dim = d_model // len(scales)
        remainder = d_model % len(scales)
        
        dims = [base_dim] * len(scales)
        # Добавляем остаток к последнему
        if remainder > 0:
            dims[-1] += remainder
            
        self.embeddings = nn.ModuleList([
            nn.Conv1d(n_features, dim, kernel_size=scale, stride=scale // 2)
            for scale, dim in zip(scales, dims)
        ])
        
        # Проекция теперь точно соответствует суммарной размерности
        total_dim = sum(dims)
        self.projection = nn.Linear(total_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        multi_scale_features = []
        for embed in self.embeddings:
            features = embed(x)  # (batch, d_model/n_scales, n_patches)
            features = features.transpose(1, 2)  # (batch, n_patches, d_model/n_scales)
            multi_scale_features.append(features)
        
        # Выравниваем длины последовательностей
        min_len = min(f.size(1) for f in multi_scale_features)
        multi_scale_features = [f[:, :min_len, :] for f in multi_scale_features]
        
        # Конкатенация по feature dimension
        combined = torch.cat(multi_scale_features, dim=-1)  # (batch, min_len, d_model)
        
        # Проекция и нормализация
        output = self.projection(combined)
        output = self.norm(output)
        
        return output


class AttentionPooling(nn.Module):
    """Attention-based pooling вместо простого усреднения"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        scores = self.attention(x)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        output = (x * weights).sum(dim=1)  # (batch, d_model)
        
        return output


class DirectionSpecificEncoder(nn.Module):
    """Специализированный энкодер для задачи определения направления"""
    
    def __init__(self, 
                 d_model: int = 512,
                 n_heads: int = 8,
                 e_layers: int = 4,
                 d_ff: int = 2048,
                 dropout: float = 0.3):
        super().__init__()
        
        # Transformer encoder layers с увеличенной глубиной
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(e_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
            
        # Final normalization
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class TemporalConsistencyModule(nn.Module):
    """Модуль для обеспечения согласованности предсказаний между таймфреймами"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, short_term: torch.Tensor, long_term: torch.Tensor) -> torch.Tensor:
        # Cross-attention: short-term обращает внимание на long-term паттерны
        attended, _ = self.cross_attention(
            query=short_term,
            key=long_term,
            value=long_term
        )
        
        # Residual connection
        output = self.norm(short_term + attended)
        
        return output


class DirectionPredictor(nn.Module):
    """
    Специализированная модель для предсказания направления движения цены
    Оптимизирована для максимальной directional accuracy
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Конфигурация
        self.n_features = config.get('n_features', 171)
        self.context_window = config.get('context_window', 168)
        self.d_model = 512  # Увеличенная размерность для лучшего представления
        self.n_heads = 8
        self.e_layers = 4
        self.d_ff = 2048
        self.dropout = 0.3
        
        # Multi-scale patch embedding
        self.multiscale_patches = MultiScalePatchEmbedding(
            n_features=self.n_features,
            d_model=self.d_model,
            scales=[4, 16, 64]  # 1h, 4h, 16h паттерны для 15м данных
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.context_window
        )
        
        # Специализированный encoder
        self.encoder = DirectionSpecificEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout
        )
        
        # Temporal consistency для согласованности между таймфреймами
        self.temporal_consistency = TemporalConsistencyModule(self.d_model)
        
        # Attention pooling
        self.attention_pool = AttentionPooling(self.d_model)
        
        # Глубокая classification голова с skip connections
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Linear(256 + self.d_model, 128),  # Skip connection
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(128 + 256, 64),  # Skip connection
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        ])
        
        # Выходные головы для разных таймфреймов
        self.output_heads = nn.ModuleDict({
            '15m': nn.Linear(64, 3),  # UP, DOWN, FLAT
            '1h': nn.Linear(64, 3),
            '4h': nn.Linear(64, 3),
            '12h': nn.Linear(64, 3)
        })
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Careful weight initialization для стабильного обучения"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_len, n_features)
            return_features: вернуть промежуточные признаки для анализа
            
        Returns:
            Dict с предсказаниями для каждого таймфрейма
        """
        batch_size = x.shape[0]
        
        # Multi-scale embedding
        embedded = self.multiscale_patches(x)
        
        # Positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Encode
        encoded = self.encoder(embedded)
        
        # Разделяем на короткие и длинные таймфреймы для temporal consistency
        seq_len = encoded.size(1)
        short_term = encoded[:, :seq_len//2, :]
        long_term = encoded[:, seq_len//2:, :]
        
        # Обеспечиваем temporal consistency
        consistent_features = self.temporal_consistency(short_term, long_term)
        
        # Объединяем обратно
        enhanced_encoded = torch.cat([consistent_features, long_term], dim=1)
        
        # Attention pooling
        pooled = self.attention_pool(enhanced_encoded)
        
        # Classification с skip connections
        features = [pooled]
        x_cls = pooled
        
        for i, layer in enumerate(self.classifier):
            if i == 1:
                # Первый skip connection
                x_cls = torch.cat([x_cls, pooled], dim=-1)
            elif i == 2:
                # Второй skip connection
                x_cls = torch.cat([x_cls, features[1]], dim=-1)
                
            x_cls = layer(x_cls)
            features.append(x_cls)
        
        # Выходные предсказания для каждого таймфрейма
        outputs = {
            'direction_15m': self.output_heads['15m'](x_cls),
            'direction_1h': self.output_heads['1h'](x_cls),
            'direction_4h': self.output_heads['4h'](x_cls),
            'direction_12h': self.output_heads['12h'](x_cls)
        }
        
        if return_features:
            outputs['features'] = pooled
            outputs['attention_weights'] = self.get_attention_weights()
        
        return outputs
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Получить веса attention для визуализации"""
        # TODO: Implement attention weight extraction
        return None
    
    def predict_with_confidence(self, x: torch.Tensor) -> Dict[str, Tuple[int, float]]:
        """
        Предсказание с оценкой уверенности
        
        Returns:
            Dict с (predicted_class, confidence) для каждого таймфрейма
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            results = {}
            for key, logits in outputs.items():
                if key.startswith('direction_'):
                    probs = F.softmax(logits, dim=-1)
                    confidence, predicted = probs.max(dim=-1)
                    results[key] = (predicted.item(), confidence.item())
                    
        return results


class DirectionalTradingLoss(nn.Module):
    """
    Специализированная loss функция для максимизации прибыльности
    Учитывает потенциальный P&L от торговых решений
    """
    
    def __init__(self, 
                 commission: float = 0.001,
                 class_weights: Optional[List[float]] = None,
                 profit_focus_weight: float = 10.0):
        super().__init__()
        self.commission = commission
        self.profit_focus_weight = profit_focus_weight
        
        # Веса классов (можно настроить для баланса)
        if class_weights is None:
            self.class_weights = torch.tensor([1.0, 1.0, 0.5])  # Меньше вес для FLAT
        else:
            self.class_weights = torch.tensor(class_weights)
            
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                price_changes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Вычисление loss с учетом потенциальной прибыли
        
        Args:
            predictions: Dict с логитами для каждого таймфрейма
            targets: Dict с истинными метками (0=UP, 1=DOWN, 2=FLAT)
            price_changes: Dict с процентными изменениями цены
            
        Returns:
            Общий loss
        """
        total_loss = 0.0
        timeframe_weights = {'15m': 0.2, '1h': 0.3, '4h': 0.35, '12h': 0.15}
        
        for timeframe in ['15m', '1h', '4h', '12h']:
            key = f'direction_{timeframe}'
            
            # Base cross entropy с весами классов
            ce_loss = F.cross_entropy(
                predictions[key], 
                targets[key],
                weight=self.class_weights.to(predictions[key].device),
                reduction='none'
            )
            
            # Рассчитываем потенциальный P&L
            predicted_direction = predictions[key].argmax(dim=1)
            actual_direction = targets[key]
            price_change = price_changes[timeframe]
            
            # Убеждаемся что price_change имеет правильную размерность
            if price_change.dim() > 1:
                price_change = price_change.squeeze(-1)
            
            # Потенциальная прибыль/убыток от каждого решения
            potential_pnl = torch.zeros_like(ce_loss)
            
            # LONG позиции (predicted=0)
            long_mask = predicted_direction == 0
            potential_pnl[long_mask] = price_change[long_mask] - self.commission
            
            # SHORT позиции (predicted=1)
            short_mask = predicted_direction == 1  
            potential_pnl[short_mask] = -price_change[short_mask] - self.commission
            
            # Веса на основе величины движения цены
            # Больше штраф за ошибки на крупных движениях
            movement_weight = 1 + torch.abs(price_change) * self.profit_focus_weight
            
            # Дополнительный штраф за неправильное направление
            wrong_direction_mask = (
                ((predicted_direction == 0) & (actual_direction == 1)) |  # Предсказали UP, было DOWN
                ((predicted_direction == 1) & (actual_direction == 0))    # Предсказали DOWN, было UP
            )
            direction_penalty = wrong_direction_mask.float() * torch.abs(price_change) * 2
            
            # Штраф за false positives (торговля когда нужно было ждать)
            false_positive_penalty = ((predicted_direction != 2) & (actual_direction == 2)).float() * 0.5
            
            # Комбинированный loss для таймфрейма
            timeframe_loss = (ce_loss * movement_weight + direction_penalty + false_positive_penalty).mean()
            
            # Взвешенный вклад в общий loss
            total_loss += timeframe_loss * timeframe_weights[timeframe]
            
        return total_loss


class DirectionEnsemble(nn.Module):
    """Ансамбль моделей для улучшения точности предсказаний направления"""
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_outputs = []
        
        for model in self.models:
            outputs = model(x)
            all_outputs.append(outputs)
            
        # Weighted averaging
        ensemble_outputs = {}
        for key in all_outputs[0].keys():
            if key.startswith('direction_'):
                weighted_sum = sum(
                    self.weights[i] * F.softmax(out[key], dim=-1) 
                    for i, out in enumerate(all_outputs)
                )
                ensemble_outputs[key] = torch.log(weighted_sum + 1e-8)  # Log для возврата логитов
                
        return ensemble_outputs