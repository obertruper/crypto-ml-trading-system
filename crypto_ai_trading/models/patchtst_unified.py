"""
Унифицированная версия PatchTST для многозадачного обучения
Решает проблему несоответствия размерностей: модель выдает точное количество выходов из конфига
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math

# Импортируем из components.py вместо несуществующего patchtst.py
from .components import PositionalEncoding
# PatchEmbedding не найден, определим локально
EINOPS_AVAILABLE = False


class RevIN(nn.Module):
    """Reversible Instance Normalization для временных рядов"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        x: [Batch, Length, Features]
        mode: 'norm' для нормализации, 'denorm' для денормализации
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
            x = (x - self.mean) / self.std
            
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
                
            x = x * self.std + self.mean
            
        return x


class PatchTSTEncoder(nn.Module):
    """Encoder для PatchTST с остаточными соединениями"""
    
    def __init__(self, 
                 e_layers: int = 3,
                 d_model: int = 256,
                 n_heads: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 res_attention: bool = True):
        super().__init__()
        
        self.e_layers = e_layers
        self.d_model = d_model
        self.res_attention = res_attention
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                res_attention=res_attention
            ) for _ in range(e_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Length, Features]
        """
        # Encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            
        x = self.norm(x)
        
        return x


class EncoderLayer(nn.Module):
    """Слой энкодера с multi-head attention"""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 res_attention: bool = True):
        super().__init__()
        
        self.res_attention = res_attention
        
        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Length, Features]
        """
        # Self attention with residual
        attn_out, _ = self.self_attention(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed forward with residual
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x


class UnifiedPatchTSTForTrading(nn.Module):
    """
    Единая модель PatchTST для торговли с настраиваемым количеством выходов
    
    Архитектура:
    1. Общий энкодер для извлечения признаков
    2. Многозадачные головы для разных типов предсказаний
    3. Правильное соответствие с целевыми переменными из датасета
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        
        # Добавляем логгер
        from utils.logger import get_logger
        self.logger = get_logger('UnifiedPatchTSTForTrading')
        
        # Базовые параметры
        self.n_features = model_config.get('input_size', 159)
        self.context_window = model_config.get('context_window', 168)
        self.patch_len = model_config.get('patch_len', 16)
        self.stride = model_config.get('stride', 8)
        
        # Параметры трансформера
        self.d_model = model_config.get('d_model', 256)
        self.n_heads = model_config.get('n_heads', 4)
        self.e_layers = model_config.get('e_layers', 3)
        self.d_ff = model_config.get('d_ff', 512)
        self.dropout = model_config.get('dropout', 0.1)
        self.activation = model_config.get('activation', 'gelu')
        
        # ВАЖНО: Количество выходов берем из конфигурации или используем 20 по умолчанию для v4.0
        self.n_outputs = model_config.get('output_size', 20)
        
        # Нормализация
        self.revin = RevIN(
            num_features=self.n_features,
            eps=1e-5,
            affine=True
        )
        
        # Патч эмбеддинги
        self.patch_embedding = nn.Conv1d(
            in_channels=self.n_features,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            stride=self.stride,
            padding=0,
            bias=False
        )
        
        # Позиционное кодирование
        self.n_patches = (self.context_window - self.patch_len) // self.stride + 1
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.n_patches
        )
        
        # Основной энкодер
        self.encoder = PatchTSTEncoder(
            e_layers=self.e_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            res_attention=True
        )
        
        # Многозадачные выходные слои для v4.0 (20 выходов)
        # Группируем выходы по типам для лучшего обучения
        
        # 1. Future returns (регрессия) - 4 выхода [0-3]
        self.future_returns_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 4),
            nn.Tanh()  # Ограничиваем выходы для стабильности
        )
        
        # 2. Направления движения - УСИЛЕННАЯ АРХИТЕКТУРА ПРОТИВ КОЛЛАПСА
        # Отдельные головы для каждого таймфрейма для независимого обучения
        self.direction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),  # Больше capacity
                nn.LayerNorm(self.d_model),  # Нормализация для стабильности
                nn.GELU(),  # GELU для лучших градиентов
                nn.Dropout(self.dropout * 0.5),  # Меньше dropout
                nn.Linear(self.d_model, self.d_model // 2),
                nn.LayerNorm(self.d_model // 2),
                nn.GELU(),
                nn.Dropout(self.dropout * 0.5),
                nn.Linear(self.d_model // 2, 3)  # 3 класса: LONG, SHORT, FLAT
            ) for _ in range(4)  # 15m, 1h, 4h, 12h
        ])

        # Резервная голова для совместимости
        self.direction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 4 * 3)
        )
        
        # 3. Достижение уровней LONG (бинарная классификация) - 4 выхода [8-11]
        self.long_levels_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 4)
            # Логиты, sigmoid применяется в loss
        )
        
        # 4. Достижение уровней SHORT (бинарная классификация) - 4 выхода [12-15]
        self.short_levels_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 4)
        )
        
        # 5. Риск-метрики (регрессия) - 4 выхода
        self.risk_metrics_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 4)
        )
        
        # Финальный слой для объединения всех предсказаний
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        
        # Layer normalization
        self.ln = nn.LayerNorm(self.d_model)
        
        # Temperature scaling для калибровки уверенности
        if model_config.get('temperature_scaling', False):
            # Инициализируем температуру из конфига (по умолчанию 2.0)
            # Большая температура = менее уверенные предсказания = меньше FLAT
            temp_value = model_config.get('temperature', 2.0)
            self.temperature = nn.Parameter(torch.ones(1) * temp_value)
        else:
            self.temperature = None
            
        # Confidence head для предсказания собственной уверенности
        self.confidence_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 4),  # 4 значения уверенности
            nn.Tanh()  # Ограничиваем выход в диапазоне [-1, 1] для стабильности
        )
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов модели с учетом конфигурации"""
        # Проверяем конфигурацию для direction head init
        direction_init_config = self.config.get('model', {}).get('direction_head_init', {})
        init_method = direction_init_config.get('method', 'balanced')
        bias_init = direction_init_config.get('bias_init', 'balanced')
        weight_scale = direction_init_config.get('weight_scale', 0.1)
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Специальная инициализация для direction head
                if 'direction_head' in name:
                    if module.out_features == 12:  # Финальный слой direction head
                        # Инициализация весов с малой дисперсией для стабильности
                        nn.init.xavier_uniform_(module.weight, gain=weight_scale)
                        
                        if module.bias is not None and bias_init in ['balanced', 'neutral']:
                            # Инициализация bias для direction head
                            with torch.no_grad():
                                bias = module.bias.view(4, 3)  # 4 таймфрейма × 3 класса
                                
                                if bias_init == 'neutral':
                                    # Нейтральная инициализация - равные шансы для всех классов
                                    bias.fill_(0.0)
                                    self.logger.info("🔧 Используется нейтральная инициализация bias (все = 0.0)")
                                    
                                elif bias_init == 'balanced':
                                    # Используем веса классов для смещения
                                    class_weights = self.config.get('loss', {}).get('class_weights', [1.0, 1.0, 1.0])
                                    # Обратная пропорция весов для смещения
                                    bias[:, 0] = np.log(1.0 / class_weights[0])  # LONG
                                    bias[:, 1] = np.log(1.0 / class_weights[1])  # SHORT  
                                    bias[:, 2] = np.log(1.0 / class_weights[2])  # FLAT
                                    self.logger.info(f"🎯 Смещение bias на основе весов: {bias[0].tolist()}")
                                    
                        elif module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                            self.logger.info("✅ Direction head bias инициализирован")
                    else:
                        # Промежуточные слои direction head
                        nn.init.xavier_uniform_(module.weight, gain=0.8)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                else:
                    # Стандартная инициализация для остальных слоев
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                        
            elif isinstance(module, nn.Conv1d):
                # Kaiming инициализация с уменьшенным масштабом
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                # Дополнительное масштабирование для стабильности
                with torch.no_grad():
                    module.weight.mul_(0.7)
                
    def forward(self, x: torch.Tensor, return_dict: bool = False) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, seq_len, n_features)
            return_dict: игнорируется, для совместимости с trainer

        Returns:
            output: (batch_size, n_outputs) - все целевые переменные
        """
        batch_size = x.shape[0]
        
        # Нормализация
        x = self.revin(x, 'norm')
        
        # Перестановка для Conv1d: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        
        # Создание патчей
        x = self.patch_embedding(x)  # (B, d_model, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, d_model)
        
        # Позиционное кодирование
        x = self.positional_encoding(x)
        
        # Трансформер энкодер
        x = self.encoder(x)  # (B, n_patches, d_model)
        
        # Глобальное представление (среднее по патчам)
        x_global = x.mean(dim=1)  # (B, d_model)
        
        # Нормализация
        x_global = self.ln(x_global)
        
        # Проекция для лучшего представления
        x_projected = self.output_projection(x_global)
        
        # Многозадачные предсказания v4.0 (20 выходов)
        future_returns = self.future_returns_head(x_projected)  # (B, 4)
        
        # Direction head - используем новые отдельные головы если они есть
        if hasattr(self, 'direction_heads') and len(self.direction_heads) > 0:
            # Используем отдельные головы для каждого таймфрейма
            direction_outputs = []
            for i, head in enumerate(self.direction_heads):
                logits = head(x_projected)  # (B, 3)
                direction_outputs.append(logits)
            # Объединяем в один тензор
            direction_logits = torch.cat(direction_outputs, dim=1)  # (B, 12)
        else:
            # Fallback на старую голову
            direction_logits = self.direction_head(x_projected)  # (B, 12) = 4 таймфрейма * 3 класса
        
        # ОТЛАДКА: Проверяем выходы direction head
        if not hasattr(self, '_logged_direction_head') and self.training:
            self._logged_direction_head = True
            with torch.no_grad():
                logits_stats = {
                    'min': direction_logits.min().item(),
                    'max': direction_logits.max().item(), 
                    'mean': direction_logits.mean().item(),
                    'std': direction_logits.std().item()
                }
                print(f"🔍 Direction head логиты статистика: {logits_stats}")
                # Проверяем веса последнего слоя
                last_layer = list(self.direction_head.children())[-1]
                if hasattr(last_layer, 'weight'):
                    weight_stats = {
                        'min': last_layer.weight.min().item(),
                        'max': last_layer.weight.max().item(),
                        'mean': last_layer.weight.mean().item(),
                        'std': last_layer.weight.std().item()
                    }
                    bias_stats = {
                        'min': last_layer.bias.min().item(),
                        'max': last_layer.bias.max().item(),
                        'mean': last_layer.bias.mean().item()
                    } if last_layer.bias is not None else None
                    print(f"🔍 Direction head веса: {weight_stats}")
                    print(f"🔍 Direction head bias: {bias_stats}")
        
        # Преобразуем логиты в классы для совместимости со старым форматом
        direction_logits_reshaped = direction_logits.view(batch_size, 4, 3)  # (B, 4, 3)

        # КРИТИЧЕСКИ ВАЖНО: Применяем direction_bias если есть в конфиге
        direction_bias = self.config.get('model', {}).get('direction_bias', None)
        if direction_bias is not None and len(direction_bias) == 3:
            bias_tensor = torch.tensor(direction_bias, device=direction_logits_reshaped.device, dtype=direction_logits_reshaped.dtype)
            direction_logits_reshaped = direction_logits_reshaped + bias_tensor.view(1, 1, 3)  # Broadcast across batch and timeframes

            # Логирование bias (только один раз)
            if not hasattr(self, '_bias_logged'):
                self._bias_logged = True
                self.logger.info(f"🎯 Direction bias применён: {direction_bias}")

        # Применяем temperature scaling если включено
        if self.temperature is not None:
            # Temperature scaling делает предсказания более сбалансированными (Т > 1)
            direction_logits_reshaped = direction_logits_reshaped / self.temperature
            
            # Логирование температуры (только один раз)
            if not hasattr(self, '_temp_logged'):
                self._temp_logged = True
                self.logger.info(f"🌡️ Temperature scaling активирован: T={self.temperature.item():.2f}")
        
        # Применяем softmax для получения вероятностей
        direction_probs = torch.softmax(direction_logits_reshaped, dim=-1)  # (B, 4, 3)
        
        # Получаем предсказанные классы
        directions = torch.argmax(direction_probs, dim=-1).float()  # (B, 4)
        
        # ВАЖНО: НЕ применяем порог уверенности при обучении!
        # Это критическая ошибка, которая убивает градиенты и не дает модели учиться
        # Порог уверенности должен применяться ТОЛЬКО при инференсе/торговле

        # Получаем максимальные вероятности для каждого предсказания
        # Удалена вся логика confidence_threshold - она вызывала коллапс модели
        # Модель теперь предсказывает классы напрямую без фильтрации по уверенности
        
        # Предсказываем уверенность для каждого таймфрейма
        confidence_scores = self.confidence_head(x_projected)  # (B, 4)
        
        long_levels = self.long_levels_head(x_projected)  # (B, 4)
        short_levels = self.short_levels_head(x_projected)  # (B, 4)
        risk_metrics = self.risk_metrics_head(x_projected)  # (B, 4)
        
        # Объединяем все выходы в один тензор (20 выходов)
        outputs = torch.cat([
            future_returns,    # 0-3: future_return_15m, 1h, 4h, 12h (регрессия)
            directions,        # 4-7: direction_15m, 1h, 4h, 12h (классы 0,1,2)
            long_levels,       # 8-11: long_will_reach_1pct_4h, 2pct_4h, 3pct_12h, 5pct_12h (вероятности)
            short_levels,      # 12-15: short_will_reach_1pct_4h, 2pct_4h, 3pct_12h, 5pct_12h (вероятности)
            risk_metrics       # 16-19: max_drawdown_1h, max_rally_1h, max_drawdown_4h, max_rally_4h (регрессия)
        ], dim=1)
        
        # КРИТИЧНО: Клиппинг выходов для предотвращения взрыва градиентов
        # Ограничиваем выходы в разумных пределах перед возвратом
        outputs = torch.clamp(outputs, min=-10.0, max=10.0)
        
        # ВАЖНО: clamp создает новый тензор, поэтому устанавливаем атрибуты ПОСЛЕ clamp
        # Для обучения сохраняем логиты direction для правильной loss функции
        outputs._direction_logits = direction_logits_reshaped  # (B, 4, 3)
        outputs._confidence_scores = confidence_scores  # (B, 4) - уверенность для каждого таймфрейма
        
        # v4.0: Комментарии по активациям для 20 выходов
        # Future returns (0-3) - без активации (регрессия)
        # Directions (4-7) - без активации (категориальные как числа)
        # Long levels (8-11) - логиты (sigmoid в BCEWithLogitsLoss)
        # Short levels (12-15) - логиты (sigmoid в BCEWithLogitsLoss) 
        # Risk metrics (16-19) - без активации (регрессия)
        
        # Возвращаем сырые логиты для всех выходов
        
        return outputs
    
    def get_direction_l2_loss(self) -> torch.Tensor:
        """Вычисляет L2 регуляризацию для direction head"""
        l2_loss = 0.0
        
        # Добавляем L2 регуляризацию только для direction head
        for name, param in self.direction_head.named_parameters():
            if 'weight' in name:
                l2_loss += torch.norm(param, 2) ** 2
                
        return l2_loss * self.config.get('model', {}).get('direction_l2_weight', 0.001)
    
    def get_output_names(self) -> List[str]:
        """Возвращает имена всех выходов в правильном порядке (20 переменных v4.0)"""
        return [
            # A. Базовые возвраты (0-3)
            'future_return_15m', 'future_return_1h', 'future_return_4h', 'future_return_12h',
            # B. Направление движения (4-7)
            'direction_15m', 'direction_1h', 'direction_4h', 'direction_12h',
            # C. Достижение уровней прибыли LONG (8-11)
            'long_will_reach_1pct_4h', 'long_will_reach_2pct_4h', 
            'long_will_reach_3pct_12h', 'long_will_reach_5pct_12h',
            # D. Достижение уровней прибыли SHORT (12-15)
            'short_will_reach_1pct_4h', 'short_will_reach_2pct_4h',
            'short_will_reach_3pct_12h', 'short_will_reach_5pct_12h',
            # E. Риск-метрики (16-19)
            'max_drawdown_1h', 'max_rally_1h', 'max_drawdown_4h', 'max_rally_4h'
        ]


# НЕИСПОЛЬЗУЕМЫЙ КЛАСС UnifiedTradingLoss ЗАКОММЕНТИРОВАН
# class UnifiedTradingLoss(nn.Module):
#     """
#     Унифицированная loss функция для 20 выходов v4.0
#     Комбинирует regression и classification losses с weighted focus на крупные движения
#     """
#     
#     def __init__(self, config: Dict):
#         super().__init__()
#         self.config = config
#         loss_config = config.get('loss', {})
#         
        # Веса для разных компонентов
#         self.future_return_weight = loss_config.get('future_return_weight', 1.0)
#         self.tp_weight = loss_config.get('tp_weight', 0.8)
#         self.sl_weight = loss_config.get('sl_weight', 1.2)
#         self.signal_weight = loss_config.get('signal_weight', 0.6)
#         
        # Веса для крупных движений
#         self.large_move_threshold = loss_config.get('large_move_threshold', 0.02)  # 2%
#         self.large_move_weight = loss_config.get('large_move_weight', 5.0)
#         
        # Штраф за неправильное направление
#         self.wrong_direction_penalty = loss_config.get('wrong_direction_penalty', 2.0)
#         
        # Loss функции (исправлено для mixed precision)
#         self.mse_loss = nn.MSELoss(reduction='none')
#         self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')  # Безопасно для autocast
#         
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
#                 price_changes: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Вычисление loss для v4.0 с weighted focus на крупные движения
#         
#         Args:
#             predictions: (batch_size, 20)
#             targets: (batch_size, 20)
#             price_changes: (batch_size, 4) - опциональные изменения цен для взвешивания
#             
#         Returns:
#             loss: скаляр
#         """
#         assert predictions.shape == targets.shape, \
#             f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
#         
#         batch_size = predictions.shape[0]
#         losses = []
#         
        # ИСПРАВЛЕНИЕ для v4.0: Обработка 20 переменных
        # Нормализация целевых переменных где необходимо
#         normalized_targets = targets.clone()
#         
        # 1. Future returns (индексы 0-3) - регрессия с weighted loss
        # Нормализация из процентов в доли
#         normalized_targets[:, :4] = targets[:, :4] / 100.0
#         
        # Базовый MSE loss
#         future_return_loss = self.mse_loss(
#             predictions[:, :4], 
#             normalized_targets[:, :4]
#         )
#         
        # Взвешивание для крупных движений
#         if price_changes is not None:
            # price_changes уже в долях
#             large_move_mask = torch.abs(price_changes) > self.large_move_threshold
#             
            # Применяем больший вес к крупным движениям
#             weights = torch.ones_like(future_return_loss)
#             weights[large_move_mask] = self.large_move_weight
#             
            # Взвешенный loss
#             future_return_loss = (future_return_loss * weights).mean()
#         else:
            # Альтернативный подход - использовать сами целевые значения для взвешивания
#             weights = 1.0 + torch.abs(normalized_targets[:, :4]) * 10.0
#             future_return_loss = (future_return_loss * weights).mean()
#         
#         losses.append(future_return_loss * self.future_return_weight)
#         
        # 2. Направления движения (индексы 4-7) - классификация с штрафом за неправильное направление
        # Получаем предсказанные и истинные направления
#         pred_directions = predictions[:, 4:8]
#         true_directions = targets[:, 4:8]
#         
        # Специальный loss для направлений
#         direction_losses = []
#         
#         for i in range(4):  # Для каждого таймфрейма
            # Базовый MSE
#             base_loss = self.mse_loss(
#                 pred_directions[:, i],
#                 true_directions[:, i] / 2.0
#             )
#             
            # Дополнительный штраф за противоположное направление
            # Если истинное = 0 (UP), а предсказано = 1 (DOWN) или наоборот
#             pred_class = torch.round(pred_directions[:, i] * 2).clamp(0, 2)
#             true_class = true_directions[:, i]
#             
#             wrong_direction_mask = (
#                 ((true_class == 0) & (pred_class == 1)) |  # True UP, Pred DOWN
#                 ((true_class == 1) & (pred_class == 0))    # True DOWN, Pred UP
#             )
#             
            # Применяем штраф
#             penalty_weights = torch.ones_like(base_loss)
#             penalty_weights[wrong_direction_mask] = self.wrong_direction_penalty
#             
#             weighted_loss = (base_loss * penalty_weights).mean()
#             direction_losses.append(weighted_loss)
#         
#         direction_loss = sum(direction_losses) / len(direction_losses)
#         losses.append(direction_loss * self.signal_weight)
#         
        # 3. Достижение уровней LONG (индексы 8-11) - бинарная классификация
#         long_levels_loss = self.bce_with_logits_loss(
#             predictions[:, 8:12],
#             targets[:, 8:12]
#         )
#         
        # Взвешивание для положительных примеров (когда цель достигнута)
#         positive_weight = 2.0  # Больше веса на правильное предсказание достижения уровня
#         weights = torch.ones_like(long_levels_loss)
#         weights[targets[:, 8:12] == 1] = positive_weight
#         
#         long_levels_loss = (long_levels_loss * weights).mean()
#         losses.append(long_levels_loss * self.tp_weight)
#         
        # 4. Достижение уровней SHORT (индексы 12-15) - бинарная классификация
#         short_levels_loss = self.bce_with_logits_loss(
#             predictions[:, 12:16],
#             targets[:, 12:16]
#         )
#         
        # Аналогичное взвешивание
#         weights = torch.ones_like(short_levels_loss)
#         weights[targets[:, 12:16] == 1] = positive_weight
#         
#         short_levels_loss = (short_levels_loss * weights).mean()
#         losses.append(short_levels_loss * self.tp_weight)
#         
        # 5. Риск-метрики (индексы 16-19) - регрессия с фокусом на большие drawdowns
#         risk_metrics_loss = self.mse_loss(
#             predictions[:, 16:20],
#             targets[:, 16:20]  # Уже нормализованы в feature_engineering
#         )
#         
        # Больше веса на правильное предсказание больших drawdowns
#         risk_weights = 1.0 + torch.abs(targets[:, 16:20]) * 5.0
#         risk_metrics_loss = (risk_metrics_loss * risk_weights).mean()
#         
#         losses.append(risk_metrics_loss * self.signal_weight)
#         
        # Общий loss с адаптивным взвешиванием
        # Можно добавить динамическое изменение весов в процессе обучения
#         total_loss = sum(losses) / len(losses)
#         
        # L2 регуляризация для предотвращения переобучения
#         l2_lambda = self.config.get('model', {}).get('l2_regularization', 0.001)
#         if l2_lambda > 0 and hasattr(self, 'model') and self.model is not None:
#             l2_reg = 0.0
#             for param in self.model.parameters():
#                 if param.requires_grad:
#                     l2_reg += torch.norm(param, 2) ** 2
#             total_loss += l2_lambda * l2_reg
#         
#         return total_loss
# 

# НЕИСПОЛЬЗУЕМЫЙ КЛАСС DirectionalTradingLoss ЗАКОММЕНТИРОВАН
# class DirectionalTradingLoss(nn.Module):
#     """
#     Специализированная loss функция для максимизации прибыльности
#     Учитывает потенциальный P&L от торговых решений
#     """
    
#     def __init__(self, 
#                  commission: float = 0.001,
#                  class_weights: Optional[List[float]] = None,
#                  profit_focus_weight: float = 10.0):
#         super().__init__()
#         self.commission = commission
#         self.profit_focus_weight = profit_focus_weight
        
        # Веса классов (можно настроить для баланса)
#         if class_weights is None:
#             self.class_weights = torch.tensor([1.0, 1.0, 0.5])  # Меньше вес для FLAT
#         else:
#             self.class_weights = torch.tensor(class_weights)
            
#     def forward(self, 
#                 predictions: Dict[str, torch.Tensor], 
#                 targets: Dict[str, torch.Tensor],
#                 price_changes: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#         Вычисление loss с учетом потенциальной прибыли
        
#         Args:
#             predictions: Dict с логитами для каждого таймфрейма
#             targets: Dict с истинными метками (0=UP, 1=DOWN, 2=FLAT)
#             price_changes: Dict с процентными изменениями цены
            
#         Returns:
#             Общий loss
#         """
#         total_loss = 0.0
#         timeframe_weights = {'15m': 0.2, '1h': 0.3, '4h': 0.35, '12h': 0.15}
        
#         for timeframe in ['15m', '1h', '4h', '12h']:
#             key = f'direction_{timeframe}'
            
            # Base cross entropy с весами классов
#             ce_loss = F.cross_entropy(
#                 predictions[key], 
#                 targets[key],
#                 weight=self.class_weights.to(predictions[key].device),
#                 reduction='none'
#             )
            
            # Рассчитываем потенциальный P&L
#             predicted_direction = predictions[key].argmax(dim=1)
#             actual_direction = targets[key]
#             price_change = price_changes[timeframe]
            
            # Убеждаемся что price_change имеет правильную размерность
#             if price_change.dim() > 1:
#                 price_change = price_change.squeeze(-1)
            
            # Потенциальная прибыль/убыток от каждого решения
#             potential_pnl = torch.zeros_like(ce_loss)
            
            # LONG позиции (predicted=0)
#             long_mask = predicted_direction == 0
#             potential_pnl[long_mask] = price_change[long_mask] - self.commission
            
            # SHORT позиции (predicted=1)
#             short_mask = predicted_direction == 1  
#             potential_pnl[short_mask] = -price_change[short_mask] - self.commission
            
            # Веса на основе величины движения цены
            # Больше штраф за ошибки на крупных движениях
#             movement_weight = 1 + torch.abs(price_change) * self.profit_focus_weight
            
            # Дополнительный штраф за неправильное направление
#             wrong_direction_mask = (
#                 ((predicted_direction == 0) & (actual_direction == 1)) |  # Предсказали UP, было DOWN
#                 ((predicted_direction == 1) & (actual_direction == 0))    # Предсказали DOWN, было UP
#             )
#             direction_penalty = wrong_direction_mask.float() * torch.abs(price_change) * 2
            
            # Штраф за false positives (торговля когда нужно было ждать)
#             false_positive_penalty = ((predicted_direction != 2) & (actual_direction == 2)).float() * 0.5
            
            # Комбинированный loss для таймфрейма
#             timeframe_loss = (ce_loss * movement_weight + direction_penalty + false_positive_penalty).mean()
            
            # Взвешенный вклад в общий loss
#             total_loss += timeframe_loss * timeframe_weights[timeframe]
            
#         return total_loss


class DirectionalMultiTaskLoss(nn.Module):
    """
    Гибридная loss функция для multi-task learning:
    - MSE для regression переменных (returns, risk metrics)
    - CrossEntropy для classification переменных (directions)
    - BCE для binary classification (levels)
    
    Основана на научных исследованиях 2024 года по crypto direction prediction
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Добавляем логгер для отладки
        from utils.logger import get_logger
        self.logger = get_logger('DirectionalMultiTaskLoss')
        
        # Веса для разных типов задач из конфигурации
        task_weights = config.get('loss', {}).get('task_weights', {})
        self.future_returns_weight = task_weights.get('future_returns', 1.0)
        self.directions_weight = task_weights.get('directions', 3.0)  # Больший вес для направлений
        self.long_levels_weight = task_weights.get('long_levels', 1.0)
        self.short_levels_weight = task_weights.get('short_levels', 1.0)
        self.risk_metrics_weight = task_weights.get('risk_metrics', 0.5)
        
        # Параметры для direction focus
        self.large_move_weight = config.get('loss', {}).get('large_move_weight', 5.0)
        self.min_movement_threshold = config.get('loss', {}).get('large_move_threshold', 0.003)
        
        # Loss функции
        self.mse_loss = nn.MSELoss(reduction='none')
        
        # Веса классов для борьбы с дисбалансом
        # Основано на анализе реальных данных: LONG≈19%, SHORT≈18%, FLAT≈63%
        # Рекомендуемые веса из analyze_class_distribution: [1.16, 1.20, 0.64]
        
        # Используем веса из конфига или сбалансированные по умолчанию
        config_weights = config.get('loss', {}).get('class_weights', [1.3, 1.3, 0.7])
        class_weights = torch.tensor(config_weights)  # LONG, SHORT, FLAT
        
        # КРИТИЧНО: Логируем фактические веса классов
        print(f"🔥 DirectionalMultiTaskLoss инициализирована с весами классов: {config_weights}")
        print(f"   - LONG weight: {config_weights[0]}")
        print(f"   - SHORT weight: {config_weights[1]}")
        print(f"   - FLAT weight: {config_weights[2]}")
        
        # Метод 2: Динамическая адаптация весов на основе батча
        # Это позволит модели адаптироваться к локальным распределениям
        self.use_dynamic_weights = config.get('loss', {}).get('use_dynamic_class_weights', True)
        self.class_weight_momentum = 0.9  # Экспоненциальное скользящее среднее
        
        # Инициализируем начальные веса
        self.register_buffer('class_weights', class_weights)
        self.register_buffer('running_class_counts', torch.zeros(3))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
        # CrossEntropy с начальными весами
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=self.class_weights.cuda() if torch.cuda.is_available() else self.class_weights,
            reduction='none'
        )
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Focal Loss параметры для несбалансированных классов direction
        self.focal_alpha = config.get('loss', {}).get('focal_alpha', 0.25)
        self.focal_gamma = config.get('loss', {}).get('focal_gamma', 2.0)
        
        # Сохраняем веса классов для использования в focal loss
        self.class_weights = class_weights
        
        # Счетчик эпох для динамического веса direction loss
        self.current_epoch = 0
        self.warmup_epochs = 10  # Постепенное увеличение веса direction loss
        
        # Label smoothing параметр  
        self.label_smoothing = config.get('model', {}).get('label_smoothing', 0.2)
        
        # Штраф за неправильное направление
        self.wrong_direction_penalty = config.get('loss', {}).get('wrong_direction_penalty', 3.0)
        
        # Активные losses для поэтапного обучения
        self.active_losses = ["all"]  # По умолчанию все losses активны
        
        # КРИТИЧНО: Параметры для предотвращения схлопывания модели
        self.auto_adjust_on_collapse = config.get('loss', {}).get('auto_adjust_on_collapse', False)
        self.collapse_threshold = config.get('loss', {}).get('collapse_threshold', 0.75)
        self.min_entropy = config.get('loss', {}).get('min_entropy', 0.6)
        self.entropy_min_weight = config.get('loss', {}).get('entropy_min_weight', 0.2)
        self.min_entropy_threshold = config.get('model', {}).get('min_entropy_threshold', 0.5)
        
        print(f"⚡ Параметры предотвращения схлопывания:")
        print(f"   - auto_adjust_on_collapse: {self.auto_adjust_on_collapse}")
        print(f"   - collapse_threshold: {self.collapse_threshold}")  
        print(f"   - min_entropy: {self.min_entropy}")
        print(f"   - entropy_min_weight: {self.entropy_min_weight}")
        print(f"   - min_entropy_threshold: {self.min_entropy_threshold}")
        
    def set_active_losses(self, active_losses: List[str]):
        """
        Устанавливает активные loss компоненты для поэтапного обучения
        
        Args:
            active_losses: список активных losses, например:
                ["directions"] - только direction loss
                ["directions", "future_returns"] - direction + returns
                ["all"] - все losses
        """
        self.active_losses = active_losses
        print(f"🎯 Активные losses установлены: {active_losses}")
        
    def set_epoch(self, epoch: int):
        """Установка текущей эпохи для динамических весов"""
        self.current_epoch = epoch
    
    def update_class_weights(self, targets: torch.Tensor):
        """
        Динамическое обновление весов классов на основе текущего батча
        
        Args:
            targets: направления классов из текущего батча (batch_size, 4)
        """
        if not self.use_dynamic_weights:
            return
        
        # Извлекаем direction targets
        if isinstance(targets, dict):
            direction_targets = targets.get('directions', targets.get('direction_15m', None))
            if direction_targets is None:
                return  # Нет direction targets для обновления весов
            device = direction_targets.device
        else:
            direction_targets = targets[:, 4:8]  # direction columns
            device = targets.device

        # Если direction_targets 1D, расширяем до 4 таймфреймов
        if direction_targets.dim() == 1:
            direction_targets = direction_targets.unsqueeze(1).expand(-1, 4)

        # Подсчитываем классы в текущем батче
        batch_counts = torch.zeros(3, device=device)
        for i in range(direction_targets.shape[1]):  # По всем таймфреймам
            for c in range(3):  # По всем классам
                batch_counts[c] += (direction_targets[:, i] == c).sum().float()
        
        # Обновляем скользящее среднее (перемещаем на правильное устройство)
        self.running_class_counts = self.running_class_counts.to(device)
        self.total_samples = self.total_samples.to(device)
        
        self.running_class_counts = (self.class_weight_momentum * self.running_class_counts + 
                                     (1 - self.class_weight_momentum) * batch_counts)
        self.total_samples = self.total_samples + direction_targets.numel()
        
        # Вычисляем текущие частоты
        if self.total_samples > 100:  # Начинаем адаптацию после 100 примеров
            current_frequencies = self.running_class_counts / self.running_class_counts.sum()
            current_frequencies = current_frequencies.clamp(min=0.01)  # Избегаем деления на 0
            
            # Вычисляем новые веса: inverse frequency с квадратным корнем
            new_weights = torch.sqrt(1.0 / current_frequencies)
            new_weights = new_weights / new_weights.mean()  # Нормализация
            
            # Плавное обновление весов
            weight_update_rate = 0.1  # Скорость адаптации
            self.class_weights = self.class_weights.to(targets.device)
            self.class_weights = (1 - weight_update_rate) * self.class_weights + weight_update_rate * new_weights
            
            # НЕ обновляем cross_entropy_loss здесь - будем использовать веса напрямую в focal_loss
        
    def get_dynamic_direction_weight(self) -> float:
        """Получение динамического веса для direction loss с warmup"""
        base_weight = self.directions_weight
        
        if self.current_epoch < self.warmup_epochs:
            # Постепенное увеличение от 1.0 до base_weight
            warmup_factor = self.current_epoch / self.warmup_epochs
            return 1.0 + (base_weight - 1.0) * warmup_factor
        else:
            return base_weight
        
    def apply_label_smoothing(self, targets: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
        """
        Применяет label smoothing к целевым меткам
        
        Args:
            targets: (batch_size,) - целевые метки классов
            num_classes: количество классов
            
        Returns:
            smoothed_targets: (batch_size, num_classes) - сглаженные метки
        """
        if self.label_smoothing == 0:
            # Если label smoothing выключен, возвращаем one-hot
            return F.one_hot(targets, num_classes).float()
        
        # Создаем сглаженное распределение
        confidence = 1.0 - self.label_smoothing
        smoothed_targets = torch.full(
            (targets.size(0), num_classes), 
            self.label_smoothing / (num_classes - 1),
            device=targets.device
        )
        
        # Устанавливаем основную вероятность для истинного класса
        smoothed_targets.scatter_(1, targets.unsqueeze(1), confidence)
        
        return smoothed_targets
    
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, dynamic_weights: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss для борьбы с несбалансированными классами direction
        С учетом динамических весов классов и label smoothing
        """
        # Применяем label smoothing
        if self.label_smoothing > 0:
            smoothed_targets = self.apply_label_smoothing(targets, num_classes=3)
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(smoothed_targets * log_probs).sum(dim=-1)
            
            # Применяем динамические веса
            target_weights = dynamic_weights[targets]
            ce_loss = ce_loss * target_weights
        else:
            # Обычный weighted cross entropy с динамическими весами
            ce_loss = F.cross_entropy(logits, targets, weight=dynamic_weights, reduction='none')
        
        # Focal loss модификация
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss
        
    def forward(self, outputs: torch.Tensor, targets: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Вычисление multi-task loss с confidence-aware механизмом

        Args:
            outputs: (batch_size, 20) - выходы модели
            targets: (batch_size, 20) - целевые значения ИЛИ словарь с целевыми значениями

        Returns:
            loss: скаляр
        """
        losses = []

        # Проверяем активные losses
        use_all_losses = "all" in self.active_losses

        # Извлекаем confidence scores если доступны
        confidence_scores = None
        if hasattr(outputs, '_confidence_scores'):
            confidence_scores = outputs._confidence_scores  # (batch_size, 4)

        # Обработка targets как словаря или тензора
        if isinstance(targets, dict):
            # Если targets - словарь, извлекаем значения
            future_returns_target = targets.get('future_returns', targets.get('future_return_15m', None))
            if future_returns_target is not None and future_returns_target.dim() == 1:
                # Если только один целевой возврат, используем его для всех горизонтов
                future_returns_target = future_returns_target.unsqueeze(1).expand(-1, 4)
            direction_target = targets.get('directions', targets.get('direction_15m', None))
            long_levels_target = targets.get('long_levels', None)
            short_levels_target = targets.get('short_levels', None)
            risk_metrics_target = targets.get('risk_metrics', None)
        else:
            # Если targets - тензор, используем индексы
            future_returns_target = targets[:, 0:4] / 100.0  # Конвертируем из % в доли
            direction_target = targets[:, 4:8]
            long_levels_target = targets[:, 8:12]
            short_levels_target = targets[:, 12:16]
            risk_metrics_target = targets[:, 16:20]

        # 1. Future Returns Loss (индексы 0-3) - MSE для регрессии
        if use_all_losses or "future_returns" in self.active_losses:
            if future_returns_target is not None:
                future_returns_pred = outputs[:, 0:4]
                # target уже нормализован выше, дублирование убрано
            
            # Взвешивание для крупных движений
            abs_returns = torch.abs(future_returns_target)
            large_move_mask = abs_returns > self.min_movement_threshold
            
            mse_loss = self.mse_loss(future_returns_pred, future_returns_target)
            
            # Применяем больший вес к крупным движениям
            movement_weights = torch.ones_like(mse_loss)
            movement_weights[large_move_mask] = self.large_move_weight
            
            future_returns_loss = (mse_loss * movement_weights).mean()
            losses.append(future_returns_loss * self.future_returns_weight)
        
        # 2. Direction Loss (индексы 4-7) - CrossEntropy для 3-классовой классификации
        if use_all_losses or "directions" in self.active_losses:
            if hasattr(outputs, '_direction_logits'):
                direction_logits = outputs._direction_logits  # (batch_size, 4, 3)
                # Используем уже извлеченные direction_target
                if direction_target is not None:
                    direction_targets = direction_target.long() if not isinstance(targets, dict) else direction_target

                    # Проверка и исправление размерности
                    if direction_targets.dim() == 1:
                        # Если 1D (только один таймфрейм), расширяем до всех 4 таймфреймов
                        batch_size = direction_targets.shape[0]
                        direction_targets = direction_targets.unsqueeze(1).expand(-1, 4)  # (batch_size, 4)

                    # Получаем текущие веса (могут быть изменены динамически)
                    current_class_weights = self.class_weights.to(direction_logits.device)

                    # Энтропийная регуляризация и экстренная коррекция
                    entropy_weight = self.config.get('model', {}).get('entropy_weight', 0.1)
                    entropy_loss = 0.0
                    if entropy_weight > 0:
                        probs = torch.softmax(direction_logits, dim=-1)
                        log_probs = torch.log(probs + 1e-8)
                        entropy = -torch.sum(probs * log_probs, dim=-1)
                        max_entropy = np.log(3)
                        normalized_entropy = entropy / max_entropy
                        mean_entropy = normalized_entropy.mean().item()

                        if self.auto_adjust_on_collapse and mean_entropy < self.collapse_threshold:
                            pred_classes = torch.argmax(probs, dim=-1)
                            flat_ratio = (pred_classes == 2).float().mean().item()
                            if flat_ratio > 0.85:
                                print(f"🚨 ОБНАРУЖЕНО СХЛОПЫВАНИЕ (flat_ratio={flat_ratio:.2f})! Экстренная корректировка весов.")
                                emergency_factor = 1.5 + (flat_ratio - 0.85) * 10
                                current_class_weights[0] *= emergency_factor
                                current_class_weights[1] *= emergency_factor
                                current_class_weights[2] /= emergency_factor
                                current_class_weights = current_class_weights / current_class_weights.mean()

                        entropy_penalty = (self.min_entropy_threshold - normalized_entropy).clamp(min=0)
                        entropy_loss = entropy_penalty.mean() * entropy_weight

                    direction_loss = 0
                    for i in range(4):  # Для каждого таймфрейма
                        focal_loss_val = self.focal_loss(direction_logits[:, i, :], direction_targets[:, i], current_class_weights)

                        pred_classes = torch.argmax(direction_logits[:, i, :], dim=-1)
                        true_classes = direction_targets[:, i]

                        # ОТЛАДКА: Логирование логитов для первого батча первой эпохи
                        if not hasattr(self, '_debug_logged') and i == 0:
                            self._debug_logged = True
                            logits_sample = direction_logits[0, i, :].detach().cpu().numpy()
                            probs_sample = torch.softmax(direction_logits[0, i, :], dim=-1).detach().cpu().numpy()
                            self.logger.info(f"🔍 DEBUG - Пример логитов и вероятностей для первого сэмпла:")
                            self.logger.info(f"   Логиты: LONG={logits_sample[0]:.3f}, SHORT={logits_sample[1]:.3f}, FLAT={logits_sample[2]:.3f}")
                            self.logger.info(f"   Вероятности: LONG={probs_sample[0]:.3f}, SHORT={probs_sample[1]:.3f}, FLAT={probs_sample[2]:.3f}")
                            self.logger.info(f"   Предсказанный класс: {pred_classes[0].item()}, Истинный класс: {true_classes[0].item()}")
                            self.logger.info(f"   Веса классов: {current_class_weights.cpu().numpy()}")

                        wrong_direction_penalty = (
                            ((pred_classes == 0) & (true_classes == 1)) |
                            ((pred_classes == 1) & (true_classes == 0))
                        ).float() * self.wrong_direction_penalty

                        timeframe_loss = focal_loss_val + wrong_direction_penalty

                        if confidence_scores is not None:
                            confidence_weight = 2.0 - confidence_scores[:, i]
                            timeframe_loss = timeframe_loss * confidence_weight

                        direction_loss += timeframe_loss.mean()

                    direction_loss = direction_loss / 4 + entropy_loss

                    dynamic_weight = self.get_dynamic_direction_weight()
                    losses.append(direction_loss * dynamic_weight)
            else:
                # КРИТИЧНО: Если нет логитов, это ошибка архитектуры!
                # Не должны использовать MSE для классификации направлений
                raise RuntimeError(
                    "🚨 КРИТИЧЕСКАЯ ОШИБКА: outputs не содержит _direction_logits! "
                    "Это означает, что модель не генерирует логиты для классификации направлений. "
                    "Проверьте, что UnifiedPatchTSTForTrading правильно устанавливает outputs._direction_logits"
                )
        
        # 3. Long Levels Loss (индексы 8-11) - BCE для бинарной классификации
        if use_all_losses or "long_levels" in self.active_losses:
            if long_levels_target is not None:
                long_levels_pred = outputs[:, 8:12]
                long_levels_loss = self.bce_with_logits_loss(long_levels_pred, long_levels_target).mean()
                losses.append(long_levels_loss * self.long_levels_weight)
        
        # 4. Short Levels Loss (индексы 12-15) - BCE для бинарной классификации
        if use_all_losses or "short_levels" in self.active_losses:
            if short_levels_target is not None:
                short_levels_pred = outputs[:, 12:16]
                short_levels_loss = self.bce_with_logits_loss(short_levels_pred, short_levels_target).mean()
                losses.append(short_levels_loss * self.short_levels_weight)
        
        # 5. Risk Metrics Loss (индексы 16-19) - MSE для регрессии
        if use_all_losses or "risk_metrics" in self.active_losses:
            if risk_metrics_target is not None:
                risk_metrics_pred = outputs[:, 16:20]
                # Нормализуем если в процентах и если targets - тензор
                if not isinstance(targets, dict):
                    risk_metrics_target = risk_metrics_target / 100.0
                risk_metrics_loss = self.mse_loss(risk_metrics_pred, risk_metrics_target).mean()
                losses.append(risk_metrics_loss * self.risk_metrics_weight)
        
        # 6. Confidence Loss - обучаем предсказывать правильность предсказаний
        if confidence_scores is not None and hasattr(outputs, '_direction_logits') and direction_target is not None:
            direction_logits = outputs._direction_logits
            pred_classes = torch.argmax(direction_logits, dim=-1)
            # Используем уже извлеченный direction_target
            true_classes = direction_target.long() if direction_target is not None else None

            if true_classes is not None:
                # Проверка размерности
                if true_classes.dim() == 1:
                    true_classes = true_classes.unsqueeze(1).expand(-1, 4)

                correct_predictions = (pred_classes == true_classes).float()
                confidence_targets = correct_predictions * 2 - 1
                confidence_loss = F.mse_loss(confidence_scores, confidence_targets, reduction='mean')
                losses.append(confidence_loss * 0.1)
        
        if len(losses) > 0:
            total_loss = sum(losses)
            if hasattr(self, 'model') and hasattr(self.model, 'get_direction_l2_loss'):
                l2_loss = self.model.get_direction_l2_loss()
                total_loss = total_loss + l2_loss
        else:
            total_loss = torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        return total_loss


def create_unified_model(config: Dict) -> UnifiedPatchTSTForTrading:
    """Создание унифицированной модели

    Args:
        config: Может быть как полный конфиг с секцией 'model',
                так и только содержимое секции 'model'
    """
    # Если передали только model config, оборачиваем его
    if 'model' not in config:
        config = {'model': config}

    return UnifiedPatchTSTForTrading(config)