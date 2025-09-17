"""
Иерархическая голова предсказаний
4-уровневая структура: Market Regime → Direction → Targets → Returns
С поддержкой teacher forcing и confidence scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Any
from utils.logger import get_logger


class HierarchicalPredictionHead(nn.Module):
    """
    4-уровневая иерархическая голова для предсказаний
    
    Уровень 1: Market Regime (bull/bear/sideways) - 3 класса
    Уровень 2: Direction (LONG/SHORT/FLAT) - 12 выходов (4 TF × 3 класса)
    Уровень 3: Profit Targets (вероятности достижения) - 8 выходов
    Уровень 4: Expected Returns - 4 выхода
    
    Каждый уровень conditioned на предыдущие с teacher forcing
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 config: Dict = None):
        super().__init__()
        
        self.d_model = d_model
        self.config = config or {}
        self.logger = get_logger("HierarchicalHead")
        
        # Конфигурация иерархии
        hierarchical_config = self.config.get('hierarchical', {})
        self.use_teacher_forcing = hierarchical_config.get('teacher_forcing', True)
        self.teacher_dropout = hierarchical_config.get('teacher_dropout', 0.1)
        self.loss_weights = hierarchical_config.get('loss_weights', [1.0, 1.0, 0.5, 0.5])
        
        # Dropout для teacher forcing
        self.teacher_forcing_dropout = nn.Dropout(self.teacher_dropout)
        
        # УРОВЕНЬ 1: MARKET REGIME (3 класса: bull/bear/sideways)
        # ======================================================
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 3)  # bull, bear, sideways
        )
        
        # УРОВЕНЬ 2: DIRECTION (conditioned on regime)
        # ============================================
        # Input: d_model + 3 (regime probs)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 12)  # 4 timeframes × 3 classes (UP/DOWN/FLAT)
        )
        
        # УРОВЕНЬ 3: PROFIT TARGETS (conditioned on regime + direction)
        # ============================================================
        # Input: d_model + 3 (regime) + 12 (direction)
        self.long_targets_head = nn.Sequential(
            nn.Linear(d_model + 3 + 12, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 4)  # 4 long profit levels
        )
        
        self.short_targets_head = nn.Sequential(
            nn.Linear(d_model + 3 + 12, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 4)  # 4 short profit levels
        )
        
        # УРОВЕНЬ 4: EXPECTED RETURNS (использует всю информацию)
        # =======================================================
        # Input: d_model + 3 (regime) + 12 (direction) + 8 (targets)
        self.returns_head = nn.Sequential(
            nn.Linear(d_model + 3 + 12 + 8, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 4),  # 4 timeframe returns
            nn.Tanh()  # Ограничиваем возвраты в [-1, 1]
        )
        
        # CONFIDENCE HEADS для каждого уровня
        # ===================================
        self.confidence_heads = nn.ModuleDict({
            'regime': nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            ),
            'direction': nn.Sequential(
                nn.Linear(d_model + 3, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 4),  # 4 timeframes
                nn.Sigmoid()
            ),
            'targets': nn.Sequential(
                nn.Linear(d_model + 3 + 12, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 8),  # 4 long + 4 short
                nn.Sigmoid()
            ),
            'returns': nn.Sequential(
                nn.Linear(d_model + 3 + 12 + 8, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 4),  # 4 timeframes
                nn.Sigmoid()
            )
        })
        
        # Temperature scaling для калибровки
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)
        
        self.logger.info("🏗️ Иерархическая голова инициализирована:")
        self.logger.info(f"   - Teacher forcing: {self.use_teacher_forcing}")
        self.logger.info(f"   - Loss weights: {self.loss_weights}")
        
    def forward(self, 
                x: torch.Tensor,
                teacher_forcing: Optional[Dict[str, torch.Tensor]] = None,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass с иерархической структурой
        
        Args:
            x: Основное представление [batch_size, d_model]
            teacher_forcing: Ground truth для обучения (опционально)
            return_intermediates: Возвращать промежуточные представления
            
        Returns:
            Dict с предсказаниями всех уровней
        """
        batch_size = x.size(0)
        results = {}
        intermediates = {} if return_intermediates else None
        
        # УРОВЕНЬ 1: MARKET REGIME
        # ========================
        regime_logits = self.regime_head(x)  # [B, 3]
        
        # Temperature scaling
        regime_logits_scaled = regime_logits / self.temperature
        regime_probs = F.softmax(regime_logits_scaled, dim=-1)
        
        # Confidence для regime
        regime_confidence = self.confidence_heads['regime'](x).squeeze(-1)  # [B]
        
        results['regime'] = {
            'logits': regime_logits,
            'probs': regime_probs,
            'confidence': regime_confidence
        }
        
        if return_intermediates:
            intermediates['regime_representation'] = x
        
        # УРОВЕНЬ 2: DIRECTION (conditioned on regime)
        # ============================================
        
        # Используем teacher forcing или предсказания
        if self.training and teacher_forcing and 'regime' in teacher_forcing:
            # Teacher forcing: используем ground truth
            regime_input = teacher_forcing['regime']
            if self.teacher_dropout > 0:
                regime_input = self.teacher_forcing_dropout(regime_input)
        else:
            # Inference: используем предсказанные вероятности
            regime_input = regime_probs
        
        # Объединяем с основным представлением
        x_with_regime = torch.cat([x, regime_input], dim=-1)  # [B, d_model + 3]
        
        direction_logits = self.direction_head(x_with_regime)  # [B, 12]
        
        # Reshape для 4 timeframes × 3 classes
        direction_logits_reshaped = direction_logits.view(batch_size, 4, 3)  # [B, 4, 3]
        direction_probs = F.softmax(direction_logits_reshaped, dim=-1)  # [B, 4, 3]
        
        # Confidence для direction (по каждому timeframe)
        direction_confidence = self.confidence_heads['direction'](x_with_regime)  # [B, 4]
        
        results['direction'] = {
            'logits': direction_logits,
            'logits_reshaped': direction_logits_reshaped,
            'probs': direction_probs,
            'confidence': direction_confidence
        }
        
        if return_intermediates:
            intermediates['direction_representation'] = x_with_regime
        
        # УРОВЕНЬ 3: PROFIT TARGETS (conditioned on regime + direction)
        # ============================================================
        
        # Teacher forcing для direction
        if self.training and teacher_forcing and 'direction' in teacher_forcing:
            direction_input = teacher_forcing['direction']
            if self.teacher_dropout > 0:
                direction_input = self.teacher_forcing_dropout(direction_input)
        else:
            direction_input = direction_logits  # Используем логиты для стабильности
        
        x_with_direction = torch.cat([x_with_regime, direction_input], dim=-1)  # [B, d_model + 3 + 12]
        
        # Long targets
        long_targets_logits = self.long_targets_head(x_with_direction)  # [B, 4]
        long_targets_probs = torch.sigmoid(long_targets_logits)
        
        # Short targets  
        short_targets_logits = self.short_targets_head(x_with_direction)  # [B, 4]
        short_targets_probs = torch.sigmoid(short_targets_logits)
        
        # Объединяем targets
        targets_logits = torch.cat([long_targets_logits, short_targets_logits], dim=-1)  # [B, 8]
        targets_probs = torch.cat([long_targets_probs, short_targets_probs], dim=-1)  # [B, 8]
        
        # Confidence для targets
        targets_confidence = self.confidence_heads['targets'](x_with_direction)  # [B, 8]
        
        results['targets'] = {
            'logits': targets_logits,
            'probs': targets_probs,
            'long_logits': long_targets_logits,
            'long_probs': long_targets_probs,
            'short_logits': short_targets_logits,
            'short_probs': short_targets_probs,
            'confidence': targets_confidence
        }
        
        if return_intermediates:
            intermediates['targets_representation'] = x_with_direction
        
        # УРОВЕНЬ 4: EXPECTED RETURNS (финальный уровень)
        # ===============================================
        
        # Teacher forcing для targets
        if self.training and teacher_forcing and 'targets' in teacher_forcing:
            targets_input = teacher_forcing['targets']
            if self.teacher_dropout > 0:
                targets_input = self.teacher_forcing_dropout(targets_input)
        else:
            targets_input = targets_logits  # Используем логиты
        
        x_full = torch.cat([x_with_direction, targets_input], dim=-1)  # [B, d_model + 3 + 12 + 8]
        
        returns_values = self.returns_head(x_full)  # [B, 4]
        returns_confidence = self.confidence_heads['returns'](x_full)  # [B, 4]
        
        results['returns'] = {
            'values': returns_values,
            'confidence': returns_confidence
        }
        
        if return_intermediates:
            intermediates['final_representation'] = x_full
            results['intermediates'] = intermediates
        
        return results
    
    def compute_loss(self, 
                     predictions: Dict[str, Any],
                     targets: Dict[str, torch.Tensor],
                     weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Вычисление иерархического loss
        
        Args:
            predictions: Результат forward pass
            targets: Ground truth targets
            weights: Веса для разных компонентов loss
            
        Returns:
            Dict с компонентами loss и общим loss
        """
        if weights is None:
            weights = {
                'regime': self.loss_weights[0],
                'direction': self.loss_weights[1], 
                'targets': self.loss_weights[2],
                'returns': self.loss_weights[3]
            }
        
        losses = {}
        
        # 1. REGIME LOSS (Cross Entropy)
        if 'regime' in targets:
            regime_loss = F.cross_entropy(
                predictions['regime']['logits'],
                targets['regime']
            )
            losses['regime'] = regime_loss
        
        # 2. DIRECTION LOSS (Cross Entropy с учетом timeframes)
        if 'direction' in targets:
            # Targets должны быть [B, 4] с индексами классов для каждого TF
            batch_size = predictions['direction']['logits_reshaped'].size(0)
            
            direction_loss = 0
            for tf in range(4):  # 4 timeframes
                tf_logits = predictions['direction']['logits_reshaped'][:, tf, :]  # [B, 3]
                tf_targets = targets['direction'][:, tf]  # [B]
                
                tf_loss = F.cross_entropy(tf_logits, tf_targets)
                direction_loss += tf_loss
            
            direction_loss /= 4  # Среднее по timeframes
            losses['direction'] = direction_loss
        
        # 3. TARGETS LOSS (Binary Cross Entropy)
        if 'targets' in targets:
            targets_loss = F.binary_cross_entropy_with_logits(
                predictions['targets']['logits'],
                targets['targets']
            )
            losses['targets'] = targets_loss
        
        # 4. RETURNS LOSS (MSE)
        if 'returns' in targets:
            returns_loss = F.mse_loss(
                predictions['returns']['values'],
                targets['returns']
            )
            losses['returns'] = returns_loss
        
        # ОБЩИЙ LOSS с весами
        total_loss = sum(
            weights[name] * loss 
            for name, loss in losses.items() 
            if name in weights
        )
        
        losses['total'] = total_loss
        
        return losses
    
    def get_predictions_dict(self, 
                           hierarchical_output: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Преобразование иерархического выхода в стандартный формат
        для совместимости с существующим кодом
        """
        predictions = {}
        
        # Regime (3 values - bull/bear/sideways scores)
        predictions['regime_logits'] = hierarchical_output['regime']['logits']
        predictions['regime_probs'] = hierarchical_output['regime']['probs']
        
        # Direction (12 values - 4 TF × 3 classes, flattened)
        predictions['direction_logits'] = hierarchical_output['direction']['logits']
        predictions['direction_probs'] = hierarchical_output['direction']['probs'].view(-1, 12)
        
        # Targets (8 values - 4 long + 4 short)
        predictions['targets_logits'] = hierarchical_output['targets']['logits']
        predictions['targets_probs'] = hierarchical_output['targets']['probs']
        
        # Returns (4 values)
        predictions['returns'] = hierarchical_output['returns']['values']
        
        # Confidence scores
        predictions['regime_confidence'] = hierarchical_output['regime']['confidence']
        predictions['direction_confidence'] = hierarchical_output['direction']['confidence']
        predictions['targets_confidence'] = hierarchical_output['targets']['confidence']
        predictions['returns_confidence'] = hierarchical_output['returns']['confidence']
        
        return predictions
    
    def interpret_predictions(self, 
                            hierarchical_output: Dict[str, Any],
                            confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Интерпретация предсказаний для генерации торговых сигналов
        """
        batch_size = hierarchical_output['regime']['probs'].size(0)
        interpretations = []
        
        for i in range(batch_size):
            # Market Regime
            regime_probs = hierarchical_output['regime']['probs'][i]
            regime_idx = torch.argmax(regime_probs).item()
            regime_names = ['bull', 'bear', 'sideways']
            regime = regime_names[regime_idx]
            regime_conf = hierarchical_output['regime']['confidence'][i].item()
            
            # Direction для каждого timeframe
            direction_probs = hierarchical_output['direction']['probs'][i]  # [4, 3]
            direction_names = ['UP', 'DOWN', 'FLAT']
            timeframe_names = ['15m', '1h', '4h', '12h']
            
            directions = {}
            for tf in range(4):
                tf_probs = direction_probs[tf]
                tf_idx = torch.argmax(tf_probs).item()
                tf_conf = hierarchical_output['direction']['confidence'][i, tf].item()
                
                directions[timeframe_names[tf]] = {
                    'direction': direction_names[tf_idx],
                    'confidence': tf_conf,
                    'probabilities': tf_probs.tolist()
                }
            
            # Profit Targets
            targets_probs = hierarchical_output['targets']['probs'][i]  # [8]
            long_targets = targets_probs[:4].tolist()  # First 4 are long
            short_targets = targets_probs[4:].tolist()  # Last 4 are short
            
            # Expected Returns
            returns = hierarchical_output['returns']['values'][i].tolist()  # [4]
            returns_conf = hierarchical_output['returns']['confidence'][i].tolist()  # [4]
            
            interpretation = {
                'market_regime': {
                    'regime': regime,
                    'confidence': regime_conf,
                    'scores': {
                        'bull': regime_probs[0].item(),
                        'bear': regime_probs[1].item(), 
                        'sideways': regime_probs[2].item()
                    }
                },
                'directions': directions,
                'profit_targets': {
                    'long': {f'level_{j+1}': prob for j, prob in enumerate(long_targets)},
                    'short': {f'level_{j+1}': prob for j, prob in enumerate(short_targets)}
                },
                'expected_returns': {
                    timeframe_names[j]: {
                        'return': ret,
                        'confidence': conf
                    } for j, (ret, conf) in enumerate(zip(returns, returns_conf))
                },
                'trading_signal': self._generate_trading_signal(
                    regime, regime_conf, directions, confidence_threshold
                )
            }
            
            interpretations.append(interpretation)
        
        return interpretations
    
    def _generate_trading_signal(self, 
                               regime: str,
                               regime_confidence: float,
                               directions: Dict,
                               confidence_threshold: float) -> Dict:
        """Генерация торгового сигнала на основе иерархических предсказаний"""
        
        # Получаем основное направление (4h timeframe как основной)
        main_direction = directions.get('4h', {})
        main_signal = main_direction.get('direction', 'FLAT')
        main_conf = main_direction.get('confidence', 0.0)
        
        # Корректируем на основе режима рынка
        adjusted_confidence = main_conf
        
        if regime == 'bear' and main_signal == 'UP':
            adjusted_confidence *= 0.7  # Снижаем уверенность в лонгах в медвежьем рынке
        elif regime == 'bull' and main_signal == 'DOWN':
            adjusted_confidence *= 0.7  # Снижаем уверенность в шортах в бычьем рынке
        elif regime == 'sideways':
            adjusted_confidence *= 0.8  # Общее снижение уверенности в боковике
        
        # Финальное решение
        if adjusted_confidence >= confidence_threshold:
            action = 'LONG' if main_signal == 'UP' else 'SHORT' if main_signal == 'DOWN' else 'HOLD'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'confidence': adjusted_confidence,
            'raw_signal': main_signal,
            'raw_confidence': main_conf,
            'regime_adjustment': regime,
            'confidence_threshold': confidence_threshold
        }


def hierarchical_loss(predictions: Dict[str, Any], 
                     targets: Dict[str, torch.Tensor],
                     weights: list = [1.0, 1.0, 0.5, 0.5]) -> torch.Tensor:
    """
    Standalone функция для вычисления иерархического loss
    Для совместимости с существующим кодом
    """
    loss_dict = {}
    
    # Regime loss
    if 'regime' in predictions and 'regime' in targets:
        loss_dict['regime'] = F.cross_entropy(
            predictions['regime']['logits'],
            targets['regime']
        )
    
    # Direction loss
    if 'direction' in predictions and 'direction' in targets:
        loss_dict['direction'] = F.cross_entropy(
            predictions['direction']['logits'],
            targets['direction']
        )
    
    # Targets loss
    if 'targets' in predictions and 'targets' in targets:
        loss_dict['targets'] = F.binary_cross_entropy_with_logits(
            predictions['targets']['logits'],
            targets['targets']
        )
    
    # Returns loss
    if 'returns' in predictions and 'returns' in targets:
        loss_dict['returns'] = F.mse_loss(
            predictions['returns']['values'],
            targets['returns']
        )
    
    # Weighted sum
    total_loss = 0
    for i, (name, weight) in enumerate(zip(['regime', 'direction', 'targets', 'returns'], weights)):
        if name in loss_dict:
            total_loss += weight * loss_dict[name]
    
    return total_loss