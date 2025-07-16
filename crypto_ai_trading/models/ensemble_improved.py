"""
Улучшенный Ensemble модуль для объединения предсказаний нескольких моделей
Оптимизирован для fine-tuning и production использования
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

from models.patchtst_unified import UnifiedPatchTST
from utils.logger import get_logger


class ImprovedModelEnsemble(nn.Module):
    """Улучшенный ансамбль из нескольких моделей"""
    
    def __init__(self, 
                 config: Dict,
                 checkpoint_paths: List[str],
                 weights: Optional[List[float]] = None,
                 voting: str = 'soft',
                 device: Optional[torch.device] = None):
        """
        Args:
            config: конфигурация модели
            checkpoint_paths: список путей к checkpoint файлам
            weights: веса для каждой модели (None = равные веса)
            voting: 'soft' (усреднение вероятностей) или 'hard' (мажоритарное голосование)
            device: устройство для вычислений
        """
        super().__init__()
        
        self.logger = get_logger("ImprovedModelEnsemble")
        self.config = config
        self.voting = voting
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загружаем модели
        self.models = nn.ModuleList()
        self.model_weights = []
        self.model_metrics = []
        
        for idx, checkpoint_path in enumerate(checkpoint_paths):
            model, metrics = self._load_model(checkpoint_path, idx)
            self.models.append(model)
            self.model_metrics.append(metrics)
            
            # Веса для взвешенного голосования
            if weights is not None and idx < len(weights):
                self.model_weights.append(weights[idx])
            else:
                # Автоматические веса на основе метрик
                self.model_weights.append(self._calculate_weight(metrics))
        
        # Нормализуем веса
        weight_sum = sum(self.model_weights)
        self.model_weights = [w / weight_sum for w in self.model_weights]
        
        # Temperature для калибровки
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # Learnable weights для адаптивного ensemble
        self.learnable_weights = nn.Parameter(torch.tensor(self.model_weights))
        
        self.logger.info(f"✅ Ensemble создан из {len(self.models)} моделей")
        self.logger.info(f"   - Voting: {self.voting}")
        self.logger.info(f"   - Веса: {[f'{w:.3f}' for w in self.model_weights]}")
    
    def _load_model(self, checkpoint_path: str, idx: int) -> tuple:
        """Загрузка одной модели из checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Создаем модель
        model = UnifiedPatchTST(self.config).to(self.device)
        
        # Загружаем веса
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'ema_state_dict' in checkpoint:
            # Предпочитаем EMA веса если есть
            model.load_state_dict(checkpoint['ema_state_dict'])
            self.logger.info(f"   Model {idx}: использованы EMA веса")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Извлекаем метрики
        metrics = {}
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            self.logger.info(f"   Model {idx}: Direction Acc = {metrics.get('direction_accuracy', 0):.3f}, "
                           f"Win Rate = {metrics.get('win_rate', 0):.3f}")
        
        return model, metrics
    
    def _calculate_weight(self, metrics: Dict) -> float:
        """Вычисление веса модели на основе её метрик"""
        # Используем комбинацию метрик для веса
        direction_acc = metrics.get('direction_accuracy', 0.33)  # baseline 33%
        win_rate = metrics.get('win_rate', 0.5)  # baseline 50%
        
        # Нормализуем метрики
        norm_direction = max(0, (direction_acc - 0.33) / 0.67)  # 0 to 1
        norm_win_rate = max(0, (win_rate - 0.45) / 0.55)  # 0 to 1
        
        # Комбинированный вес
        weight = 0.7 * norm_direction + 0.3 * norm_win_rate + 0.1  # минимум 0.1
        
        return weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass через ансамбль"""
        batch_size = x.shape[0]
        
        # Используем learnable weights если обучаемся
        if self.training:
            weights = F.softmax(self.learnable_weights, dim=0)
        else:
            weights = self.model_weights
        
        # Собираем предсказания от всех моделей
        all_outputs = []
        all_direction_logits = []
        all_confidence_scores = []
        
        for model in self.models:
            outputs = model(x)
            all_outputs.append(outputs)
            
            # Сохраняем direction логиты и confidence если есть
            if hasattr(outputs, '_direction_logits'):
                all_direction_logits.append(outputs._direction_logits)
            if hasattr(outputs, '_confidence_scores'):
                all_confidence_scores.append(outputs._confidence_scores)
        
        # Объединяем предсказания
        if self.voting == 'soft':
            # Soft voting - взвешенное усреднение
            ensemble_output = torch.zeros_like(all_outputs[0])
            
            for idx, outputs in enumerate(all_outputs):
                ensemble_output += outputs * weights[idx]
            
            # Объединяем direction логиты с temperature scaling
            if all_direction_logits:
                ensemble_direction_logits = torch.zeros_like(all_direction_logits[0])
                
                for idx, logits in enumerate(all_direction_logits):
                    # Применяем temperature scaling
                    scaled_logits = logits / self.temperature
                    ensemble_direction_logits += scaled_logits * weights[idx]
                
                # Сохраняем для loss функции
                ensemble_output._direction_logits = ensemble_direction_logits
            
            # Взвешенное усреднение confidence scores
            if all_confidence_scores:
                ensemble_confidence = torch.zeros_like(all_confidence_scores[0])
                for idx, conf in enumerate(all_confidence_scores):
                    ensemble_confidence += conf * weights[idx]
                ensemble_output._confidence_scores = ensemble_confidence
        
        else:  # hard voting
            # Для направлений используем взвешенное мажоритарное голосование
            if all_direction_logits:
                # Собираем взвешенные голоса
                weighted_votes = torch.zeros(batch_size, 4, 3, device=self.device)
                
                for idx, logits in enumerate(all_direction_logits):
                    probs = F.softmax(logits / self.temperature, dim=-1)
                    weighted_votes += probs * weights[idx]
                
                # Выбираем класс с максимальным взвешенным голосом
                ensemble_preds = torch.argmax(weighted_votes, dim=-1)
                
                # Создаем логиты для консистентности
                ensemble_direction_logits = torch.log(weighted_votes + 1e-8)
                ensemble_output._direction_logits = ensemble_direction_logits
            
            # Для остальных выходов используем взвешенное среднее
            ensemble_output = torch.zeros_like(all_outputs[0])
            for idx, outputs in enumerate(all_outputs):
                ensemble_output += outputs * weights[idx]
        
        return ensemble_output
    
    def get_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Оценка неопределенности через разброс предсказаний"""
        with torch.no_grad():
            all_predictions = []
            
            for model in self.models:
                outputs = model(x)
                all_predictions.append(outputs)
            
            # Стек предсказаний
            stacked = torch.stack(all_predictions, dim=0)
            
            # Вычисляем стандартное отклонение
            std = torch.std(stacked, dim=0)
            mean = torch.mean(stacked, dim=0)
            
            # Коэффициент вариации как мера неопределенности
            uncertainty = std / (torch.abs(mean) + 1e-8)
            
            return {
                'prediction_std': std,
                'prediction_uncertainty': uncertainty,
                'mean_prediction': mean
            }
    
    def calibrate_temperature(self, val_loader, criterion, num_batches: int = 50):
        """Калибровка temperature на валидационных данных"""
        self.eval()
        
        best_temperature = 1.0
        best_loss = float('inf')
        
        temperatures = np.logspace(-0.5, 0.5, 11)  # от 0.316 до 3.16
        
        for temp in temperatures:
            self.temperature.data = torch.tensor([temp])
            total_loss = 0.0
            
            with torch.no_grad():
                for i, (inputs, targets, _) in enumerate(val_loader):
                    if i >= num_batches:
                        break
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
            
            avg_loss = total_loss / min(num_batches, len(val_loader))
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_temperature = temp
        
        self.temperature.data = torch.tensor([best_temperature])
        self.logger.info(f"✅ Temperature откалибрована: {best_temperature:.3f}")
        
        return best_temperature


def create_improved_ensemble(config: Dict, 
                           checkpoint_paths: List[str],
                           weights: Optional[List[float]] = None,
                           voting: str = 'soft',
                           top_k: int = 5) -> ImprovedModelEnsemble:
    """
    Создание улучшенного ансамбля с автоматическим выбором лучших моделей
    
    Args:
        config: конфигурация
        checkpoint_paths: список путей к checkpoints
        weights: веса моделей (None для автоматического расчета)
        voting: тип голосования
        top_k: количество лучших моделей для ансамбля
    """
    
    # Загружаем метрики всех моделей
    model_info = []
    for path in checkpoint_paths:
        if Path(path).exists():
            try:
                checkpoint = torch.load(path, map_location='cpu')
                metrics = checkpoint.get('metrics', {})
                model_info.append({
                    'path': path,
                    'metrics': metrics,
                    'score': metrics.get('direction_accuracy', 0) * 0.7 + 
                            metrics.get('win_rate', 0) * 0.3
                })
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {path}: {e}")
    
    # Сортируем по score и берем top_k
    model_info.sort(key=lambda x: x['score'], reverse=True)
    selected_models = model_info[:min(top_k, len(model_info))]
    
    if len(selected_models) < 2:
        raise ValueError(f"Недостаточно моделей для ансамбля. Найдено: {len(selected_models)}")
    
    # Пути к выбранным моделям
    selected_paths = [m['path'] for m in selected_models]
    
    print(f"✅ Выбрано {len(selected_paths)} лучших моделей для ансамбля")
    for m in selected_models:
        print(f"   - Score: {m['score']:.3f}, Path: {Path(m['path']).name}")
    
    ensemble = ImprovedModelEnsemble(
        config=config,
        checkpoint_paths=selected_paths,
        weights=weights,
        voting=voting
    )
    
    return ensemble