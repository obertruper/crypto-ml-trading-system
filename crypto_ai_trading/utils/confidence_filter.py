"""
Утилиты для фильтрации предсказаний по уверенности
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from utils.logger import get_logger

logger = get_logger("ConfidenceFilter")


def filter_predictions_by_confidence(
    predictions: Dict[str, torch.Tensor],
    confidence_threshold: float = 0.6,
    return_mask: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Фильтрует предсказания по порогу уверенности
    
    Args:
        predictions: Словарь с предсказаниями модели
        confidence_threshold: Минимальный порог уверенности
        return_mask: Возвращать ли маску фильтрации
        
    Returns:
        Отфильтрованные предсказания
    """
    # Извлекаем confidence scores
    if 'confidence_scores' not in predictions:
        logger.warning("Confidence scores не найдены в предсказаниях")
        return predictions
    
    confidence_scores = predictions['confidence_scores']
    
    # confidence_scores в диапазоне [-1, 1] благодаря Tanh
    # -1 = неуверенный/неправильный, +1 = уверенный/правильный
    # Преобразуем в вероятности [0, 1] для порога
    confidence_probs = (confidence_scores + 1) / 2
    
    # Создаем маску для фильтрации
    # confidence_probs имеет размерность (batch_size, 4) для 4 таймфреймов
    confidence_mask = confidence_probs > confidence_threshold
    
    # Статистика
    total_predictions = confidence_scores.numel()
    confident_predictions = confidence_mask.sum().item()
    confidence_ratio = confident_predictions / total_predictions
    
    logger.info(f"📊 Фильтрация по уверенности:")
    logger.info(f"   Порог: {confidence_threshold:.2f}")
    logger.info(f"   Уверенных предсказаний: {confident_predictions}/{total_predictions} ({confidence_ratio:.1%})")
    
    # Создаем отфильтрованные предсказания
    filtered_predictions = predictions.copy()
    
    # Для direction классов: заменяем неуверенные на FLAT (2)
    if 'direction_classes' in predictions:
        direction_classes = predictions['direction_classes'].clone()
        # Заменяем неуверенные предсказания на FLAT
        direction_classes[~confidence_mask] = 2  # FLAT
        filtered_predictions['direction_classes'] = direction_classes
        
        # Статистика по классам после фильтрации
        for i in range(4):  # 4 таймфрейма
            mask_tf = confidence_mask[:, i]
            confident_count = mask_tf.sum().item()
            total_count = mask_tf.shape[0]
            
            if confident_count > 0:
                classes = direction_classes[:, i][mask_tf]
                long_count = (classes == 0).sum().item()
                short_count = (classes == 1).sum().item()
                flat_count = (classes == 2).sum().item()
                
                logger.info(f"   Таймфрейм {i+1}: {confident_count}/{total_count} уверенных "
                          f"(LONG: {long_count}, SHORT: {short_count}, FLAT: {flat_count})")
    
    # Для вероятностей уровней: обнуляем неуверенные
    for key in ['long_levels', 'short_levels']:
        if key in predictions:
            levels = predictions[key].clone()
            levels[~confidence_mask] = 0.0  # Обнуляем вероятности
            filtered_predictions[key] = levels
    
    if return_mask:
        filtered_predictions['confidence_mask'] = confidence_mask
    
    return filtered_predictions


def get_high_confidence_signals(
    predictions: Dict[str, torch.Tensor],
    min_confidence: float = 0.7,
    min_price_change: float = 0.01  # 1%
) -> Dict[str, torch.Tensor]:
    """
    Извлекает только высокоуверенные торговые сигналы
    
    Args:
        predictions: Предсказания модели
        min_confidence: Минимальная уверенность
        min_price_change: Минимальное ожидаемое изменение цены
        
    Returns:
        Словарь с высокоуверенными сигналами
    """
    signals = {
        'long_signals': [],
        'short_signals': [],
        'confidence': [],
        'expected_return': []
    }
    
    if 'confidence_scores' not in predictions:
        return signals
    
    confidence_scores = predictions['confidence_scores']
    # Преобразуем из [-1, 1] в [0, 1] (так как используем Tanh в модели)
    confidence_probs = (confidence_scores + 1) / 2
    
    direction_classes = predictions.get('direction_classes', None)
    future_returns = predictions.get('future_returns', None)
    
    if direction_classes is None or future_returns is None:
        return signals
    
    batch_size = confidence_scores.shape[0]
    
    for i in range(batch_size):
        for tf in range(4):  # 4 таймфрейма
            conf = confidence_probs[i, tf].item()
            
            if conf >= min_confidence:
                direction = int(direction_classes[i, tf].item())
                expected_return = future_returns[i, tf].item()
                
                # Проверяем минимальное движение
                if abs(expected_return) >= min_price_change:
                    if direction == 0:  # LONG
                        signals['long_signals'].append({
                            'sample': i,
                            'timeframe': tf,
                            'confidence': conf,
                            'expected_return': expected_return
                        })
                    elif direction == 1:  # SHORT
                        signals['short_signals'].append({
                            'sample': i,
                            'timeframe': tf,
                            'confidence': conf,
                            'expected_return': expected_return
                        })
    
    # Логирование результатов
    total_long = len(signals['long_signals'])
    total_short = len(signals['short_signals'])
    
    if total_long + total_short > 0:
        logger.info(f"🎯 Высокоуверенные сигналы (conf>{min_confidence:.2f}, move>{min_price_change:.1%}):")
        logger.info(f"   LONG: {total_long} сигналов")
        logger.info(f"   SHORT: {total_short} сигналов")
        
        # Средняя уверенность
        all_conf = [s['confidence'] for s in signals['long_signals'] + signals['short_signals']]
        avg_conf = np.mean(all_conf) if all_conf else 0
        logger.info(f"   Средняя уверенность: {avg_conf:.3f}")
    else:
        logger.info("❌ Нет высокоуверенных сигналов")
    
    return signals


def apply_confidence_based_position_sizing(
    base_position_size: float,
    confidence_score: float,
    min_confidence: float = 0.5,
    max_confidence: float = 0.9
) -> float:
    """
    Корректирует размер позиции на основе уверенности
    
    Args:
        base_position_size: Базовый размер позиции
        confidence_score: Уверенность предсказания [0, 1]
        min_confidence: Минимальная уверенность для торговли
        max_confidence: Уверенность для максимальной позиции
        
    Returns:
        Скорректированный размер позиции
    """
    if confidence_score < min_confidence:
        return 0.0  # Не торгуем при низкой уверенности
    
    # Линейная интерполяция между min и max
    confidence_factor = (confidence_score - min_confidence) / (max_confidence - min_confidence)
    confidence_factor = np.clip(confidence_factor, 0.0, 1.0)
    
    # Размер позиции от 50% до 100% базового
    adjusted_size = base_position_size * (0.5 + 0.5 * confidence_factor)
    
    return adjusted_size