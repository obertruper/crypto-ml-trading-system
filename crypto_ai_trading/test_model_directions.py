#!/usr/bin/env python3
"""
Тестирование предсказаний direction модели
"""

import torch
import yaml
from pathlib import Path
import numpy as np

from models.patchtst_unified import UnifiedPatchTSTForTrading
from utils.logger import get_logger

def main():
    logger = get_logger("TestDirections")
    
    # Загружаем конфигурацию
    config_path = "config/config_production.yaml" if Path("config/config_production.yaml").exists() else "config/config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✅ Загружена конфигурация из {config_path}")
    
    # Параметры модели
    model_config = config['model']
    # Используем input_size из конфига
    
    # Создаем модель
    model = UnifiedPatchTSTForTrading(model_config).cuda()
    logger.info("✅ Модель создана")
    
    # Проверяем начальные веса direction_head
    for name, param in model.named_parameters():
        if 'direction_head' in name and 'bias' in name:
            logger.info(f"\n📊 {name}:")
            if param.shape[0] == 12:  # 4 таймфрейма × 3 класса
                bias = param.view(4, 3)
                for i in range(4):
                    logger.info(f"   Таймфрейм {i}: LONG={bias[i, 0]:.3f}, SHORT={bias[i, 1]:.3f}, FLAT={bias[i, 2]:.3f}")
    
    # Создаем тестовый батч
    batch_size = 32
    seq_len = 168
    n_features = model_config['input_size']  # Берем из конфига
    
    # Случайные данные
    x = torch.randn(batch_size, seq_len, n_features).cuda()
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        
        # Извлекаем direction логиты
        if hasattr(outputs, '_direction_logits'):
            direction_logits = outputs._direction_logits  # (B, 4, 3)
            
            logger.info("\n📊 Анализ предсказаний direction:")
            
            for i, tf in enumerate(['15m', '1h', '4h', '12h']):
                logits = direction_logits[:, i, :]  # (B, 3)
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                
                # Подсчет классов
                unique, counts = torch.unique(predictions, return_counts=True)
                class_dist = {0: 0, 1: 0, 2: 0}
                for cls, cnt in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                    class_dist[int(cls)] = cnt
                
                logger.info(f"\n🕐 Таймфрейм {tf}:")
                logger.info(f"   LONG (0):  {class_dist[0]:3d} ({class_dist[0]/batch_size*100:5.1f}%)")
                logger.info(f"   SHORT (1): {class_dist[1]:3d} ({class_dist[1]/batch_size*100:5.1f}%)")
                logger.info(f"   FLAT (2):  {class_dist[2]:3d} ({class_dist[2]/batch_size*100:5.1f}%)")
                
                # Статистика логитов
                mean_logits = logits.mean(dim=0)
                logger.info(f"   Средние логиты: LONG={mean_logits[0]:.3f}, SHORT={mean_logits[1]:.3f}, FLAT={mean_logits[2]:.3f}")
                
                # Статистика вероятностей
                mean_probs = probs.mean(dim=0)
                logger.info(f"   Средние вероятности: LONG={mean_probs[0]:.3f}, SHORT={mean_probs[1]:.3f}, FLAT={mean_probs[2]:.3f}")
    
    # Попробуем загрузить обученную модель если есть
    checkpoint_path = Path("models_saved/best_model.pth")
    if checkpoint_path.exists():
        logger.info(f"\n📥 Загружаем checkpoint из {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        logger.info("\n📊 Предсказания ОБУЧЕННОЙ модели:")
        
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            
            if hasattr(outputs, '_direction_logits'):
                direction_logits = outputs._direction_logits
                
                for i, tf in enumerate(['15m', '1h', '4h', '12h']):
                    logits = direction_logits[:, i, :]
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(probs, dim=-1)
                    
                    # Подсчет классов
                    unique, counts = torch.unique(predictions, return_counts=True)
                    class_dist = {0: 0, 1: 0, 2: 0}
                    for cls, cnt in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                        class_dist[int(cls)] = cnt
                    
                    logger.info(f"\n🕐 Таймфрейм {tf}:")
                    logger.info(f"   LONG (0):  {class_dist[0]:3d} ({class_dist[0]/batch_size*100:5.1f}%)")
                    logger.info(f"   SHORT (1): {class_dist[1]:3d} ({class_dist[1]/batch_size*100:5.1f}%)")
                    logger.info(f"   FLAT (2):  {class_dist[2]:3d} ({class_dist[2]/batch_size*100:5.1f}%)")
                    
                    # Проверяем есть ли вообще разнообразие
                    if len(unique) == 1:
                        logger.warning(f"   ⚠️ Модель предсказывает ТОЛЬКО класс {unique[0].item()}!")

if __name__ == "__main__":
    main()