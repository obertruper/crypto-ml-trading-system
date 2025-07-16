#!/usr/bin/env python3
"""
Анализ предсказаний обученной модели
"""

import torch
import yaml
from pathlib import Path
import h5py
import numpy as np

from utils.logger import get_logger

def main():
    logger = get_logger("AnalyzeModel")
    
    # Загружаем checkpoint
    checkpoint_path = Path("models_saved/best_model.pth")
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint не найден: {checkpoint_path}")
        return
        
    logger.info(f"📥 Загружаем checkpoint из {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Анализируем содержимое
    logger.info("\n📊 Содержимое checkpoint:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            logger.info(f"   {key}: {len(checkpoint[key])} параметров")
            # Ищем direction_head bias
            for param_name, param in checkpoint[key].items():
                if 'direction_head' in param_name and 'bias' in param_name:
                    logger.info(f"\n   📍 {param_name}: shape={param.shape}")
                    if param.shape[0] == 12:
                        bias = param.view(4, 3)
                        for i in range(4):
                            logger.info(f"      TF{i}: LONG={bias[i,0]:.3f}, SHORT={bias[i,1]:.3f}, FLAT={bias[i,2]:.3f}")
        else:
            logger.info(f"   {key}: {checkpoint[key] if not isinstance(checkpoint[key], dict) else len(checkpoint[key])}")
    
    # Загружаем валидационные данные для проверки
    val_file = Path("cache/precomputed/val_w168_s4.h5")
    if val_file.exists():
        logger.info(f"\n📊 Загружаем валидационные данные из {val_file}")
        
        with h5py.File(val_file, 'r') as f:
            X = f['X'][:100]  # Первые 100 примеров
            y = f['y'][:100]
            
            logger.info(f"   X shape: {X.shape}")
            logger.info(f"   y shape: {y.shape}")
            
            # Анализируем распределение классов в данных
            if y.shape[2] >= 8:  # Проверяем что есть direction переменные
                for i, tf in enumerate(['15m', '1h', '4h', '12h']):
                    direction_idx = 4 + i
                    if direction_idx < y.shape[2]:
                        directions = y[:, 0, direction_idx]
                        unique, counts = np.unique(directions, return_counts=True)
                        
                        logger.info(f"\n   🕐 {tf} - распределение в данных:")
                        for cls, cnt in zip(unique, counts):
                            logger.info(f"      Класс {int(cls)}: {cnt} ({cnt/len(directions)*100:.1f}%)")
    
    # Создаем модель и загружаем веса
    config_path = Path("config/config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Обновляем input_size из checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        if 'model' in saved_config and 'input_size' in saved_config['model']:
            config['model']['input_size'] = saved_config['model']['input_size']
            logger.info(f"\n✅ Используем input_size из checkpoint: {config['model']['input_size']}")
    
    # Пробуем загрузить модель
    try:
        from models.patchtst_unified import UnifiedPatchTSTForTrading
        model = UnifiedPatchTSTForTrading(config['model']).cuda()
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        logger.info("✅ Модель загружена")
        
        # Тестируем на случайных данных
        batch_size = 32
        seq_len = 168
        n_features = config['model']['input_size']
        
        x = torch.randn(batch_size, seq_len, n_features).cuda()
        
        with torch.no_grad():
            outputs = model(x)
            
            if hasattr(outputs, '_direction_logits'):
                direction_logits = outputs._direction_logits
                
                logger.info("\n📊 Предсказания модели на случайных данных:")
                for i, tf in enumerate(['15m', '1h', '4h', '12h']):
                    logits = direction_logits[:, i, :]
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(probs, dim=-1)
                    
                    unique, counts = torch.unique(predictions, return_counts=True)
                    
                    logger.info(f"\n   🕐 {tf}:")
                    class_dist = {0: 0, 1: 0, 2: 0}
                    for cls, cnt in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                        class_dist[int(cls)] = cnt
                    
                    logger.info(f"      LONG:  {class_dist[0]} ({class_dist[0]/batch_size*100:.1f}%)")
                    logger.info(f"      SHORT: {class_dist[1]} ({class_dist[1]/batch_size*100:.1f}%)")
                    logger.info(f"      FLAT:  {class_dist[2]} ({class_dist[2]/batch_size*100:.1f}%)")
                    
                    # Проверяем логиты
                    mean_logits = logits.mean(dim=0).cpu()
                    logger.info(f"      Средние логиты: [{mean_logits[0]:.3f}, {mean_logits[1]:.3f}, {mean_logits[2]:.3f}]")
                    
                    if len(unique) == 1:
                        logger.warning(f"      ⚠️ Модель предсказывает ТОЛЬКО класс {unique[0].item()}!")
                        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")

if __name__ == "__main__":
    main()