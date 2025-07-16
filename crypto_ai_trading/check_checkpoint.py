#!/usr/bin/env python3
"""Проверка содержимого checkpoint"""

import torch
from pathlib import Path

# Находим последний checkpoint
models_dir = Path("models_saved")
model_files = list(models_dir.glob("*.pth"))
model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

if model_files:
    checkpoint_path = model_files[0]
    print(f"Проверяем: {checkpoint_path.name}")
    
    # Загружаем checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        print("\nКлючи в checkpoint:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        # Проверяем model_config
        if 'model_config' in checkpoint:
            print("\nКонфигурация модели:")
            model_config = checkpoint['model_config']
            if 'model' in model_config:
                print(f"  input_size: {model_config['model'].get('input_size', 'НЕ НАЙДЕНО')}")
                print(f"  output_size: {model_config['model'].get('output_size', 'НЕ НАЙДЕНО')}")
        
        # Проверяем размеры модели
        if 'model_state_dict' in checkpoint:
            print("\nРазмеры ключевых слоев:")
            state_dict = checkpoint['model_state_dict']
            
            # RevIN слои
            if 'revin.affine_weight' in state_dict:
                print(f"  revin.affine_weight: {state_dict['revin.affine_weight'].shape}")
            
            # Patch embedding
            if 'patch_embedding.weight' in state_dict:
                print(f"  patch_embedding.weight: {state_dict['patch_embedding.weight'].shape}")
            
            # Output projection
            if 'output_projection.weight' in state_dict:
                print(f"  output_projection.weight: {state_dict['output_projection.weight'].shape}")
    else:
        print("Checkpoint не является словарем!")