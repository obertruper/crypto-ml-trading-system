#!/usr/bin/env python3
"""
Комплексная диагностика проблем обучения
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class TrainingDiagnostics:
    def __init__(self):
        self.issues = []
        self.recommendations = []
        
    def analyze_checkpoint(self, checkpoint_path="models_saved/best_model_20250701_120952.pth"):
        """Детальный анализ checkpoint"""
        console.print("\n[bold cyan]🔍 Анализ checkpoint модели...[/bold cyan]")
        
        if not Path(checkpoint_path).exists():
            self.issues.append("Checkpoint не найден")
            return
            
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Анализ весов
        model_state = ckpt['model_state_dict']
        dead_weights = []
        gradient_issues = []
        
        for name, param in model_state.items():
            if 'weight' in name and param.dim() > 1:
                std = param.std().item()
                mean = param.mean().item()
                
                # Проверка на мертвые веса
                if std < 0.001:
                    dead_weights.append(name)
                    
                # Проверка на взрывающие/исчезающие градиенты
                if std > 10 or std < 0.0001:
                    gradient_issues.append((name, std))
        
        if dead_weights:
            self.issues.append(f"Найдено {len(dead_weights)} слоев с мертвыми весами")
            
        if gradient_issues:
            self.issues.append(f"Проблемы с градиентами в {len(gradient_issues)} слоях")
            
        # Анализ learning rate
        if 'optimizer_state_dict' in ckpt:
            lr = ckpt['optimizer_state_dict']['param_groups'][0]['lr']
            if lr < 0.0001:
                self.issues.append(f"Слишком низкий learning rate: {lr}")
                self.recommendations.append("Увеличить learning rate до 0.001-0.003")
                
        console.print(f"✅ Val loss: {ckpt.get('best_val_loss', 'N/A')}")
        console.print(f"✅ Эпоха: {ckpt.get('epoch', 'N/A')}")
        
    def analyze_data_distribution(self):
        """Анализ распределения данных"""
        console.print("\n[bold cyan]📊 Анализ данных...[/bold cyan]")
        
        train_path = Path("data/processed/train_data.pkl")
        if not train_path.exists():
            self.issues.append("Данные для обучения не найдены")
            return
            
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
            
        # Анализ целевых переменных
        target_cols = [col for col in train_data.columns if any(x in col for x in ['tp', 'sl', 'hit', 'reached', 'direction'])]
        
        imbalanced_targets = []
        for col in target_cols:
            if col in train_data.columns:
                value_counts = train_data[col].value_counts(normalize=True)
                # Проверка на дисбаланс классов
                if len(value_counts) == 2:  # Бинарная классификация
                    min_class = value_counts.min()
                    if min_class < 0.1:  # Менее 10%
                        imbalanced_targets.append((col, min_class))
                        
        if imbalanced_targets:
            self.issues.append(f"Дисбаланс классов в {len(imbalanced_targets)} целевых переменных")
            self.recommendations.append("Использовать weighted loss или SMOTE для балансировки")
            
        # Проверка на NaN и выбросы
        nan_cols = train_data.columns[train_data.isna().any()].tolist()
        if nan_cols:
            self.issues.append(f"NaN значения в {len(nan_cols)} колонках")
            
        console.print(f"✅ Размер данных: {train_data.shape}")
        console.print(f"✅ Целевых переменных: {len(target_cols)}")
        
    def analyze_training_dynamics(self):
        """Анализ динамики обучения"""
        console.print("\n[bold cyan]📈 Анализ динамики обучения...[/bold cyan]")
        
        # Из логов видно:
        # Эпоха 1: train_loss=0.3814, val_loss=0.3308
        # Эпоха 2: train_loss=0.3754, val_loss=0.3308
        # Эпоха 3: train_loss=0.3754, val_loss=0.3308
        
        train_losses = [0.3814, 0.3754, 0.3754]
        val_losses = [0.3308, 0.3308, 0.3308]
        
        # Проверка на застревание
        if len(set(val_losses)) == 1:
            self.issues.append("Val loss не меняется - модель застряла")
            self.recommendations.append("Использовать OneCycleLR или CosineAnnealingLR scheduler")
            
        # Проверка на недообучение
        if all(tl > vl for tl, vl in zip(train_losses, val_losses)):
            self.issues.append("Train loss выше val loss - возможно недообучение")
            self.recommendations.append("Увеличить capacity модели или уменьшить regularization")
            
    def generate_report(self):
        """Генерация отчета с рекомендациями"""
        console.print("\n[bold red]❌ ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:[/bold red]")
        for i, issue in enumerate(self.issues, 1):
            console.print(f"{i}. {issue}")
            
        console.print("\n[bold green]💡 РЕКОМЕНДАЦИИ:[/bold green]")
        for i, rec in enumerate(self.recommendations, 1):
            console.print(f"{i}. {rec}")
            
        # Дополнительные рекомендации
        console.print("\n[bold yellow]🚀 КОМПЛЕКСНОЕ РЕШЕНИЕ:[/bold yellow]")
        console.print("1. Остановить текущее обучение (Ctrl+C)")
        console.print("2. Использовать оптимизированный конфиг (будет создан)")
        console.print("3. Перезапустить с новыми параметрами")
        console.print("4. Мониторить градиенты и loss dynamics")
        
        return self.issues, self.recommendations

if __name__ == "__main__":
    console.print("[bold magenta]🔧 ДИАГНОСТИКА ПРОБЛЕМ ОБУЧЕНИЯ[/bold magenta]")
    
    diagnostics = TrainingDiagnostics()
    diagnostics.analyze_checkpoint()
    diagnostics.analyze_data_distribution()
    diagnostics.analyze_training_dynamics()
    
    issues, recommendations = diagnostics.generate_report()
    
    # Сохранение отчета
    report = {
        'issues': issues,
        'recommendations': recommendations,
        'timestamp': pd.Timestamp.now()
    }
    
    with open('training_diagnostics_report.pkl', 'wb') as f:
        pickle.dump(report, f)
        
    console.print("\n✅ Отчет сохранен в training_diagnostics_report.pkl")