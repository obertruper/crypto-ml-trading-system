#!/usr/bin/env python3
"""
Анализ переобучения модели и качества предсказаний
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import yaml

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger('OverfittingAnalysis')

class ModelAnalyzer:
    """Анализатор модели для выявления переобучения"""
    
    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        self.model_path = Path(model_path)
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.results = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def load_model(self):
        """Загрузка модели"""
        logger.info(f"Загрузка модели из {self.model_path}")
        
        # Импорт модели
        from models.patchtst_unified import UnifiedPatchTST
        
        # Создание модели
        self.model = UnifiedPatchTST(self.config['model'])
        
        # Загрузка весов
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        logger.info("✅ Модель загружена успешно")
        
    def analyze_predictions(self, data_loader, dataset_name: str = "test"):
        """Анализ предсказаний модели"""
        logger.info(f"📊 Анализ предсказаний на {dataset_name} датасете...")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (X, y, _) in enumerate(data_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Предсказание
                outputs = self.model(X)
                
                # Сохранение результатов
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logger.info(f"  Обработано батчей: {batch_idx + 1}/{len(data_loader)}")
        
        # Объединение результатов
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Анализ результатов
        self.results[dataset_name] = {
            'predictions': predictions,
            'targets': targets,
            'metrics': self._calculate_metrics(predictions, targets)
        }
        
        return predictions, targets
        
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Расчет метрик"""
        metrics = {}
        
        # Определение типов таргетов из конфига
        regression_targets = ['future_return_15m', 'future_return_1h', 
                            'future_return_4h', 'future_return_12h']
        categorical_targets = ['direction_15m', 'direction_1h', 
                             'direction_4h', 'direction_12h']
        binary_targets = ['long_will_reach_1pct_4h', 'long_will_reach_2pct_4h',
                         'short_will_reach_1pct_4h', 'short_will_reach_2pct_4h']
        
        # Индексы таргетов (первые 20 из config v4.0)
        reg_idx = slice(0, 4)  # future_returns
        cat_idx = slice(4, 8)  # directions
        bin_idx = slice(8, 20)  # binary targets
        
        # Метрики для регрессионных таргетов
        reg_pred = predictions[:, reg_idx]
        reg_true = targets[:, reg_idx]
        
        metrics['regression'] = {
            'mae': np.mean(np.abs(reg_pred - reg_true)),
            'rmse': np.sqrt(np.mean((reg_pred - reg_true) ** 2)),
            'correlation': np.mean([np.corrcoef(reg_pred[:, i], reg_true[:, i])[0, 1] 
                                   for i in range(reg_pred.shape[1])])
        }
        
        # Метрики для категориальных таргетов (directions)
        cat_pred = np.argmax(predictions[:, cat_idx], axis=1)
        cat_true = np.argmax(targets[:, cat_idx], axis=1) if len(targets.shape) > 1 else targets
        
        # Метрики для бинарных таргетов
        bin_pred = (predictions[:, bin_idx] > 0.5).astype(int)
        bin_true = targets[:, bin_idx]
        
        metrics['binary'] = {
            'accuracy': np.mean(bin_pred == bin_true),
            'precision': np.sum((bin_pred == 1) & (bin_true == 1)) / (np.sum(bin_pred == 1) + 1e-8),
            'recall': np.sum((bin_pred == 1) & (bin_true == 1)) / (np.sum(bin_true == 1) + 1e-8)
        }
        
        # Общая статистика
        metrics['overall'] = {
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'mean_target': np.mean(targets),
            'std_target': np.std(targets)
        }
        
        return metrics
        
    def visualize_results(self, save_dir: str = "analysis_results"):
        """Визуализация результатов"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for dataset_name, results in self.results.items():
            predictions = results['predictions']
            targets = results['targets']
            
            # 1. Распределение предсказаний vs реальных значений
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Future returns (первые 4 таргета)
            for i in range(4):
                ax = axes[i // 2, i % 2]
                ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=1)
                ax.plot([targets[:, i].min(), targets[:, i].max()], 
                       [targets[:, i].min(), targets[:, i].max()], 'r--', lw=2)
                ax.set_xlabel(f'True Return {["15m", "1h", "4h", "12h"][i]}')
                ax.set_ylabel(f'Predicted Return {["15m", "1h", "4h", "12h"][i]}')
                ax.set_title(f'Predictions vs Targets - {["15m", "1h", "4h", "12h"][i]}')
                
                # Добавляем корреляцию
                corr = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]
                ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
                
            plt.tight_layout()
            plt.savefig(save_path / f'{dataset_name}_scatter_{timestamp}.png', dpi=150)
            plt.close()
            
            # 2. Распределения предсказаний
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            for i in range(4):
                ax = axes[i // 2, i % 2]
                ax.hist(predictions[:, i], bins=50, alpha=0.7, label='Predictions', density=True)
                ax.hist(targets[:, i], bins=50, alpha=0.7, label='Targets', density=True)
                ax.set_xlabel(f'Return {["15m", "1h", "4h", "12h"][i]}')
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution - {["15m", "1h", "4h", "12h"][i]}')
                ax.legend()
                
            plt.tight_layout()
            plt.savefig(save_path / f'{dataset_name}_distributions_{timestamp}.png', dpi=150)
            plt.close()
            
            # 3. Метрики в текстовом виде
            with open(save_path / f'{dataset_name}_metrics_{timestamp}.txt', 'w') as f:
                f.write(f"=== Анализ {dataset_name} датасета ===\n\n")
                
                metrics = results['metrics']
                
                f.write("📊 Регрессионные метрики:\n")
                f.write(f"  MAE: {metrics['regression']['mae']:.6f}\n")
                f.write(f"  RMSE: {metrics['regression']['rmse']:.6f}\n")
                f.write(f"  Correlation: {metrics['regression']['correlation']:.3f}\n\n")
                
                f.write("📊 Бинарные метрики:\n")
                f.write(f"  Accuracy: {metrics['binary']['accuracy']:.3f}\n")
                f.write(f"  Precision: {metrics['binary']['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['binary']['recall']:.3f}\n\n")
                
                f.write("📊 Общая статистика:\n")
                f.write(f"  Mean prediction: {metrics['overall']['mean_prediction']:.6f}\n")
                f.write(f"  Std prediction: {metrics['overall']['std_prediction']:.6f}\n")
                f.write(f"  Mean target: {metrics['overall']['mean_target']:.6f}\n")
                f.write(f"  Std target: {metrics['overall']['std_target']:.6f}\n")
                
        logger.info(f"✅ Результаты сохранены в {save_path}")
        
    def check_overfitting_signs(self):
        """Проверка признаков переобучения"""
        logger.info("🔍 Проверка признаков переобучения...")
        
        signs = []
        
        if 'train' in self.results and 'val' in self.results:
            train_metrics = self.results['train']['metrics']
            val_metrics = self.results['val']['metrics']
            
            # Проверка MAE
            mae_ratio = val_metrics['regression']['mae'] / (train_metrics['regression']['mae'] + 1e-8)
            if mae_ratio > 2.0:
                signs.append(f"⚠️ MAE ratio (val/train): {mae_ratio:.2f} > 2.0")
                
            # Проверка корреляции
            corr_diff = train_metrics['regression']['correlation'] - val_metrics['regression']['correlation']
            if corr_diff > 0.2:
                signs.append(f"⚠️ Correlation drop: {corr_diff:.3f} > 0.2")
                
            # Проверка дисперсии предсказаний
            std_ratio = val_metrics['overall']['std_prediction'] / (train_metrics['overall']['std_prediction'] + 1e-8)
            if std_ratio < 0.5 or std_ratio > 2.0:
                signs.append(f"⚠️ Std ratio (val/train): {std_ratio:.2f}")
                
        if signs:
            logger.warning("🚨 Обнаружены признаки переобучения:")
            for sign in signs:
                logger.warning(f"  {sign}")
        else:
            logger.info("✅ Явных признаков переобучения не обнаружено")
            
        return signs


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ переобучения модели')
    parser.add_argument('--model', type=str, default='models_saved/best_model.pth',
                       help='Путь к модели')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Директория с данными')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Размер батча для анализа')
    
    args = parser.parse_args()
    
    # Создание анализатора
    analyzer = ModelAnalyzer(args.model)
    analyzer.load_model()
    
    # Загрузка данных
    from data.precomputed_dataset import create_precomputed_data_loaders
    
    # Здесь нужно загрузить данные из parquet файлов
    # Для примера используем заглушку
    logger.info("📥 Загрузка данных...")
    
    # TODO: Реализовать загрузку данных и создание data_loader'ов
    # train_loader, val_loader, test_loader = create_data_loaders(args.data_dir)
    
    # Анализ на разных датасетах
    # analyzer.analyze_predictions(test_loader, "test")
    # analyzer.analyze_predictions(val_loader, "val")
    # analyzer.analyze_predictions(train_loader, "train")
    
    # Визуализация результатов
    # analyzer.visualize_results()
    
    # Проверка признаков переобучения
    # analyzer.check_overfitting_signs()
    
    logger.info("✅ Анализ завершен!")


if __name__ == "__main__":
    main()