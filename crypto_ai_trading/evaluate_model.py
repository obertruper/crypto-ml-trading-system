#!/usr/bin/env python3
"""
Скрипт для детальной оценки обученной модели PatchTST
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading')

from config.config_loader import ConfigLoader
from models.patchtst_unified import UnifiedPatchTST
from data.precomputed_dataset import create_precomputed_dataloaders
from utils.logger import Logger

def load_config(config_path):
    config_loader = ConfigLoader()
    return config_loader.load(config_path)

logger = Logger(level="INFO", name=__name__)

class ModelEvaluator:
    def __init__(self, config_path='config/config.yaml'):
        """Инициализация оценщика модели"""
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def load_model(self, checkpoint_path='models_saved/best_model.pth'):
        """Загрузка обученной модели"""
        logger.info(f"📥 Загрузка модели из {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Создаем модель
        self.model = UnifiedPatchTST(
            input_size=self.config['model']['input_size'],
            output_size=self.config['model']['output_size'],
            seq_len=self.config['model']['seq_len'],
            pred_len=self.config['model']['pred_len'],
            patch_len=self.config['model']['patch_len'],
            stride=self.config['model']['stride'],
            d_model=self.config['model']['d_model'],
            n_heads=self.config['model']['n_heads'],
            e_layers=self.config['model']['e_layers'],
            d_ff=self.config['model']['d_ff'],
            dropout=self.config['model']['dropout'],
            activation=self.config['model']['activation']
        )
        
        # Загружаем веса
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Сохраняем историю обучения
        self.training_history = checkpoint.get('history', {})
        
        logger.info(f"✅ Модель загружена. Эпоха: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"📊 Лучший val_loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
        
        return self.model
        
    def load_data(self):
        """Загрузка тестовых данных"""
        logger.info("📊 Загрузка данных...")
        
        # Используем PrecomputedDataLoaders для быстрой загрузки
        data_loaders = PrecomputedDataLoaders(self.config)
        _, _, self.test_loader = data_loaders.get_loaders()
        
        logger.info(f"✅ Загружено {len(self.test_loader.dataset)} тестовых примеров")
        
        return self.test_loader
        
    def evaluate_predictions(self):
        """Оценка предсказаний модели"""
        logger.info("🔍 Начало оценки модели на тестовых данных...")
        
        all_predictions = []
        all_targets = []
        all_features = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Предсказание
                outputs = self.model(features)
                
                # Сохраняем для анализа
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_features.append(features.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logger.info(f"   Обработано батчей: {batch_idx + 1}/{len(self.test_loader)}")
        
        # Объединяем все предсказания
        self.predictions = np.vstack(all_predictions)
        self.targets = np.vstack(all_targets)
        self.features = np.vstack(all_features)
        
        logger.info(f"✅ Оценка завершена. Форма предсказаний: {self.predictions.shape}")
        
        return self.predictions, self.targets
        
    def calculate_metrics(self):
        """Расчет метрик для каждой целевой переменной"""
        logger.info("📈 Расчет метрик по каждой целевой переменной...")
        
        # Названия целевых переменных (20 штук)
        target_names = [
            'future_return_15m', 'future_return_1h', 'future_return_4h', 'future_return_12h',
            'direction_15m', 'direction_1h', 'direction_4h', 'direction_12h',
            'volatility_15m', 'volatility_1h', 'volatility_4h', 'volatility_12h',
            'volume_change_15m', 'volume_change_1h', 'volume_change_4h', 'volume_change_12h',
            'price_range_15m', 'price_range_1h', 'price_range_4h', 'price_range_12h'
        ]
        
        metrics_results = {}
        
        for i, target_name in enumerate(target_names):
            pred = self.predictions[:, i]
            true = self.targets[:, i]
            
            if 'direction' in target_name:
                # Для бинарной классификации
                pred_binary = (pred > 0.5).astype(int)
                true_binary = (true > 0.5).astype(int)
                
                accuracy = accuracy_score(true_binary, pred_binary)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_binary, pred_binary, average='binary'
                )
                
                metrics_results[target_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'type': 'classification'
                }
            else:
                # Для регрессии
                mse = mean_squared_error(true, pred)
                mae = mean_absolute_error(true, pred)
                r2 = r2_score(true, pred)
                
                metrics_results[target_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'type': 'regression'
                }
        
        self.metrics_results = metrics_results
        
        # Выводим сводку
        logger.info("\n📊 СВОДКА МЕТРИК:")
        logger.info("=" * 70)
        
        # Регрессионные метрики
        logger.info("\n🎯 Регрессионные целевые переменные:")
        for name, metrics in metrics_results.items():
            if metrics['type'] == 'regression':
                logger.info(f"{name:20s} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}")
        
        # Классификационные метрики
        logger.info("\n🎯 Классификационные целевые переменные:")
        for name, metrics in metrics_results.items():
            if metrics['type'] == 'classification':
                logger.info(f"{name:20s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
        
        return metrics_results
        
    def create_visualizations(self):
        """Создание визуализаций результатов"""
        logger.info("🎨 Создание визуализаций...")
        
        output_dir = Path('experiments/evaluation_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. График истории обучения
        if self.training_history:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.training_history.get('train_loss', []), label='Train Loss')
            plt.plot(self.training_history.get('val_loss', []), label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('История обучения')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            if 'learning_rates' in self.training_history:
                plt.plot(self.training_history['learning_rates'])
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title('Изменение Learning Rate')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'training_history.png', dpi=300)
            plt.close()
        
        # 2. Scatter plots для регрессионных переменных
        regression_vars = [name for name, m in self.metrics_results.items() if m['type'] == 'regression']
        
        n_cols = 4
        n_rows = (len(regression_vars) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(16, 4 * n_rows))
        
        for idx, var_name in enumerate(regression_vars[:8]):  # Первые 8 переменных
            plt.subplot(n_rows, n_cols, idx + 1)
            
            var_idx = [i for i, name in enumerate(self.metrics_results.keys()) if name == var_name][0]
            
            # Случайная выборка для визуализации
            sample_size = min(1000, len(self.predictions))
            sample_idx = np.random.choice(len(self.predictions), sample_size, replace=False)
            
            plt.scatter(self.targets[sample_idx, var_idx], 
                       self.predictions[sample_idx, var_idx], 
                       alpha=0.5, s=10)
            
            # Линия идеального предсказания
            min_val = min(self.targets[sample_idx, var_idx].min(), 
                         self.predictions[sample_idx, var_idx].min())
            max_val = max(self.targets[sample_idx, var_idx].max(), 
                         self.predictions[sample_idx, var_idx].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.xlabel('Реальные значения')
            plt.ylabel('Предсказания')
            plt.title(f'{var_name}\nR²={self.metrics_results[var_name]["r2"]:.3f}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'predictions_vs_actual.png', dpi=300)
        plt.close()
        
        # 3. Confusion matrices для классификационных переменных
        classification_vars = [name for name, m in self.metrics_results.items() if m['type'] == 'classification']
        
        if classification_vars:
            n_vars = len(classification_vars)
            plt.figure(figsize=(5 * n_vars, 4))
            
            for idx, var_name in enumerate(classification_vars):
                plt.subplot(1, n_vars, idx + 1)
                
                var_idx = [i for i, name in enumerate(self.metrics_results.keys()) if name == var_name][0]
                
                pred_binary = (self.predictions[:, var_idx] > 0.5).astype(int)
                true_binary = (self.targets[:, var_idx] > 0.5).astype(int)
                
                cm = confusion_matrix(true_binary, pred_binary)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Предсказанные')
                plt.ylabel('Реальные')
                plt.title(f'{var_name}\nAccuracy={self.metrics_results[var_name]["accuracy"]:.3f}')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrices.png', dpi=300)
            plt.close()
        
        # 4. Сводная таблица метрик
        plt.figure(figsize=(12, 8))
        
        # Подготовка данных для таблицы
        table_data = []
        for name, metrics in self.metrics_results.items():
            if metrics['type'] == 'regression':
                table_data.append([
                    name,
                    'Regression',
                    f"{metrics['mae']:.4f}",
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['r2']:.4f}",
                    '-'
                ])
            else:
                table_data.append([
                    name,
                    'Classification',
                    '-',
                    '-',
                    '-',
                    f"{metrics['accuracy']:.4f}"
                ])
        
        df_metrics = pd.DataFrame(table_data, 
                                 columns=['Variable', 'Type', 'MAE', 'RMSE', 'R²', 'Accuracy'])
        
        # Создаем таблицу
        ax = plt.subplot(111)
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df_metrics.values,
                        colLabels=df_metrics.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Сводная таблица метрик модели', fontsize=14, pad=20)
        plt.savefig(output_dir / 'metrics_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Визуализации сохранены в {output_dir}")
        
    def save_results(self):
        """Сохранение результатов оценки"""
        output_dir = Path('experiments/evaluation_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Сохраняем метрики в JSON
        results = {
            'timestamp': timestamp,
            'model_path': 'models_saved/best_model.pth',
            'test_samples': len(self.predictions),
            'metrics': self.metrics_results,
            'summary': {
                'avg_mae_regression': np.mean([m['mae'] for m in self.metrics_results.values() if m['type'] == 'regression']),
                'avg_r2_regression': np.mean([m['r2'] for m in self.metrics_results.values() if m['type'] == 'regression']),
                'avg_accuracy_classification': np.mean([m['accuracy'] for m in self.metrics_results.values() if m['type'] == 'classification'])
            }
        }
        
        with open(output_dir / f'evaluation_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Сохраняем детальный отчет
        with open(output_dir / f'evaluation_report_{timestamp}.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ ПО ОЦЕНКЕ МОДЕЛИ PatchTST\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Дата оценки: {timestamp}\n")
            f.write(f"Модель: models_saved/best_model.pth\n")
            f.write(f"Количество тестовых примеров: {len(self.predictions)}\n\n")
            
            f.write("СВОДНЫЕ МЕТРИКИ:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Средний MAE (регрессия): {results['summary']['avg_mae_regression']:.4f}\n")
            f.write(f"Средний R² (регрессия): {results['summary']['avg_r2_regression']:.4f}\n")
            f.write(f"Средняя точность (классификация): {results['summary']['avg_accuracy_classification']:.4f}\n\n")
            
            f.write("ДЕТАЛЬНЫЕ МЕТРИКИ ПО ПЕРЕМЕННЫМ:\n")
            f.write("-" * 40 + "\n")
            
            for name, metrics in self.metrics_results.items():
                f.write(f"\n{name}:\n")
                for metric_name, value in metrics.items():
                    if metric_name != 'type':
                        f.write(f"  {metric_name}: {value:.4f}\n")
        
        logger.info(f"✅ Результаты сохранены в {output_dir}")
        
    def run_full_evaluation(self):
        """Запуск полной оценки модели"""
        logger.info("🚀 Запуск полной оценки модели...")
        
        # 1. Загрузка модели
        self.load_model()
        
        # 2. Загрузка данных
        self.load_data()
        
        # 3. Получение предсказаний
        self.evaluate_predictions()
        
        # 4. Расчет метрик
        self.calculate_metrics()
        
        # 5. Создание визуализаций
        self.create_visualizations()
        
        # 6. Сохранение результатов
        self.save_results()
        
        logger.info("✅ Оценка модели завершена успешно!")
        

def main():
    """Основная функция"""
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()