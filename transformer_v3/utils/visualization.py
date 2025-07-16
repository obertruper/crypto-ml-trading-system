"""
Визуализация для Transformer v3
Адаптировано из VisualizationCallback в train_universal_transformer.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tensorflow import keras
import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')

from config import VISUALIZATION_PARAMS

logger = logging.getLogger(__name__)


class VisualizationCallback(keras.callbacks.Callback):
    """Callback для визуализации процесса обучения"""
    
    def __init__(self, log_dir: Path, model_name: str, 
                 update_freq: int = 5, task: str = 'regression'):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        self.update_freq = update_freq
        self.task = task
        self.epoch_count = 0
        
        # Создаем директорию для графиков
        self.plots_dir = self.log_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # История обучения
        self.history = {
            'loss': [],
            'val_loss': [],
            'lr': []
        }
        
        if task == 'regression':
            self.history.update({
                'mae': [],
                'val_mae': [],
                'rmse': [],
                'val_rmse': []
            })
        else:  # classification
            self.history.update({
                'accuracy': [],
                'val_accuracy': [],
                'precision': [],
                'val_precision': [],
                'recall': [],
                'val_recall': [],
                'auc': [],
                'val_auc': []
            })
        
        # Настройка стиля
        plt.style.use(VISUALIZATION_PARAMS['style'])
        
        # Создаем фигуру
        self.fig, self.axes = plt.subplots(2, 2, figsize=VISUALIZATION_PARAMS['figure_size'])
        self.fig.suptitle(f'Training Progress: {model_name}', fontsize=16)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        
        # Сохраняем метрики
        for key in self.history:
            if key == 'lr':
                # Learning rate
                lr = self.model.optimizer.learning_rate
                if hasattr(lr, 'numpy'):
                    self.history['lr'].append(lr.numpy())
                else:
                    self.history['lr'].append(float(lr))
            else:
                # Остальные метрики из logs
                value = logs.get(key, 0)
                self.history[key].append(value)
        
        # Сохраняем в CSV
        self._save_metrics_csv()
        
        # Обновляем графики
        if self.epoch_count % self.update_freq == 0:
            self.update_plots()
    
    def _save_metrics_csv(self):
        """Сохранение метрик в CSV"""
        metrics_df = pd.DataFrame(self.history)
        metrics_path = self.log_dir / f'{self.model_name}_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
    
    def update_plots(self):
        """Обновление графиков"""
        epochs = range(1, len(self.history['loss']) + 1)
        
        # Очищаем оси
        for ax in self.axes.flat:
            ax.clear()
        
        # График 1: Loss
        self.axes[0, 0].plot(epochs, self.history['loss'], 'b-', label='Train Loss', linewidth=2)
        self.axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        self.axes[0, 0].set_title('Model Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Основная метрика
        if self.task == 'regression':
            self._plot_regression_metric(self.axes[0, 1], epochs)
        else:
            self._plot_classification_metric(self.axes[0, 1], epochs)
        
        # График 3: Learning Rate
        self.axes[1, 0].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        self.axes[1, 0].set_title('Learning Rate Schedule')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 0].set_yscale('log')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Статистика
        self._plot_statistics(self.axes[1, 1])
        
        plt.tight_layout()
        
        # Сохраняем
        progress_path = self.plots_dir / 'training_progress.png'
        plt.savefig(progress_path, dpi=VISUALIZATION_PARAMS['save_dpi'], bbox_inches='tight')
        
        # Сохраняем snapshot эпохи
        epoch_path = self.plots_dir / f'epoch_{self.epoch_count:03d}.png'
        plt.savefig(epoch_path, dpi=100)
        
        logger.info(f"📊 График обновлен: эпоха {self.epoch_count}")
    
    def _plot_regression_metric(self, ax, epochs):
        """График для регрессии"""
        ax.plot(epochs, self.history['mae'], 'b-', label='Train MAE', linewidth=2)
        ax.plot(epochs, self.history['val_mae'], 'r-', label='Val MAE', linewidth=2)
        ax.set_title('Mean Absolute Error (%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_classification_metric(self, ax, epochs):
        """График для классификации"""
        ax.plot(epochs, self.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        ax.plot(epochs, self.history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        
        if 'auc' in self.history:
            ax.plot(epochs, self.history['auc'], 'b--', label='Train AUC', linewidth=1)
            ax.plot(epochs, self.history['val_auc'], 'r--', label='Val AUC', linewidth=1)
        
        ax.set_title('Model Performance')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_statistics(self, ax):
        """График статистики"""
        ax.axis('off')
        
        if self.task == 'regression':
            stats_text = self._get_regression_stats()
        else:
            stats_text = self._get_classification_stats()
        
        ax.text(0.1, 0.5, stats_text, fontsize=12,
               verticalalignment='center', family='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
    
    def _get_regression_stats(self) -> str:
        """Статистика для регрессии"""
        current_epoch = len(self.history['loss'])
        
        # Находим лучшую эпоху
        best_epoch = np.argmin(self.history['val_loss']) + 1
        
        stats = f"""
Current Epoch: {current_epoch}

Loss:
  Train:  {self.history['loss'][-1]:.4f}
  Val:    {self.history['val_loss'][-1]:.4f}
  
MAE (%):
  Train:  {self.history['mae'][-1]:.3f}%
  Val:    {self.history['val_mae'][-1]:.3f}%

RMSE (%):
  Train:  {self.history['rmse'][-1]:.3f}%
  Val:    {self.history['val_rmse'][-1]:.3f}%

Best Results:
  Val Loss: {min(self.history['val_loss']):.4f}
  Val MAE:  {min(self.history['val_mae']):.3f}%
  Epoch:    {best_epoch}

Learning Rate: {self.history['lr'][-1]:.2e}
"""
        return stats
    
    def _get_classification_stats(self) -> str:
        """Статистика для классификации"""
        current_epoch = len(self.history['loss'])
        
        # Находим лучшую эпоху
        best_epoch = np.argmax(self.history['val_accuracy']) + 1 if self.history['val_accuracy'] else 0
        
        stats = f"""
Current Epoch: {current_epoch}

Loss:
  Train:  {self.history['loss'][-1]:.4f}
  Val:    {self.history['val_loss'][-1]:.4f}
  
Accuracy:
  Train:  {self.history['accuracy'][-1]:.3f}
  Val:    {self.history['val_accuracy'][-1]:.3f}

Precision:
  Train:  {self.history.get('precision', [0])[-1]:.3f}
  Val:    {self.history.get('val_precision', [0])[-1]:.3f}
  
Recall:
  Train:  {self.history.get('recall', [0])[-1]:.3f}
  Val:    {self.history.get('val_recall', [0])[-1]:.3f}

AUC:
  Train:  {self.history.get('auc', [0])[-1]:.3f}
  Val:    {self.history.get('val_auc', [0])[-1]:.3f}

Best Results:
  Val Loss: {min(self.history['val_loss']):.4f}
  Val Acc:  {max(self.history['val_accuracy']) if self.history['val_accuracy'] else 0:.3f}
  Epoch:    {best_epoch}

Learning Rate: {self.history['lr'][-1]:.2e}
"""
        return stats
    
    def on_train_end(self, logs=None):
        """Финальное сохранение графиков"""
        self.update_plots()
        plt.close(self.fig)
        logger.info("📊 Визуализация обучения завершена")


def plot_training_history(history: Dict, save_path: Path, model_name: str = "Model"):
    """
    Построение графиков истории обучения
    
    Args:
        history: История обучения (keras history.history)
        save_path: Путь для сохранения
        model_name: Название модели
    """
    # Определяем количество эпох
    epochs = range(1, len(history['loss']) + 1)
    
    # Создаем фигуру
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training History: {model_name}', fontsize=14)
    
    # Loss
    axes[0, 0].plot(epochs, history['loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Основная метрика
    if 'mae' in history:
        # Регрессия
        axes[0, 1].plot(epochs, history['mae'], 'b-', label='Train MAE')
        axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_ylabel('MAE')
    elif 'accuracy' in history:
        # Классификация
        axes[0, 1].plot(epochs, history['accuracy'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Дополнительные метрики
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 График истории сохранен: {save_path}")