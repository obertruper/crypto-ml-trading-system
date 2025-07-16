"""
Визуализация результатов обучения и анализа
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import confusion_matrix, roc_curve, auc

logger = logging.getLogger(__name__)

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Класс для создания визуализаций"""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
    def plot_training_history(self, history: Dict[str, List[float]], 
                            model_name: str = "model",
                            save: bool = True) -> plt.Figure:
        """График истории обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax = axes[0]
        if 'train' in history and 'metric' in history['train']:
            ax.plot(history['train']['metric'], label='Train', alpha=0.8)
        if 'val' in history and 'metric' in history['val']:
            ax.plot(history['val']['metric'], label='Validation', alpha=0.8)
            
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_name} - Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning curves
        ax = axes[1]
        if 'train' in history and 'metric' in history['train']:
            train_metric = history['train']['metric']
            val_metric = history['val']['metric'] if 'val' in history else None
            
            # Скользящее среднее для сглаживания
            window = min(50, len(train_metric) // 10)
            if window > 1:
                train_smooth = pd.Series(train_metric).rolling(window).mean()
                ax.plot(train_smooth, label='Train (smoothed)', linewidth=2)
                
                if val_metric:
                    val_smooth = pd.Series(val_metric).rolling(window).mean()
                    ax.plot(val_smooth, label='Validation (smoothed)', linewidth=2)
                    
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Metric')
        ax.set_title(f'{model_name} - Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save and self.save_dir:
            save_path = self.save_dir / f"{model_name}_training_history.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 График сохранен: {save_path}")
            
        return fig
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "model",
                            normalize: bool = True,
                            save: bool = True) -> plt.Figure:
        """Матрица ошибок"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', square=True, cbar=True,
                   xticklabels=['Не входить', 'Входить'],
                   yticklabels=['Не входить', 'Входить'])
        
        ax.set_xlabel('Предсказано')
        ax.set_ylabel('Фактически')
        ax.set_title(f'{model_name} - Confusion Matrix' + 
                    (' (Normalized)' if normalize else ''))
        
        if save and self.save_dir:
            save_path = self.save_dir / f"{model_name}_confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 График сохранен: {save_path}")
            
        return fig
        
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "model",
                      save: bool = True) -> plt.Figure:
        """ROC кривая"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Находим оптимальную точку
        gmean = np.sqrt(tpr * (1 - fpr))
        optimal_idx = np.argmax(gmean)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # ROC кривая
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Диагональ
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random classifier')
        
        # Оптимальная точка
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                  color='red', s=100, zorder=5,
                  label=f'Optimal threshold = {thresholds[optimal_idx]:.3f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save and self.save_dir:
            save_path = self.save_dir / f"{model_name}_roc_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 График сохранен: {save_path}")
            
        return fig
        
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                              top_n: int = 20,
                              model_name: str = "model",
                              save: bool = True) -> plt.Figure:
        """График важности признаков"""
        # Берем топ N признаков
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Горизонтальная гистограмма
        bars = ax.barh(top_features['feature'], top_features['importance'])
        
        # Цветовая карта
        colors = plt.cm.viridis(top_features['importance'] / top_features['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{model_name} - Top {top_n} Feature Importance')
        ax.invert_yaxis()
        
        # Добавляем значения на бары
        for i, (feature, importance) in enumerate(zip(top_features['feature'], 
                                                     top_features['importance'])):
            ax.text(importance, i, f' {importance:.0f}', 
                   va='center', fontsize=9)
            
        plt.tight_layout()
        
        if save and self.save_dir:
            save_path = self.save_dir / f"{model_name}_feature_importance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 График сохранен: {save_path}")
            
        return fig
        
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str = "model",
                                   task_type: str = "classification",
                                   save: bool = True) -> plt.Figure:
        """Распределение предсказаний"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if task_type == "classification":
            # Для классификации - распределение вероятностей
            ax = axes[0, 0]
            ax.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='Class 0', density=True)
            ax.hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Class 1', density=True)
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Density')
            ax.set_title('Probability Distribution by Class')
            ax.legend()
            
            # Калибровочный график
            ax = axes[0, 1]
            self._plot_calibration_curve(ax, y_true, y_pred)
            
        else:
            # Для регрессии - scatter plot
            ax = axes[0, 0]
            ax.scatter(y_true, y_pred, alpha=0.5, s=10)
            ax.plot([y_true.min(), y_true.max()], 
                   [y_true.min(), y_true.max()], 
                   'r--', lw=2)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title('Predictions vs True Values')
            
            # Распределение ошибок
            ax = axes[0, 1]
            errors = y_pred - y_true
            ax.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Frequency')
            ax.set_title('Error Distribution')
            
        # QQ-plot для проверки нормальности ошибок
        ax = axes[1, 0]
        if task_type == "regression":
            from scipy import stats
            stats.probplot(y_pred - y_true, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot of Errors')
        else:
            # Для классификации - показываем пороговые метрики
            self._plot_threshold_metrics(ax, y_true, y_pred)
            
        # Статистика предсказаний
        ax = axes[1, 1]
        ax.axis('off')
        
        if task_type == "classification":
            stats_text = self._get_classification_stats(y_true, y_pred)
        else:
            stats_text = self._get_regression_stats(y_true, y_pred)
            
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{model_name} - Prediction Analysis', fontsize=16)
        plt.tight_layout()
        
        if save and self.save_dir:
            save_path = self.save_dir / f"{model_name}_prediction_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 График сохранен: {save_path}")
            
        return fig
        
    def plot_ensemble_comparison(self, model_metrics: List[Dict],
                               save: bool = True) -> plt.Figure:
        """Сравнение моделей в ансамбле"""
        metrics_df = pd.DataFrame(model_metrics)
        
        # Выбираем ключевые метрики для визуализации
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        if not available_metrics:
            logger.warning("Нет доступных метрик для визуализации")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Групповая гистограмма
        x = np.arange(len(metrics_df))
        width = 0.15
        
        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics)/2) * width
            ax.bar(x + offset, metrics_df[metric], width, 
                  label=metric.upper(), alpha=0.8)
            
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Ensemble Models Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Model {i+1}' for i in range(len(metrics_df))])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем среднее значение
        for i, metric in enumerate(available_metrics):
            mean_val = metrics_df[metric].mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', 
                      alpha=0.5, label=f'Mean {metric.upper()}' if i == 0 else "")
            
        plt.tight_layout()
        
        if save and self.save_dir:
            save_path = self.save_dir / "ensemble_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 График сохранен: {save_path}")
            
        return fig
        
    def _plot_calibration_curve(self, ax, y_true, y_pred_proba):
        """Калибровочная кривая"""
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10, strategy='uniform'
        )
        
        ax.plot(mean_predicted_value, fraction_of_positives, 
               marker='o', linewidth=1, label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_threshold_metrics(self, ax, y_true, y_pred_proba):
        """Метрики в зависимости от порога"""
        thresholds = np.linspace(0, 1, 100)
        metrics = {'precision': [], 'recall': [], 'f1': []}
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
            
        ax.plot(thresholds, metrics['precision'], label='Precision')
        ax.plot(thresholds, metrics['recall'], label='Recall')
        ax.plot(thresholds, metrics['f1'], label='F1-Score')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Metrics vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _get_classification_stats(self, y_true, y_pred_proba):
        """Статистика для классификации"""
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        stats = f"""Classification Statistics:
        
Total samples: {len(y_true):,}
Positive class: {(y_true == 1).sum():,} ({(y_true == 1).mean()*100:.1f}%)
Predicted positive: {(y_pred == 1).sum():,} ({(y_pred == 1).mean()*100:.1f}%)

Probability Statistics:
Mean: {y_pred_proba.mean():.3f}
Std: {y_pred_proba.std():.3f}
Min: {y_pred_proba.min():.3f}
Max: {y_pred_proba.max():.3f}

Performance:
Accuracy: {accuracy_score(y_true, y_pred):.3f}
Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}
Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}"""
        
        return stats
        
    def _get_regression_stats(self, y_true, y_pred):
        """Статистика для регрессии"""
        errors = y_pred - y_true
        
        from sklearn.metrics import mean_absolute_error, r2_score
        
        stats = f"""Regression Statistics:
        
Total samples: {len(y_true):,}

True Values:
Mean: {y_true.mean():.3f}
Std: {y_true.std():.3f}
Range: [{y_true.min():.3f}, {y_true.max():.3f}]

Predictions:
Mean: {y_pred.mean():.3f}
Std: {y_pred.std():.3f}
Range: [{y_pred.min():.3f}, {y_pred.max():.3f}]

Errors:
MAE: {mean_absolute_error(y_true, y_pred):.3f}
Mean Error: {errors.mean():.3f}
Std Error: {errors.std():.3f}
R²: {r2_score(y_true, y_pred):.3f}"""
        
        return stats
        
    def create_summary_report(self, figures: Dict[str, plt.Figure],
                            save_path: Optional[Path] = None):
        """Создание сводного PDF отчета"""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            
            if save_path is None and self.save_dir:
                save_path = self.save_dir / "summary_report.pdf"
                
            if save_path:
                with PdfPages(save_path) as pdf:
                    for name, fig in figures.items():
                        if fig is not None:
                            pdf.savefig(fig, bbox_inches='tight')
                            
                logger.info(f"📄 Сводный отчет сохранен: {save_path}")
        except Exception as e:
            logger.error(f"Ошибка при создании PDF отчета: {e}")


def plot_walk_forward_results(buy_splits: list, sell_splits: list, output_dir: str):
    """
    Визуализирует результаты walk-forward анализа.
    
    Args:
        buy_splits: Результаты для модели покупки
        sell_splits: Результаты для модели продажи
        output_dir: Директория для сохранения
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Извлекаем метрики по splits
    buy_metrics = {
        'roc_auc': [s['metrics']['roc_auc'] for s in buy_splits],
        'accuracy': [s['metrics']['accuracy'] for s in buy_splits],
        'precision': [s['metrics']['precision'] for s in buy_splits],
        'recall': [s['metrics']['recall'] for s in buy_splits]
    }
    
    sell_metrics = {
        'roc_auc': [s['metrics']['roc_auc'] for s in sell_splits],
        'accuracy': [s['metrics']['accuracy'] for s in sell_splits],
        'precision': [s['metrics']['precision'] for s in sell_splits],
        'recall': [s['metrics']['recall'] for s in sell_splits]
    }
    
    splits_x = range(1, len(buy_splits) + 1)
    
    # ROC-AUC
    ax = axes[0, 0]
    ax.plot(splits_x, buy_metrics['roc_auc'], 'o-', label='Buy', color='green', linewidth=2)
    ax.plot(splits_x, sell_metrics['roc_auc'], 's-', label='Sell', color='red', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Split')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('ROC-AUC по Walk-Forward Splits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 0.8])
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(splits_x, buy_metrics['accuracy'], 'o-', label='Buy', color='green', linewidth=2)
    ax.plot(splits_x, sell_metrics['accuracy'], 's-', label='Sell', color='red', linewidth=2)
    ax.set_xlabel('Split')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy по Walk-Forward Splits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision
    ax = axes[1, 0]
    ax.plot(splits_x, buy_metrics['precision'], 'o-', label='Buy', color='green', linewidth=2)
    ax.plot(splits_x, sell_metrics['precision'], 's-', label='Sell', color='red', linewidth=2)
    ax.set_xlabel('Split')
    ax.set_ylabel('Precision')
    ax.set_title('Precision по Walk-Forward Splits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes[1, 1]
    ax.plot(splits_x, buy_metrics['recall'], 'o-', label='Buy', color='green', linewidth=2)
    ax.plot(splits_x, sell_metrics['recall'], 's-', label='Sell', color='red', linewidth=2)
    ax.set_xlabel('Split')
    ax.set_ylabel('Recall')
    ax.set_title('Recall по Walk-Forward Splits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Walk-Forward Analysis Results', fontsize=16)
    plt.tight_layout()
    
    save_path = Path(output_dir) / 'walk_forward_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ График walk-forward результатов сохранен: {save_path}")


def plot_feature_importance(buy_features: list, sell_features: list, output_dir: str):
    """
    Визуализирует важность признаков для buy и sell моделей.
    
    Args:
        buy_features: Топ признаки для buy модели
        sell_features: Топ признаки для sell модели
        output_dir: Директория для сохранения
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Buy features
    y_pos = np.arange(len(buy_features))
    ax1.barh(y_pos, range(len(buy_features), 0, -1), color='green', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(buy_features, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Относительная важность')
    ax1.set_title('Топ признаки - Модель BUY')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Sell features
    y_pos = np.arange(len(sell_features))
    ax2.barh(y_pos, range(len(sell_features), 0, -1), color='red', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sell_features, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Относительная важность')
    ax2.set_title('Топ признаки - Модель SELL')
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.suptitle('Важность признаков в моделях направления', fontsize=14)
    plt.tight_layout()
    
    save_path = Path(output_dir) / 'feature_importance_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ График важности признаков сохранен: {save_path}")