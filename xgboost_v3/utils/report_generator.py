"""
Генератор отчетов о результатах обучения
"""

import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Класс для генерации отчетов"""
    
    def __init__(self, config, log_dir: Path):
        self.config = config
        self.log_dir = Path(log_dir)
        
    def generate_report(self, results: Dict[str, Any]):
        """Генерация финального отчета"""
        logger.info("📝 Генерация отчета...")
        
        report_path = self.log_dir / "final_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write("="*60 + "\n")
            f.write("XGBoost v3.0 - ОТЧЕТ О РЕЗУЛЬТАТАХ ОБУЧЕНИЯ\n")
            f.write("="*60 + "\n\n")
            
            # Время
            f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Конфигурация: {self.config.training.task_type}\n")
            f.write(f"Режим: {'Тест' if self.config.training.test_mode else 'Полный'}\n\n")
            
            # Данные
            f.write("ДАННЫЕ:\n")
            f.write("-"*40 + "\n")
            f.write(f"Количество записей: {results.get('n_samples', 0):,}\n")
            f.write(f"Количество признаков: {results.get('n_features', 0)}\n")
            f.write(f"Порог классификации: {self.config.training.classification_threshold}%\n\n")
            
            # Метрики BUY
            f.write("РЕЗУЛЬТАТЫ МОДЕЛИ BUY:\n")
            f.write("-"*40 + "\n")
            self._write_metrics(f, results.get('buy', {}))
            
            # Метрики SELL
            f.write("\nРЕЗУЛЬТАТЫ МОДЕЛИ SELL:\n")
            f.write("-"*40 + "\n")
            self._write_metrics(f, results.get('sell', {}))
            
            # Топ признаки
            if 'feature_names' in results:
                f.write("\nТОП-20 ПРИЗНАКОВ:\n")
                f.write("-"*40 + "\n")
                for i, feat in enumerate(results['feature_names'][:20], 1):
                    f.write(f"{i:2d}. {feat}\n")
                    
            # Анализ важности по категориям
            if 'feature_importance_analysis' in results:
                f.write("\nАНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ПО КАТЕГОРИЯМ:\n")
                f.write("-"*40 + "\n")
                analysis = results['feature_importance_analysis']
                for category, info in analysis.items():
                    f.write(f"\n{category}:\n")
                    f.write(f"  Количество: {info.get('count', 0)} ({info.get('percentage', 0):.1f}%)\n")
                    f.write(f"  Суммарная важность: {info.get('total_importance', 0):.3f}\n")
                    f.write(f"  Топ признаки: {', '.join(info.get('top_features', [])[:5])}\n")
                    
            # Параметры модели
            f.write("\nПАРАМЕТРЫ МОДЕЛИ:\n")
            f.write("-"*40 + "\n")
            f.write(f"Max Depth: {self.config.model.max_depth}\n")
            f.write(f"Learning Rate: {self.config.model.learning_rate}\n")
            f.write(f"N Estimators: {self.config.model.n_estimators}\n")
            f.write(f"Ensemble Size: {self.config.training.ensemble_size}\n")
            
            f.write("\n" + "="*60 + "\n")
            
        # Сохраняем метрики в JSON
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"✅ Отчет сохранен: {report_path}")
        logger.info(f"📊 Метрики сохранены: {metrics_path}")
        
    def _write_metrics(self, f, metrics: Dict[str, float]):
        """Запись метрик в файл"""
        if self.config.training.task_type == "regression":
            f.write(f"MAE: {metrics.get('mae', 0):.4f}\n")
            f.write(f"RMSE: {metrics.get('rmse', 0):.4f}\n")
            f.write(f"R²: {metrics.get('r2', 0):.4f}\n")
            f.write(f"Direction Accuracy: {metrics.get('direction_accuracy', 0):.1%}\n")
        else:
            f.write(f"ROC-AUC: {metrics.get('roc_auc', 0):.4f}\n")
            f.write(f"Accuracy: {metrics.get('accuracy', 0):.1%}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.1%}\n")
            f.write(f"Recall: {metrics.get('recall', 0):.1%}\n")
            f.write(f"F1-Score: {metrics.get('f1', 0):.3f}\n")
            
            if 'total_trades' in metrics:
                f.write(f"\nТрейдинг метрики:\n")
                f.write(f"Всего сделок: {metrics.get('total_trades', 0)}\n")
                f.write(f"Win Rate: {metrics.get('win_rate', 0):.1%}\n")