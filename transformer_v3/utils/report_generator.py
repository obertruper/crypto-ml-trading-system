"""
Генератор отчетов для Transformer v3
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json
import pandas as pd
import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Генератор итоговых отчетов"""
    
    def __init__(self, config: Config, log_dir: Path):
        self.config = config
        self.log_dir = Path(log_dir)
        
    def generate_report(self, results: Dict[str, Any]):
        """
        Генерация полного отчета
        
        Args:
            results: Словарь с результатами обучения
        """
        logger.info("\n📝 Генерация итогового отчета...")
        
        # Текстовый отчет
        text_report = self._create_text_report(results)
        report_path = self.log_dir / 'final_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        # JSON отчет для программной обработки
        json_report = self._create_json_report(results)
        json_path = self.log_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # CSV с основными метриками
        if 'models' in results:
            metrics_df = self._create_metrics_dataframe(results['models'])
            csv_path = self.log_dir / 'model_metrics.csv'
            metrics_df.to_csv(csv_path, index=False)
        
        logger.info(f"✅ Отчеты сохранены в {self.log_dir}")
        
        # Выводим краткую сводку в консоль
        self._print_summary(results)
    
    def _create_text_report(self, results: Dict[str, Any]) -> str:
        """Создание текстового отчета"""
        report = []
        report.append("=" * 80)
        report.append("ИТОГОВЫЙ ОТЧЕТ ПО ОБУЧЕНИЮ TRANSFORMER V3")
        report.append("=" * 80)
        report.append("")
        
        # Общая информация
        report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Лог директория: {self.log_dir}")
        report.append(f"Тип задачи: {self.config.training.task_type.upper()}")
        report.append("")
        
        # Архитектура
        report.append("АРХИТЕКТУРА:")
        report.append(f"- Модель: Temporal Fusion Transformer (TFT)")
        report.append(f"- Sequence Length: {self.config.model.sequence_length}")
        report.append(f"- Model Dimension: {self.config.model.d_model}")
        report.append(f"- Number of Heads: {self.config.model.num_heads}")
        report.append(f"- Transformer Blocks: {self.config.model.num_transformer_blocks}")
        report.append(f"- Dropout Rate: {self.config.model.dropout_rate}")
        report.append("")
        
        # Параметры обучения
        report.append("ПАРАМЕТРЫ ОБУЧЕНИЯ:")
        report.append(f"- Batch Size: {self.config.training.batch_size}")
        report.append(f"- Learning Rate: {self.config.training.learning_rate}")
        report.append(f"- Optimizer: {self.config.training.optimizer}")
        report.append(f"- Loss Function: {self.config.training.loss_function}")
        report.append(f"- Early Stopping Patience: {self.config.training.early_stopping_patience}")
        report.append("")
        
        # Данные
        if 'data_info' in results:
            report.append("ИНФОРМАЦИЯ О ДАННЫХ:")
            info = results['data_info']
            report.append(f"- Общее количество записей: {info.get('total_records', 'N/A'):,}")
            report.append(f"- Количество символов: {info.get('n_symbols', 'N/A')}")
            report.append(f"- Количество признаков: {info.get('n_features', 'N/A')}")
            report.append(f"- Размер обучающей выборки: {info.get('train_size', 'N/A'):,}")
            report.append(f"- Размер валидационной выборки: {info.get('val_size', 'N/A'):,}")
            report.append(f"- Размер тестовой выборки: {info.get('test_size', 'N/A'):,}")
            report.append("")
        
        # Результаты моделей
        if 'models' in results:
            report.append("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
            for model_name, model_results in results['models'].items():
                report.append(f"\n{model_name.upper()}:")
                
                if 'test_metrics' in model_results:
                    metrics = model_results['test_metrics']
                    
                    if self.config.training.task_type == 'regression':
                        report.append(f"- MAE: {metrics.get('mae', 'N/A'):.4f}%")
                        report.append(f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}%")
                        report.append(f"- R²: {metrics.get('r2', 'N/A'):.4f}")
                        report.append(f"- Direction Accuracy: {metrics.get('direction_accuracy', 'N/A'):.2%}")
                    else:
                        report.append(f"- Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                        report.append(f"- Precision: {metrics.get('precision', 'N/A'):.4f}")
                        report.append(f"- Recall: {metrics.get('recall', 'N/A'):.4f}")
                        report.append(f"- F1-Score: {metrics.get('f1', 'N/A'):.4f}")
                        report.append(f"- AUC: {metrics.get('auc', 'N/A'):.4f}")
                
                if 'best_epoch' in model_results:
                    report.append(f"- Best Epoch: {model_results['best_epoch']}")
        
        report.append("")
        report.append("=" * 80)
        report.append("ФАЙЛЫ И АРТЕФАКТЫ:")
        report.append(f"- Модели: {self.log_dir}/models/")
        report.append(f"- Графики: {self.log_dir}/plots/")
        report.append(f"- Метрики: {self.log_dir}/*_metrics.csv")
        report.append(f"- TensorBoard: tensorboard --logdir {self.log_dir}/tensorboard/")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _create_json_report(self, results: Dict[str, Any]) -> Dict:
        """Создание JSON отчета"""
        json_report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__
            },
            'results': results
        }
        
        # Преобразуем numpy типы и объекты в обычные Python типы
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif hasattr(obj, '__dict__'):
                # Для объектов классов - берем только их атрибуты
                return {k: convert_types(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            elif hasattr(obj, '__class__') and obj.__class__.__name__ in ['TFTEnsemble', 'TFTTrainer']:
                # Для специфичных объектов возвращаем только имя класса
                return f"<{obj.__class__.__name__} object>"
            else:
                return obj
        
        return convert_types(json_report)
    
    def _create_metrics_dataframe(self, models: Dict) -> pd.DataFrame:
        """Создание DataFrame с метриками всех моделей"""
        rows = []
        
        for model_name, model_results in models.items():
            if 'test_metrics' in model_results:
                row = {'model': model_name}
                row.update(model_results['test_metrics'])
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Вывод краткой сводки в консоль"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 КРАТКАЯ СВОДКА РЕЗУЛЬТАТОВ")
        logger.info("=" * 60)
        
        if 'models' in results:
            for model_name, model_results in results['models'].items():
                if 'test_metrics' in model_results:
                    metrics = model_results['test_metrics']
                    
                    logger.info(f"\n{model_name}:")
                    if self.config.training.task_type == 'regression':
                        logger.info(f"  MAE: {metrics.get('mae', 'N/A'):.3f}%")
                        logger.info(f"  Direction Accuracy: {metrics.get('direction_accuracy', 'N/A'):.1%}")
                    else:
                        logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                        logger.info(f"  F1-Score: {metrics.get('f1', 'N/A'):.3f}")
        
        logger.info("\n" + "=" * 60)