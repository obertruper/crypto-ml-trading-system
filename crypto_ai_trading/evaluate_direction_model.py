"""
Скрипт для детальной оценки обученной Direction модели
Включает confusion matrix, метрики по классам и визуализацию
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from models.direction_predictor import DirectionPredictor
from train_direction_model import DirectionDatasetAdapter
from data.data_loader import CryptoDataLoader
from utils.logger import get_logger, setup_logging
from utils.config import load_config


class DirectionModelEvaluator:
    """Детальная оценка Direction модели"""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device = None):
        self.model = model
        self.config = config
        self.logger = get_logger("DirectionEvaluator")
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        
        # Маппинг классов
        self.class_names = ['UP', 'DOWN', 'FLAT']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
    def evaluate(self, dataloader: DataLoader, 
                dataset_name: str = 'test') -> Dict:
        """Полная оценка модели на датасете"""
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Оценка модели на {dataset_name} датасете")
        self.logger.info(f"{'='*60}")
        
        all_predictions = {
            '15m': {'pred': [], 'true': [], 'probs': []},
            '1h': {'pred': [], 'true': [], 'probs': []},
            '4h': {'pred': [], 'true': [], 'probs': []},
            '12h': {'pred': [], 'true': [], 'probs': []}
        }
        
        all_profits = []
        all_symbols = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                
                # Получаем предсказания
                outputs = self.model(inputs)
                
                # Обрабатываем каждый таймфрейм
                for timeframe in ['15m', '1h', '4h', '12h']:
                    key = f'direction_{timeframe}'
                    if key in outputs and key in targets:
                        logits = outputs[key]
                        true_labels = targets[key].to(self.device).squeeze()
                        
                        # Softmax для вероятностей
                        probs = torch.softmax(logits, dim=-1)
                        predictions = logits.argmax(dim=-1)
                        
                        # Сохраняем результаты
                        all_predictions[timeframe]['pred'].extend(predictions.cpu().numpy())
                        all_predictions[timeframe]['true'].extend(true_labels.cpu().numpy())
                        all_predictions[timeframe]['probs'].extend(probs.cpu().numpy())
                
                # Расчет прибыли (для основного таймфрейма 4h)
                if 'price_changes' in info and '4h' in info['price_changes']:
                    price_changes = info['price_changes']['4h'].squeeze()
                    predictions_4h = outputs['direction_4h'].argmax(dim=-1)
                    
                    # Простой расчет P&L
                    profits = self._calculate_profits(predictions_4h, price_changes)
                    all_profits.extend(profits.cpu().numpy())
                
                # Символы для анализа по инструментам
                if 'symbol' in info:
                    all_symbols.extend(info['symbol'])
        
        # Анализ результатов
        results = {}
        
        for timeframe in ['15m', '1h', '4h', '12h']:
            if len(all_predictions[timeframe]['pred']) > 0:
                results[timeframe] = self._analyze_predictions(
                    np.array(all_predictions[timeframe]['pred']),
                    np.array(all_predictions[timeframe]['true']),
                    np.array(all_predictions[timeframe]['probs']),
                    timeframe
                )
        
        # Торговый анализ
        if all_profits:
            results['trading'] = self._analyze_trading_performance(
                np.array(all_profits),
                all_symbols
            )
        
        # Общая статистика
        results['summary'] = self._create_summary(results)
        
        return results
    
    def _analyze_predictions(self, predictions: np.ndarray, 
                           true_labels: np.ndarray,
                           probabilities: np.ndarray,
                           timeframe: str) -> Dict:
        """Анализ предсказаний для таймфрейма"""
        
        self.logger.info(f"\n📊 Анализ для {timeframe}:")
        
        # Базовые метрики
        accuracy = accuracy_score(true_labels, predictions)
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Детальный отчет по классам
        report = classification_report(
            true_labels, predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Directional accuracy (без FLAT)
        non_flat_mask = true_labels != 2
        if non_flat_mask.sum() > 0:
            directional_accuracy = accuracy_score(
                true_labels[non_flat_mask],
                predictions[non_flat_mask]
            )
        else:
            directional_accuracy = 0.0
        
        # Confidence анализ
        confidence_stats = self._analyze_confidence(predictions, probabilities, true_labels)
        
        # Вывод результатов
        self.logger.info(f"  Overall Accuracy: {accuracy:.2%}")
        self.logger.info(f"  Directional Accuracy (UP/DOWN): {directional_accuracy:.2%}")
        
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                self.logger.info(f"  {class_name}: Precision={metrics['precision']:.2%}, "
                               f"Recall={metrics['recall']:.2%}, F1={metrics['f1-score']:.2%}")
        
        return {
            'accuracy': accuracy,
            'directional_accuracy': directional_accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'confidence_stats': confidence_stats
        }
    
    def _analyze_confidence(self, predictions: np.ndarray,
                          probabilities: np.ndarray,
                          true_labels: np.ndarray) -> Dict:
        """Анализ уверенности модели в предсказаниях"""
        
        # Максимальная вероятность как мера уверенности
        max_probs = probabilities.max(axis=1)
        
        # Статистика по уверенности
        confidence_stats = {
            'mean_confidence': max_probs.mean(),
            'std_confidence': max_probs.std(),
            'min_confidence': max_probs.min(),
            'max_confidence': max_probs.max()
        }
        
        # Уверенность для правильных и неправильных предсказаний
        correct_mask = predictions == true_labels
        confidence_stats['correct_confidence'] = max_probs[correct_mask].mean() if correct_mask.sum() > 0 else 0
        confidence_stats['incorrect_confidence'] = max_probs[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
        
        # Распределение уверенности по классам
        for i, class_name in enumerate(self.class_names):
            class_mask = predictions == i
            if class_mask.sum() > 0:
                confidence_stats[f'{class_name}_confidence'] = max_probs[class_mask].mean()
        
        return confidence_stats
    
    def _calculate_profits(self, predictions: torch.Tensor, 
                         price_changes: torch.Tensor) -> torch.Tensor:
        """Расчет прибыли от торговых решений"""
        
        commission = self.config['bybit']['fees']['taker']
        profits = torch.zeros_like(price_changes)
        
        # LONG позиции
        long_mask = predictions == 0
        profits[long_mask] = price_changes[long_mask] - commission
        
        # SHORT позиции
        short_mask = predictions == 1
        profits[short_mask] = -price_changes[short_mask] - commission
        
        # HOLD (нет торговли)
        # profits остаются 0
        
        return profits
    
    def _analyze_trading_performance(self, profits: np.ndarray,
                                   symbols: List[str]) -> Dict:
        """Анализ торговой производительности"""
        
        self.logger.info("\n💰 Торговый анализ:")
        
        # Общие метрики
        total_trades = (profits != 0).sum()
        winning_trades = (profits > 0).sum()
        losing_trades = (profits < 0).sum()
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_profit = profits[profits > 0].mean() if winning_trades > 0 else 0
            avg_loss = abs(profits[profits < 0].mean()) if losing_trades > 0 else 0
            profit_factor = (avg_profit * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
            
            # Sharpe Ratio (упрощенный)
            if profits.std() > 0:
                sharpe_ratio = (profits.mean() * 252) / (profits.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
            
            # Maximum Drawdown
            cumulative_returns = (1 + profits).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
        else:
            win_rate = avg_profit = avg_loss = profit_factor = sharpe_ratio = max_drawdown = 0
        
        # Анализ по символам
        symbol_performance = {}
        if symbols:
            df = pd.DataFrame({'symbol': symbols, 'profit': profits})
            symbol_stats = df.groupby('symbol')['profit'].agg(['mean', 'std', 'count'])
            
            for symbol in symbol_stats.index[:10]:  # Топ 10 символов
                stats = symbol_stats.loc[symbol]
                symbol_performance[symbol] = {
                    'avg_profit': stats['mean'],
                    'std': stats['std'],
                    'trades': int(stats['count'])
                }
        
        # Вывод результатов
        self.logger.info(f"  Total Trades: {total_trades}")
        self.logger.info(f"  Win Rate: {win_rate:.2%}")
        self.logger.info(f"  Average Win: {avg_profit:.4%}")
        self.logger.info(f"  Average Loss: {avg_loss:.4%}")
        self.logger.info(f"  Profit Factor: {profit_factor:.2f}")
        self.logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        self.logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return {
            'total_trades': int(total_trades),
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'symbol_performance': symbol_performance
        }
    
    def _create_summary(self, results: Dict) -> Dict:
        """Создание общего резюме"""
        
        # Основной таймфрейм для оценки
        main_tf = '4h'
        
        if main_tf in results:
            main_results = results[main_tf]
            
            # Определяем готовность к торговле
            is_profitable = (
                main_results['directional_accuracy'] > 0.55 and
                results.get('trading', {}).get('win_rate', 0) > 0.50 and
                results.get('trading', {}).get('profit_factor', 0) > 1.2
            )
            
            summary = {
                'main_timeframe': main_tf,
                'directional_accuracy': main_results['directional_accuracy'],
                'overall_accuracy': main_results['accuracy'],
                'is_profitable': is_profitable,
                'recommendation': 'READY for trading' if is_profitable else 'NOT ready for trading'
            }
            
            # Добавляем торговые метрики если есть
            if 'trading' in results:
                summary.update({
                    'win_rate': results['trading']['win_rate'],
                    'profit_factor': results['trading']['profit_factor'],
                    'sharpe_ratio': results['trading']['sharpe_ratio']
                })
        else:
            summary = {
                'error': 'No results for main timeframe',
                'is_profitable': False,
                'recommendation': 'NOT ready for trading'
            }
        
        return summary
    
    def visualize_results(self, results: Dict, save_path: Path):
        """Визуализация результатов оценки"""
        
        # Создаем фигуру с подграфиками
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrices для разных таймфреймов
        timeframes = ['15m', '1h', '4h', '12h']
        for i, tf in enumerate(timeframes):
            if tf in results:
                ax = plt.subplot(3, 4, i+1)
                cm = results[tf]['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=self.class_names,
                           yticklabels=self.class_names,
                           ax=ax)
                ax.set_title(f'Confusion Matrix - {tf}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
        
        # 2. Accuracy сравнение
        ax = plt.subplot(3, 4, 5)
        accuracies = {tf: results[tf]['accuracy'] for tf in timeframes if tf in results}
        dir_accuracies = {tf: results[tf]['directional_accuracy'] for tf in timeframes if tf in results}
        
        x = list(accuracies.keys())
        y1 = list(accuracies.values())
        y2 = list(dir_accuracies.values())
        
        x_pos = np.arange(len(x))
        width = 0.35
        
        ax.bar(x_pos - width/2, y1, width, label='Overall Accuracy', alpha=0.8)
        ax.bar(x_pos + width/2, y2, width, label='Directional Accuracy', alpha=0.8)
        ax.axhline(y=0.55, color='r', linestyle='--', label='Profitable Threshold')
        
        ax.set_xlabel('Timeframe')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Timeframe')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Class-wise Performance
        ax = plt.subplot(3, 4, 6)
        if '4h' in results:
            report = results['4h']['classification_report']
            
            classes = []
            precisions = []
            recalls = []
            f1_scores = []
            
            for class_name in self.class_names:
                if class_name in report:
                    classes.append(class_name)
                    precisions.append(report[class_name]['precision'])
                    recalls.append(report[class_name]['recall'])
                    f1_scores.append(report[class_name]['f1-score'])
            
            x = np.arange(len(classes))
            width = 0.25
            
            ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
            ax.bar(x, recalls, width, label='Recall', alpha=0.8)
            ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
            
            ax.set_xlabel('Class')
            ax.set_ylabel('Score')
            ax.set_title('Class-wise Performance (4h)')
            ax.set_xticks(x)
            ax.set_xticklabels(classes)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Trading Performance
        if 'trading' in results:
            # Win Rate и Profit Factor
            ax = plt.subplot(3, 4, 7)
            metrics = ['Win Rate', 'Profit Factor']
            values = [
                results['trading']['win_rate'],
                results['trading']['profit_factor'] / 3  # Нормализуем для визуализации
            ]
            thresholds = [0.5, 1.2 / 3]
            
            bars = ax.bar(metrics, values, alpha=0.8)
            
            # Пороговые линии
            ax.axhline(y=thresholds[0], color='r', linestyle='--', alpha=0.5)
            
            # Цвет баров в зависимости от порога
            for i, (bar, val, thresh) in enumerate(zip(bars, values, thresholds)):
                if val > thresh:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            ax.set_ylabel('Value')
            ax.set_title('Trading Metrics')
            ax.set_ylim(0, 1)
            
            # Аннотации с реальными значениями
            ax.text(0, values[0] + 0.02, f'{results["trading"]["win_rate"]:.2%}', 
                   ha='center', va='bottom')
            ax.text(1, values[1] + 0.02, f'{results["trading"]["profit_factor"]:.2f}', 
                   ha='center', va='bottom')
        
        # 5. Confidence Distribution
        ax = plt.subplot(3, 4, 8)
        if '4h' in results and 'confidence_stats' in results['4h']:
            conf_stats = results['4h']['confidence_stats']
            
            labels = ['Correct\nPredictions', 'Incorrect\nPredictions']
            values = [
                conf_stats.get('correct_confidence', 0),
                conf_stats.get('incorrect_confidence', 0)
            ]
            
            bars = ax.bar(labels, values, alpha=0.8)
            bars[0].set_color('green')
            bars[1].set_color('red')
            
            ax.set_ylabel('Average Confidence')
            ax.set_title('Model Confidence Analysis')
            ax.set_ylim(0, 1)
            
            # Аннотации
            for i, (label, value) in enumerate(zip(labels, values)):
                ax.text(i, value + 0.02, f'{value:.2%}', ha='center', va='bottom')
        
        # 6. Summary Text
        ax = plt.subplot(3, 1, 3)
        ax.axis('off')
        
        if 'summary' in results:
            summary = results['summary']
            
            summary_text = f"""
EVALUATION SUMMARY
==================

Main Timeframe: {summary.get('main_timeframe', 'N/A')}
Directional Accuracy: {summary.get('directional_accuracy', 0):.2%}
Overall Accuracy: {summary.get('overall_accuracy', 0):.2%}

TRADING METRICS:
Win Rate: {summary.get('win_rate', 0):.2%}
Profit Factor: {summary.get('profit_factor', 0):.2f}
Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}

STATUS: {summary.get('recommendation', 'Unknown')}

{'✅ Model is PROFITABLE and ready for trading!' if summary.get('is_profitable', False) 
 else '❌ Model needs more training to be profitable'}
"""
            
            ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                   fontsize=14, verticalalignment='center',
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   family='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path / 'evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Визуализация сохранена: {save_path / 'evaluation_results.png'}")
    
    def save_results(self, results: Dict, save_path: Path):
        """Сохранение результатов в файл"""
        
        # Конвертируем numpy arrays в списки для JSON
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        # Сохраняем JSON
        json_path = save_path / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"📄 Результаты сохранены: {json_path}")
        
        # Сохраняем текстовый отчет
        report_path = save_path / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("DIRECTION MODEL EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Summary
            if 'summary' in results:
                f.write("SUMMARY:\n")
                for key, value in results['summary'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Detailed results by timeframe
            for tf in ['15m', '1h', '4h', '12h']:
                if tf in results:
                    f.write(f"\n{tf} RESULTS:\n")
                    f.write("-"*30 + "\n")
                    f.write(f"  Accuracy: {results[tf]['accuracy']:.2%}\n")
                    f.write(f"  Directional Accuracy: {results[tf]['directional_accuracy']:.2%}\n")
                    
                    # Classification report
                    if 'classification_report' in results[tf]:
                        f.write("\n  Per-class metrics:\n")
                        for class_name in self.class_names:
                            if class_name in results[tf]['classification_report']:
                                metrics = results[tf]['classification_report'][class_name]
                                f.write(f"    {class_name}:\n")
                                f.write(f"      Precision: {metrics['precision']:.2%}\n")
                                f.write(f"      Recall: {metrics['recall']:.2%}\n")
                                f.write(f"      F1-Score: {metrics['f1-score']:.2%}\n")
            
            # Trading performance
            if 'trading' in results:
                f.write("\n\nTRADING PERFORMANCE:\n")
                f.write("-"*30 + "\n")
                trading = results['trading']
                f.write(f"  Total Trades: {trading['total_trades']}\n")
                f.write(f"  Win Rate: {trading['win_rate']:.2%}\n")
                f.write(f"  Average Win: {trading['avg_profit']:.4%}\n")
                f.write(f"  Average Loss: {trading['avg_loss']:.4%}\n")
                f.write(f"  Profit Factor: {trading['profit_factor']:.2f}\n")
                f.write(f"  Sharpe Ratio: {trading['sharpe_ratio']:.2f}\n")
                f.write(f"  Max Drawdown: {trading['max_drawdown']:.2%}\n")
        
        self.logger.info(f"📄 Отчет сохранен: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Direction Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset to evaluate on')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for evaluation')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Specific symbols to evaluate')
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    
    # Логирование
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/evaluation_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir, "evaluation")
    logger = get_logger("EvaluateDirection")
    
    logger.info("🎯 Начинаем оценку Direction модели")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Загрузка checkpoint
    logger.info("📂 Загрузка модели...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Извлекаем конфигурацию модели
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        model_config = config['model']
    
    # Создание модели
    model = DirectionPredictor(model_config)
    
    # Загрузка весов
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("✅ Модель загружена")
    
    # Информация о checkpoint
    if 'history' in checkpoint:
        history = checkpoint['history']
        logger.info(f"📊 История обучения:")
        logger.info(f"  - Эпох обучено: {len(history['train_loss'])}")
        logger.info(f"  - Лучший val_loss: {min(history['val_loss']):.4f}")
        if 'val_metrics' in history and history['val_metrics']:
            last_metrics = history['val_metrics'][-1]
            if 'directional_accuracy' in last_metrics:
                logger.info(f"  - Последняя directional accuracy: {last_metrics['directional_accuracy']:.2%}")
    
    # Загрузка данных
    logger.info(f"📊 Загрузка {args.dataset} датасета...")
    data_loader = CryptoDataLoader(config)
    
    if args.symbols:
        data = data_loader.load_data(symbols=args.symbols)
    else:
        # Используем те же символы что и при обучении
        data = data_loader.load_data()
    
    # Разделение данных
    train_data, val_data, test_data = data_loader.split_data(data)
    
    # Выбор нужного датасета
    if args.dataset == 'train':
        eval_data = train_data
    elif args.dataset == 'val':
        eval_data = val_data
    else:
        eval_data = test_data
    
    logger.info(f"✅ Загружено {len(eval_data)} записей")
    
    # Создание dataset и dataloader
    dataset = DirectionDatasetAdapter(
        eval_data,
        context_window=model_config.get('context_window', 168),
        feature_cols=data_loader.feature_columns,
        target_cols=[col for col in data_loader.target_columns if col.startswith('direction_')],
        stride=1  # Используем все данные для оценки
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluator
    evaluator = DirectionModelEvaluator(model, config)
    
    # Оценка
    logger.info("🔍 Начинаем оценку...")
    results = evaluator.evaluate(dataloader, args.dataset)
    
    # Визуализация
    logger.info("📊 Создание визуализаций...")
    evaluator.visualize_results(results, log_dir)
    
    # Сохранение результатов
    evaluator.save_results(results, log_dir)
    
    # Финальный вывод
    logger.info("\n" + "="*60)
    logger.info("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    logger.info("="*60)
    
    if 'summary' in results:
        summary = results['summary']
        logger.info(f"Directional Accuracy (4h): {summary.get('directional_accuracy', 0):.2%}")
        logger.info(f"Win Rate: {summary.get('win_rate', 0):.2%}")
        logger.info(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
        logger.info(f"\n🎯 {summary.get('recommendation', 'Unknown status')}")
        
        if summary.get('is_profitable', False):
            logger.info("✅ Модель готова к использованию!")
        else:
            logger.info("❌ Модель требует дополнительного обучения")
    
    logger.info(f"\n📁 Все результаты сохранены в: {log_dir}")


if __name__ == "__main__":
    main()