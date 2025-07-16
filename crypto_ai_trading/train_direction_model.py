"""
Скрипт для обучения специализированной модели предсказания направления
Фокус на максимизации directional accuracy и прибыльности
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models.direction_predictor import DirectionPredictor, DirectionalTradingLoss
from data.dataset import TimeSeriesDataset
from data.data_loader import CryptoDataLoader
from training.trainer import Trainer
from utils.logger import get_logger, setup_logging
from utils.config import load_config
from utils.metrics import MetricsCalculator


class DirectionDatasetAdapter(TimeSeriesDataset):
    """Адаптер для использования существующего датасета только с direction targets"""
    
    def __init__(self, *args, **kwargs):
        # Фильтруем только direction targets
        if 'target_cols' in kwargs:
            kwargs['target_cols'] = [col for col in kwargs['target_cols'] 
                                   if col.startswith('direction_')]
        
        # Передаем normalize параметр
        if 'normalize' not in kwargs:
            kwargs['normalize'] = False
            
        super().__init__(*args, **kwargs)
        
        # Добавляем информацию о ценовых изменениях для loss
        self.price_change_cols = [
            'future_return_15m', 'future_return_1h', 
            'future_return_4h', 'future_return_12h'
        ]
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Dict]:
        """Возвращает features, targets и price changes"""
        # Вызываем базовый метод для получения основных данных
        features_tensor, targets_tensor, base_info = super().__getitem__(idx)
        
        # Преобразуем targets в словарь для direction модели
        targets_dict = {}
        
        # Если у нас есть direction targets
        if targets_tensor.numel() > 0:
            # targets_tensor имеет форму (1, n_targets) или (n_targets,)
            if targets_tensor.dim() > 1:
                targets_tensor = targets_tensor.squeeze(0)
                
            for i, col in enumerate(self.target_cols):
                if i < targets_tensor.shape[0]:
                    # Конвертируем в LongTensor для классификации
                    # Каждый target должен быть скаляром
                    # Берем только первое значение, так как это категориальный код
                    target_value = targets_tensor[i]
                    if target_value.numel() > 1:
                        target_value = target_value[0]  # Берем первое значение
                    targets_dict[col] = target_value.long()
        
        # Получаем информацию о индексах из базового класса
        index_info = self.indices[idx]
        context_end = index_info['context_end_idx']
        
        # Price changes для loss
        price_changes = {}
        for col in self.price_change_cols:
            if col in self.data.columns:
                change = self.data.iloc[context_end][col]
                timeframe = col.split('_')[-1]  # 15m, 1h, etc.
                price_changes[timeframe] = torch.FloatTensor([change / 100.0])  # Конвертируем в доли
        
        # Обновляем info
        info = base_info.copy()
        info['price_changes'] = price_changes
        
        return features_tensor, targets_dict, info


class DirectionModelTrainer(Trainer):
    """Специализированный trainer для direction модели"""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device = None):
        super().__init__(model, config, device)
        
        # Специальная loss для direction
        self.criterion = DirectionalTradingLoss(
            commission=config.get('loss', {}).get('commission', 0.00055),
            profit_focus_weight=config.get('loss', {}).get('profit_focus_weight', 10.0)
        )
        
        # Метрики для отслеживания
        self.metrics_history = {
            'directional_accuracy': [],
            'profit_factor': [],
            'win_rate': [],
            'avg_profit': [],
            'avg_loss': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Обучение с расчетом торговых метрик"""
        self.model.train()
        
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        all_profits = []
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (inputs, targets, info) in enumerate(progress_bar):
            # Перенос на устройство
            inputs = inputs.to(self.device)
            targets = {k: v.to(self.device).squeeze() for k, v in targets.items()}
            price_changes = {k: v.to(self.device) for k, v in info['price_changes'].items()}
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets, price_changes)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, price_changes)
            
            # Backward pass
            loss = loss / self.gradient_accumulation_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
            
            # Сохраняем для метрик
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            
            # Расчет directional accuracy и profit
            with torch.no_grad():
                for timeframe in ['4h']:  # Фокус на 4h как основном
                    key = f'direction_{timeframe}'
                    if key in outputs:
                        pred_direction = outputs[key].argmax(dim=1)
                        true_direction = targets[key]
                        
                        # Directional accuracy
                        correct = (pred_direction == true_direction).float()
                        all_predictions.extend(pred_direction.cpu().numpy())
                        all_targets.extend(true_direction.cpu().numpy())
                        
                        # Расчет прибыли (упрощенный)
                        price_change = price_changes[timeframe].squeeze()
                        profits = torch.zeros_like(price_change)
                        
                        # LONG
                        long_mask = pred_direction == 0
                        profits[long_mask] = price_change[long_mask] - 0.001  # минус комиссия
                        
                        # SHORT
                        short_mask = pred_direction == 1
                        profits[short_mask] = -price_change[short_mask] - 0.001
                        
                        all_profits.extend(profits.cpu().numpy())
            
            # Обновление progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dir_acc': f'{np.mean(correct.cpu().numpy()):.2%}'
            })
        
        # Расчет метрик эпохи
        metrics = self._calculate_trading_metrics(all_predictions, all_targets, all_profits)
        metrics['loss'] = epoch_loss / len(train_loader)
        
        return metrics
    
    def _calculate_trading_metrics(self, predictions: List, targets: List, profits: List) -> Dict:
        """Расчет торговых метрик"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        profits = np.array(profits)
        
        # Directional accuracy (исключая FLAT)
        non_flat_mask = targets != 2
        if non_flat_mask.sum() > 0:
            directional_accuracy = (predictions[non_flat_mask] == targets[non_flat_mask]).mean()
        else:
            directional_accuracy = 0.0
        
        # Торговые метрики
        trading_mask = predictions != 2  # Когда модель решила торговать
        if trading_mask.sum() > 0:
            trading_profits = profits[trading_mask]
            
            winning_trades = trading_profits > 0
            win_rate = winning_trades.mean()
            
            if winning_trades.sum() > 0:
                avg_profit = trading_profits[winning_trades].mean()
            else:
                avg_profit = 0.0
                
            losing_trades = trading_profits < 0
            if losing_trades.sum() > 0:
                avg_loss = abs(trading_profits[losing_trades].mean())
            else:
                avg_loss = 0.0
                
            if avg_loss > 0:
                profit_factor = avg_profit / avg_loss
            else:
                profit_factor = float('inf') if avg_profit > 0 else 0.0
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        return {
            'directional_accuracy': directional_accuracy,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': trading_mask.sum()
        }
    
    def log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Логирование с фокусом на торговые метрики"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Эпоха {epoch + 1} завершена")
        self.logger.info(f"{'='*60}")
        
        self.logger.info("📊 Основные метрики:")
        self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        self.logger.info(f"  Train Dir Acc: {train_metrics['directional_accuracy']:.2%}, "
                        f"Val Dir Acc: {val_metrics['directional_accuracy']:.2%}")
        
        self.logger.info("\n💰 Торговые метрики (Val):")
        self.logger.info(f"  Win Rate: {val_metrics['win_rate']:.2%}")
        self.logger.info(f"  Profit Factor: {val_metrics['profit_factor']:.2f}")
        self.logger.info(f"  Avg Profit: {val_metrics['avg_profit']:.4%}")
        self.logger.info(f"  Avg Loss: {val_metrics['avg_loss']:.4%}")
        self.logger.info(f"  Total Trades: {val_metrics['total_trades']}")
        
        # Проверка готовности к торговле
        is_profitable = (
            val_metrics['directional_accuracy'] > 0.55 and
            val_metrics['win_rate'] > 0.50 and
            val_metrics['profit_factor'] > 1.2
        )
        
        if is_profitable:
            self.logger.info("\n✅ Модель показывает ПРИБЫЛЬНЫЕ результаты!")
        else:
            self.logger.info("\n⚠️ Модель еще НЕ готова к прибыльной торговле")


def create_direction_config(base_config: Dict) -> Dict:
    """Создание специализированной конфигурации для direction модели"""
    config = base_config.copy()
    
    # Модификации для direction модели
    config['model']['name'] = 'DirectionPredictor'
    # n_features будет установлен позже из данных
    config['model']['n_features'] = 254  # Временно, будет обновлено
    
    # Оптимизированные параметры для direction
    config['model']['learning_rate'] = 1e-4  # Выше для быстрой сходимости
    config['model']['batch_size'] = 256  # Меньше для лучшей генерализации
    config['model']['dropout'] = 0.3  # Больше dropout против переобучения
    
    # Scheduler для direction
    config['scheduler'] = {
        'name': 'CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 1e-6
        }
    }
    
    return config


def visualize_results(history: Dict, save_path: Path):
    """Визуализация результатов обучения"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Direction Model Training Results', fontsize=16)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True)
    
    # Directional Accuracy
    ax = axes[0, 1]
    train_acc = [m.get('directional_accuracy', 0) for m in history['train_metrics']]
    val_acc = [m.get('directional_accuracy', 0) for m in history['val_metrics']]
    ax.plot(train_acc, label='Train Accuracy')
    ax.plot(val_acc, label='Val Accuracy')
    ax.axhline(y=0.55, color='r', linestyle='--', label='Profitable Threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Directional Accuracy')
    ax.set_title('Directional Accuracy Progress')
    ax.legend()
    ax.grid(True)
    
    # Win Rate
    ax = axes[1, 0]
    val_wr = [m.get('win_rate', 0) for m in history['val_metrics']]
    ax.plot(val_wr, label='Val Win Rate')
    ax.axhline(y=0.50, color='r', linestyle='--', label='Break Even')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate Progress')
    ax.legend()
    ax.grid(True)
    
    # Profit Factor
    ax = axes[1, 1]
    val_pf = [m.get('profit_factor', 0) for m in history['val_metrics']]
    ax.plot(val_pf, label='Val Profit Factor')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Break Even')
    ax.axhline(y=1.5, color='g', linestyle='--', label='Good')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Profit Factor')
    ax.set_title('Profit Factor Progress')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'direction_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохранение финального отчета
    final_metrics = history['val_metrics'][-1] if history['val_metrics'] else {}
    report = f"""
DIRECTION MODEL TRAINING REPORT
==============================

Final Validation Metrics:
- Directional Accuracy: {final_metrics.get('directional_accuracy', 0):.2%}
- Win Rate: {final_metrics.get('win_rate', 0):.2%}
- Profit Factor: {final_metrics.get('profit_factor', 0):.2f}
- Average Profit: {final_metrics.get('avg_profit', 0):.4%}
- Average Loss: {final_metrics.get('avg_loss', 0):.4%}

Model Status: {'PROFITABLE' if final_metrics.get('directional_accuracy', 0) > 0.55 else 'NOT YET PROFITABLE'}
Recommended for Trading: {'YES' if final_metrics.get('profit_factor', 0) > 1.5 else 'NO'}
"""
    
    with open(save_path / 'direction_model_report.txt', 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Train Direction Prediction Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Train on specific symbol only')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    config = create_direction_config(config)
    
    if args.epochs:
        config['model']['epochs'] = args.epochs
    
    # Логирование
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/direction_training_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Настройка логирования
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    logger = get_logger("DirectionTraining")
    
    logger.info("🎯 Начинаем обучение Direction Prediction Model")
    logger.info(f"Конфигурация: {json.dumps(config['model'], indent=2)}")
    
    # Загрузка данных
    logger.info("📊 Загрузка данных...")
    
    # Проверяем существующие обработанные данные
    processed_train = Path("data/processed/train_data.parquet")
    processed_val = Path("data/processed/val_data.parquet")
    processed_test = Path("data/processed/test_data.parquet")
    
    if processed_train.exists() and processed_val.exists() and not args.symbol:
        logger.info("📂 Найдены обработанные данные, загружаем...")
        
        try:
            import pandas as pd
            train_data = pd.read_parquet(processed_train)
            val_data = pd.read_parquet(processed_val)
            test_data = pd.read_parquet(processed_test)
            
            logger.info(f"✅ Загружено из processed: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
            logger.info(f"📅 Период данных: {train_data['datetime'].min()} - {test_data['datetime'].max()}")
            logger.info(f"🪙 Символы: {train_data['symbol'].unique()}")
            
            # Получаем списки колонок из данных
            feature_columns = [col for col in train_data.columns 
                             if col not in ['id', 'symbol', 'datetime', 'timestamp']
                             and not col.startswith(('target_', 'future_', 'direction_', 'optimal_'))]
            
            target_columns = [col for col in train_data.columns 
                            if col.startswith('direction_')]
            
            logger.info(f"📊 Признаков: {len(feature_columns)}, Целевых: {len(target_columns)}")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки processed данных: {e}")
            logger.info("Загружаем данные стандартным способом...")
            
            data_loader = CryptoDataLoader(config)
            if args.symbol:
                data = data_loader.load_data(symbols=[args.symbol])
            else:
                # Начинаем с топ символов
                top_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
                data = data_loader.load_data(symbols=top_symbols)
            
            train_data, val_data, test_data = data_loader.split_data(data)
            feature_columns = data_loader.feature_columns
            target_columns = data_loader.target_columns
    else:
        # Стандартная загрузка
        data_loader = CryptoDataLoader(config)
        
        if args.symbol:
            data = data_loader.load_data(symbols=[args.symbol])
        else:
            # Начинаем с топ символов
            top_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
            data = data_loader.load_data(symbols=top_symbols)
        
        # Подготовка датасетов
        logger.info("🔧 Подготовка датасетов...")
        train_data, val_data, test_data = data_loader.split_data(data)
        feature_columns = data_loader.feature_columns
        target_columns = data_loader.target_columns
    
    # Создание датасетов с адаптером
    train_dataset = DirectionDatasetAdapter(
        train_data,
        context_window=config['model']['context_window'],
        feature_cols=feature_columns,
        target_cols=target_columns,
        stride=1,  # Используем все данные
        normalize=False  # Отключаем нормализацию для простоты
    )
    
    val_dataset = DirectionDatasetAdapter(
        val_data,
        context_window=config['model']['context_window'],
        feature_cols=feature_columns,
        target_cols=target_columns,
        stride=4,  # Меньше данных для валидации
        normalize=False  # Отключаем нормализацию для простоты
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=config['performance']['num_workers'],
        pin_memory=config['performance']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False,
        num_workers=config['performance']['num_workers'],
        pin_memory=config['performance']['pin_memory']
    )
    
    logger.info(f"✅ Данные загружены: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Обновляем n_features в конфиге
    config['model']['n_features'] = len(feature_columns)
    
    # Создание модели
    logger.info("🏗️ Создание модели...")
    model = DirectionPredictor(config['model'])
    
    # Trainer
    trainer = DirectionModelTrainer(model, config)
    
    # Resume если указано
    start_epoch = 0
    if args.resume:
        logger.info(f"📂 Загрузка checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Обучение
    logger.info("🚀 Начинаем обучение...")
    
    for epoch in range(start_epoch, config['model']['epochs']):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Update learning rate
        if trainer.scheduler:
            if hasattr(trainer.scheduler, 'step'):
                if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    trainer.scheduler.step(val_metrics['loss'])
                else:
                    trainer.scheduler.step()
        
        # Logging
        trainer.log_epoch_results(epoch, train_metrics, val_metrics)
        
        # Save checkpoint if best
        if val_metrics.get('directional_accuracy', 0) > getattr(trainer, 'best_metric', 0):
            trainer.best_metric = val_metrics['directional_accuracy']
            # Сохраняем с правильным именем для direction модели
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = Path("models_saved") / f"best_direction_model_{timestamp}.pth"
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
                'config': config,
                'history': trainer.history
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Сохранена лучшая модель с accuracy: {trainer.best_metric:.2%}")
            logger.info(f"📁 Путь: {checkpoint_path}")
        
        # Early stopping на основе directional accuracy
        if epoch > 10 and val_metrics.get('directional_accuracy', 0) < 0.52:
            logger.warning("⚠️ Directional accuracy слишком низкая, требуется пересмотр подхода")
    
    # Финальная визуализация
    logger.info("📊 Создание финального отчета...")
    visualize_results(trainer.history, log_dir)
    
    logger.info("✅ Обучение завершено!")
    logger.info(f"📁 Результаты сохранены в: {log_dir}")


if __name__ == "__main__":
    main()