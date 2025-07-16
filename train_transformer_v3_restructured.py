#!/usr/bin/env python3
"""
Реструктурированная версия Temporal Fusion Transformer v3.0
Использует лучшие практики из XGBoost v3:
- Упрощенные таргеты
- Иерархический отбор признаков  
- Временное разделение данных
- Улучшенная визуализация
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к XGBoost модулям
sys.path.append('xgboost_v3')
from data.target_calculator import TargetCalculator
from utils.feature_selector import FeatureSelector

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedGatedLinearUnit(nn.Module):
    """Улучшенный GLU с нормализацией и dropout"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class ImprovedGatedResidualNetwork(nn.Module):
    """Улучшенный GRN с skip connections"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1, use_time_distributed=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_time_distributed = use_time_distributed
        
        # Основные слои
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        
        # GLU для нелинейной обработки
        self.glu = ImprovedGatedLinearUnit(hidden_dim, output_dim, dropout_rate)
        
        # Skip connection если размерности совпадают
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # Сохраняем для skip connection
        residual = x
        
        # Основная обработка
        x = self.fc1(x)
        x = self.elu(x)
        x = self.glu(x)
        
        # Skip connection
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
            
        x = x + residual
        x = self.layer_norm(x)
        
        return x


class ImprovedVariableSelectionNetwork(nn.Module):
    """Улучшенная сеть выбора переменных с механизмом внимания"""
    def __init__(self, input_dim, num_inputs, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        
        # Flattening для обработки всех входов вместе
        self.flattened_grn = ImprovedGatedResidualNetwork(
            input_dim * num_inputs,
            hidden_dim,
            num_inputs,
            dropout_rate
        )
        
        # Отдельные GRN для каждой переменной
        self.variable_grns = nn.ModuleList([
            ImprovedGatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout_rate)
            for _ in range(num_inputs)
        ])
        
        # Softmax для весов
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, inputs):
        # inputs shape: (batch_size, num_inputs, input_dim)
        batch_size = inputs.size(0)
        
        # Flatten и получаем веса переменных
        flattened = inputs.view(batch_size, -1)
        variable_weights = self.softmax(self.flattened_grn(flattened))
        
        # Обрабатываем каждую переменную отдельно
        processed_inputs = []
        for i in range(self.num_inputs):
            processed = self.variable_grns[i](inputs[:, i, :])
            processed_inputs.append(processed)
            
        processed_inputs = torch.stack(processed_inputs, dim=1)
        
        # Взвешенная комбинация
        weighted_inputs = processed_inputs * variable_weights.unsqueeze(-1)
        combined = weighted_inputs.sum(dim=1)
        
        return combined, variable_weights


class TemporalFusionTransformerV3(nn.Module):
    """
    Реструктурированный Temporal Fusion Transformer
    с улучшениями из XGBoost v3
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Размерности
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.dropout_rate = config['dropout_rate']
        self.output_dim = config['output_dim']
        
        # Variable Selection для статических признаков
        self.static_vsn = ImprovedVariableSelectionNetwork(
            self.input_dim,
            1,  # Все статические признаки как один вход
            self.hidden_dim,
            self.dropout_rate
        )
        
        # LSTM Encoder для локальной обработки
        self.lstm_encoder = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=2,
            dropout=self.dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-Head Attention для долгосрочных зависимостей
        self.self_attention = nn.MultiheadAttention(
            self.hidden_dim * 2,  # bidirectional LSTM
            self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Decoder GRN
        self.decoder_grn = ImprovedGatedResidualNetwork(
            self.hidden_dim * 2,
            self.hidden_dim,
            self.hidden_dim,
            self.dropout_rate
        )
        
        # Output layers в зависимости от задачи
        if config['task'] == 'regression':
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            )
        else:  # classification
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.output_dim),
                nn.Sigmoid() if self.output_dim == 1 else nn.Softmax(dim=-1)
            )
            
    def forward(self, x_static):
        batch_size = x_static.size(0)
        
        # 1. Variable Selection для статических признаков
        selected_features, feature_weights = self.static_vsn(x_static.unsqueeze(1))
        
        # 2. Создаем последовательность для LSTM (простое повторение)
        # В реальности здесь должны быть временные признаки
        sequence_length = 10  # фиксированная длина последовательности
        lstm_input = selected_features.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # 3. LSTM Encoder
        lstm_output, (hidden, cell) = self.lstm_encoder(lstm_input)
        
        # 4. Self-Attention
        attn_output, attn_weights = self.self_attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # 5. Decoder GRN
        decoded = self.decoder_grn(attn_output)
        
        # 6. Агрегация по времени (среднее)
        aggregated = decoded.mean(dim=1)
        
        # 7. Output
        output = self.output_layer(aggregated)
        
        return output, {
            'feature_weights': feature_weights,
            'attention_weights': attn_weights
        }


class TransformerTrainer:
    """Тренер для Temporal Fusion Transformer с визуализацией"""
    
    def __init__(self, model, config, log_dir):
        self.model = model
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Оптимизатор
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss functions
        if config['task'] == 'regression':
            self.criterion = nn.MSELoss()
        elif config['output_dim'] == 1:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
    def train_epoch(self, train_loader):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(data)
            
            # Loss
            if self.config['task'] == 'classification' and self.config['output_dim'] > 1:
                loss = self.criterion(output, target.long())
            else:
                loss = self.criterion(output.squeeze(), target)
                
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Сохраняем предсказания
            predictions.extend(output.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())
            
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
        
    def validate(self, val_loader):
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.model(data)
                
                if self.config['task'] == 'classification' and self.config['output_dim'] > 1:
                    loss = self.criterion(output, target.long())
                else:
                    loss = self.criterion(output.squeeze(), target)
                    
                total_loss += loss.item()
                
                predictions.extend(output.detach().cpu().numpy())
                targets.extend(target.detach().cpu().numpy())
                
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
        
    def _calculate_metrics(self, predictions, targets):
        """Расчет метрик в зависимости от задачи"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if self.config['task'] == 'regression':
            return {
                'mae': mean_absolute_error(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions)),
                'r2': r2_score(targets, predictions)
            }
        else:
            if self.config['output_dim'] == 1:
                # Бинарная классификация
                preds_binary = (predictions > 0.5).astype(int)
                return {
                    'accuracy': accuracy_score(targets, preds_binary),
                    'precision': precision_score(targets, preds_binary, zero_division=0),
                    'recall': recall_score(targets, preds_binary, zero_division=0),
                    'f1': f1_score(targets, preds_binary, zero_division=0),
                    'roc_auc': roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0
                }
            else:
                # Мультиклассовая классификация
                preds_class = np.argmax(predictions, axis=1)
                return {
                    'accuracy': accuracy_score(targets, preds_class),
                    'macro_f1': f1_score(targets, preds_class, average='macro', zero_division=0)
                }
                
    def train(self, train_loader, val_loader, epochs):
        """Основной цикл обучения"""
        logger.info(f"🚀 Начало обучения на {epochs} эпох")
        logger.info(f"📊 Устройство: {self.device}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Обучение
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Валидация
            val_loss, val_metrics = self.validate(val_loader)
            
            # Обновление scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
            
            # Логирование
            logger.info(f"\nЭпоха {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            if self.config['task'] == 'regression':
                logger.info(f"Val MAE: {val_metrics['mae']:.4f} | "
                          f"Val RMSE: {val_metrics['rmse']:.4f} | "
                          f"Val R²: {val_metrics['r2']:.4f}")
            else:
                logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
                
            # Сохранение лучшей модели
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss, val_metrics)
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= 10:
                logger.info("⏹️ Early stopping triggered")
                break
                
            # Визуализация каждые 5 эпох
            if (epoch + 1) % 5 == 0:
                self._plot_training_progress(epoch + 1)
                
        # Финальная визуализация
        self._plot_final_results()
        self._save_training_summary()
        
        logger.info("✅ Обучение завершено!")
        
    def _save_checkpoint(self, epoch, val_loss, val_metrics):
        """Сохранение чекпоинта модели"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        checkpoint_path = self.log_dir / 'best_model.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Сохранен чекпоинт: {checkpoint_path}")
        
    def _plot_training_progress(self, epoch):
        """Визуализация прогресса обучения"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning Rate
        axes[0, 1].plot(self.history['learning_rates'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True)
        
        # Primary Metric
        if self.config['task'] == 'regression':
            train_metric = [m['mae'] for m in self.history['train_metrics']]
            val_metric = [m['mae'] for m in self.history['val_metrics']]
            metric_name = 'MAE'
        else:
            train_metric = [m['accuracy'] for m in self.history['train_metrics']]
            val_metric = [m['accuracy'] for m in self.history['val_metrics']]
            metric_name = 'Accuracy'
            
        axes[1, 0].plot(train_metric, label=f'Train {metric_name}')
        axes[1, 0].plot(val_metric, label=f'Val {metric_name}')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel(metric_name)
        axes[1, 0].set_title(f'{metric_name} Progress')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Additional Metrics
        if self.config['task'] == 'regression':
            val_r2 = [m['r2'] for m in self.history['val_metrics']]
            axes[1, 1].plot(val_r2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].set_title('Validation R² Score')
            axes[1, 1].grid(True)
        else:
            if 'f1' in self.history['val_metrics'][0]:
                val_f1 = [m['f1'] for m in self.history['val_metrics']]
                axes[1, 1].plot(val_f1)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('F1 Score')
                axes[1, 1].set_title('Validation F1 Score')
                axes[1, 1].grid(True)
                
        plt.tight_layout()
        plt.savefig(self.log_dir / f'training_progress_epoch_{epoch}.png')
        plt.close()
        
    def _plot_final_results(self):
        """Финальная визуализация результатов"""
        # Загружаем лучшую модель
        checkpoint = torch.load(self.log_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("📊 Создание финальных визуализаций...")
        
        # Здесь можно добавить дополнительные визуализации
        # например, confusion matrix для классификации
        # или scatter plot предсказаний для регрессии
        
    def _save_training_summary(self):
        """Сохранение итогового отчета"""
        summary = {
            'config': self.config,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_train_metrics': self.history['train_metrics'][-1],
            'final_val_metrics': self.history['val_metrics'][-1],
            'total_epochs': len(self.history['train_loss']),
            'best_val_loss': min(self.history['val_loss']),
            'best_epoch': self.history['val_loss'].index(min(self.history['val_loss'])) + 1
        }
        
        with open(self.log_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Текстовый отчет
        with open(self.log_dir / 'final_report.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TEMPORAL FUSION TRANSFORMER V3 - ФИНАЛЬНЫЙ ОТЧЕТ\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Задача: {self.config['task']}\n")
            f.write(f"Всего эпох: {summary['total_epochs']}\n")
            f.write(f"Лучшая эпоха: {summary['best_epoch']}\n")
            f.write(f"Лучший Val Loss: {summary['best_val_loss']:.4f}\n\n")
            
            f.write("Финальные метрики валидации:\n")
            for metric, value in summary['final_val_metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")


def prepare_data_for_transformer(df, target_type='threshold_binary', test_size=0.2):
    """
    Подготовка данных для трансформера
    Использует упрощенные таргеты из XGBoost v3
    """
    logger.info("📊 Подготовка данных для трансформера")
    
    # Расчет таргетов
    target_calculator = TargetCalculator(lookahead_bars=4, price_threshold=0.5)
    df_with_targets = target_calculator.calculate_all_targets(df)
    
    # Выбор колонки таргета
    target_column = f"target_{target_type}"
    if target_column not in df_with_targets.columns:
        raise ValueError(f"Неизвестный тип таргета: {target_type}")
        
    # Отбор признаков (используем FeatureSelector из XGBoost)
    feature_columns = [col for col in df.columns if col not in ['symbol', 'timestamp', 'close'] 
                      and not col.startswith('target_')]
    
    # Иерархический отбор признаков
    feature_selector = FeatureSelector(method='hierarchical', top_k=100)
    X = df_with_targets[feature_columns]
    y = df_with_targets[target_column]
    
    # Удаляем строки с NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    X_selected, selected_features = feature_selector.select_features(X, y, feature_columns)
    
    logger.info(f"✅ Отобрано {len(selected_features)} признаков из {len(feature_columns)}")
    
    # Временное разделение (как в XGBoost)
    n_samples = len(X_selected)
    train_size = int(n_samples * (1 - test_size))
    
    X_train = X_selected.iloc[:train_size]
    X_test = X_selected.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Статистика
    logger.info(f"📈 Train: {len(X_train)} примеров")
    logger.info(f"📈 Test: {len(X_test)} примеров")
    
    if target_type != 'simple_regression':
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info("📊 Распределение классов в train:")
        for val, count in zip(unique, counts):
            logger.info(f"   Класс {val}: {count} ({count/len(y_train)*100:.1f}%)")
            
    return (X_train_scaled, X_test_scaled, y_train.values, y_test.values, 
            selected_features, scaler)


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Temporal Fusion Transformer v3')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Тип задачи')
    parser.add_argument('--target-type', type=str, default='threshold_binary',
                       choices=['simple_binary', 'threshold_binary', 
                               'direction_multiclass', 'simple_regression'],
                       help='Тип таргета')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Количество эпох')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Размер батча')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Размерность скрытого слоя')
    
    args = parser.parse_args()
    
    # Конфигурация модели
    config = {
        'task': args.task,
        'input_dim': 100,  # будет обновлено после отбора признаков
        'hidden_dim': args.hidden_dim,
        'num_heads': 8,
        'dropout_rate': 0.1,
        'output_dim': 1,  # будет обновлено в зависимости от задачи
        'learning_rate': args.learning_rate,
        'weight_decay': 0.0001
    }
    
    # Определение output_dim
    if args.target_type == 'direction_multiclass':
        config['output_dim'] = 5
        config['task'] = 'classification'
    elif args.target_type == 'simple_regression':
        config['output_dim'] = 1
        config['task'] = 'regression'
    else:
        config['output_dim'] = 1
        config['task'] = 'classification'
        
    # Создание директории для логов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/transformer_v3_{timestamp}"
    
    logger.info("=" * 60)
    logger.info("🚀 TEMPORAL FUSION TRANSFORMER V3")
    logger.info(f"📊 Задача: {config['task']}")
    logger.info(f"🎯 Тип таргета: {args.target_type}")
    logger.info("=" * 60)
    
    # Загрузка данных (используем данные из processed_market_data)
    import psycopg2
    conn = psycopg2.connect(
        host="localhost",
        port=5555,
        database="crypto_trading",
        user="ruslan"
    )
    
    query = """
    SELECT * FROM processed_market_data 
    WHERE symbol IN ('BTCUSDT', 'ETHUSDT')
    ORDER BY timestamp
    LIMIT 100000
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    logger.info(f"✅ Загружено {len(df)} записей")
    
    # Подготовка данных
    (X_train, X_test, y_train, y_test, 
     selected_features, scaler) = prepare_data_for_transformer(df, args.target_type)
    
    # Обновляем input_dim
    config['input_dim'] = X_train.shape[1]
    
    # Создание DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Создание модели
    model = TemporalFusionTransformerV3(config)
    logger.info(f"📊 Параметров модели: {sum(p.numel() for p in model.parameters()):,}")
    
    # Создание тренера
    trainer = TransformerTrainer(model, config, log_dir)
    
    # Обучение
    trainer.train(train_loader, val_loader, args.epochs)
    
    # Сохранение конфигурации и признаков
    with open(f"{log_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
        
    with open(f"{log_dir}/selected_features.json", 'w') as f:
        json.dump(selected_features, f, indent=2)
        
    logger.info(f"\n✅ Результаты сохранены в: {log_dir}")
    

if __name__ == "__main__":
    import torch.nn.functional as F
    main()