#!/usr/bin/env python3
"""
Серверная версия Temporal Fusion Transformer v3.0
Адаптирована для работы на GPU сервере с оптимизациями
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
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI на сервере
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import psycopg2
import gc
import time
warnings.filterwarnings('ignore')

# Настройка для сервера
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Для лучшей диагностики GPU ошибок

# Добавляем путь к XGBoost модулям
sys.path.append('xgboost_v3')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('transformer_v3_server.log')
    ]
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Проверка доступности GPU"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"🎮 Обнаружено GPU: {gpu_count}")
        for i in range(gpu_count):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"   Память: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        return True
    else:
        logger.warning("⚠️ GPU не обнаружен, используется CPU")
        return False


class TargetCalculator:
    """Упрощенная версия TargetCalculator для сервера"""
    def __init__(self, lookahead_bars=4, price_threshold=0.5):
        self.lookahead_bars = lookahead_bars
        self.price_threshold = price_threshold
        
    def calculate_threshold_binary(self, df):
        """Бинарная классификация с порогом"""
        df = df.copy()
        
        # Рассчитываем будущее изменение цены
        future_return = (df['close'].shift(-self.lookahead_bars) / df['close'] - 1) * 100
        
        # Бинарный таргет
        df['target_threshold_binary'] = (future_return > self.price_threshold).astype(int)
        
        return df
        
    def calculate_simple_regression(self, df):
        """Простая регрессия на процентное изменение"""
        df = df.copy()
        
        # Будущее изменение цены в процентах
        df['target_simple_regression'] = (df['close'].shift(-self.lookahead_bars) / df['close'] - 1) * 100
        
        return df


class HierarchicalFeatureSelector:
    """Упрощенная версия для сервера"""
    def __init__(self, top_k=100):
        self.top_k = top_k
        
    def select_features(self, X, y, feature_names):
        """Простой отбор топ признаков по корреляции"""
        # Рассчитываем корреляции
        correlations = []
        for col in X.columns:
            if X[col].std() > 0:
                corr = abs(X[col].corr(y))
                correlations.append((col, corr))
                
        # Сортируем по корреляции
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Отбираем топ-K
        selected = [col for col, _ in correlations[:self.top_k]]
        
        logger.info(f"✅ Отобрано {len(selected)} признаков")
        
        return X[selected], selected


class ImprovedGatedLinearUnit(nn.Module):
    """Улучшенный GLU с нормализацией и dropout"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        import torch.nn.functional as F
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class ImprovedGatedResidualNetwork(nn.Module):
    """Улучшенный GRN с skip connections"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.glu = ImprovedGatedLinearUnit(hidden_dim, output_dim, dropout_rate)
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        residual = x
        
        x = self.fc1(x)
        x = self.elu(x)
        x = self.glu(x)
        
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
            
        x = x + residual
        x = self.layer_norm(x)
        
        return x


class ImprovedVariableSelectionNetwork(nn.Module):
    """Улучшенная сеть выбора переменных"""
    def __init__(self, input_dim, num_inputs, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        
        self.flattened_grn = ImprovedGatedResidualNetwork(
            input_dim * num_inputs,
            hidden_dim,
            num_inputs,
            dropout_rate
        )
        
        self.variable_grns = nn.ModuleList([
            ImprovedGatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout_rate)
            for _ in range(num_inputs)
        ])
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        
        flattened = inputs.view(batch_size, -1)
        variable_weights = self.softmax(self.flattened_grn(flattened))
        
        processed_inputs = []
        for i in range(self.num_inputs):
            processed = self.variable_grns[i](inputs[:, i, :])
            processed_inputs.append(processed)
            
        processed_inputs = torch.stack(processed_inputs, dim=1)
        weighted_inputs = processed_inputs * variable_weights.unsqueeze(-1)
        combined = weighted_inputs.sum(dim=1)
        
        return combined, variable_weights


class TemporalFusionTransformerV3(nn.Module):
    """TFT модель оптимизированная для сервера"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.dropout_rate = config['dropout_rate']
        self.output_dim = config['output_dim']
        
        # Variable Selection
        self.static_vsn = ImprovedVariableSelectionNetwork(
            self.input_dim,
            1,
            self.hidden_dim,
            self.dropout_rate
        )
        
        # LSTM Encoder (оптимизирован для GPU)
        self.lstm_encoder = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=2,
            dropout=self.dropout_rate,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-Head Attention
        self.self_attention = nn.MultiheadAttention(
            self.hidden_dim * 2,
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
        
        # Output layers
        if config['task'] == 'regression':
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.output_dim),
                nn.Sigmoid() if self.output_dim == 1 else nn.Softmax(dim=-1)
            )
            
    def forward(self, x_static):
        batch_size = x_static.size(0)
        
        # Variable Selection
        selected_features, feature_weights = self.static_vsn(x_static.unsqueeze(1))
        
        # Создаем последовательность
        sequence_length = 10
        lstm_input = selected_features.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # LSTM Encoder
        lstm_output, (hidden, cell) = self.lstm_encoder(lstm_input)
        
        # Self-Attention
        attn_output, attn_weights = self.self_attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # Decoder GRN
        decoded = self.decoder_grn(attn_output)
        
        # Агрегация
        aggregated = decoded.mean(dim=1)
        
        # Output
        output = self.output_layer(aggregated)
        
        return output, {
            'feature_weights': feature_weights,
            'attention_weights': attn_weights
        }


class TransformerTrainer:
    """Оптимизированный тренер для сервера"""
    def __init__(self, model, config, log_dir):
        self.model = model
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Оптимизатор с gradient accumulation для больших батчей
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
            
        # Определяем устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Включаем DataParallel для нескольких GPU
        if torch.cuda.device_count() > 1:
            logger.info(f"🚀 Используется {torch.cuda.device_count()} GPU")
            self.model = nn.DataParallel(self.model)
            
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'gpu_memory': []
        }
        
    def train_epoch(self, train_loader):
        """Обучение одной эпохи с мониторингом GPU"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        # Gradient accumulation steps
        accumulation_steps = 4
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output, _ = self.model(data)
            
            # Loss
            if self.config['task'] == 'classification' and self.config['output_dim'] > 1:
                loss = self.criterion(output, target.long())
            else:
                loss = self.criterion(output.squeeze(), target)
                
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            total_loss += loss.item() * accumulation_steps
            
            # Сохраняем предсказания
            predictions.extend(output.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())
            
            # Логируем использование памяти GPU
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"   Batch {batch_idx}/{len(train_loader)} | "
                          f"GPU Memory: {memory_used:.1f}GB / {memory_cached:.1f}GB")
                
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
        """Расчет метрик"""
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
                preds_binary = (predictions > 0.5).astype(int).flatten()
                targets_flat = targets.flatten().astype(int)
                return {
                    'accuracy': accuracy_score(targets_flat, preds_binary),
                    'precision': precision_score(targets_flat, preds_binary, zero_division=0),
                    'recall': recall_score(targets_flat, preds_binary, zero_division=0),
                    'f1': f1_score(targets_flat, preds_binary, zero_division=0),
                    'roc_auc': roc_auc_score(targets_flat, predictions.flatten()) if len(np.unique(targets_flat)) > 1 else 0
                }
            else:
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
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Обучение
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Валидация
            val_loss, val_metrics = self.validate(val_loader)
            
            # Обновление scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # GPU memory stats
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                self.history['gpu_memory'].append(memory_used)
                
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
            
            # Время эпохи
            epoch_time = time.time() - epoch_start
            
            # Логирование
            logger.info(f"\n{'='*60}")
            logger.info(f"Эпоха {epoch+1}/{epochs} | Время: {epoch_time:.1f}с")
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            if self.config['task'] == 'regression':
                logger.info(f"Val MAE: {val_metrics['mae']:.4f} | "
                          f"Val RMSE: {val_metrics['rmse']:.4f} | "
                          f"Val R²: {val_metrics['r2']:.4f}")
            else:
                logger.info(f"Val Accuracy: {val_metrics.get('accuracy', 0):.4f} | "
                          f"Val F1: {val_metrics.get('f1', val_metrics.get('macro_f1', 0)):.4f}")
                
            # Сохранение лучшей модели
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss, val_metrics)
                logger.info("✅ Сохранена лучшая модель!")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= 10:
                logger.info("⏹️ Early stopping triggered")
                break
                
            # Визуализация каждые 5 эпох
            if (epoch + 1) % 5 == 0:
                self._plot_training_progress(epoch + 1)
                
            # Очистка GPU кэша
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Общее время обучения
        total_time = time.time() - start_time
        logger.info(f"\n⏱️ Общее время обучения: {total_time/60:.1f} минут")
        
        # Финальная визуализация
        self._plot_final_results()
        self._save_training_summary()
        
        logger.info("✅ Обучение завершено!")
        
    def _save_checkpoint(self, epoch, val_loss, val_metrics):
        """Сохранение чекпоинта модели"""
        # Если модель обернута в DataParallel, извлекаем оригинальную модель
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
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
            train_metric = [m.get('accuracy', 0) for m in self.history['train_metrics']]
            val_metric = [m.get('accuracy', 0) for m in self.history['val_metrics']]
            metric_name = 'Accuracy'
            
        axes[1, 0].plot(train_metric, label=f'Train {metric_name}')
        axes[1, 0].plot(val_metric, label=f'Val {metric_name}')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel(metric_name)
        axes[1, 0].set_title(f'{metric_name} Progress')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # GPU Memory Usage
        if self.history['gpu_memory']:
            axes[1, 1].plot(self.history['gpu_memory'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('GPU Memory (GB)')
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].grid(True)
            
        plt.tight_layout()
        plt.savefig(self.log_dir / f'training_progress_epoch_{epoch}.png', dpi=150)
        plt.close()
        
        # Очистка памяти
        del fig
        gc.collect()
        
    def _plot_final_results(self):
        """Финальная визуализация результатов"""
        logger.info("📊 Создание финальных визуализаций...")
        
        # Сводный график
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Summary')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_summary.png', dpi=150)
        plt.close()
        
    def _save_training_summary(self):
        """Сохранение итогового отчета"""
        summary = {
            'config': self.config,
            'final_train_loss': float(self.history['train_loss'][-1]),
            'final_val_loss': float(self.history['val_loss'][-1]),
            'final_train_metrics': {k: float(v) for k, v in self.history['train_metrics'][-1].items()},
            'final_val_metrics': {k: float(v) for k, v in self.history['val_metrics'][-1].items()},
            'total_epochs': len(self.history['train_loss']),
            'best_val_loss': float(min(self.history['val_loss'])),
            'best_epoch': self.history['val_loss'].index(min(self.history['val_loss'])) + 1,
            'device': str(self.device),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        with open(self.log_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Текстовый отчет
        with open(self.log_dir / 'final_report.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TEMPORAL FUSION TRANSFORMER V3 - ФИНАЛЬНЫЙ ОТЧЕТ\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Устройство: {summary['device']}\n")
            f.write(f"Количество GPU: {summary['gpu_count']}\n")
            f.write(f"Задача: {self.config['task']}\n")
            f.write(f"Всего эпох: {summary['total_epochs']}\n")
            f.write(f"Лучшая эпоха: {summary['best_epoch']}\n")
            f.write(f"Лучший Val Loss: {summary['best_val_loss']:.4f}\n\n")
            
            f.write("Финальные метрики валидации:\n")
            for metric, value in summary['final_val_metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")


def prepare_data_for_transformer(df, target_type='threshold_binary', test_size=0.2):
    """Подготовка данных для трансформера"""
    logger.info("📊 Подготовка данных для трансформера")
    
    # Расчет таргетов
    target_calculator = TargetCalculator(lookahead_bars=4, price_threshold=0.5)
    
    if target_type == 'threshold_binary':
        df = target_calculator.calculate_threshold_binary(df)
    elif target_type == 'simple_regression':
        df = target_calculator.calculate_simple_regression(df)
    else:
        raise ValueError(f"Неизвестный тип таргета: {target_type}")
        
    target_column = f"target_{target_type}"
    
    # Отбор признаков
    feature_columns = [col for col in df.columns if col not in ['symbol', 'timestamp', 'close'] 
                      and not col.startswith('target_')]
    
    # Иерархический отбор признаков
    feature_selector = HierarchicalFeatureSelector(top_k=100)
    X = df[feature_columns]
    y = df[target_column]
    
    # Удаляем строки с NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    X_selected, selected_features = feature_selector.select_features(X, y, feature_columns)
    
    # Временное разделение
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
    """Главная функция для сервера"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Temporal Fusion Transformer v3 Server')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression'])
    parser.add_argument('--target-type', type=str, default='threshold_binary',
                       choices=['threshold_binary', 'simple_regression'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)  # Увеличен для GPU
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=256)  # Увеличен для GPU
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=500000,
                       help='Лимит записей из БД')
    
    args = parser.parse_args()
    
    # Проверка GPU
    check_gpu()
    
    # Конфигурация модели
    config = {
        'task': args.task,
        'input_dim': 100,  # будет обновлено
        'hidden_dim': args.hidden_dim,
        'num_heads': 8,
        'dropout_rate': 0.1,
        'output_dim': 1,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.0001
    }
    
    # Создание директории для логов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/transformer_v3_{timestamp}"
    
    logger.info("=" * 60)
    logger.info("🚀 TEMPORAL FUSION TRANSFORMER V3 - SERVER VERSION")
    logger.info(f"📊 Задача: {config['task']}")
    logger.info(f"🎯 Тип таргета: {args.target_type}")
    logger.info("=" * 60)
    
    # Загрузка данных
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5555,
            database="crypto_trading",
            user="ruslan"
        )
        
        # Загружаем все символы для production mode
        query = f"""
        SELECT * FROM processed_market_data 
        WHERE symbol NOT LIKE '%TEST%'
        ORDER BY timestamp
        LIMIT {args.limit}
        """
        
        logger.info(f"📥 Загрузка данных (лимит: {args.limit})...")
        df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"✅ Загружено {len(df)} записей")
        logger.info(f"📊 Символы: {df['symbol'].nunique()}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке данных: {e}")
        return
        
    # Подготовка данных
    (X_train, X_test, y_train, y_test, 
     selected_features, scaler) = prepare_data_for_transformer(df, args.target_type)
    
    # Обновляем input_dim
    config['input_dim'] = X_train.shape[1]
    
    # Создание DataLoaders с pin_memory для GPU
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
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
        
    # Сохранение scaler
    import joblib
    joblib.dump(scaler, f"{log_dir}/scaler.pkl")
    
    logger.info(f"\n✅ Результаты сохранены в: {log_dir}")
    

if __name__ == "__main__":
    main()