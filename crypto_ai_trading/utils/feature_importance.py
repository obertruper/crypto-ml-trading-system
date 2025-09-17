"""
Анализ важности признаков для модели
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

class FeatureImportanceAnalyzer:
    """Анализ важности признаков для понимания какие индикаторы влияют на предсказания"""
    
    def __init__(self):
        self.logger = get_logger("FeatureImportance")
        self.feature_importance = {}
        self.gradient_importance = {}
    
    def analyze_gradient_importance(self, model, dataloader, feature_names: List[str], 
                                   device: str = 'cuda', num_batches: int = 5) -> Dict[str, float]:
        """
        Анализ важности признаков через градиенты
        
        Args:
            model: обученная модель
            dataloader: загрузчик данных
            feature_names: список имен признаков
            device: устройство для вычислений
            num_batches: количество батчей для анализа
            
        Returns:
            Словарь с важностью каждого признака
        """
        # Оставляем модель в eval режиме, но включаем градиенты
        model.eval()
        
        # Получаем размерность входных признаков из первого батча
        try:
            first_batch = next(iter(dataloader))
            n_features = first_batch[0].shape[-1]  # Последняя размерность - признаки
        except StopIteration:
            self.logger.error("Dataloader пуст!")
            return {}
            
        gradient_sum = torch.zeros(n_features).to(device)
        
        # Подсчет обработанных батчей
        processed_batches = 0
        
        with torch.enable_grad():
            for batch_idx, (inputs, targets, _) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                    
                inputs = inputs.to(device).requires_grad_(True)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Обрабатываем multi-dimensional outputs корректно
                # Убеждаемся, что targets имеет правильную размерность
                if targets.ndim == 3 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)  # (batch, n_targets)
                
                # Приводим outputs к той же размерности что и targets
                if outputs.ndim != targets.ndim:
                    if outputs.ndim == 3 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze(1)
                
                # Используем MSE loss для multi-target regression
                # Это работает для любого количества целевых переменных
                try:
                    # Убеждаемся что размерности совпадают
                    if outputs.shape != targets.shape:
                        # Обрезаем до минимального размера
                        min_size = min(outputs.shape[-1], targets.shape[-1])
                        outputs = outputs[..., :min_size]
                        targets = targets[..., :min_size]
                    
                    # Простой MSE loss для всех выходов
                    loss = torch.nn.functional.mse_loss(outputs, targets)
                    
                except Exception as e:
                    self.logger.warning(f"Ошибка при вычислении loss: {e}")
                    # Fallback: используем только первый выход
                    if outputs.shape[-1] > 0 and targets.shape[-1] > 0:
                        loss = torch.nn.functional.mse_loss(outputs[..., 0], targets[..., 0])
                    else:
                        continue
                
                # Backward pass
                loss.backward()
                
                # Суммируем абсолютные градиенты по батчу и временному измерению
                if inputs.grad is not None:
                    grad_importance = inputs.grad.abs().mean(dim=[0, 1])  # (n_features,)
                    # Проверяем на NaN
                    if not torch.isnan(grad_importance).any():
                        gradient_sum += grad_importance
                        processed_batches += 1
                    else:
                        self.logger.warning(f"NaN градиенты обнаружены в батче {batch_idx}")
                else:
                    self.logger.warning(f"Градиенты отсутствуют в батче {batch_idx}")
                
                # Очищаем градиенты
                model.zero_grad()
                if inputs.grad is not None:
                    inputs.grad.zero_()
        
        # Нормализация
        if processed_batches > 0:
            gradient_sum = gradient_sum / processed_batches
            if gradient_sum.sum() > 0:
                gradient_sum = gradient_sum / gradient_sum.sum()
            else:
                self.logger.error("Сумма градиентов равна 0!")
                # Возвращаем равные веса если градиенты нулевые
                gradient_sum = torch.ones_like(gradient_sum) / len(gradient_sum)
        else:
            self.logger.error("Не удалось обработать ни одного батча!")
            gradient_sum = torch.ones_like(gradient_sum) / len(gradient_sum)
        
        # Создаем словарь важности
        importance_dict = {}
        # Проверяем соответствие размерностей
        if len(feature_names) != len(gradient_sum):
            self.logger.warning(f"Несоответствие размерностей: feature_names={len(feature_names)}, gradients={len(gradient_sum)}")
            # Обрезаем или дополняем
            min_len = min(len(feature_names), len(gradient_sum))
            for i in range(min_len):
                importance_dict[feature_names[i]] = gradient_sum[i].item()
        else:
            for i, name in enumerate(feature_names):
                importance_dict[name] = gradient_sum[i].item()
        
        # Сортируем по важности
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True))
        
        self.gradient_importance = importance_dict
        return importance_dict
    
    def analyze_permutation_importance(self, model, dataloader, feature_names: List[str],
                                      device: str = 'cuda', num_batches: int = 3) -> Dict[str, float]:
        """
        Анализ важности признаков через перестановку (permutation importance)
        Более надежный метод когда градиенты не работают
        """
        model.eval()
        importance_scores = {}
        
        # Получаем базовую метрику
        with torch.no_grad():
            base_losses = []
            for batch_idx, (inputs, targets, _) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                
                # Упрощенная loss - просто accuracy для direction_15m
                if outputs.shape[-1] >= 8 and targets.ndim == 3:
                    targets = targets.squeeze(1)
                    preds = outputs[:, 4:7].argmax(dim=-1)  # direction predictions
                    correct = (preds == targets[:, 4].long()).float().mean()
                    base_losses.append(1 - correct.item())  # Используем error rate
                    
            base_loss = np.mean(base_losses) if base_losses else 0.5
        
        # Проверяем важность каждого признака
        n_features = next(iter(dataloader))[0].shape[-1]
        
        for feature_idx in range(min(n_features, len(feature_names))):
            feature_losses = []
            
            with torch.no_grad():
                for batch_idx, (inputs, targets, _) in enumerate(dataloader):
                    if batch_idx >= num_batches:
                        break
                    
                    inputs = inputs.to(device).clone()
                    targets = targets.to(device)
                    
                    # Перемешиваем значения конкретного признака
                    batch_size = inputs.shape[0]
                    perm_idx = torch.randperm(batch_size).to(device)
                    inputs[:, :, feature_idx] = inputs[perm_idx, :, feature_idx]
                    
                    outputs = model(inputs)
                    
                    if outputs.shape[-1] >= 8 and targets.ndim == 3:
                        targets = targets.squeeze(1)
                        preds = outputs[:, 4:7].argmax(dim=-1)
                        correct = (preds == targets[:, 4].long()).float().mean()
                        feature_losses.append(1 - correct.item())
            
            feature_loss = np.mean(feature_losses) if feature_losses else 0.5
            # Важность = насколько ухудшается метрика при перестановке
            importance = max(0, feature_loss - base_loss)
            
            if feature_idx < len(feature_names):
                importance_scores[feature_names[feature_idx]] = importance
        
        # Нормализация
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        else:
            # Если все важности нулевые, даем равные веса
            n = len(importance_scores)
            importance_scores = {k: 1.0/n for k in importance_scores.keys()}
        
        # Сортировка
        importance_scores = dict(sorted(importance_scores.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
        
        return importance_scores
    
    def analyze_statistical_importance(self, data: pd.DataFrame, 
                                      target_col: str = 'direction_1h',
                                      top_n: int = 50) -> Dict[str, float]:
        """
        Статистический анализ важности признаков через корреляцию и mutual information
        
        Args:
            data: DataFrame с признаками и целевыми переменными
            target_col: целевая переменная для анализа
            top_n: количество топ признаков для анализа
            
        Returns:
            Словарь с важностью каждого признака
        """
        from sklearn.feature_selection import mutual_info_classif
        
        # Исключаем служебные колонки
        exclude_cols = ['id', 'timestamp', 'datetime', 'symbol', 'symbol_id', 
                       'open', 'high', 'low', 'close', 'volume', 'turnover']
        
        feature_cols = [col for col in data.columns 
                       if col not in exclude_cols and not col.startswith('future_') 
                       and not col.startswith('direction_') and not col.startswith('will_reach_')
                       and not col.startswith('max_')]
        
        if target_col not in data.columns:
            self.logger.warning(f"Целевая колонка {target_col} не найдена")
            return {}
        
        # Подготавливаем данные
        X = data[feature_cols].fillna(0)
        y = data[target_col].fillna(0)
        
        # Вычисляем mutual information
        mi_scores = mutual_info_classif(X, y, discrete_features=False, n_neighbors=3)
        
        # Создаем словарь важности
        importance_dict = {}
        for i, col in enumerate(feature_cols):
            importance_dict[col] = mi_scores[i]
        
        # Сортируем и берем топ N
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:top_n])
        
        self.feature_importance = importance_dict
        return importance_dict
    
    def log_top_features(self, importance_dict: Dict[str, float], 
                        top_n: int = 25, 
                        title: str = "Топ важных признаков"):
        """
        Логирование топ важных признаков
        
        Args:
            importance_dict: словарь с важностью признаков
            top_n: количество топ признаков для вывода
            title: заголовок для логов
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 {title}")
        self.logger.info(f"{'='*60}")
        
        # Группировка признаков по категориям
        categories = {
            'Технические индикаторы': ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'adx', 
                                       'cci', 'willr', 'stoch', 'obv', 'cmf', 'mfi'],
            'Ценовые признаки': ['returns', 'log_returns', 'volatility', 'high_low', 
                                 'close_open', 'price_position', 'price_ma_ratio'],
            'Объемные признаки': ['volume_', 'vwap', 'volume_ratio', 'turnover', 
                                 'buy_sell', 'volume_ma'],
            'Паттерны': ['pattern', 'candle', 'support', 'resistance', 'breakout'],
            'Микроструктура': ['spread', 'imbalance', 'toxicity', 'bid_ask', 'liquidity'],
            'Временные': ['hour', 'minute', 'day', 'month', 'session', 'weekend'],
            'Рыночный контекст': ['btc_', 'sector_', 'market_', 'relative_', 'correlation']
        }
        
        # Подсчет важности по категориям
        category_importance = {cat: 0.0 for cat in categories}
        category_counts = {cat: 0 for cat in categories}
        
        for i, (feature, importance) in enumerate(list(importance_dict.items())[:top_n], 1):
            # Определяем категорию
            feature_category = "Другое"
            for category, keywords in categories.items():
                if any(keyword in feature.lower() for keyword in keywords):
                    feature_category = category
                    category_importance[category] += importance
                    category_counts[category] += 1
                    break
            
            # Форматированный вывод
            self.logger.info(f"{i:2}. {feature:<40} | Важность: {importance:.4f} | {feature_category}")
        
        # Статистика по категориям
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📈 Важность по категориям:")
        self.logger.info(f"{'='*60}")
        
        sorted_categories = sorted(category_importance.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)
        
        for category, total_importance in sorted_categories:
            if category_counts[category] > 0:
                avg_importance = total_importance / category_counts[category]
                self.logger.info(f"{category:<25} | Признаков: {category_counts[category]:2} | "
                               f"Средняя важность: {avg_importance:.4f}")
        
        self.logger.info(f"{'='*60}\n")
        
        return importance_dict
    
    def get_feature_statistics(self, data: pd.DataFrame, 
                              feature_names: List[str]) -> pd.DataFrame:
        """
        Получение статистики по признакам
        
        Args:
            data: DataFrame с данными
            feature_names: список признаков для анализа
            
        Returns:
            DataFrame со статистикой
        """
        stats = []
        
        for feature in feature_names:
            if feature in data.columns:
                col = data[feature]
                stats.append({
                    'Feature': feature,
                    'Mean': col.mean(),
                    'Std': col.std(),
                    'Min': col.min(),
                    'Max': col.max(),
                    'NaN%': (col.isna().sum() / len(col)) * 100,
                    'Zeros%': ((col == 0).sum() / len(col)) * 100,
                    'Unique': col.nunique()
                })
        
        stats_df = pd.DataFrame(stats)
        return stats_df