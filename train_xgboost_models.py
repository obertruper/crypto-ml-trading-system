#!/usr/bin/env python3
"""
Обучение XGBoost моделей для предсказания прибыльности сделок
- Поддержка регрессии и классификации (бинарной/мультиклассовой)
- Использует данные из PostgreSQL
- Совместим с текущей системой
"""

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
import logging
import joblib
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score,
                           roc_curve)
import argparse
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f'logs/xgboost_training_{current_time}'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f'{log_dir}/plots', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Директории для сохранения моделей
MODEL_DIR = 'trained_model/xgboost'
os.makedirs(MODEL_DIR, exist_ok=True)


class XGBoostTrainer:
    def __init__(self, config_path='config.yaml', mode='binary'):
        """Инициализация тренера XGBoost моделей
        
        Args:
            config_path: путь к конфигурации
            mode: режим обучения ('regression', 'binary', 'multiclass')
        """
        # Загружаем конфигурацию
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mode = mode
        logger.info(f"🎯 Режим обучения: {mode}")
        
        self.db_config = self.config['database'].copy()
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
        
        # Список технических индикаторов (как в основной системе)
        self.TECHNICAL_INDICATORS = [
            # Трендовые индикаторы
            'ema_15', 'adx_val', 'adx_plus_di', 'adx_minus_di',
            'macd_val', 'macd_signal', 'macd_hist', 'sar',
            'ichimoku_conv', 'ichimoku_base', 'aroon_up', 'aroon_down',
            'dpo',
            
            # Осцилляторы
            'rsi_val', 'stoch_k', 'stoch_d', 'cci_val', 'williams_r',
            'roc', 'ult_osc', 'mfi',
            
            # Волатильность
            'atr_val', 'bb_upper', 'bb_lower', 'bb_basis',
            'donchian_upper', 'donchian_lower',
            
            # Объемные индикаторы
            'obv', 'cmf', 'volume_sma', 'volume_ratio',
            
            # Vortex индикаторы
            'vortex_vip', 'vortex_vin',
            
            # Производные индикаторы
            'macd_signal_ratio', 'adx_diff', 'bb_position',
            'rsi_dist_from_mid', 'stoch_diff', 'vortex_ratio',
            'ichimoku_diff', 'atr_norm',
            
            # Временные признаки
            'hour', 'day_of_week', 'is_weekend',
            
            # Ценовые паттерны
            'price_change_1', 'price_change_4', 'price_change_16',
            'volatility_4', 'volatility_16'
        ]
        
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_importance = {}
        
    def prepare_binary_labels(self, returns, threshold=0.3):
        """Преобразование в бинарные метки для классификации"""
        return (returns > threshold).astype(int)
    
    def prepare_multiclass_labels(self, returns):
        """Преобразование в мультиклассовые метки"""
        labels = np.zeros(len(returns))
        labels[returns < -0.5] = 0  # Убыточные
        labels[(returns >= -0.5) & (returns < 0.5)] = 1  # Около нуля
        labels[(returns >= 0.5) & (returns < 1.5)] = 2  # Малоприбыльные
        labels[(returns >= 1.5) & (returns < 3)] = 3  # Прибыльные
        labels[returns >= 3] = 4  # Высокоприбыльные
        return labels.astype(int)
        
    def connect_db(self):
        """Подключение к PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            logger.info("✅ Подключение к PostgreSQL установлено")
            return conn
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
            raise
    
    def load_data(self):
        """Загрузка данных из PostgreSQL"""
        logger.info("📊 Загрузка данных из PostgreSQL...")
        
        conn = self.connect_db()
        
        query = """
        SELECT 
            p.symbol, p.timestamp, p.datetime,
            p.technical_indicators,
            p.buy_expected_return,
            p.sell_expected_return,
            p.open, p.high, p.low, p.close, p.volume
        FROM processed_market_data p
        JOIN raw_market_data r ON p.raw_data_id = r.id
        WHERE p.technical_indicators IS NOT NULL
          AND r.market_type = 'futures'
          AND p.buy_expected_return IS NOT NULL
          AND p.sell_expected_return IS NOT NULL
        ORDER BY p.symbol, p.timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"✅ Загружено {len(df)} записей")
        
        # Статистика по символам
        symbol_counts = df['symbol'].value_counts()
        logger.info("📊 Распределение по символам:")
        for symbol, count in symbol_counts.head(10).items():
            logger.info(f"   {symbol}: {count:,} записей")
        
        return df
    
    def prepare_features(self, df):
        """Подготовка признаков из technical_indicators"""
        logger.info("🔧 Подготовка признаков...")
        
        # Извлекаем признаки из JSON
        features = []
        for _, row in df.iterrows():
            feature_values = []
            indicators = row['technical_indicators']
            
            for indicator in self.TECHNICAL_INDICATORS:
                value = indicators.get(indicator, 0.0)
                if value is None or pd.isna(value):
                    value = 0.0
                feature_values.append(float(value))
            
            # Добавляем инженерные признаки
            rsi = indicators.get('rsi_val', 50.0)
            feature_values.append(1.0 if rsi is not None and rsi < 30 else 0.0)  # RSI oversold
            feature_values.append(1.0 if rsi is not None and rsi > 70 else 0.0)  # RSI overbought
            
            macd = indicators.get('macd_val', 0.0)
            macd_signal = indicators.get('macd_signal', 0.0)
            feature_values.append(1.0 if macd is not None and macd_signal is not None and macd > macd_signal else 0.0)
            
            bb_position = indicators.get('bb_position', 0.5)
            feature_values.append(1.0 if bb_position is not None and bb_position < 0.2 else 0.0)
            feature_values.append(1.0 if bb_position is not None and bb_position > 0.8 else 0.0)
            
            adx = indicators.get('adx_val', 0.0)
            feature_values.append(1.0 if adx is not None and adx > 25 else 0.0)
            
            volume_ratio = indicators.get('volume_ratio', 1.0)
            feature_values.append(1.0 if volume_ratio is not None and volume_ratio > 2.0 else 0.0)
            
            features.append(feature_values)
        
        # Создаем DataFrame с признаками
        feature_names = self.TECHNICAL_INDICATORS + [
            'rsi_oversold', 'rsi_overbought', 'macd_bullish',
            'bb_near_lower', 'bb_near_upper', 'strong_trend', 'high_volume'
        ]
        
        X = pd.DataFrame(features, columns=feature_names)
        
        # Целевые переменные
        y_buy = df['buy_expected_return'].values
        y_sell = df['sell_expected_return'].values
        
        logger.info(f"✅ Подготовлено {len(X)} примеров с {len(feature_names)} признаками")
        
        return X, y_buy, y_sell, df['symbol'].values, df['timestamp'].values
    
    def train_model(self, X_train, y_train, X_val, y_val, model_name):
        """Обучение XGBoost модели"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Обучение модели: {model_name}")
        logger.info(f"{'='*60}")
        
        # Параметры XGBoost
        if self.mode == 'regression':
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 50,
                'eval_metric': ['mae', 'rmse']
            }
            model = xgb.XGBRegressor(**params)
        
        elif self.mode == 'binary':
            # Подсчет весов классов для балансировки
            pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
            logger.info(f"📊 Баланс классов - 0: {np.sum(y_train == 0)}, 1: {np.sum(y_train == 1)}")
            logger.info(f"⚖️ Scale pos weight: {pos_ratio:.2f}")
            
            params = {
                'objective': 'binary:logistic',
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': pos_ratio,
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 50,
                'eval_metric': ['auc', 'error']
            }
            model = xgb.XGBClassifier(**params)
            
        else:  # multiclass
            num_classes = len(np.unique(y_train))
            logger.info(f"📊 Количество классов: {num_classes}")
            
            params = {
                'objective': 'multi:softprob',
                'num_class': num_classes,
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 50,
                'eval_metric': ['mlogloss', 'merror']
            }
            model = xgb.XGBClassifier(**params)
        
        # Обучение с early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100
        )
        
        # Предсказания на валидации
        if self.mode == 'regression':
            y_pred = model.predict(X_val)
            
            # Метрики регрессии
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)
            direction_accuracy = np.mean((y_pred > 0) == (y_val > 0))
            
            logger.info(f"\n📊 Результаты для {model_name}:")
            logger.info(f"   MAE: {mae:.4f}%")
            logger.info(f"   RMSE: {rmse:.4f}%")
            logger.info(f"   R²: {r2:.4f}")
            logger.info(f"   Direction Accuracy: {direction_accuracy:.2%}")
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
            
        elif self.mode == 'binary':
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Метрики классификации
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            logger.info(f"\n📊 Результаты для {model_name}:")
            logger.info(f"   Accuracy: {accuracy:.2%}")
            logger.info(f"   Precision: {precision:.2%}")
            logger.info(f"   Recall: {recall:.2%}")
            logger.info(f"   F1-Score: {f1:.4f}")
            logger.info(f"   ROC-AUC: {auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_val, y_pred)
            logger.info(f"   Confusion Matrix:")
            logger.info(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
            logger.info(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm,
                'y_pred_proba': y_pred_proba
            }
            
        else:  # multiclass
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            
            # Метрики мультиклассовой классификации
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"\n📊 Результаты для {model_name}:")
            logger.info(f"   Accuracy: {accuracy:.2%}")
            logger.info(f"   Weighted Precision: {precision:.2%}")
            logger.info(f"   Weighted Recall: {recall:.2%}")
            logger.info(f"   Weighted F1-Score: {f1:.4f}")
            
            # Per-class metrics
            report = classification_report(y_val, y_pred, 
                                         target_names=['Убыточные', 'Около нуля', 'Малоприбыльные', 
                                                      'Прибыльные', 'Высокоприбыльные'])
            logger.info(f"\n   Подробный отчет по классам:\n{report}")
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'classification_report': report,
                'y_pred_proba': y_pred_proba
            }
        
        # Сохраняем важность признаков
        self.feature_importance[model_name] = model.feature_importances_
        # Сохраняем модель в словарь
        self.models[model_name] = model
        
        return model, metrics
    
    def plot_results(self, y_true, y_pred, model_name, metrics):
        """Визуализация результатов"""
        if self.mode == 'regression':
            self._plot_regression_results(y_true, y_pred, model_name, metrics)
        elif self.mode == 'binary':
            self._plot_binary_results(y_true, metrics, model_name)
        else:
            self._plot_multiclass_results(y_true, y_pred, metrics, model_name)
    
    def _plot_regression_results(self, y_true, y_pred, model_name, metrics):
        """Визуализация для регрессии"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'XGBoost Regression Model: {model_name}', fontsize=16)
        
        # График 1: Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Return (%)')
        axes[0, 0].set_ylabel('Predicted Return (%)')
        axes[0, 0].set_title(f'Predictions vs True (R² = {metrics["r2"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Распределение ошибок
        errors = y_pred - y_true
        axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Prediction Error (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Error Distribution (MAE = {metrics["mae"]:.3f}%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Важность признаков (топ-20)
        feature_names = self.TECHNICAL_INDICATORS + [
            'rsi_oversold', 'rsi_overbought', 'macd_bullish',
            'bb_near_lower', 'bb_near_upper', 'strong_trend', 'high_volume'
        ]
        importance = self.feature_importance[model_name]
        indices = np.argsort(importance)[-20:]
        
        axes[1, 0].barh(range(20), importance[indices])
        axes[1, 0].set_yticks(range(20))
        axes[1, 0].set_yticklabels([feature_names[i] for i in indices])
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Top 20 Features')
        axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Статистика
        axes[1, 1].axis('off')
        stats_text = f"""
Performance Summary:
  MAE: {metrics['mae']:.3f}%
  RMSE: {metrics['rmse']:.3f}%
  R²: {metrics['r2']:.3f}
  Direction Accuracy: {metrics['direction_accuracy']:.1%}
  
Model: XGBoost Regression
Trees: {len(self.models[model_name].get_booster().get_dump())}
Max Depth: 8
Learning Rate: 0.05
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{log_dir}/plots/{model_name}_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_binary_results(self, y_true, metrics, model_name):
        """Визуализация для бинарной классификации"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'XGBoost Binary Classification: {model_name}', fontsize=16)
        
        # График 1: ROC кривая
        if 'y_pred_proba' in metrics:
            fpr, tpr, thresholds = roc_curve(y_true, metrics['y_pred_proba'])
            axes[0, 0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {metrics["auc"]:.3f})')
            axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Confusion Matrix
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xticklabels(['Не входить', 'Входить'])
        axes[0, 1].set_yticklabels(['Не входить', 'Входить'])
        
        # График 3: Распределение вероятностей
        if 'y_pred_proba' in metrics:
            axes[1, 0].hist(metrics['y_pred_proba'][y_true == 0], bins=50, alpha=0.7, 
                           label='Класс 0 (Не входить)', density=True)
            axes[1, 0].hist(metrics['y_pred_proba'][y_true == 1], bins=50, alpha=0.7, 
                           label='Класс 1 (Входить)', density=True)
            axes[1, 0].set_xlabel('Predicted Probability')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Probability Distribution by Class')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Метрики
        axes[1, 1].axis('off')
        stats_text = f"""
Performance Summary:
  Accuracy: {metrics['accuracy']:.2%}
  Precision: {metrics['precision']:.2%}
  Recall: {metrics['recall']:.2%}
  F1-Score: {metrics['f1']:.3f}
  ROC-AUC: {metrics['auc']:.3f}
  
Confusion Matrix:
  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}
  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}
  
Model: XGBoost Binary
Trees: {len(self.models[model_name].get_booster().get_dump())}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{log_dir}/plots/{model_name}_binary_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_multiclass_results(self, y_true, y_pred, metrics, model_name):
        """Визуализация для мультиклассовой классификации"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'XGBoost Multiclass Classification: {model_name}', fontsize=16)
        
        # График 1: Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title('Confusion Matrix')
        class_names = ['Убыточные', 'Около нуля', 'Малоприбыльные', 'Прибыльные', 'Высокоприбыльные']
        axes[0, 0].set_xticklabels(class_names, rotation=45)
        axes[0, 0].set_yticklabels(class_names, rotation=0)
        
        # График 2: Per-class accuracy
        per_class_acc = []
        for i in range(len(cm)):
            if cm[i].sum() > 0:
                acc = cm[i, i] / cm[i].sum()
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0)
        
        axes[0, 1].bar(range(len(per_class_acc)), per_class_acc)
        axes[0, 1].set_xticks(range(len(class_names)))
        axes[0, 1].set_xticklabels(class_names, rotation=45)
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Per-Class Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Feature Importance
        feature_names = self.TECHNICAL_INDICATORS + [
            'rsi_oversold', 'rsi_overbought', 'macd_bullish',
            'bb_near_lower', 'bb_near_upper', 'strong_trend', 'high_volume'
        ]
        importance = self.feature_importance[model_name]
        indices = np.argsort(importance)[-15:]
        
        axes[1, 0].barh(range(15), importance[indices])
        axes[1, 0].set_yticks(range(15))
        axes[1, 0].set_yticklabels([feature_names[i] for i in indices])
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Top 15 Features')
        axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Метрики
        axes[1, 1].axis('off')
        stats_text = f"""
Performance Summary:
  Accuracy: {metrics['accuracy']:.2%}
  Weighted Precision: {metrics['precision']:.2%}
  Weighted Recall: {metrics['recall']:.2%}
  Weighted F1-Score: {metrics['f1']:.3f}
  
Model: XGBoost Multiclass
Classes: 5
Trees: {len(self.models[model_name].get_booster().get_dump())}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{log_dir}/plots/{model_name}_multiclass_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def simulate_trading(self, X_test, y_true, y_pred_proba, threshold=0.6):
        """Симуляция торговли по сигналам модели для бинарной классификации"""
        trades = []
        
        # Входим только если вероятность > threshold
        signals = y_pred_proba > threshold
        
        # Считаем метрики торговли
        total_signals = np.sum(signals)
        correct_signals = np.sum(signals & (y_true == 1))
        
        if total_signals > 0:
            win_rate = correct_signals / total_signals
            profit_factor = correct_signals / (total_signals - correct_signals) if (total_signals - correct_signals) > 0 else np.inf
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            'total_signals': total_signals,
            'correct_signals': correct_signals,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'threshold': threshold
        }
    
    def train(self):
        """Основной процесс обучения"""
        logger.info("="*80)
        logger.info("🚀 НАЧАЛО ОБУЧЕНИЯ XGBOOST МОДЕЛЕЙ")
        if self.mode == 'regression':
            logger.info("📊 Задача: предсказание expected returns (регрессия)")
        elif self.mode == 'binary':
            logger.info("📊 Задача: бинарная классификация (входить/не входить)")
        else:
            logger.info("📊 Задача: мультиклассовая классификация (5 классов)")
        logger.info("="*80)
        
        try:
            # Загружаем данные
            df = self.load_data()
            
            # Подготавливаем признаки
            X, y_buy, y_sell, symbols, timestamps = self.prepare_features(df)
            
            # Нормализация
            logger.info("🔄 Нормализация признаков...")
            X_scaled = self.scaler.fit_transform(X)
            
            # Временное разделение (70/15/15)
            n = len(X_scaled)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            
            X_train = X_scaled[:train_end]
            X_val = X_scaled[train_end:val_end]
            X_test = X_scaled[val_end:]
            
            # Преобразование меток в зависимости от режима
            if self.mode == 'binary':
                logger.info("🔄 Преобразование в бинарные метки...")
                y_buy_binary = self.prepare_binary_labels(y_buy, threshold=0.3)
                y_sell_binary = self.prepare_binary_labels(y_sell, threshold=0.3)
                
                logger.info(f"   Buy - Класс 0 (не входить): {np.sum(y_buy_binary == 0):,} ({np.mean(y_buy_binary == 0):.1%})")
                logger.info(f"   Buy - Класс 1 (входить): {np.sum(y_buy_binary == 1):,} ({np.mean(y_buy_binary == 1):.1%})")
                logger.info(f"   Sell - Класс 0 (не входить): {np.sum(y_sell_binary == 0):,} ({np.mean(y_sell_binary == 0):.1%})")
                logger.info(f"   Sell - Класс 1 (входить): {np.sum(y_sell_binary == 1):,} ({np.mean(y_sell_binary == 1):.1%})")
                
                model_configs = [
                    ('buy_classifier', y_buy_binary),
                    ('sell_classifier', y_sell_binary)
                ]
                
            elif self.mode == 'multiclass':
                logger.info("🔄 Преобразование в мультиклассовые метки...")
                y_buy_multi = self.prepare_multiclass_labels(y_buy)
                y_sell_multi = self.prepare_multiclass_labels(y_sell)
                
                logger.info("   Распределение классов Buy:")
                for i in range(5):
                    class_names = ['Убыточные', 'Около нуля', 'Малоприбыльные', 'Прибыльные', 'Высокоприбыльные']
                    logger.info(f"     Класс {i} ({class_names[i]}): {np.sum(y_buy_multi == i):,} ({np.mean(y_buy_multi == i):.1%})")
                
                model_configs = [
                    ('buy_multiclass', y_buy_multi),
                    ('sell_multiclass', y_sell_multi)
                ]
                
            else:  # regression
                model_configs = [
                    ('buy_return_predictor', y_buy),
                    ('sell_return_predictor', y_sell)
                ]
            
            results = {}
            
            for model_name, y_values in model_configs:
                # Разделение целевых значений
                y_train = y_values[:train_end]
                y_val = y_values[train_end:val_end]
                y_test = y_values[val_end:]
                
                # Статистика
                logger.info(f"\n📊 Статистика для {model_name}:")
                logger.info(f"   Train: {len(y_train)} примеров")
                logger.info(f"   Val: {len(y_val)} примеров")
                logger.info(f"   Test: {len(y_test)} примеров")
                logger.info(f"   Среднее: {np.mean(y_train):.3f}%")
                logger.info(f"   Std: {np.std(y_train):.3f}%")
                
                # Обучение
                model, metrics = self.train_model(X_train, y_train, X_val, y_val, model_name)
                
                # Оценка на тесте
                if self.mode == 'regression':
                    y_pred_test = model.predict(X_test)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_direction = np.mean((y_pred_test > 0) == (y_test > 0))
                    
                    test_metrics = {
                        'mae': test_mae,
                        'rmse': test_rmse,
                        'r2': test_r2,
                        'direction_accuracy': test_direction
                    }
                    
                    logger.info(f"\n📊 Результаты на тесте для {model_name}:")
                    logger.info(f"   MAE: {test_mae:.4f}%")
                    logger.info(f"   RMSE: {test_rmse:.4f}%")
                    logger.info(f"   R²: {test_r2:.4f}")
                    logger.info(f"   Direction Accuracy: {test_direction:.2%}")
                    
                elif self.mode == 'binary':
                    y_pred_test = model.predict(X_test)
                    y_pred_test_proba = model.predict_proba(X_test)[:, 1]
                    
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
                    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
                    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
                    test_auc = roc_auc_score(y_test, y_pred_test_proba)
                    test_cm = confusion_matrix(y_test, y_pred_test)
                    
                    test_metrics = {
                        'accuracy': test_accuracy,
                        'precision': test_precision,
                        'recall': test_recall,
                        'f1': test_f1,
                        'auc': test_auc,
                        'confusion_matrix': test_cm,
                        'y_pred_proba': y_pred_test_proba
                    }
                    
                    logger.info(f"\n📊 Результаты на тесте для {model_name}:")
                    logger.info(f"   Accuracy: {test_accuracy:.2%}")
                    logger.info(f"   Precision: {test_precision:.2%}")
                    logger.info(f"   Recall: {test_recall:.2%}")
                    logger.info(f"   F1-Score: {test_f1:.4f}")
                    logger.info(f"   ROC-AUC: {test_auc:.4f}")
                    
                else:  # multiclass
                    y_pred_test = model.predict(X_test)
                    y_pred_test_proba = model.predict_proba(X_test)
                    
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
                    test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
                    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
                    
                    test_metrics = {
                        'accuracy': test_accuracy,
                        'precision': test_precision,
                        'recall': test_recall,
                        'f1': test_f1,
                        'y_pred_proba': y_pred_test_proba
                    }
                    
                    logger.info(f"\n📊 Результаты на тесте для {model_name}:")
                    logger.info(f"   Accuracy: {test_accuracy:.2%}")
                    logger.info(f"   Weighted Precision: {test_precision:.2%}")
                    logger.info(f"   Weighted Recall: {test_recall:.2%}")
                    logger.info(f"   Weighted F1-Score: {test_f1:.4f}")
                
                # Визуализация
                self.plot_results(y_test, y_pred_test, model_name, test_metrics)
                
                # Результаты
                results[model_name] = {
                    'val_metrics': metrics,
                    'test_metrics': test_metrics
                }
            
            # Сохраняем модели
            self.save_models(results)
            
            # Создаем отчет
            self.create_report(results)
            
            logger.info("\n✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            logger.info(f"📊 Результаты сохранены в: {log_dir}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении: {e}")
            raise
    
    def save_models(self, results):
        """Сохранение моделей и метаданных"""
        logger.info("\n💾 Сохранение моделей...")
        
        # Сохраняем модели
        for name, model in self.models.items():
            model_path = f'{MODEL_DIR}/{name}_xgboost.pkl'
            joblib.dump(model, model_path)
            logger.info(f"   ✅ Модель сохранена: {model_path}")
        
        # Сохраняем scaler
        scaler_path = f'{MODEL_DIR}/scaler_xgboost.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"   ✅ Scaler сохранен: {scaler_path}")
        
        # Сохраняем метаданные
        # Конвертируем numpy массивы в списки для JSON сериализации
        json_safe_results = {}
        for model_name, result in results.items():
            json_safe_results[model_name] = {
                'val_metrics': {},
                'test_metrics': {}
            }
            
            # Конвертируем метрики валидации
            for key, value in result['val_metrics'].items():
                if isinstance(value, np.ndarray):
                    continue  # Пропускаем массивы типа y_pred_proba
                elif isinstance(value, (np.float32, np.float64)):
                    json_safe_results[model_name]['val_metrics'][key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    json_safe_results[model_name]['val_metrics'][key] = int(value)
                else:
                    json_safe_results[model_name]['val_metrics'][key] = value
            
            # Конвертируем метрики теста
            for key, value in result['test_metrics'].items():
                if isinstance(value, np.ndarray):
                    if key == 'confusion_matrix':
                        json_safe_results[model_name]['test_metrics'][key] = value.tolist()
                    else:
                        continue  # Пропускаем другие массивы
                elif isinstance(value, (np.float32, np.float64)):
                    json_safe_results[model_name]['test_metrics'][key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    json_safe_results[model_name]['test_metrics'][key] = int(value)
                else:
                    json_safe_results[model_name]['test_metrics'][key] = value
        
        metadata = {
            'type': 'xgboost',
            'mode': self.mode,
            'features': self.TECHNICAL_INDICATORS + [
                'rsi_oversold', 'rsi_overbought', 'macd_bullish',
                'bb_near_lower', 'bb_near_upper', 'strong_trend', 'high_volume'
            ],
            'created_at': datetime.now().isoformat(),
            'results': json_safe_results,
            'feature_importance': {
                name: importance.tolist() 
                for name, importance in self.feature_importance.items()
            }
        }
        
        with open(f'{MODEL_DIR}/metadata_xgboost.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("   ✅ Метаданные сохранены")
    
    def create_report(self, results):
        """Создание итогового отчета"""
        mode_name = {
            'regression': 'Regression (предсказание expected returns)',
            'binary': 'Binary Classification (входить/не входить)',
            'multiclass': 'Multiclass Classification (5 классов)'
        }[self.mode]
        
        report = f"""
{'='*80}
ИТОГОВЫЙ ОТЧЕТ - XGBOOST МОДЕЛИ
{'='*80}

Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Тип: XGBoost {mode_name}

РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:
"""
        
        for model_name, result in results.items():
            test_metrics = result['test_metrics']
            report += f"\n{model_name.upper()}:\n"
            
            if self.mode == 'regression':
                report += f"""
- MAE: {test_metrics['mae']:.4f}%
- RMSE: {test_metrics['rmse']:.4f}%
- R²: {test_metrics['r2']:.4f}
- Direction Accuracy: {test_metrics['direction_accuracy']:.2%}
"""
            elif self.mode == 'binary':
                report += f"""
- Accuracy: {test_metrics['accuracy']:.2%}
- Precision: {test_metrics['precision']:.2%}
- Recall: {test_metrics['recall']:.2%}
- F1-Score: {test_metrics['f1']:.4f}
- ROC-AUC: {test_metrics['auc']:.4f}
"""
            else:  # multiclass
                report += f"""
- Accuracy: {test_metrics['accuracy']:.2%}
- Weighted Precision: {test_metrics['precision']:.2%}
- Weighted Recall: {test_metrics['recall']:.2%}
- Weighted F1-Score: {test_metrics['f1']:.4f}
"""
        
        report += f"""
{'='*80}
Модели сохранены в: {MODEL_DIR}
Логи и графики: {log_dir}
{'='*80}
"""
        
        with open(f'{log_dir}/final_report.txt', 'w') as f:
            f.write(report)
        
        print(report)


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Обучение XGBoost моделей')
    parser.add_argument('--mode', choices=['regression', 'binary', 'multiclass'], 
                       default='binary',
                       help='Режим обучения: regression, binary, multiclass')
    parser.add_argument('--config', default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    logger.info("🚀 Запуск обучения XGBoost моделей")
    logger.info(f"🎯 Режим: {args.mode}")
    
    trainer = XGBoostTrainer(config_path=args.config, mode=args.mode)
    trainer.train()


if __name__ == "__main__":
    main()