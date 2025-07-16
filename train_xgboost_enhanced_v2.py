#!/usr/bin/env python3
"""
Enhanced XGBoost для криптотрейдинга v2.1
Исправлена проблема с "монеткой" и улучшена работа с несбалансированными классами

Особенности:
- Все 92 признака из TFT v2.1 (technical + market + OHLC + symbol)
- Загрузка данных батчами из PostgreSQL
- Поддержка regression и binary classification
- Ensemble моделей с взвешенным голосованием
- Feature importance visualization
- GPU поддержка через tree_method='gpu_hist'
- Продвинутая оптимизация порога (G-mean, profit-based)
- Калибровка вероятностей
- SMOTE для балансировки классов
- Focal Loss для работы с дисбалансом
"""

import os
import sys
import time
import json
import pickle
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import yaml
import warnings
import argparse
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
    average_precision_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import joblib
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Импортируем наши утилиты
    calculate_scale_pos_weight,
    find_optimal_threshold_gmean,
    find_optimal_threshold_profit,
    calibrate_probabilities,
    apply_smote,
    apply_random_oversampler,
    apply_smote_tomek,
    create_focal_loss_objective,
    ensemble_predictions_weighted,
    validate_binary_features
)
from fix_binary_features import recreate_binary_features, separate_features_for_smote, clip_technical_indicators

# Настройка логирования
log_dir = f"logs/xgboost_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f"{log_dir}/plots", exist_ok=True)
os.makedirs("trained_model", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Подавляем предупреждения
warnings.filterwarnings('ignore')

# Проверка GPU
try:
    import GPUtil
    gpus = GPUtil.getGPUs()
    if gpus:
        logger.info(f"🖥️ GPU доступен: {gpus[0].name}")
        USE_GPU = True
    else:
        logger.info("⚠️ GPU не найден, используется CPU")
        USE_GPU = False
except:
    logger.info("⚠️ GPUtil не установлен, проверка GPU пропущена")
    USE_GPU = False


class CacheManager:
    """Менеджер кэширования данных для ускорения повторных запусков"""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_key(self, params: Dict) -> str:
        """Генерация уникального ключа на основе параметров"""
        key_data = {
            'symbols_count': params.get('symbols_count'),
            'date_range': params.get('date_range'),
            'features_version': 'v2.1',  # версия feature engineering
            'task': params.get('task')
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def save(self, data, key: str, description: str):
        """Сохранение данных с метаданными"""
        cache_file = self.cache_dir / f'{key}_{description}.pkl'
        metadata = {
            'created_at': datetime.now(),
            'data_shape': data.shape if hasattr(data, 'shape') else len(data),
            'description': description
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump({'data': data, 'metadata': metadata}, f)
        
        logger.info(f"💾 Сохранено в кэш: {cache_file.name}")
        
    def load(self, key: str, description: str, max_age_days: int = 7):
        """Загрузка данных с проверкой актуальности"""
        cache_file = self.cache_dir / f'{key}_{description}.pkl'
        
        if not cache_file.exists():
            return None
            
        # Проверка возраста файла
        age_days = (datetime.now() - datetime.fromtimestamp(
            cache_file.stat().st_mtime)).days
        
        if age_days > max_age_days:
            logger.warning(f"⚠️ Кэш устарел ({age_days} дней): {cache_file.name}")
            return None
            
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
            
        logger.info(f"📦 Загружено из кэша: {cache_file.name}")
        logger.info(f"   Создан: {cached['metadata']['created_at']}")
        logger.info(f"   Размер: {cached['metadata']['data_shape']}")
        
        return cached['data']


class AdvancedVisualizer:
    """Расширенная визуализация для анализа обучения"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.plots_dir = self.log_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
    def plot_data_overview(self, df: pd.DataFrame):
        """Обзор загруженных данных"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Overview', fontsize=16)
        
        # 1. Распределение по символам
        symbol_counts = df['symbol'].value_counts().head(20)
        axes[0,0].bar(range(len(symbol_counts)), symbol_counts.values)
        axes[0,0].set_title('Топ-20 символов по количеству данных')
        axes[0,0].set_xlabel('Символы')
        axes[0,0].set_ylabel('Количество записей')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Временное покрытие
        if 'timestamp' in df.columns:
            try:
                # Проверяем диапазон timestamp
                min_ts = df['timestamp'].min()
                max_ts = df['timestamp'].max()
                logger.info(f"   Диапазон timestamp: {min_ts} - {max_ts}")
                
                # Фильтруем невалидные timestamp (слишком большие или маленькие)
                valid_mask = (df['timestamp'] > 0) & (df['timestamp'] < 2147483647)  # Unix timestamp до 2038 года
                if not valid_mask.all():
                    logger.warning(f"   ⚠️ Обнаружены невалидные timestamp: {(~valid_mask).sum()} записей")
                    df_valid = df[valid_mask].copy()
                else:
                    df_valid = df
                    
                df_valid['date'] = pd.to_datetime(df_valid['timestamp'], unit='s', errors='coerce')
                date_range = df_valid.groupby('symbol')['date'].agg(['min', 'max'])
                date_range['days'] = (date_range['max'] - date_range['min']).dt.days
                
                axes[0,1].scatter(range(len(date_range.head(20))), 
                                date_range['days'].head(20))
                axes[0,1].set_title('Временное покрытие по символам (дни)')
                axes[0,1].set_xlabel('Символы')
                axes[0,1].set_ylabel('Дни')
            except Exception as e:
                logger.warning(f"   ⚠️ Ошибка при обработке дат: {e}")
                axes[0,1].text(0.5, 0.5, 'Ошибка обработки дат', 
                             horizontalalignment='center', verticalalignment='center')
                axes[0,1].set_title('Временное покрытие недоступно')
        
        # 3. Распределение expected returns
        if 'buy_expected_return' in df.columns and 'sell_expected_return' in df.columns:
            axes[1,0].hist([df['buy_expected_return'].dropna(), 
                          df['sell_expected_return'].dropna()], 
                          bins=50, alpha=0.7, label=['Buy', 'Sell'])
            axes[1,0].set_title('Распределение Expected Returns')
            axes[1,0].set_xlabel('Return %')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].legend()
            axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Корреляция основных признаков
        key_features = ['rsi_val', 'macd_hist', 'adx_val', 'volume_ratio']
        available_features = [f for f in key_features if f in df.columns]
        if available_features:
            corr_data = df[available_features].corr()
            sns.heatmap(corr_data, annot=True, ax=axes[1,1], cmap='coolwarm',
                       center=0, vmin=-1, vmax=1)
            axes[1,1].set_title('Корреляция ключевых признаков')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'data_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 График сохранен: data_overview.png")
        
    def plot_feature_distributions(self, X: np.ndarray, feature_names: List[str], 
                                 sample_size: int = 10000):
        """Распределения важных признаков"""
        # Случайная выборка для скорости
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Выбираем топ признаки для визуализации
        important_features = [
            'momentum_score', 'volume_strength_score', 'volatility_regime_score',
            'rsi_val', 'macd_bullish', 'adx_val', 'bb_position', 'atr_norm'
        ]
        
        available_features = []
        feature_indices = []
        for feature in important_features:
            if feature in feature_names:
                available_features.append(feature)
                feature_indices.append(feature_names.index(feature))
        
        if not available_features:
            logger.warning("⚠️ Не найдены важные признаки для визуализации")
            return
            
        n_features = len(available_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        fig.suptitle('Feature Distributions', fontsize=16)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (idx, feature) in enumerate(zip(feature_indices, available_features)):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            data = X_sample[:, idx]
            ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribution: {feature}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
            # Добавляем статистику
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}', linewidth=2)
            ax.axvline(median_val, color='green', linestyle='--', 
                      label=f'Median: {median_val:.2f}', linewidth=2)
            ax.legend()
        
        # Убираем пустые subplot'ы
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 График сохранен: feature_distributions.png")
        
    def plot_training_comparison(self, metrics_history: Dict):
        """Сравнение метрик разных моделей"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Метрики для отображения
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            for model_name, history in metrics_history.items():
                if metric in history:
                    values = history[metric]
                    if isinstance(values, list):
                        ax.plot(values, label=model_name, marker='o')
                    else:
                        ax.bar(model_name, values)
                    
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xlabel('Model/Iteration')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 График сохранен: model_comparison.png")


class XGBoostEnhancedTrainer:
    """Enhanced XGBoost trainer с поддержкой всех фич из TFT v2.1"""
    
    def __init__(self, config_path='config.yaml'):
        """Инициализация тренера"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_config = self.config['database'].copy()
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
            
        # Признаки из TFT v2.1 - исправленные названия из БД
        self.TECHNICAL_INDICATORS = [
            # Трендовые индикаторы
            'ema_15', 'adx_val', 'adx_plus_di', 'adx_minus_di', 'adx_diff',
            'macd_val', 'macd_signal', 'macd_hist', 'macd_signal_ratio',
            'sar', 'ichimoku_conv', 'ichimoku_base', 'ichimoku_diff',
            'aroon_up', 'aroon_down', 'dpo',
            # Осцилляторы
            'rsi_val', 'rsi_dist_from_mid', 'stoch_k', 'stoch_d', 'stoch_diff',
            'cci_val', 'roc', 'williams_r', 'ult_osc', 'mfi',
            # Волатильность
            'atr_val', 'atr_norm', 'bb_position', 'bb_upper', 'bb_lower', 'bb_basis',
            'donchian_upper', 'donchian_lower',
            # Объем
            'obv', 'cmf', 'volume_sma',
            # Vortex
            'vortex_vip', 'vortex_vin', 'vortex_ratio',
            # Изменения цены и волатильность
            'price_change_1', 'price_change_4', 'price_change_16',
            'volatility_4', 'volatility_16', 'volume_ratio',
            # Временные признаки
            'hour', 'day_of_week', 'is_weekend'
        ]
        
        self.MARKET_FEATURES = [
            # BTC корреляции и метрики (эти рассчитываются отдельно)
            'btc_correlation_20', 'btc_correlation_60',
            'btc_return_1h', 'btc_return_4h', 'btc_volatility',
            'relative_strength_btc',
            'market_regime_low_vol', 'market_regime_med_vol', 'market_regime_high_vol',
            # Циклические временные признаки
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        ]
        
        self.OHLC_FEATURES = [
            'open_ratio', 'high_ratio', 'low_ratio', 'hl_spread',
            'body_size', 'upper_shadow', 'lower_shadow', 'is_bullish',
            'log_return', 'log_volume',
            'price_to_ema15', 'price_to_ema50', 'price_to_vwap'
        ]
        
        # Symbol features будут добавлены динамически
        self.TOP_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                           'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT']
        
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.cache_manager = None
        self.visualizer = None
        self.use_cache = False
        self.force_reload = False
        
    def connect_db(self):
        """Подключение к PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            logger.info("✅ Подключение к PostgreSQL установлено")
            return conn
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к БД: {e}")
            raise
            
    def load_data_batch(self, symbols: List[str], conn) -> pd.DataFrame:
        """Загрузка данных для списка символов с извлечением технических индикаторов"""
        # ВАЖНО: Явно указываем ТОЛЬКО безопасные колонки
        # НЕ загружаем: buy/sell_profit_target, buy/sell_loss_target, max_profit, realized_profit
        query = """
        SELECT 
            pm.symbol, pm.timestamp, pm.datetime,
            pm.technical_indicators,
            pm.buy_expected_return,  -- ТОЛЬКО для целевой переменной
            pm.sell_expected_return, -- ТОЛЬКО для целевой переменной
            rm.open, rm.high, rm.low, rm.close, rm.volume
        FROM processed_market_data pm
        JOIN raw_market_data rm ON pm.raw_data_id = rm.id
        WHERE pm.symbol = ANY(%s)
        ORDER BY pm.timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(symbols,))
        logger.info(f"   ✅ Загружено {len(df):,} записей для {symbols}")
        
        # Сохраняем целевые переменные отдельно перед их удалением
        target_columns = ['buy_expected_return', 'sell_expected_return']
        
        # Извлекаем технические индикаторы из JSON
        if 'technical_indicators' in df.columns:
            logger.info("   📊 Извлечение технических индикаторов из JSON...")
            indicators_df = pd.json_normalize(df['technical_indicators'])
            
            # ВАЖНО: Проверяем, что в technical_indicators нет целевых переменных
            for target_col in target_columns:
                if target_col in indicators_df.columns:
                    logger.warning(f"   ⚠️ ОБНАРУЖЕНА УТЕЧКА: {target_col} найден в technical_indicators! Удаляем...")
                    indicators_df = indicators_df.drop(columns=[target_col])
            
            # Объединяем с основным DataFrame
            df = pd.concat([df, indicators_df], axis=1)
            
            # Логируем найденные индикаторы
            found_indicators = [col for col in indicators_df.columns if col in self.TECHNICAL_INDICATORS]
            logger.info(f"   ✅ Найдено {len(found_indicators)} из {len(self.TECHNICAL_INDICATORS)} индикаторов")
            
            # Детальная диагностика для первого батча
            if len(symbols) == 2 and 'BTCUSDT' in symbols:
                missing_indicators = [ind for ind in self.TECHNICAL_INDICATORS if ind not in indicators_df.columns]
                if missing_indicators:
                    logger.warning(f"   ⚠️ Не найдено {len(missing_indicators)} индикаторов: {missing_indicators[:5]}...")
                
                extra_indicators = [col for col in indicators_df.columns if col not in self.TECHNICAL_INDICATORS]
                if extra_indicators and len(extra_indicators) < 10:
                    logger.info(f"   🔍 Дополнительные индикаторы в БД: {extra_indicators}")
                    
                # Диагностика значений ключевых индикаторов
                logger.info("   📊 Проверка значений индикаторов:")
                if 'rsi_val' in indicators_df.columns:
                    rsi_stats = indicators_df['rsi_val'].describe()
                    logger.info(f"      RSI: min={rsi_stats['min']:.2f}, max={rsi_stats['max']:.2f}, mean={rsi_stats['mean']:.2f}")
                    
                if 'macd_hist' in indicators_df.columns:
                    macd_stats = indicators_df['macd_hist'].describe()
                    logger.info(f"      MACD hist: min={macd_stats['min']:.4f}, max={macd_stats['max']:.4f}, mean={macd_stats['mean']:.4f}")
                    
                if 'adx_val' in indicators_df.columns:
                    adx_stats = indicators_df['adx_val'].describe()
                    logger.info(f"      ADX: min={adx_stats['min']:.2f}, max={adx_stats['max']:.2f}, mean={adx_stats['mean']:.2f}")
            
            # Добавляем отсутствующие индикаторы со значениями по умолчанию
            for indicator in self.TECHNICAL_INDICATORS:
                if indicator not in df.columns:
                    df[indicator] = 0.0
        
        # Логируем доступные колонки для отладки
        if len(symbols) == 2 and 'BTCUSDT' in symbols:  # Только для тестового режима
            logger.info(f"   📋 Всего колонок: {len(df.columns)}")
            if 'buy_expected_return' in df.columns and 'sell_expected_return' in df.columns:
                logger.info("   ✅ Целевые колонки найдены и будут использованы ТОЛЬКО как targets")
            
        return df
        
    def load_market_data(self, conn) -> Dict[str, pd.DataFrame]:
        """Загрузка данных BTC и других топ монет для корреляций"""
        market_data = {}
        
        # Загружаем BTC
        query = """
        SELECT timestamp, close, volume,
               (high - low) / close as volatility
        FROM raw_market_data
        WHERE symbol = 'BTCUSDT'
        ORDER BY timestamp
        """
        btc_df = pd.read_sql_query(query, conn)
        btc_df['return_1h'] = btc_df['close'].pct_change(4)
        btc_df['return_4h'] = btc_df['close'].pct_change(16)
        btc_df['volatility_20'] = btc_df['volatility'].rolling(20).mean()
        market_data['BTCUSDT'] = btc_df
        logger.info(f"✅ Загружены данные BTC: {len(btc_df)} записей")
        
        # Загружаем топ альткоины для корреляций
        alt_symbols = ['ETHUSDT', 'BNBUSDT', 'XRPUSDT']
        for symbol in alt_symbols:
            query = f"""
            SELECT timestamp, close
            FROM raw_market_data
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
            """
            market_data[symbol] = pd.read_sql_query(query, conn)
            
        logger.info(f"✅ Загружены корреляции с: {alt_symbols}")
        return market_data
        
    def calculate_market_features(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Расчет market features как в TFT v2.1"""
        logger.info("🔧 Расчет рыночных признаков и OHLC features...")
        
        # Временные признаки
        # Проверяем тип timestamp и конвертируем правильно
        if df['timestamp'].dtype == 'int64' or df['timestamp'].dtype == 'float64':
            # Проверяем масштаб timestamp
            sample_ts = df['timestamp'].iloc[0]
            if sample_ts > 1e10:  # Вероятно миллисекунды
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            else:  # Секунды
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        else:
            df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        # Извлекаем временные признаки
        # hour и day_of_week уже могут быть в БД из prepare_dataset.py
        if 'hour' not in df.columns and 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
        if 'day_of_week' not in df.columns and 'datetime' in df.columns:
            df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Циклические признаки (используем day_of_week вместо dow)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        if 'day_of_week' in df.columns:
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        # is_weekend уже создается в prepare_dataset.py и сохраняется в БД
        # Не создаем дубликат
        # df['is_weekend'] = (df['dow'] >= 5).astype(int)
        
        # Удаляем временную колонку datetime если она не нужна
        if 'datetime' in df.columns and 'datetime' not in self.MARKET_FEATURES:
            df = df.drop(columns=['datetime'])
        
        # Merge с BTC данными
        btc_df = market_data['BTCUSDT']
        df = df.merge(
            btc_df[['timestamp', 'return_1h', 'return_4h', 'volatility_20']],
            on='timestamp',
            how='left',
            suffixes=('', '_btc')
        )
        df.rename(columns={
            'return_1h': 'btc_return_1h',
            'return_4h': 'btc_return_4h',
            'volatility_20': 'btc_volatility'
        }, inplace=True)
        
        # Корреляции с BTC (упрощенная версия для скорости)
        # Просто добавляем BTC цену к каждой записи для расчета корреляций
        df = df.merge(
            btc_df[['timestamp', 'close']].rename(columns={'close': 'btc_close'}),
            on='timestamp',
            how='left'
        )
        
        # Проверяем на дублированные колонки после merge
        if df.columns.duplicated().any():
            logger.warning(f"Найдены дублированные колонки: {df.columns[df.columns.duplicated()].tolist()}")
            # Удаляем дубликаты, оставляя первую
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Рассчитываем корреляции через rolling (без concat для сохранения всех колонок)
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        for window in [20, 60]:
            # Инициализируем колонку
            df[f'btc_correlation_{window}'] = 0.0
            
            # Рассчитываем для каждого символа
            for symbol in df['symbol'].unique():
                mask = df['symbol'] == symbol
                # Рассчитываем корреляцию и сразу заполняем NaN нулями
                df.loc[mask, f'btc_correlation_{window}'] = (
                    df.loc[mask, 'close'].rolling(window)
                    .corr(df.loc[mask, 'btc_close'])
                    .fillna(0)
                )
        
        df.drop('btc_close', axis=1, inplace=True)
            
        # Относительная сила к BTC
        btc_returns = btc_df.set_index('timestamp')['close'].pct_change(20)
        df['symbol_return_20'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(20))
        df = df.merge(
            btc_returns.reset_index().rename(columns={'close': 'btc_return_20'}),
            on='timestamp',
            how='left'
        )
        df['relative_strength_btc'] = df['symbol_return_20'] / df['btc_return_20'].replace(0, np.nan)
        df.drop(['symbol_return_20', 'btc_return_20'], axis=1, inplace=True)
        
        # Проверка на дубликаты после всех merge операций
        if df.columns.duplicated().any():
            logger.warning(f"Дубликаты после merge: {df.columns[df.columns.duplicated()].tolist()}")
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Market regime
        btc_vol = df['btc_volatility'].fillna(df['btc_volatility'].mean())
        vol_percentiles = btc_vol.quantile([0.33, 0.67])
        df['market_regime_low_vol'] = (btc_vol <= vol_percentiles[0.33]).astype(int)
        df['market_regime_med_vol'] = ((btc_vol > vol_percentiles[0.33]) & (btc_vol <= vol_percentiles[0.67])).astype(int)
        df['market_regime_high_vol'] = (btc_vol > vol_percentiles[0.67]).astype(int)
        
        # OHLC features
        df['open_ratio'] = df['open'] / df['close']
        df['high_ratio'] = df['high'] / df['close']
        df['low_ratio'] = df['low'] / df['close']
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = np.abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['log_return'] = df.groupby('symbol')['close'].transform(
            lambda x: np.log(x / x.shift(1).replace(0, np.nan))
        )
        df['log_volume'] = np.log1p(df['volume'])
        
        # Циклические временные признаки
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            logger.info("✅ Добавлены циклические часовые признаки")
        
        if 'day_of_week' in df.columns:
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            logger.info("✅ Добавлены циклические дневные признаки")
        
        # Относительно индикаторов (проверяем наличие колонок)
        if 'ema_15' in df.columns:
            df['price_to_ema15'] = df['close'] / df['ema_15'].replace(0, np.nan)
        else:
            df['price_to_ema15'] = 1.0
            
        # EMA50 нет в БД, но можно рассчитать
        # Пока используем только ema_15
        df['price_to_ema50'] = 1.0
        df['price_to_vwap'] = 1.0
        
        # Заполняем NaN и inf значения
        # Сначала заменяем inf на NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Заполняем NaN значения
        # Для корреляций используем 0
        for col in df.columns:
            if 'correlation' in col:
                df[col].fillna(0, inplace=True)
            elif 'ratio' in col or 'relative' in col:
                df[col].fillna(1, inplace=True)
            else:
                # Для остальных - forward fill, затем 0
                df[col].fillna(method='ffill', inplace=True)
                df[col].fillna(0, inplace=True)
        
        # КРИТИЧЕСКИ ВАЖНО: Проверяем, что целевые переменные не попали в market features
        target_columns = ['buy_expected_return', 'sell_expected_return', 'expected_return_buy', 'expected_return_sell']
        for target_col in target_columns:
            if target_col in df.columns and target_col not in ['buy_expected_return', 'sell_expected_return']:
                logger.error(f"🚨 ОБНАРУЖЕНА УТЕЧКА в market features: {target_col}! Это может привести к переобучению!")
                logger.warning(f"   Колонка {target_col} не должна использоваться как признак!")
        
        logger.info("✅ Рыночные признаки рассчитаны")
        return df
        
    def load_data_parallel(self, symbols: List[str], conn, batch_size: int = 10) -> pd.DataFrame:
        """Параллельная загрузка данных по символам для ускорения"""
        logger.info(f"⚡ Параллельная загрузка данных для {len(symbols)} символов...")
        
        # Разбиваем символы на батчи
        symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for batch in symbol_batches:
                future = executor.submit(self.load_data_batch, batch, conn)
                futures.append(future)
            
            all_data = []
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Загрузка батчей"):
                try:
                    df_batch = future.result()
                    all_data.append(df_batch)
                except Exception as e:
                    logger.error(f"❌ Ошибка загрузки батча: {e}")
                    
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            logger.info(f"✅ Загружено всего {len(df):,} записей")
            return df
        else:
            raise Exception("Не удалось загрузить данные")
        
    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизация использования памяти DataFrame"""
        logger.info("💾 Оптимизация памяти...")
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if col not in ['timestamp', 'datetime']:  # Не трогаем временные метки
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                            
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"   Память уменьшена с {start_mem:.1f} MB до {end_mem:.1f} MB "
                   f"({100 * (start_mem - end_mem) / start_mem:.1f}% экономии)")
        
        return df
        
    def create_weighted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание взвешенных комбинаций признаков для поиска паттернов"""
        logger.info("🔄 Создание взвешенных комбинаций признаков...")
        
        # 1. Мультипликативные взаимодействия (a*b)
        if 'rsi_val' in df.columns and 'macd_hist' in df.columns:
            df['rsi_macd_interaction'] = df['rsi_val'] * df['macd_hist']
        
        if 'volume_ratio' in df.columns and 'volatility_4' in df.columns:
            df['volume_volatility_interaction'] = df['volume_ratio'] * df['volatility_4']
            
        if 'adx_val' in df.columns and 'strong_trend' in df.columns:
            df['adx_trend_strength'] = df['adx_val'] * df['strong_trend']
        
        # 2. Соотношения (a/b)
        if 'rsi_val' in df.columns and 'adx_val' in df.columns:
            df['rsi_to_adx'] = df['rsi_val'] / (df['adx_val'] + 1e-8)
            
        if 'volume_ratio' in df.columns and 'volatility_16' in df.columns:
            df['volume_to_volatility'] = df['volume_ratio'] / (df['volatility_16'] + 1e-8)
            
        if 'price_change_4' in df.columns and 'price_change_16' in df.columns:
            df['price_momentum_ratio'] = df['price_change_4'] / (df['price_change_16'] + 1e-8)
        
        # 3. Аддитивные комбинации (a+b)
        momentum_features = ['rsi_val', 'macd_hist', 'roc_val']
        if all(f in df.columns for f in momentum_features):
            df['momentum_composite'] = df[momentum_features].sum(axis=1)
            
        volatility_features = ['atr_val', 'volatility_4']
        if all(f in df.columns for f in volatility_features):
            df['volatility_composite'] = df[volatility_features].sum(axis=1)
        
        # 4. Сложные паттерны с весами для точек входа
        if all(f in df.columns for f in ['rsi_val', 'macd_bullish', 'volume_spike', 'bb_position']):
            # Паттерн oversold reversal
            df['oversold_reversal_score'] = (
                np.maximum(0, 30 - df['rsi_val']) * 0.4 +  # Чем ниже RSI от 30, тем выше вес
                df['macd_bullish'] * 20 +                   # Bullish MACD crossover
                df['volume_spike'] * 15 +                   # Volume confirmation
                (df['bb_position'] < 0.2).astype(int) * 15  # Near lower Bollinger band
            )
            
        if all(f in df.columns for f in ['bb_position', 'adx_val', 'volume_spike', 'rsi_val']):
            # Паттерн breakout
            df['breakout_score'] = (
                (df['bb_position'] > 0.8).astype(int) * 20 +  # Near upper band
                (df['adx_val'] > 25).astype(int) * 15 +       # Strong trend
                df['volume_spike'] * 20 +                      # Volume breakout
                (df['rsi_val'] > 50).astype(int) * 10         # Momentum confirmation
            )
            
        # 5. Market regime паттерны
        if all(f in df.columns for f in ['market_regime_low_vol', 'adx_val', 'volume_ratio']):
            # Паттерн для range trading - исправлено
            # market_regime_low_vol уже является бинарным (0 или 1)
            df['range_trading_score'] = (
                df['market_regime_low_vol'].astype(float) * 30 +
                (df['adx_val'] < 20).astype(float) * 20 +
                (df['volume_ratio'] < 1).astype(float) * 10
            )
            
            # Проверка что score не всегда 0
            if df['range_trading_score'].sum() == 0:
                logger.warning("⚠️ range_trading_score всегда 0, проверьте признаки")
                logger.info(f"   market_regime_low_vol mean: {df['market_regime_low_vol'].mean():.3f}")
                logger.info(f"   adx_val < 20: {(df['adx_val'] < 20).mean():.3f}")
                logger.info(f"   volume_ratio < 1: {(df['volume_ratio'] < 1).mean():.3f}")
            
        # 6. Divergence паттерны
        if 'rsi_val' in df.columns and 'price_change_4' in df.columns:
            # RSI divergence (цена растет, RSI падает)
            df['rsi_divergence'] = (
                (df['price_change_4'] > 0) & 
                (df['rsi_val'].diff(4) < 0)
            ).astype(int)
            
        logger.info(f"✅ Создано {len([c for c in df.columns if any(p in c for p in ['interaction', 'score', 'composite', 'ratio'])])} взвешенных признаков")
        return df
    
    def add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление скользящих статистик для индикаторов"""
        logger.info("📊 Добавление скользящих статистик...")
        
        # RSI скользящие статистики
        if 'rsi_val' in df.columns:
            df['rsi_rolling_mean_10'] = df.groupby('symbol')['rsi_val'].transform(lambda x: x.rolling(10, min_periods=1).mean())
            df['rsi_rolling_std_10'] = df.groupby('symbol')['rsi_val'].transform(lambda x: x.rolling(10, min_periods=1).std())
            df['rsi_rolling_max_10'] = df.groupby('symbol')['rsi_val'].transform(lambda x: x.rolling(10, min_periods=1).max())
            df['rsi_rolling_min_10'] = df.groupby('symbol')['rsi_val'].transform(lambda x: x.rolling(10, min_periods=1).min())
        
        # Volume скользящие статистики
        if 'volume' in df.columns:
            df['volume_rolling_mean_20'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
            df['volume_rolling_std_20'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=1).std())
            # Volume trend
            df['volume_trend'] = df['volume_rolling_mean_20'] / df.groupby('symbol')['volume'].transform(lambda x: x.rolling(60, min_periods=1).mean())
        
        # Price momentum статистики
        if 'close' in df.columns:
            df['price_momentum_10'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(10))
            df['price_momentum_20'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(20))
            df['momentum_acceleration'] = df['price_momentum_10'] - df['price_momentum_20']
        
        # ATR скользящие статистики
        if 'atr_val' in df.columns:
            df['atr_rolling_mean_10'] = df.groupby('symbol')['atr_val'].transform(lambda x: x.rolling(10, min_periods=1).mean())
            df['atr_expansion'] = df['atr_val'] / df['atr_rolling_mean_10']
        
        logger.info("✅ Добавлены скользящие статистики")
        return df
    
    def add_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление дивергенций между ценой и индикаторами"""
        logger.info("🔄 Добавление дивергенций...")
        
        # RSI дивергенция
        if 'rsi_val' in df.columns and 'price_change_4' in df.columns:
            # Bullish divergence: цена падает, RSI растет
            df['rsi_bullish_divergence'] = (
                (df['price_change_4'] < 0) & 
                (df.groupby('symbol')['rsi_val'].transform(lambda x: x.diff(4)) > 0)
            ).astype(int)
            
            # Bearish divergence: цена растет, RSI падает
            df['rsi_bearish_divergence'] = (
                (df['price_change_4'] > 0) & 
                (df.groupby('symbol')['rsi_val'].transform(lambda x: x.diff(4)) < 0)
            ).astype(int)
        
        # MACD дивергенция
        if 'macd_hist' in df.columns and 'price_change_4' in df.columns:
            df['macd_bullish_divergence'] = (
                (df['price_change_4'] < 0) & 
                (df.groupby('symbol')['macd_hist'].transform(lambda x: x.diff(4)) > 0)
            ).astype(int)
            
            df['macd_bearish_divergence'] = (
                (df['price_change_4'] > 0) & 
                (df.groupby('symbol')['macd_hist'].transform(lambda x: x.diff(4)) < 0)
            ).astype(int)
        
        # Volume-Price дивергенция
        if 'volume' in df.columns and 'close' in df.columns:
            volume_change = df.groupby('symbol')['volume'].transform(lambda x: x.pct_change(4))
            price_change = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(4))
            
            # Volume растет, цена падает - подозрительно
            df['volume_price_divergence'] = (
                (volume_change > 0.5) & (price_change < -0.01)
            ).astype(int)
        
        logger.info("✅ Добавлены дивергенции")
        return df
    
    def add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление паттернов свечей"""
        logger.info("🕯️ Добавление паттернов свечей...")
        
        # Hammer pattern
        df['is_hammer'] = (
            (df['lower_shadow'] > df['body_size'] * 2) &
            (df['upper_shadow'] < df['body_size'] * 0.5) &
            (df['body_size'] < df['hl_spread'] * 0.3)
        ).astype(int)
        
        # Doji pattern
        df['is_doji'] = (df['body_size'] < df['hl_spread'] * 0.1).astype(int)
        
        # Engulfing pattern
        prev_body = df.groupby('symbol')['body_size'].shift(1)
        prev_close = df.groupby('symbol')['close'].shift(1)
        prev_open = df.groupby('symbol')['open'].shift(1)
        
        # Bullish engulfing
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &  # Текущая свеча зеленая
            (prev_close < prev_open) &    # Предыдущая свеча красная
            (df['body_size'] > prev_body * 1.5) &  # Текущее тело больше
            (df['close'] > prev_open) &   # Закрытие выше предыдущего открытия
            (df['open'] < prev_close)     # Открытие ниже предыдущего закрытия
        ).astype(int)
        
        # Bearish engulfing
        df['bearish_engulfing'] = (
            (df['close'] < df['open']) &  # Текущая свеча красная
            (prev_close > prev_open) &    # Предыдущая свеча зеленая
            (df['body_size'] > prev_body * 1.5) &
            (df['close'] < prev_open) &
            (df['open'] > prev_close)
        ).astype(int)
        
        # Pin bar pattern
        df['pin_bar'] = (
            ((df['lower_shadow'] > df['body_size'] * 3) | 
             (df['upper_shadow'] > df['body_size'] * 3)) &
            (df['body_size'] < df['hl_spread'] * 0.25)
        ).astype(int)
        
        logger.info("✅ Добавлены паттерны свечей")
        return df
    
    def add_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление volume profile признаков"""
        logger.info("📈 Добавление volume profile...")
        
        # VWAP distance
        if 'vwap' in df.columns:
            df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        elif 'close' in df.columns and 'volume' in df.columns:
            # Рассчитываем VWAP если его нет
            df['vwap'] = df.groupby('symbol').apply(
                lambda x: (x['close'] * x['volume']).rolling(20, min_periods=1).sum() / 
                         x['volume'].rolling(20, min_periods=1).sum()
            ).reset_index(level=0, drop=True)
            df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Volume concentration
        if 'volume' in df.columns:
            # Концентрация объема в последних N барах
            df['volume_concentration_5'] = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(5, min_periods=1).sum() / x.rolling(20, min_periods=1).sum()
            )
        
        logger.info("✅ Добавлены volume profile признаки")
        return df
        
    def create_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание symbol features (one-hot encoding)"""
        # One-hot для топ символов
        for symbol in self.TOP_SYMBOLS:
            df[f'is_{symbol.replace("USDT", "").lower()}'] = (df['symbol'] == symbol).astype(int)
            
        # Категории монет
        major_coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        meme_coins = ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT']
        defi_coins = ['UNIUSDT', 'AAVEUSDT', 'CRVUSDT', 'COMPUSDT']
        
        df['is_major'] = df['symbol'].isin(major_coins).astype(int)
        df['is_meme'] = df['symbol'].isin(meme_coins).astype(int)
        df['is_defi'] = df['symbol'].isin(defi_coins).astype(int)
        df['is_alt'] = (~df['symbol'].isin(major_coins)).astype(int)
        
        # Дублируем market regime для совместимости
        df['market_regime_low_vol'] = df['market_regime_low_vol']
        df['market_regime_med_vol'] = df['market_regime_med_vol']
        df['market_regime_high_vol'] = df['market_regime_high_vol']
        
        return df
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Подготовка признаков для XGBoost"""
        logger.info(f"\n📊 Подготовка признаков. Размер данных: {df.shape}")
        
        # Диагностика: какие колонки есть в данных
        tech_indicators = [col for col in df.columns if col in self.TECHNICAL_INDICATORS]
        logger.info(f"\n🔍 Найдено технических индикаторов: {len(tech_indicators)}")
        
        # Проверяем наличие ключевых индикаторов
        key_indicators = ['rsi_val', 'macd_hist', 'adx_val', 'bb_upper', 'bb_lower']
        missing_key = [ind for ind in key_indicators if ind not in df.columns]
        if missing_key:
            logger.warning(f"⚠️ Отсутствуют ключевые индикаторы: {missing_key}")
        
        # КРИТИЧЕСКИ ВАЖНО: Сначала извлекаем и сохраняем целевые переменные
        if 'buy_expected_return' not in df.columns:
            logger.error("❌ Отсутствует колонка 'buy_expected_return'")
            logger.info(f"Доступные колонки: {list(df.columns)}")
            raise KeyError("buy_expected_return not found in DataFrame")
            
        if 'sell_expected_return' not in df.columns:
            logger.error("❌ Отсутствует колонка 'sell_expected_return'")
            raise KeyError("sell_expected_return not found in DataFrame")
            
        # Сохраняем целевые переменные
        y_buy = df['buy_expected_return'].values
        y_sell = df['sell_expected_return'].values
        
        # ЛОГИРОВАНИЕ: Показываем статистику целевых переменных
        logger.info("\n📊 ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ:")
        logger.info(f"   buy_expected_return: min={y_buy.min():.2f}, max={y_buy.max():.2f}, mean={y_buy.mean():.2f}, std={y_buy.std():.2f}")
        logger.info(f"   sell_expected_return: min={y_sell.min():.2f}, max={y_sell.max():.2f}, mean={y_sell.mean():.2f}, std={y_sell.std():.2f}")
        
        # ВАЖНО: Расширенный список колонок для исключения (ВСЕ колонки с информацией о будущем)
        columns_to_exclude = [
            # Целевые переменные
            'buy_expected_return', 'sell_expected_return',
            'expected_return_buy', 'expected_return_sell',
            # Бинарные метки (тоже содержат информацию о будущем!)
            'buy_profit_target', 'buy_loss_target',
            'sell_profit_target', 'sell_loss_target',
            # Реализованная и максимальная прибыль (информация о будущем!)
            'buy_max_profit', 'sell_max_profit',
            'buy_realized_profit', 'sell_realized_profit',
            # ID и метаданные (не нужны для обучения)
            'id', 'raw_data_id', 'created_at', 'updated_at',
            'processing_version'
        ]
        
        # Логируем исключаемые колонки, которые есть в DataFrame
        excluded_found = [col for col in columns_to_exclude if col in df.columns]
        logger.info(f"\n🚫 ИСКЛЮЧАЕМЫЕ КОЛОНКИ ({len(excluded_found)}):")
        for col in excluded_found:
            if col in df.columns:
                logger.info(f"   - {col}")
        
        # Удаляем все потенциально опасные колонки
        df_features = df.drop(columns=columns_to_exclude, errors='ignore')
        
        # Создаем symbol features
        df_features = self.create_symbol_features(df_features)
        
        # Корректируем is_bullish если он содержит -1
        if 'is_bullish' in df_features.columns:
            # Убедимся что is_bullish содержит только 0 и 1
            df_features['is_bullish'] = df_features['is_bullish'].apply(lambda x: max(0, x))
        
        # Создаем взвешенные комбинации признаков
        df_features = self.create_weighted_features(df_features)
        
        # Инженерные признаки (проверяем наличие колонок)
        # Важно: перезаписываем признаки, даже если они уже существуют
        if 'rsi_val' in df_features.columns:
            # Логируем статистику RSI для диагностики
            rsi_stats = df_features['rsi_val'].describe()
            logger.info(f"   📊 RSI статистика: mean={rsi_stats['mean']:.2f}, std={rsi_stats['std']:.2f}, min={rsi_stats['min']:.2f}, max={rsi_stats['max']:.2f}")
            df_features['rsi_oversold'] = (df_features['rsi_val'] < 30).astype(int)
            df_features['rsi_overbought'] = (df_features['rsi_val'] > 70).astype(int)
            oversold_pct = df_features['rsi_oversold'].mean() * 100
            overbought_pct = df_features['rsi_overbought'].mean() * 100
            logger.info(f"   📊 RSI oversold: {oversold_pct:.1f}%, overbought: {overbought_pct:.1f}%")
        else:
            logger.warning("⚠️ rsi_val не найден в данных!")
            df_features['rsi_oversold'] = 0
            df_features['rsi_overbought'] = 0
            
        # Перезаписываем macd_bullish, чтобы убрать возможные значения -1
        if 'macd_hist' in df_features.columns:
            # Логируем статистику macd_hist для диагностики
            macd_stats = df_features['macd_hist'].describe()
            logger.info(f"   📊 MACD hist статистика: mean={macd_stats['mean']:.4f}, std={macd_stats['std']:.4f}, min={macd_stats['min']:.4f}, max={macd_stats['max']:.4f}")
            df_features['macd_bullish'] = (df_features['macd_hist'] > 0).astype(int)
            bullish_pct = df_features['macd_bullish'].mean() * 100
            logger.info(f"   📊 MACD bullish процент: {bullish_pct:.1f}%")
        elif 'macd_bullish' in df_features.columns:
            # Если macd_bullish уже есть, корректируем значения -1 на 0
            df_features['macd_bullish'] = df_features['macd_bullish'].apply(lambda x: max(0, x))
        else:
            logger.warning("⚠️ Ни macd_hist, ни macd_bullish не найдены в данных!")
            df_features['macd_bullish'] = 0
            
        if 'bb_position' in df_features.columns:
            df_features['bb_near_lower'] = ((df_features['close'] - df_features['bb_position'] * 2) < 0.02).astype(int)
            df_features['bb_near_upper'] = ((df_features['bb_position'] * 2 - df_features['close']) < 0.02).astype(int)
        else:
            df_features['bb_near_lower'] = 0
            df_features['bb_near_upper'] = 0
            
        if 'adx_val' in df_features.columns:
            df_features['strong_trend'] = (df_features['adx_val'] > 25).astype(int)
        else:
            df_features['strong_trend'] = 0
            
        if 'volume_ratio' in df_features.columns:
            df_features['volume_spike'] = (df_features['volume_ratio'] > 2).astype(int)
        else:
            df_features['volume_spike'] = 0
        
        # Собираем все признаки
        # Исключаем is_bullish, так как он уже есть в OHLC_FEATURES
        symbol_features = [col for col in df_features.columns if col.startswith('is_') and col != 'is_bullish']
        engineered_features = ['rsi_oversold', 'rsi_overbought', 'macd_bullish',
                              'bb_near_lower', 'bb_near_upper', 'strong_trend', 'volume_spike']
        
        # Добавляем взвешенные признаки
        weighted_features = [
            'rsi_macd_interaction', 'volume_volatility_interaction', 'adx_trend_strength',
            'rsi_to_adx', 'volume_to_volatility', 'price_momentum_ratio',
            'momentum_composite', 'volatility_composite',
            'oversold_reversal_score', 'breakout_score', 'range_trading_score',
            'rsi_divergence'
        ]
        # Фильтруем только существующие взвешенные признаки
        weighted_features = [f for f in weighted_features if f in df_features.columns]
        
        # Добавляем новые продвинутые признаки
        rolling_features = [
            # RSI статистики
            'rsi_mean_10', 'rsi_std_10', 'rsi_mean_30', 'rsi_std_30', 'rsi_mean_60', 'rsi_std_60',
            # Объемные статистики
            'volume_mean_10', 'volume_spike_10', 'volume_mean_30', 'volume_spike_30', 
            'volume_mean_60', 'volume_spike_60',
            # Моментум
            'momentum_10', 'momentum_accel_10', 'momentum_30', 'momentum_accel_30',
            'momentum_60', 'momentum_accel_60',
            # ATR
            'atr_ratio_10', 'atr_ratio_30', 'atr_ratio_60'
        ]
        rolling_features = [f for f in rolling_features if f in df_features.columns]
        
        # Дивергенции
        divergence_features = [
            'rsi_bullish_divergence', 'rsi_bearish_divergence',
            'macd_price_divergence', 'volume_price_divergence'
        ]
        divergence_features = [f for f in divergence_features if f in df_features.columns]
        
        # Паттерны свечей
        candle_features = [
            'hammer_pattern', 'doji_pattern', 'bullish_engulfing', 'bearish_engulfing',
            'pin_bar_bullish', 'pin_bar_bearish'
        ]
        candle_features = [f for f in candle_features if f in df_features.columns]
        
        # Volume profile
        volume_profile_features = [
            'vwap_distance', 'volume_concentration', 'relative_volume_level',
            'accumulation_distribution', 'ad_oscillator'
        ]
        volume_profile_features = [f for f in volume_profile_features if f in df_features.columns]
        
        # Собираем только существующие признаки
        all_features = (self.TECHNICAL_INDICATORS + self.MARKET_FEATURES + 
                       self.OHLC_FEATURES + symbol_features + engineered_features + 
                       weighted_features + rolling_features + divergence_features + 
                       candle_features + volume_profile_features)
        
        # Фильтруем только те, которые есть в DataFrame
        # Удаляем дубликаты, сохраняя порядок
        features_with_duplicates = [f for f in all_features if f in df_features.columns]
        self.feature_names = list(dict.fromkeys(features_with_duplicates))
        
        # Логируем информацию о дубликатах
        if len(features_with_duplicates) != len(self.feature_names):
            logger.info(f"   ✅ Удалено дубликатов признаков: {len(features_with_duplicates) - len(self.feature_names)}")
            duplicates = [f for f in features_with_duplicates if features_with_duplicates.count(f) > 1]
            unique_duplicates = list(set(duplicates))
            if unique_duplicates:
                logger.info(f"   📋 Дублированные признаки: {unique_duplicates[:10]}")
        
        # ВАЖНО: Расширенная проверка на отсутствие целевых переменных и утечек в признаках
        dangerous_patterns = [
            'expected_return', 'profit_target', 'loss_target', 
            'max_profit', 'realized_profit', '_return_buy', '_return_sell'
        ]
        
        features_to_remove = []
        for feature in self.feature_names:
            for pattern in dangerous_patterns:
                if pattern in feature.lower():
                    features_to_remove.append(feature)
                    logger.error(f"🚨 ОПАСНЫЙ ПРИЗНАК ОБНАРУЖЕН: {feature} (содержит '{pattern}')! Удаляем...")
                    break
        
        # Удаляем опасные признаки
        for feature in features_to_remove:
            if feature in self.feature_names:
                self.feature_names.remove(feature)
        
        # Логируем отсутствующие признаки
        missing_features = [f for f in all_features if f not in df_features.columns]
        if missing_features:
            logger.warning(f"⚠️ Отсутствующие признаки ({len(missing_features)}): {missing_features[:10]}...")
        
        logger.info(f"\n✅ ФИНАЛЬНЫЙ НАБОР ПРИЗНАКОВ:")
        logger.info(f"📊 Используется {len(self.feature_names)} признаков из {len(all_features)} запланированных")
        
        # Детальное логирование используемых признаков
        logger.info("\n📋 СПИСОК ВСЕХ ИСПОЛЬЗУЕМЫХ ПРИЗНАКОВ:")
        tech_features = [f for f in self.TECHNICAL_INDICATORS if f in self.feature_names]
        market_features = [f for f in self.MARKET_FEATURES if f in self.feature_names]
        ohlc_features = [f for f in self.OHLC_FEATURES if f in self.feature_names]
        symbol_features_used = [f for f in symbol_features if f in self.feature_names]
        engineered_features_used = [f for f in engineered_features if f in self.feature_names]
        
        logger.info(f"\n   📈 Технические индикаторы ({len(tech_features)}):")
        for i, f in enumerate(tech_features[:10]):  # Показываем первые 10
            logger.info(f"      {i+1}. {f}")
        if len(tech_features) > 10:
            logger.info(f"      ... и еще {len(tech_features)-10}")
            
        logger.info(f"\n   🌍 Рыночные признаки ({len(market_features)}):")
        for f in market_features:
            logger.info(f"      - {f}")
            
        logger.info(f"\n   📊 OHLC признаки ({len(ohlc_features)}):")
        for f in ohlc_features:
            logger.info(f"      - {f}")
            
        logger.info(f"\n   🏷️ Symbol признаки ({len(symbol_features_used)}):")
        for f in symbol_features_used[:5]:
            logger.info(f"      - {f}")
        if len(symbol_features_used) > 5:
            logger.info(f"      ... и еще {len(symbol_features_used)-5}")
        
        # Логируем новые категории признаков если есть
        if weighted_features:
            logger.info(f"\n   ⚖️ Взвешенные признаки ({len(weighted_features)}):")
            for f in weighted_features[:5]:
                logger.info(f"      - {f}")
        
        if rolling_features:
            logger.info(f"\n   📈 Скользящие статистики ({len(rolling_features)}):")
            for f in rolling_features[:5]:
                logger.info(f"      - {f}")
        
        if divergence_features:
            logger.info(f"\n   🔄 Дивергенции ({len(divergence_features)}):")
            for f in divergence_features:
                logger.info(f"      - {f}")
        
        if candle_features:
            logger.info(f"\n   🕯️ Паттерны свечей ({len(candle_features)}):")
            for f in candle_features:
                logger.info(f"      - {f}")
        
        if volume_profile_features:
            logger.info(f"\n   📉 Volume profile ({len(volume_profile_features)}):")
            for f in volume_profile_features:
                logger.info(f"      - {f}")
        
        logger.info(f"📊 Финальная проверка - используется {len(self.feature_names)} признаков")
        
        # Извлекаем признаки
        X = df_features[self.feature_names].values
        
        logger.info(f"📊 Используется {len(self.feature_names)} признаков")
        logger.info(f"   - Технические индикаторы: {len([f for f in self.TECHNICAL_INDICATORS if f in self.feature_names])}")
        logger.info(f"   - Market features: {len([f for f in self.MARKET_FEATURES if f in self.feature_names])}")
        logger.info(f"   - OHLC features: {len([f for f in self.OHLC_FEATURES if f in self.feature_names])}")
        logger.info(f"   - Symbol features: {len(symbol_features)}")
        logger.info(f"   - Engineered features: {len([f for f in engineered_features if f in self.feature_names])}")
        logger.info(f"   - Weighted features: {len(weighted_features)}")
        logger.info(f"   - Rolling statistics: {len(rolling_features)}")
        logger.info(f"   - Divergences: {len(divergence_features)}")
        logger.info(f"   - Candle patterns: {len(candle_features)}")
        logger.info(f"   - Volume profile: {len(volume_profile_features)}")
        
        # Финальная проверка размерностей
        logger.info(f"📏 Размерности: X.shape={X.shape}, y_buy.shape={y_buy.shape}, y_sell.shape={y_sell.shape}")
        
        # ФИНАЛЬНАЯ ПРОВЕРКА: убеждаемся, что нет корреляции между признаками и целевыми переменными
        logger.info("\n🔍 ПРОВЕРКА НА УТЕЧКУ ДАННЫХ:")
        # Проверяем корреляцию первых 5 признаков с целевыми переменными
        for i in range(min(5, X.shape[1])):
            corr_buy = np.corrcoef(X[:, i], y_buy)[0, 1]
            corr_sell = np.corrcoef(X[:, i], y_sell)[0, 1]
            if abs(corr_buy) > 0.9 or abs(corr_sell) > 0.9:
                logger.error(f"   🚨 ВЫСОКАЯ КОРРЕЛЯЦИЯ! Признак {self.feature_names[i]}: "
                           f"corr_buy={corr_buy:.3f}, corr_sell={corr_sell:.3f}")
            else:
                logger.info(f"   ✅ Признак {self.feature_names[i]}: "
                          f"corr_buy={corr_buy:.3f}, corr_sell={corr_sell:.3f}")
        
        return X, y_buy, y_sell
        
    def create_xgboost_model(self, task: str, num_classes: int = None) -> xgb.XGBModel:
        """Создание XGBoost модели с улучшенной регуляризацией"""
        base_params = {
            'n_estimators': 3000,
            'max_depth': 8,  # Увеличено для захвата сложных паттернов
            'learning_rate': 0.01,  # Малый learning rate для точной настройки
            'subsample': 0.8,  # Увеличено для использования больше данных
            'colsample_bytree': 0.8,  # Больше признаков на дерево
            'colsample_bylevel': 0.8,
            'gamma': 0.1,  # Уменьшено для большей гибкости модели
            'reg_alpha': 0.1,  # Меньше L1 регуляризации
            'reg_lambda': 1.0,  # Умеренная L2 регуляризация
            'min_child_weight': 3,  # Уменьшено для захвата редких паттернов
            'max_delta_step': 1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': 100
        }
        
        # GPU параметры если доступен
        if USE_GPU:
            base_params['tree_method'] = 'gpu_hist'
            # predictor='gpu_predictor' устарел в новых версиях XGBoost
            
        if task == 'regression':
            model = xgb.XGBRegressor(
                **base_params,
                objective='reg:squarederror',
                eval_metric='rmse'
            )
        elif task == 'binary':
            model = xgb.XGBClassifier(
                **base_params,
                objective='binary:logistic',
                eval_metric='logloss'  # Изменено с 'auc' на 'logloss' для совместимости
            )
        else:  # multiclass
            model = xgb.XGBClassifier(
                **base_params,
                objective='multi:softprob',
                num_class=num_classes,
                eval_metric='mlogloss'
            )
            
        return model
        
    def optimize_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, method='gmean') -> float:
        """Продвинутая оптимизация порога для бинарной классификации"""
        logger.info(f"🎯 Оптимизация порога методом {method}...")
        
        if method == 'gmean':
            # Используем G-mean для балансировки sensitivity и specificity
            best_threshold = find_optimal_threshold_gmean(y_true, y_pred_proba)
        elif method == 'profit':
            # Оптимизация для максимальной прибыли
            best_threshold = find_optimal_threshold_profit(y_true, y_pred_proba, 
                                                         profit_per_tp=1.5,  # Прибыль от правильного сигнала
                                                         loss_per_fp=1.0)    # Потери от ложного сигнала
        else:
            # Классический метод через F1
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_f1 = 0
            best_threshold = 0.5
            best_precision = 0
            best_recall = 0
            
            for threshold in thresholds:
                y_pred = (y_pred_proba > threshold).astype(int)
                
                # Пропускаем если нет положительных предсказаний
                if y_pred.sum() == 0:
                    continue
                    
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Используем взвешенный критерий для трейдинга
                # Precision важнее для уменьшения ложных сигналов
                weighted_score = 0.7 * precision + 0.3 * recall
                
                if f1 > best_f1 or (f1 >= best_f1 * 0.95 and weighted_score > best_precision * 0.7 + best_recall * 0.3):
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
            
            logger.info(f"   Лучший порог: {best_threshold:.2f} (F1: {best_f1:.3f}, Precision: {best_precision:.3f}, Recall: {best_recall:.3f})")
        
        return best_threshold
        
    def plot_training_progress(self, eval_results: dict, model_name: str):
        """Визуализация процесса обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Определяем метрику в зависимости от типа задачи
        # Проверяем какие метрики доступны
        train_metrics = list(eval_results['validation_0'].keys())
        val_metrics = list(eval_results['validation_1'].keys())
        
        # Используем первую доступную метрику
        metric_name = train_metrics[0] if train_metrics else 'loss'
        
        # График loss
        epochs = range(len(eval_results['validation_0'][metric_name]))
        axes[0].plot(epochs, eval_results['validation_0'][metric_name], 'b-', label='Train')
        axes[0].plot(epochs, eval_results['validation_1'][metric_name], 'r-', label='Validation')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel(metric_name.capitalize())
        axes[0].set_title(f'Training Progress: {model_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Feature importance
        if hasattr(self.models[model_name], 'feature_importances_'):
            importance = self.models[model_name].feature_importances_
            indices = np.argsort(importance)[-20:]
            
            axes[1].barh(range(20), importance[indices])
            axes[1].set_yticks(range(20))
            axes[1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1].set_xlabel('Feature Importance')
            axes[1].set_title('Top 20 Features')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{log_dir}/plots/{model_name}_training_progress.png', dpi=150)
        plt.close()
        
    def plot_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray, model_name: str, task: str):
        """Визуализация результатов"""
        if task == 'regression':
            self._plot_regression_results(y_true, y_pred, model_name)
        else:
            self._plot_classification_results(y_true, y_pred, y_pred_proba, model_name)
            
    def _plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Визуализация для регрессии"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'XGBoost Regression Results: {model_name}', fontsize=16)
        
        # Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title('Predictions vs True Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predictions')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution
        axes[1, 0].hist(y_true, bins=50, alpha=0.5, label='True', density=True)
        axes[1, 0].hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
        axes[1, 0].set_xlabel('Expected Return (%)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        axes[1, 1].axis('off')
        metrics_text = f"""
Performance Metrics:
  MAE:  {mae:.4f}%
  RMSE: {rmse:.4f}%
  R²:   {r2:.4f}
  
  Mean True:  {np.mean(y_true):.3f}%
  Mean Pred:  {np.mean(y_pred):.3f}%
  Std True:   {np.std(y_true):.3f}%
  Std Pred:   {np.std(y_pred):.3f}%
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'{log_dir}/plots/{model_name}_evaluation.png', dpi=150)
        plt.close()
        
    def _plot_classification_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: np.ndarray, model_name: str):
        """Визуализация для классификации"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'XGBoost Classification Results: {model_name}', fontsize=16)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title('Confusion Matrix')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Probability Distribution
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.5, 
                       label='Class 0', density=True)
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.5,
                       label='Class 1', density=True)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        axes[1, 1].axis('off')
        metrics_text = f"""
Performance Metrics:
  Accuracy:  {accuracy:.2%}
  Precision: {precision:.2%}
  Recall:    {recall:.2%}
  F1-Score:  {f1:.3f}
  ROC-AUC:   {auc:.3f}
  
Confusion Matrix:
  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}
  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'{log_dir}/plots/{model_name}_evaluation.png', dpi=150)
        plt.close()
        
    def train_ensemble(self, task: str = 'classification_binary', 
                      ensemble_size: int = 1, test_mode: bool = False,
                      use_cache: bool = False, force_reload: bool = False,
                      no_smote: bool = False, classification_threshold: float = 0.5,
                      balance_method: str = 'smote'):
        """Основной метод обучения с поддержкой кэширования"""
        # Сохраняем balance_method как атрибут класса
        self.balance_method = balance_method if not no_smote else 'none'
        
        logger.info("🚀 Запуск Enhanced XGBoost v2.0")
        logger.info(f"📊 Режим: {task}")
        logger.info(f"🎯 Размер ансамбля: {ensemble_size}")
        if task == 'classification_binary':
            logger.info(f"🔄 Метод балансировки: {self.balance_method}")
            logger.info(f"🎯 Порог классификации: {classification_threshold}%")
        if test_mode:
            logger.info("⚡ ТЕСТОВЫЙ РЕЖИМ: ограниченные данные")
        if use_cache:
            logger.info("💾 Использование кэша: ВКЛ")
            if force_reload:
                logger.info("🔄 Принудительное обновление кэша")
                
        start_time = time.time()
        
        # Инициализация менеджеров
        if use_cache:
            self.cache_manager = CacheManager()
        self.visualizer = AdvancedVisualizer(log_dir)
        
        # Подключение к БД
        conn = self.connect_db()
        
        try:
            # Загрузка данных
            logger.info("📊 Загрузка данных из PostgreSQL...")
            
            # Получаем список символов
            query = "SELECT DISTINCT symbol FROM processed_market_data ORDER BY symbol"
            symbols_df = pd.read_sql_query(query, conn)
            all_symbols = symbols_df['symbol'].tolist()
            
            if test_mode:
                # В тестовом режиме используем только 2 символа
                symbols_to_load = ['BTCUSDT', 'ETHUSDT']
                logger.info(f"⚡ Тестовый режим: загружаем только {symbols_to_load}")
            else:
                symbols_to_load = all_symbols
                
            logger.info(f"📋 Найдено {len(symbols_to_load)} символов для загрузки")
            
            # Попытка загрузить из кэша
            df = None
            cache_key = None
            
            if use_cache and self.cache_manager:
                cache_key = self.cache_manager.get_cache_key({
                    'symbols_count': len(symbols_to_load),
                    'task': task,
                    'date_range': 'full'  # Можно добавить реальный диапазон дат
                })
                
                if not force_reload:
                    df = self.cache_manager.load(cache_key, 'raw_data')
            
            if df is None:
                # Загружаем данные параллельно для ускорения
                df = self.load_data_parallel(symbols_to_load, conn, batch_size=10)
                
                # Оптимизируем память
                df = self.optimize_memory(df)
                
                # Сохраняем в кэш
                if use_cache and self.cache_manager and cache_key:
                    self.cache_manager.save(df, cache_key, 'raw_data')
            
            # Загружаем market data
            logger.info("📊 Загрузка рыночных данных для корреляций...")
            market_data = self.load_market_data(conn)
            
            # Попытка загрузить подготовленные признаки из кэша
            features_cache = None
            if use_cache and self.cache_manager and not force_reload:
                features_cache = self.cache_manager.load(cache_key, 'prepared_features')
            
            if features_cache is not None:
                # Распаковываем из кэша
                df_features = features_cache['df_features']
                X = features_cache['X']
                y_buy = features_cache['y_buy']
                y_sell = features_cache['y_sell']
                self.feature_names = features_cache['feature_names']
            else:
                # Расчет market features
                df = self.calculate_market_features(df, market_data)
                
                # Добавление новых продвинутых признаков
                logger.info("🔄 Добавление продвинутых признаков...")
                df = self.add_rolling_statistics(df)
                df = self.add_divergences(df)
                df = self.add_candle_patterns(df)
                df = self.add_volume_profile(df)
                
                # Подготовка признаков
                logger.info("🔧 Подготовка признаков и целевых значений...")
                X, y_buy, y_sell = self.prepare_features(df)
                
                # Сохраняем в кэш
                if use_cache and self.cache_manager and cache_key:
                    features_cache = {
                        'df_features': df,
                        'X': X,
                        'y_buy': y_buy,
                        'y_sell': y_sell,
                        'feature_names': self.feature_names
                    }
                    self.cache_manager.save(features_cache, cache_key, 'prepared_features')
                    
            # Визуализация данных
            if hasattr(df, 'columns'):
                self.visualizer.plot_data_overview(df)
            
            # Нормализация (исключаем бинарные признаки)
            logger.info("🔄 Нормализация признаков...")
            
            # Определяем бинарные признаки для исключения из нормализации
            binary_features = [
                'rsi_oversold', 'rsi_overbought', 'macd_bullish',
                'bb_near_lower', 'bb_near_upper', 'strong_trend', 'volume_spike',
                'is_bullish', 'is_weekend', 'is_major', 'is_meme', 'is_defi', 'is_alt',
                'market_regime_low_vol', 'market_regime_med_vol', 'market_regime_high_vol'
            ]
            # Добавляем one-hot закодированные символы
            binary_features.extend([f for f in self.feature_names if f.startswith('is_')])
            # Добавляем паттерны свечей и дивергенции
            binary_features.extend([
                'is_hammer', 'is_doji', 'bullish_engulfing', 'bearish_engulfing',
                'pin_bar', 'rsi_bullish_divergence', 'rsi_bearish_divergence',
                'macd_bullish_divergence', 'macd_bearish_divergence', 'volume_price_divergence'
            ])
            
            # Находим индексы бинарных признаков
            binary_indices = []
            continuous_indices = []
            for i, feature in enumerate(self.feature_names):
                if feature in binary_features:
                    binary_indices.append(i)
                else:
                    continuous_indices.append(i)
            
            logger.info(f"   📊 Бинарных признаков: {len(binary_indices)}")
            logger.info(f"   📊 Непрерывных признаков: {len(continuous_indices)}")
            
            # Создаем копию X для масштабирования
            X_scaled = X.copy()
            
            # Нормализуем только непрерывные признаки
            if continuous_indices:
                scaler = RobustScaler()
                X_scaled[:, continuous_indices] = scaler.fit_transform(X[:, continuous_indices])
                self.scalers['features'] = scaler
                self.scalers['binary_indices'] = binary_indices
                self.scalers['continuous_indices'] = continuous_indices
            else:
                logger.warning("⚠️ Не найдено непрерывных признаков для нормализации")
            
            # Визуализация распределений признаков
            if self.feature_names:
                self.visualizer.plot_feature_distributions(X_scaled, self.feature_names)
            
            # Преобразование для классификации если нужно
            if task in ['classification_binary', 'classification_multiclass']:
                logger.info("🔄 Преобразование в метки классификации...")
                if task == 'classification_binary':
                    # Используем переданный порог для классификации
                    y_buy_class = (y_buy > classification_threshold).astype(int)
                    y_sell_class = (y_sell > classification_threshold).astype(int)
                    
                    logger.info(f"📊 Статистика бинарных меток (порог > {classification_threshold}%):")
                    logger.info(f"   Buy - Класс 1 (входить): {np.mean(y_buy_class):.1%}")
                    logger.info(f"   Sell - Класс 1 (входить): {np.mean(y_sell_class):.1%}")
                    
                    # Дополнительная статистика для анализа
                    logger.info(f"\n📈 Детальная статистика expected_return:")
                    logger.info(f"   Buy > 0%: {(y_buy > 0).mean():.1%}, Buy > 0.5%: {(y_buy > 0.5).mean():.1%}, Buy > 1%: {(y_buy > 1.0).mean():.1%}")
                    logger.info(f"   Sell > 0%: {(y_sell > 0).mean():.1%}, Sell > 0.5%: {(y_sell > 0.5).mean():.1%}, Sell > 1%: {(y_sell > 1.0).mean():.1%}")
                else:
                    # Multiclass: 4 класса для точного определения точек входа
                    # Класс 0: Не входить (< 0.5%)
                    # Класс 1: Нейтрально (0.5% - 1.5%)
                    # Класс 2: Хорошая точка (1.5% - 3%)
                    # Класс 3: Отличная точка (> 3%)
                    bins = [-np.inf, 0.5, 1.5, 3.0, np.inf]
                    y_buy_class = pd.cut(y_buy, bins=bins, labels=[0, 1, 2, 3]).astype(int)
                    y_sell_class = pd.cut(y_sell, bins=bins, labels=[0, 1, 2, 3]).astype(int)
                    
                    logger.info(f"📊 Статистика мультиклассовых меток:")
                    for direction, y_class in [('Buy', y_buy_class), ('Sell', y_sell_class)]:
                        logger.info(f"   {direction}:")
                        for i in range(4):
                            pct = (y_class == i).mean() * 100
                            logger.info(f"     Класс {i}: {pct:.1f}%")
                    
                # Разделение данных
                test_size = 0.15
                val_size = 0.15
                
                n = len(X_scaled)
                train_end = int(n * (1 - test_size - val_size))
                val_end = int(n * (1 - test_size))
                
                X_train = X_scaled[:train_end]
                X_val = X_scaled[train_end:val_end]
                X_test = X_scaled[val_end:]
                
                model_configs = [
                    ('buy', y_buy_class if task.startswith('classification') else y_buy),
                    ('sell', y_sell_class if task.startswith('classification') else y_sell)
                ]
            else:
                # Регрессия
                test_size = 0.15
                val_size = 0.15
                
                n = len(X_scaled)
                train_end = int(n * (1 - test_size - val_size))
                val_end = int(n * (1 - test_size))
                
                X_train = X_scaled[:train_end]
                X_val = X_scaled[train_end:val_end]
                X_test = X_scaled[val_end:]
                
                model_configs = [
                    ('buy_return_predictor', y_buy),
                    ('sell_return_predictor', y_sell)
                ]
                
            # Обучение моделей
            results = {}
            metrics_history = {}  # Для визуализации сравнения
            
            for direction, y_values in model_configs:
                logger.info(f"\n{'='*60}")
                logger.info(f"Обучение моделей для: {direction}")
                logger.info(f"{'='*60}")
                
                y_train = y_values[:train_end]
                y_val = y_values[train_end:val_end]
                y_test = y_values[val_end:]
                
                ensemble_predictions = []
                ensemble_models = []
                ensemble_weights = []  # Веса для взвешенного голосования
                
                # Исправляем бинарные признаки перед балансировкой
                if self.feature_names is not None:
                    binary_features = ['macd_bullish', 'rsi_oversold', 'rsi_overbought', 
                                     'strong_trend', 'volume_spike', 'is_bullish']
                    
                    for feat in binary_features:
                        if feat in self.feature_names:
                            feat_idx = self.feature_names.index(feat)
                            # Исправляем значения в X_train и X_val (преобразуем -1 в 0)
                            X_train[:, feat_idx] = np.where(X_train[:, feat_idx] > 0, 1, 0)
                            X_val[:, feat_idx] = np.where(X_val[:, feat_idx] > 0, 1, 0)
                            
                            # Логируем если были исправления
                            unique_train = np.unique(X_train[:, feat_idx])
                            if len(unique_train) != 2 or not np.array_equal(unique_train, [0, 1]):
                                logger.warning(f"⚠️ Исправлен признак {feat}: unique values = {unique_train}")
                
                # Применяем балансировку классов (только для бинарной классификации)
                if task == 'classification_binary' and direction != 'regression' and not no_smote:
                    balance_method = getattr(self, 'balance_method', 'smote')  # Получаем метод из атрибута
                    
                    if balance_method == 'random':
                        logger.info("\n🔄 Балансировка классов с помощью RandomOverSampler...")
                        X_train_balanced, y_train_balanced = apply_random_oversampler(
                            X_train, y_train, sampling_strategy=0.5
                        )
                    elif balance_method == 'smote':
                        logger.info("\n🔄 Балансировка классов с помощью SMOTE...")
                        X_train_balanced, y_train_balanced = apply_smote(
                            X_train, y_train, sampling_strategy=0.5
                        )
                        
                        # Ограничиваем значения технических индикаторов после SMOTE
                        if self.feature_names is not None:
                            logger.info("✂️ Ограничение значений технических индикаторов...")
                            X_train_balanced = clip_technical_indicators(X_train_balanced, self.feature_names)
                            
                            # Пересоздаем бинарные признаки после SMOTE и ограничения
                            logger.info("🔄 Пересоздание бинарных признаков после SMOTE...")
                            X_train_balanced = recreate_binary_features(X_train_balanced, self.feature_names, direction)
                    else:  # none
                        logger.info("\n⚠️ Балансировка отключена - используем несбалансированные данные")
                        X_train_balanced = X_train
                        y_train_balanced = y_train
                elif no_smote and task == 'classification_binary':
                    logger.info("\n⚠️ Балансировка отключена - используем несбалансированные данные")
                    X_train_balanced = X_train
                    y_train_balanced = y_train
                else:
                    X_train_balanced = X_train
                    y_train_balanced = y_train
                
                # Валидация бинарных признаков после балансировки
                if task == 'classification_binary' and self.feature_names is not None:
                    binary_features = ['macd_bullish', 'rsi_oversold', 'rsi_overbought', 
                                     'strong_trend', 'volume_spike', 'is_bullish']
                    
                    existing_binary_features = [f for f in binary_features if f in self.feature_names]
                    if existing_binary_features:
                        # Проверяем сбалансированные данные
                        df_check = pd.DataFrame(X_train_balanced[:1000], columns=self.feature_names)
                        validate_binary_features(df_check, existing_binary_features)
                    else:
                        logger.warning("⚠️ Бинарные признаки не найдены в feature_names")
                
                # Optuna оптимизация для первой модели ансамбля
                if ensemble_size > 0:
                    logger.info("\n🔧 Запуск Optuna оптимизации гиперпараметров...")
                    best_params = self.optimize_hyperparameters(
                        X_train_balanced, y_train_balanced, X_val, y_val, task, direction
                    )
                    logger.info(f"✅ Лучшие параметры найдены: {best_params}")
                else:
                    best_params = None
                
                for i in range(ensemble_size):
                    model_name = f"{direction}_xgboost_v2_{i}"
                    logger.info(f"\n🚀 Обучение модели {i+1}/{ensemble_size}: {model_name}")
                    
                    # Создаем модель
                    if task == 'regression':
                        model = self.create_xgboost_model('regression')
                    elif task == 'classification_binary':
                        model = self.create_xgboost_model('binary')
                        # Используем правильный расчет scale_pos_weight
                        scale_pos_weight = calculate_scale_pos_weight(y_train)
                        model.set_params(scale_pos_weight=scale_pos_weight)
                        
                        # Добавляем max_delta_step для стабильности
                        model.set_params(max_delta_step=1)
                    else:
                        num_classes = len(np.unique(y_train))
                        model = self.create_xgboost_model('multiclass', num_classes)
                        
                        # Рассчитываем веса для каждого класса
                        from sklearn.utils.class_weight import compute_class_weight
                        class_weights = compute_class_weight(
                            'balanced',
                            classes=np.unique(y_train),
                            y=y_train
                        )
                        sample_weights = np.ones(len(y_train))
                        for i, cls in enumerate(np.unique(y_train)):
                            sample_weights[y_train == cls] = class_weights[i]
                    
                    # Используем разные случайные семена для разнообразия в ансамбле
                    model.set_params(random_state=42 + i * 100)
                    
                    # Применяем лучшие параметры из Optuna
                    if best_params and i == 0:  # Только для первой модели
                        try:
                            # Логируем применяемые параметры
                            logger.info(f"📝 Применяем параметры Optuna: {best_params}")
                            model.set_params(**best_params)
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка при применении параметров Optuna: {e}")
                            logger.warning("Используем базовые параметры модели")
                    
                    # Логируем финальные параметры модели перед обучением
                    logger.info("📋 Параметры модели перед обучением:")
                    important_params = ['tree_method', 'objective', 'eval_metric', 'n_estimators', 
                                      'max_depth', 'learning_rate', 'scale_pos_weight', 'subsample',
                                      'colsample_bytree', 'gamma', 'min_child_weight']
                    for param in important_params:
                        if hasattr(model, param):
                            value = getattr(model, param)
                            logger.info(f"   {param}: {value}")
                    
                    # Обучение с early stopping
                    eval_set = [(X_train_balanced, y_train_balanced), (X_val, y_val)]
                    
                    if test_mode:
                        model.set_params(n_estimators=100)  # Меньше деревьев для теста
                        
                    # Для новой версии XGBoost используем callbacks
                    if task == 'classification_multiclass' and 'sample_weights' in locals():
                        # Для multiclass используем веса
                        model.fit(
                            X_train_balanced, y_train_balanced,
                            sample_weight=sample_weights,
                            eval_set=eval_set,
                            verbose=True
                        )
                    else:
                        model.fit(
                            X_train_balanced, y_train_balanced,
                            eval_set=eval_set,
                            verbose=True
                        )
                    
                    self.models[model_name] = model
                    ensemble_models.append(model)
                    
                    # Предсказания на тесте
                    # Оценка на валидационном наборе для взвешивания
                    if task == 'regression':
                        y_val_pred = model.predict(X_val)
                        val_score = r2_score(y_val, y_val_pred)
                        y_pred = model.predict(X_test)
                        ensemble_predictions.append(y_pred)
                    else:
                        y_val_pred_proba = model.predict_proba(X_val)
                        if task == 'classification_binary':
                            val_score = roc_auc_score(y_val, y_val_pred_proba[:, 1])
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            ensemble_predictions.append(y_pred_proba)
                        else:
                            val_score = accuracy_score(y_val, y_val_pred_proba.argmax(axis=1))
                            y_pred_proba = model.predict_proba(X_test)
                            ensemble_predictions.append(y_pred_proba)
                    
                    # Сохраняем вес модели на основе валидационной производительности
                    ensemble_weights.append(val_score)
                    logger.info(f"   Валидационный скор модели: {val_score:.4f}")
                    
                    # Визуализация прогресса обучения
                    if hasattr(model, 'evals_result'):
                        self.plot_training_progress(model.evals_result(), model_name)
                        
                # Нормализация весов
                if ensemble_weights:
                    weights = np.array(ensemble_weights)
                    weights = weights / weights.sum()
                    logger.info(f"\n📊 Веса моделей в ансамбле: {weights}")
                else:
                    weights = None
                
                # Взвешенное усреднение предсказаний ансамбля
                if task == 'regression':
                    y_pred_ensemble = ensemble_predictions_weighted(ensemble_predictions, weights, method='soft')
                    
                    # Метрики
                    mae = mean_absolute_error(y_test, y_pred_ensemble)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
                    r2 = r2_score(y_test, y_pred_ensemble)
                    
                    logger.info(f"\n📈 Результаты ансамбля {direction}:")
                    logger.info(f"   MAE: {mae:.4f}%")
                    logger.info(f"   RMSE: {rmse:.4f}%")
                    logger.info(f"   R²: {r2:.4f}")
                    
                    self.plot_evaluation(y_test, y_pred_ensemble, None, f"{direction}_ensemble", 'regression')
                    
                elif task == 'classification_binary':
                    # Взвешенное голосование
                    y_pred_proba_ensemble = ensemble_predictions_weighted(ensemble_predictions, weights, method='soft')
                    
                    # Калибровка вероятностей
                    logger.info("\n📊 Калибровка вероятностей...")
                    y_pred_proba_calibrated, calibrator = calibrate_probabilities(
                        ensemble_models[0], X_train_balanced, y_train_balanced, X_test, method='isotonic'
                    )
                    
                    # Используем калиброванные вероятности если они лучше
                    if roc_auc_score(y_test, y_pred_proba_calibrated) > roc_auc_score(y_test, y_pred_proba_ensemble):
                        logger.info("   ✅ Используем калиброванные вероятности")
                        y_pred_proba_ensemble = y_pred_proba_calibrated
                    
                    # Оптимизация порога
                    logger.info(f"🎯 Оптимизация порога для {direction}...")
                    best_threshold = self.optimize_threshold(y_test, y_pred_proba_ensemble, method='gmean')
                    logger.info(f"   Лучший порог: {best_threshold:.2f}")
                    
                    y_pred_ensemble = (y_pred_proba_ensemble > best_threshold).astype(int)
                    
                    # Расширенные метрики
                    accuracy = accuracy_score(y_test, y_pred_ensemble)
                    precision = precision_score(y_test, y_pred_ensemble, zero_division=0)
                    recall = recall_score(y_test, y_pred_ensemble, zero_division=0)
                    f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)
                    auc = roc_auc_score(y_test, y_pred_proba_ensemble)
                    
                    # Дополнительные метрики для несбалансированных данных
                    pr_auc = average_precision_score(y_test, y_pred_proba_ensemble)
                    mcc = matthews_corrcoef(y_test, y_pred_ensemble)
                    
                    # G-mean
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ensemble).ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    gmean = np.sqrt(sensitivity * specificity)
                    
                    logger.info(f"\n📈 Результаты ансамбля {direction}:")
                    logger.info(f"   Accuracy: {accuracy:.2%}")
                    logger.info(f"   Precision: {precision:.2%}")
                    logger.info(f"   Recall: {recall:.2%}")
                    logger.info(f"   F1-Score: {f1:.3f}")
                    logger.info(f"   ROC-AUC: {auc:.3f}")
                    logger.info(f"   PR-AUC: {pr_auc:.3f}")
                    logger.info(f"   MCC: {mcc:.3f}")
                    logger.info(f"   G-mean: {gmean:.3f}")
                    logger.info(f"   Sensitivity: {sensitivity:.3f}")
                    logger.info(f"   Specificity: {specificity:.3f}")
                    
                    self.plot_evaluation(y_test, y_pred_ensemble, y_pred_proba_ensemble,
                                       f"{direction}_ensemble", 'classification')
                    
                    # Анализ выигрышных паттернов
                    self.analyze_winning_patterns(ensemble_models[0], X_test, y_test, 
                                                y_pred_ensemble, y_pred_proba_ensemble, 
                                                f"{direction}_ensemble")
                    
                    # Фильтрация сигналов
                    X_test_df = pd.DataFrame(X_test, columns=self.feature_names)
                    filtered_signals = self.filter_trading_signals(y_pred_ensemble, y_pred_proba_ensemble, 
                                                                  X_test_df, min_confidence=0.6)
                    
                    # Метрики после фильтрации
                    filtered_accuracy = accuracy_score(y_test[filtered_signals > -1], 
                                                     filtered_signals[filtered_signals > -1])
                    logger.info(f"\n📊 Метрики после фильтрации:")
                    logger.info(f"   Accuracy: {filtered_accuracy:.2%}")
                    
                    results[direction] = {
                        'threshold': best_threshold,
                        'metrics': {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'auc': auc
                        }
                    }
                    
                    # Сохраняем метрики для визуализации
                    metrics_history[f"{direction}_ensemble"] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                    
                else:  # multiclass
                    # Усреднение вероятностей для multiclass
                    y_pred_proba_ensemble = np.mean(ensemble_predictions, axis=0)
                    y_pred_ensemble = y_pred_proba_ensemble.argmax(axis=1)
                    
                    # Метрики для multiclass
                    accuracy = accuracy_score(y_test, y_pred_ensemble)
                    
                    # Classification report для детальной статистики
                    from sklearn.metrics import classification_report
                    class_report = classification_report(y_test, y_pred_ensemble, output_dict=True)
                    
                    logger.info(f"\n📈 Результаты ансамбля {direction} (multiclass):")
                    logger.info(f"   Accuracy: {accuracy:.2%}")
                    logger.info(f"\n📊 Детальная статистика по классам:")
                    
                    class_names = ['Не входить', 'Нейтрально', 'Хорошо', 'Отлично']
                    for i in range(len(class_names)):
                        if str(i) in class_report:
                            metrics = class_report[str(i)]
                            logger.info(f"   Класс {i} ({class_names[i]}):")
                            logger.info(f"     Precision: {metrics['precision']:.2%}")
                            logger.info(f"     Recall: {metrics['recall']:.2%}")
                            logger.info(f"     F1-Score: {metrics['f1-score']:.3f}")
                            logger.info(f"     Support: {metrics['support']}")
                    
                    # Анализ паттернов для классов с хорошими сигналами (2 и 3)
                    good_signals = (y_pred_ensemble >= 2)
                    if good_signals.sum() > 0:
                        self.analyze_winning_patterns(ensemble_models[0], X_test[good_signals], 
                                                    y_test[good_signals], 
                                                    y_pred_ensemble[good_signals], 
                                                    y_pred_proba_ensemble[good_signals], 
                                                    f"{direction}_ensemble")
                    
                    results[direction] = {
                        'metrics': {
                            'accuracy': accuracy,
                            'class_report': class_report
                        }
                    }
                    
            # Сохранение моделей
            logger.info("\n💾 Сохранение моделей...")
            
            # Сохраняем модели
            for name, model in self.models.items():
                joblib.dump(model, f'trained_model/{name}.pkl')
                logger.info(f"   ✅ {name}.pkl")
                
            # Сохраняем scaler
            joblib.dump(self.scalers['features'], 'trained_model/scaler_xgboost_v2.pkl')
            
            # Сохраняем метаданные
            metadata = {
                'model_version': '2.1',
                'type': 'xgboost_enhanced_balanced',
                'task_type': task,
                'ensemble_size': ensemble_size,
                'total_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'training_time': time.time() - start_time,
                'test_mode': test_mode,
                'results': results if task.startswith('classification') else {},
                'improvements': [
                    'SMOTE для балансировки классов',
                    'Взвешенное голосование в ансамбле',
                    'Калибровка вероятностей',
                    'G-mean оптимизация порога',
                    'Расширенные метрики (PR-AUC, MCC, G-mean)'
                ]
            }
            
            with open('trained_model/metadata_xgboost_v2.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Сохраняем конфигурацию признаков
            try:
                feature_config = {
                    'technical_indicators': self.TECHNICAL_INDICATORS,
                    'market_features': self.MARKET_FEATURES,
                    'ohlc_features': self.OHLC_FEATURES,
                    'total_features': len(self.feature_names)
                }
                
                with open('trained_model/feature_config_xgboost_v2.json', 'w') as f:
                    json.dump(feature_config, f, indent=2)
                logger.info("   ✅ feature_config_xgboost_v2.json")
            except Exception as e:
                logger.error(f"❌ Ошибка при сохранении feature_config: {e}")
                logger.error(f"   Тип ошибки: {type(e).__name__}")
                import traceback
                logger.error(f"   Traceback:\n{traceback.format_exc()}")
                
            # Визуализация сравнения моделей
            if metrics_history:
                try:
                    self.visualizer.plot_training_comparison(metrics_history)
                    logger.info("   ✅ Визуализация сравнения моделей завершена")
                except Exception as e:
                    logger.error(f"❌ Ошибка при визуализации: {e}")
                    import traceback
                    logger.error(f"   Traceback:\n{traceback.format_exc()}")
            
            # Анализ feature importance для всех моделей
            try:
                self._analyze_feature_importance()
                logger.info("   ✅ Анализ feature importance завершен")
            except Exception as e:
                logger.error(f"❌ Ошибка при анализе feature importance: {e}")
                import traceback
                logger.error(f"   Traceback:\n{traceback.format_exc()}")
            
            # Финальный отчет
            try:
                self._create_final_report(metadata, results if task.startswith('classification') else {})
                logger.info("   ✅ Финальный отчет создан")
            except Exception as e:
                logger.error(f"❌ Ошибка при создании финального отчета: {e}")
                import traceback
                logger.error(f"   Traceback:\n{traceback.format_exc()}")
            
            total_time = time.time() - start_time
            logger.info(f"\n✅ Обучение завершено за {total_time/60:.1f} минут")
            
            # Принудительная запись логов
            for handler in logger.handlers:
                handler.flush()
            
        finally:
            conn.close()
    
    def _analyze_feature_importance(self):
        """Анализ и визуализация feature importance для всех моделей"""
        logger.info("\n📊 Анализ важности признаков...")
        
        if not self.models:
            logger.warning("Нет обученных моделей для анализа")
            return
        
        # Агрегируем importance по всем моделям
        aggregated_importance = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                
                # Проверяем соответствие размерностей
                if len(importance) != len(self.feature_names):
                    logger.error(f"Несоответствие размерностей для {model_name}: "
                               f"importance={len(importance)}, features={len(self.feature_names)}")
                    continue
                
                # Агрегируем
                for i, feature in enumerate(self.feature_names):
                    if feature not in aggregated_importance:
                        aggregated_importance[feature] = []
                    aggregated_importance[feature].append(importance[i])
        
        # Вычисляем средние importance
        mean_importance = {}
        for feature, values in aggregated_importance.items():
            mean_importance[feature] = np.mean(values)
        
        # Сортируем по важности
        sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Выводим топ-20
        logger.info("\n🏆 ТОП-20 ВАЖНЫХ ПРИЗНАКОВ (усреднено по всем моделям):")
        for i, (feature, importance) in enumerate(sorted_features[:20]):
            logger.info(f"{i+1:2d}. {feature:40s} {importance:.4f}")
        
        # Проверяем на подозрительные признаки
        logger.info("\n🔍 Проверка на подозрительные признаки:")
        suspicious_features = ['expected_return', 'buy_expected', 'sell_expected', 'target', 'label']
        for feature, importance in sorted_features:
            for suspicious in suspicious_features:
                if suspicious in feature.lower():
                    logger.error(f"🚨 ПОДОЗРИТЕЛЬНЫЙ ПРИЗНАК: {feature} (importance={importance:.4f})")
                    logger.error("   Возможна утечка целевой переменной!")
        
        # Создаем визуализацию
        self._plot_feature_importance(sorted_features[:30])
        
        # Сохраняем в файл (конвертируем float32 в float для JSON)
        with open(f'{log_dir}/feature_importance.json', 'w') as f:
            json.dump({k: float(v) for k, v in sorted_features}, f, indent=2)
    
    def _plot_feature_importance(self, sorted_features):
        """Создание графика feature importance"""
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        plt.figure(figsize=(12, 10))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 30 Feature Importances (Averaged across all models)')
        plt.tight_layout()
        plt.savefig(f'{log_dir}/plots/feature_importance.png', dpi=150)
        plt.close()
        
        logger.info(f"📊 График сохранен: {log_dir}/plots/feature_importance.png")
            
    def _create_final_report(self, metadata: dict, results: dict):
        """Создание финального отчета с feature importance"""
        report = []
        report.append("="*80)
        report.append("ФИНАЛЬНЫЙ ОТЧЕТ - Enhanced XGBoost v2.0")
        report.append("="*80)
        report.append(f"Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Тип задачи: {metadata['task_type']}")
        report.append(f"Размер ансамбля: {metadata['ensemble_size']}")
        report.append(f"Количество признаков: {metadata['total_features']}")
        report.append(f"Время обучения: {metadata['training_time']/60:.1f} минут")
        
        if results:
            report.append("\nРЕЗУЛЬТАТЫ:")
            for direction, data in results.items():
                report.append(f"\n{direction.upper()}:")
                report.append(f"  Оптимальный порог: {data['threshold']:.2f}")
                metrics = data['metrics']
                report.append(f"  Accuracy: {metrics['accuracy']:.2%}")
                report.append(f"  Precision: {metrics['precision']:.2%}")
                report.append(f"  Recall: {metrics['recall']:.2%}")
                report.append(f"  F1-Score: {metrics['f1']:.3f}")
                report.append(f"  ROC-AUC: {metrics['auc']:.3f}")
        
        # Добавляем топ важных признаков
        if hasattr(self, 'models') and self.models:
            report.append("\nТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
            try:
                for model_name, model in self.models.items():
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        if len(importance) != len(self.feature_names):
                            logger.warning(f"Несоответствие размерностей для {model_name}: importance={len(importance)}, features={len(self.feature_names)}")
                            continue
                        indices = np.argsort(importance)[-10:][::-1]
                        report.append(f"\n{model_name}:")
                        for idx in indices:
                            if idx < len(self.feature_names):
                                report.append(f"  - {self.feature_names[idx]}: {importance[idx]:.4f}")
                        break  # Показываем только первую модель
            except Exception as e:
                logger.error(f"Ошибка при анализе feature importance: {e}")
                report.append("  Ошибка при анализе важности признаков")
                
        report.append("\nМОДЕЛИ СОХРАНЕНЫ В:")
        report.append("  trained_model/*_xgboost_v2_*.pkl")
        report.append("  trained_model/scaler_xgboost_v2.pkl")
        report.append("  trained_model/metadata_xgboost_v2.json")
        report.append("="*80)
        
        report_text = '\n'.join(report)
        print(report_text)
        
        with open(f'{log_dir}/final_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    def analyze_winning_patterns(self, model, X_test, y_test, y_pred, y_pred_proba, model_name: str):
        """Улучшенный анализ паттернов для криптотрейдинга"""
        logger.info(f"\n🏆 Анализ выигрышных паттернов для {model_name}:")
        
        # Находим примеры с высокой уверенностью
        if len(y_pred_proba.shape) > 1:  # multiclass
            max_proba = y_pred_proba.max(axis=1)
        else:
            max_proba = y_pred_proba
        
        # Адаптивный порог на основе распределения
        percentile_75 = np.percentile(max_proba, 75)
        threshold = max(0.6, percentile_75)
            
        high_confidence = max_proba > threshold
        high_conf_correct = (y_pred == y_test) & high_confidence & (y_test > 0)  # Только правильные сигналы "входить"
        
        if high_conf_correct.sum() == 0:
            logger.warning("⚠️ Нет сигналов с высокой уверенностью")
            return
            
        # Получаем DataFrame с признаками
        X_test_df = pd.DataFrame(X_test, columns=self.feature_names)
        winning_patterns = X_test_df[high_conf_correct]
        
        # Основные статистики
        logger.info(f"📊 Найдено {high_conf_correct.sum()} выигрышных сигналов с уверенностью > {threshold:.1%}")
        
        # Анализируем расширенный набор паттернов
        pattern_features = [
            'momentum_score', 'volume_strength_score', 'volatility_regime_score',
            'oversold_reversal_score', 'breakout_score', 'range_trading_score',
            'asia_session_score', 'rsi_val', 'macd_bullish', 'volume_spike', 
            'strong_trend', 'bb_position', 'adx_val', 'atr_norm',
            'btc_correlation_20', 'market_regime_high_vol'
        ]
        
        logger.info("🔍 Характеристики выигрышных паттернов:")
        
        for feature in pattern_features:
            if feature in winning_patterns.columns:
                if feature.endswith('_score'):
                    # Для score показываем среднее и процент > порога
                    avg = winning_patterns[feature].mean()
                    threshold = 30 if 'oversold' in feature else 40
                    high_score_pct = (winning_patterns[feature] > threshold).mean() * 100
                    logger.info(f"   {feature}: среднее={avg:.1f}, >{threshold}={high_score_pct:.1f}%")
                elif feature in ['macd_bullish', 'volume_spike', 'strong_trend']:
                    # Для бинарных признаков
                    pct = winning_patterns[feature].mean() * 100
                    logger.info(f"   {feature}: {pct:.1f}% сигналов")
                else:
                    # Для остальных - среднее и стандартное отклонение
                    avg = winning_patterns[feature].mean()
                    std = winning_patterns[feature].std()
                    # Обработка случая с одним значением (std = NaN)
                    if np.isnan(std) or len(winning_patterns) == 1:
                        logger.info(f"   {feature}: {avg:.2f}")
                    else:
                        logger.info(f"   {feature}: {avg:.2f} \u00b1 {std:.2f}")
        
        # Находим наиболее частые комбинации
        if 'oversold_reversal_score' in winning_patterns.columns and 'macd_bullish' in winning_patterns.columns:
            combo1 = (winning_patterns['oversold_reversal_score'] > 30) & (winning_patterns['macd_bullish'] == 1)
            combo1_pct = combo1.mean() * 100
            logger.info(f"\n🎯 Комбинированные паттерны:")
            logger.info(f"   Oversold + MACD bullish: {combo1_pct:.1f}% сигналов")
            
        if 'breakout_score' in winning_patterns.columns and 'volume_spike' in winning_patterns.columns:
            combo2 = (winning_patterns['breakout_score'] > 40) & (winning_patterns['volume_spike'] == 1)
            combo2_pct = combo2.mean() * 100
            logger.info(f"   Breakout + Volume spike: {combo2_pct:.1f}% сигналов")
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, task, direction):
        """Оптимизация гиперпараметров с помощью Optuna - улучшенная версия"""
        
        # Отключаем логи Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            # Улучшенные диапазоны для криптотрейдинга
            params = {
                'max_depth': trial.suggest_int('max_depth', 6, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.8, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                'max_bin': trial.suggest_int('max_bin', 128, 512),
            }
            
            # НЕ включаем tree_method в params, чтобы не переопределять базовый
            # tree_method будет взят из базовой модели (gpu_hist если GPU доступен)
            
            # Создаем модель с предложенными параметрами
            if task == 'regression':
                model = xgb.XGBRegressor(
                    **params,
                    n_estimators=500,  # Баланс скорости и качества
                    objective='reg:squarederror',
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
            elif task == 'classification_binary':
                model = xgb.XGBClassifier(
                    **params,
                    n_estimators=1000,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    early_stopping_rounds=50,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                # Веса классов
                scale_pos_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1e-8)
                model.set_params(scale_pos_weight=scale_pos_weight)
            else:  # multiclass
                num_classes = len(np.unique(y_train))
                model = xgb.XGBClassifier(
                    **params,
                    n_estimators=1000,
                    objective='multi:softprob',
                    num_class=num_classes,
                    eval_metric='mlogloss',
                    early_stopping_rounds=50,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
            
            # Обучаем модель
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Оцениваем на валидации
            if task == 'regression':
                y_pred = model.predict(X_val)
                score = -mean_squared_error(y_val, y_pred)  # Минимизируем MSE
            else:
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)  # Максимизируем accuracy
            
            return score
        
        # Создаем исследование Optuna
        study = optuna.create_study(
            direction='maximize' if task != 'regression' else 'minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Запускаем оптимизацию
        logger.info(f"🔍 Начало Optuna оптимизации для {direction}...")
        study.optimize(objective, n_trials=50, show_progress_bar=True)  # Больше проб
        
        # Лучшие параметры
        best_params = study.best_params
        logger.info(f"✅ Лучшие параметры: {best_params}")
        logger.info(f"🏆 Лучший скор: {study.best_value:.4f}")
        
        return best_params
    
    def filter_trading_signals(self, predictions, probabilities, features_df, min_confidence=0.6):
        """Улучшенная фильтрация сигналов для криптотрейдинга"""
        filtered_signals = predictions.copy()
        
        # Фильтр 1: Адаптивная минимальная уверенность
        if len(probabilities.shape) > 1:  # multiclass
            max_proba = probabilities.max(axis=1)
        else:
            max_proba = probabilities
        
        # Адаптивный порог на основе распределения
        percentile_60 = np.percentile(max_proba, 60)
        adaptive_threshold = max(min_confidence, percentile_60)
            
        low_confidence = max_proba < adaptive_threshold
        filtered_signals[low_confidence] = 0  # Не входить
        
        # Фильтр 2: Многоуровневая стратегия фильтрации
        
        # 2.1 Не торговать в очень низкой волатильности
        if 'atr_norm' in features_df.columns:
            very_low_vol = features_df['atr_norm'] < features_df['atr_norm'].quantile(0.1)
            filtered_signals[very_low_vol] = 0
            
        # 2.2 Требуем хотя бы один сильный паттерн
        pattern_cols = ['momentum_score', 'volume_strength_score', 'oversold_reversal_score', 
                       'breakout_score', 'asia_session_score']
        pattern_cols = [col for col in pattern_cols if col in features_df.columns]
        
        if pattern_cols:
            # Для каждого паттерна свой порог
            pattern_thresholds = {
                'momentum_score': 0.3,
                'volume_strength_score': 0.3,
                'oversold_reversal_score': 0.4,
                'breakout_score': 0.4,
                'asia_session_score': 0.5
            }
            
            # Проверяем наличие хотя бы одного сильного паттерна
            has_strong_pattern = False
            for col in pattern_cols:
                threshold = pattern_thresholds.get(col, 0.3)
                has_strong_pattern |= (features_df[col] > threshold)
            
            filtered_signals[~has_strong_pattern] = 0
            
        # 2.3 Исключаем экстремальные случаи (возможно перекупленность/перепроданность)
        if 'rsi_val' in features_df.columns:
            extreme_rsi = (features_df['rsi_val'] < 20) | (features_df['rsi_val'] > 80)
            # Но разрешаем если есть сильный разворотный паттерн
            if 'oversold_reversal_score' in features_df.columns:
                strong_reversal = features_df['oversold_reversal_score'] > 0.5
                filtered_signals[extreme_rsi & ~strong_reversal] = 0
            
        # Логируем статистику фильтрации
        total_signals = (predictions > 0).sum()
        filtered_count = (filtered_signals > 0).sum()
        logger.info(f"\n🎯 Фильтрация сигналов:")
        logger.info(f"   Исходные сигналы: {total_signals}")
        logger.info(f"   После фильтрации: {filtered_count}")
        logger.info(f"   Отфильтровано: {total_signals - filtered_count} ({(1 - filtered_count/total_signals)*100:.1f}%)")
        
        return filtered_signals


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Enhanced XGBoost Training v2.0 с кэшированием')
    parser.add_argument('--task', type=str, default='classification_binary',
                       choices=['regression', 'classification_binary', 'classification_multiclass'],
                       help='Тип задачи')
    parser.add_argument('--ensemble_size', type=int, default=1,
                       help='Количество моделей в ансамбле')
    parser.add_argument('--test_mode', action='store_true',
                       help='Тестовый режим (2 символа, меньше эпох)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к конфигурации')
    parser.add_argument('--use-cache', action='store_true',
                       help='Использовать кэшированные данные если доступны')
    parser.add_argument('--no-cache', action='store_true',
                       help='Отключить кэширование полностью')
    parser.add_argument('--force-reload', action='store_true',
                       help='Принудительно обновить кэш')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Директория для кэша')
    parser.add_argument('--debug', action='store_true',
                       help='Режим отладки с дополнительными логами')
    parser.add_argument('--no-smote', action='store_true',
                       help='Отключить SMOTE балансировку классов')
    parser.add_argument('--balance-method', type=str, default='smote',
                       choices=['smote', 'random', 'none'],
                       help='Метод балансировки классов: smote, random или none')
    parser.add_argument('--classification-threshold', type=float, default=0.5,
                       help='Порог для бинарной классификации (по умолчанию 0.5%)')
    
    args = parser.parse_args()
    
    # Устанавливаем уровень логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("🐛 Режим отладки активирован")
    
    # Определяем режим кэширования
    if args.no_cache:
        use_cache = False
        logger.info("🚫 Кэширование отключено")
    else:
        use_cache = args.use_cache
    
    # Создаем и запускаем тренер
    trainer = XGBoostEnhancedTrainer(config_path=args.config)
    trainer.train_ensemble(
        task=args.task,
        ensemble_size=args.ensemble_size,
        test_mode=args.test_mode,
        use_cache=use_cache,
        force_reload=args.force_reload,
        no_smote=args.no_smote,
        classification_threshold=args.classification_threshold,
        balance_method=args.balance_method
    )


if __name__ == "__main__":
    main()