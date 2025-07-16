#!/usr/bin/env python3
"""
Обучение продвинутых моделей с confidence-based подходом.

Ключевые улучшения:
1. Confidence-based предсказания (модель предсказывает уверенность)
2. Ансамбль стратегий (trend, reversion, breakout, momentum)
3. Адаптивные пороги на основе волатильности
4. Учет рыночного режима
5. Multi-task learning (несколько целевых переменных одновременно)
"""

import sys
import os
import logging
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
import joblib
import json
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfidenceModel:
    """
    Модель с confidence-based предсказаниями.
    
    Идея: модель предсказывает не только направление движения,
    но и уверенность в этом предсказании.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.direction_model = None  # Предсказывает направление
        self.confidence_model = None  # Предсказывает уверенность
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, confidence_target: pd.Series):
        """
        Обучает модель направления и уверенности
        
        Args:
            X: Признаки
            y: Целевая переменная (направление)
            confidence_target: Целевая переменная уверенности
        """
        # Нормализуем признаки
        X_scaled = self.scaler.fit_transform(X)
        
        # Модель направления (классификация)
        self.direction_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.direction_model.fit(X_scaled, y)
        
        # Модель уверенности (регрессия)
        self.confidence_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse'
        )
        
        self.confidence_model.fit(X_scaled, confidence_target)
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает направление и уверенность
        
        Returns:
            direction_proba: Вероятности направления
            confidence: Уверенность модели (0-1)
        """
        X_scaled = self.scaler.transform(X)
        
        # Предсказание направления
        direction_proba = self.direction_model.predict_proba(X_scaled)[:, 1]
        
        # Предсказание уверенности
        confidence = self.confidence_model.predict(X_scaled)
        confidence = np.clip(confidence, 0, 1)  # Ограничиваем [0, 1]
        
        return direction_proba, confidence
    
    def predict_with_confidence(self, X: pd.DataFrame, min_confidence: float = 0.6) -> Dict:
        """
        Предсказания только для случаев с высокой уверенностью
        """
        direction_proba, confidence = self.predict(X)
        
        # Фильтруем по уверенности
        high_confidence_mask = confidence >= min_confidence
        
        results = {
            'all_predictions': direction_proba,
            'all_confidence': confidence,
            'high_confidence_mask': high_confidence_mask,
            'high_confidence_predictions': direction_proba[high_confidence_mask],
            'high_confidence_count': high_confidence_mask.sum(),
            'coverage': high_confidence_mask.mean()
        }
        
        return results


class StrategyEnsemble:
    """
    Ансамбль стратегий для разных рыночных условий
    """
    
    def __init__(self):
        self.models = {
            'trend_following': ConfidenceModel('trend_following'),
            'mean_reversion': ConfidenceModel('mean_reversion'),
            'breakout': ConfidenceModel('breakout'),
            'momentum': ConfidenceModel('momentum')
        }
        self.regime_model = None  # Модель для предсказания режима рынка
        
    def fit(self, X: pd.DataFrame, targets: Dict[str, pd.Series], 
            market_regime: pd.Series):
        """
        Обучает ансамбль стратегий
        
        Args:
            X: Признаки
            targets: Целевые переменные для каждой стратегии
            market_regime: Рыночный режим
        """
        logger.info("Обучение ансамбля стратегий...")
        
        # Обучаем модель режима рынка
        self._fit_regime_model(X, market_regime)
        
        # Обучаем модели стратегий
        for strategy_name, model in self.models.items():
            if strategy_name in targets:
                logger.info(f"  Обучение стратегии: {strategy_name}")
                
                y = targets[strategy_name]['direction']
                confidence_target = targets[strategy_name]['confidence']
                
                model.fit(X, y, confidence_target)
    
    def _fit_regime_model(self, X: pd.DataFrame, market_regime: pd.Series):
        """Обучает модель предсказания рыночного режима"""
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        regime_encoded = le.fit_transform(market_regime)
        
        self.regime_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.regime_model.fit(X_scaled, regime_encoded)
        self.regime_encoder = le
        self.regime_scaler = scaler
    
    def predict(self, X: pd.DataFrame, min_confidence: float = 0.6) -> Dict:
        """
        Предсказания ансамбля с учетом режима рынка
        """
        # Предсказываем режим рынка
        X_regime_scaled = self.regime_scaler.transform(X)
        predicted_regime = self.regime_model.predict(X_regime_scaled)
        regime_proba = self.regime_model.predict_proba(X_regime_scaled)
        
        # Получаем предсказания от всех стратегий
        strategy_predictions = {}
        for strategy_name, model in self.models.items():
            pred_result = model.predict_with_confidence(X, min_confidence)
            strategy_predictions[strategy_name] = pred_result
        
        # Взвешенное голосование на основе режима и уверенности
        ensemble_prediction = self._combine_predictions(
            strategy_predictions, predicted_regime, regime_proba
        )
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'strategy_predictions': strategy_predictions,
            'predicted_regime': predicted_regime,
            'regime_proba': regime_proba
        }
    
    def _combine_predictions(self, strategy_predictions: Dict, 
                           predicted_regime: np.ndarray,
                           regime_proba: np.ndarray) -> Dict:
        """Объединяет предсказания стратегий"""
        n_samples = len(predicted_regime)
        
        # Веса стратегий в зависимости от режима
        regime_weights = {
            0: {'trend_following': 0.4, 'momentum': 0.3, 'breakout': 0.2, 'mean_reversion': 0.1},
            1: {'mean_reversion': 0.4, 'trend_following': 0.3, 'breakout': 0.2, 'momentum': 0.1},
            2: {'breakout': 0.4, 'momentum': 0.3, 'trend_following': 0.2, 'mean_reversion': 0.1},
            3: {'mean_reversion': 0.5, 'breakout': 0.3, 'trend_following': 0.1, 'momentum': 0.1},
            4: {'trend_following': 0.5, 'momentum': 0.3, 'breakout': 0.1, 'mean_reversion': 0.1}
        }
        
        ensemble_proba = np.zeros(n_samples)
        ensemble_confidence = np.zeros(n_samples)
        
        for i in range(n_samples):
            regime = predicted_regime[i]
            weights = regime_weights.get(regime, regime_weights[0])
            
            weighted_proba = 0
            weighted_confidence = 0
            total_weight = 0
            
            for strategy, weight in weights.items():
                if strategy in strategy_predictions:
                    pred = strategy_predictions[strategy]
                    if i < len(pred['all_predictions']):
                        weighted_proba += pred['all_predictions'][i] * weight
                        weighted_confidence += pred['all_confidence'][i] * weight
                        total_weight += weight
            
            if total_weight > 0:
                ensemble_proba[i] = weighted_proba / total_weight
                ensemble_confidence[i] = weighted_confidence / total_weight
        
        return {
            'probabilities': ensemble_proba,
            'confidence': ensemble_confidence,
            'high_confidence_mask': ensemble_confidence >= 0.6,
            'coverage': (ensemble_confidence >= 0.6).mean()
        }


class AdvancedTrainingSystem:
    """Продвинутая система обучения"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
            
        self.db_config = {
            'host': self.config_dict['database']['host'],
            'port': self.config_dict['database']['port'],
            'database': self.config_dict['database']['database'],
            'user': self.config_dict['database']['user'],
            'password': self.config_dict['database']['password']
        }
        
        self.models = {}
        
    def load_advanced_data(self, symbols: List[str] = None, 
                          target_horizon: str = '1hour') -> pd.DataFrame:
        """Загружает продвинутые данные для обучения"""
        conn = psycopg2.connect(**self.db_config)
        
        try:
            # Загружаем advanced targets + технические индикаторы
            query = f"""
            SELECT 
                a.timestamp,
                a.symbol,
                a.close_price,
                a.atr_14,
                a.volatility_percentile,
                a.market_regime,
                a.adaptive_threshold,
                a.buy_adaptive_{target_horizon} as target_direction,
                a.buy_strong_{target_horizon} as target_strong,
                a.return_normalized_{target_horizon} as target_return,
                a.risk_adjusted_return_{target_horizon} as target_risk_adj,
                a.trend_signal,
                a.reversion_signal,
                a.breakout_signal,
                a.momentum_signal,
                -- Технические индикаторы из processed_market_data
                (p.technical_indicators->>'rsi_val')::float as rsi_val,
                (p.technical_indicators->>'macd_val')::float as macd_val,
                (p.technical_indicators->>'macd_signal')::float as macd_signal,
                (p.technical_indicators->>'bb_upper')::float as bb_upper,
                (p.technical_indicators->>'bb_lower')::float as bb_lower,
                (p.technical_indicators->>'bb_percent')::float as bb_percent,
                (p.technical_indicators->>'atr_val')::float as atr_val,
                (p.technical_indicators->>'adx_val')::float as adx_val,
                (p.technical_indicators->>'stoch_k')::float as stoch_k,
                (p.technical_indicators->>'williams_r')::float as williams_r,
                (p.technical_indicators->>'cci_val')::float as cci_val,
                (p.technical_indicators->>'mfi_val')::float as mfi_val,
                (p.technical_indicators->>'obv_val')::float as obv_val,
                (p.technical_indicators->>'ema_9')::float as ema_9,
                (p.technical_indicators->>'ema_21')::float as ema_21,
                (p.technical_indicators->>'sma_50')::float as sma_50,
                (p.technical_indicators->>'vwap')::float as vwap
            FROM advanced_targets a
            JOIN processed_market_data p ON 
                EXTRACT(EPOCH FROM a.timestamp) * 1000 = p.timestamp 
                AND a.symbol = p.symbol
            WHERE a.buy_adaptive_{target_horizon} IS NOT NULL
            """
            
            params = []
            if symbols:
                placeholders = ','.join(['%s'] * len(symbols))
                query += f" AND a.symbol IN ({placeholders})"
                params.extend(symbols)
            
            query += " ORDER BY a.timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            logger.info(f"Загружено {len(df)} записей для обучения")
            
            # Проверяем баланс классов
            if 'target_direction' in df.columns:
                class_counts = df['target_direction'].value_counts()
                logger.info("\n📊 Распределение классов:")
                for class_val, count in class_counts.items():
                    percent = count / len(df) * 100
                    logger.info(f"  - {class_val}: {count} ({percent:.1f}%)")
            
            return df
            
        finally:
            conn.close()
    
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Подготавливает признаки и целевые переменные"""
        
        # Признаки
        feature_cols = [
            'close_price', 'atr_14', 'volatility_percentile', 'adaptive_threshold',
            'rsi_val', 'macd_val', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_percent',
            'atr_val', 'adx_val', 'stoch_k', 'williams_r', 'cci_val', 'mfi_val',
            'obv_val', 'ema_9', 'ema_21', 'sma_50', 'vwap'
        ]
        
        X = df[feature_cols].fillna(0)
        
        # Целевые переменные для разных стратегий
        targets = {
            'trend_following': {
                'direction': df['trend_signal'].fillna(0).astype(int),
                'confidence': self._calculate_confidence(df, 'trend_signal')
            },
            'mean_reversion': {
                'direction': df['reversion_signal'].fillna(0).astype(int),
                'confidence': self._calculate_confidence(df, 'reversion_signal')
            },
            'breakout': {
                'direction': df['breakout_signal'].fillna(0).astype(int),
                'confidence': self._calculate_confidence(df, 'breakout_signal')
            },
            'momentum': {
                'direction': df['momentum_signal'].fillna(0).astype(int),
                'confidence': self._calculate_confidence(df, 'momentum_signal')
            }
        }
        
        # Метаданные
        metadata = {
            'timestamp': df['timestamp'],
            'symbol': df['symbol'],
            'market_regime': df['market_regime'],
            'main_target': df['target_direction'].fillna(0).astype(int),
            'strong_target': df['target_strong'].fillna(0).astype(int)
        }
        
        return X, targets, metadata
    
    def _calculate_confidence(self, df: pd.DataFrame, signal_col: str) -> pd.Series:
        """Рассчитывает уверенность для сигнала"""
        # Простая эвристика: уверенность на основе волатильности и режима
        base_confidence = 0.5
        
        # Больше уверенности при низкой волатильности
        vol_factor = 1 - df['volatility_percentile'].fillna(0.5)
        
        # Больше уверенности для трендовых режимов
        regime_factor = df['market_regime'].apply(
            lambda x: 0.8 if 'trending' in str(x).lower() else 0.6
        ).fillna(0.6)
        
        confidence = base_confidence + (vol_factor * 0.3) + (regime_factor * 0.2)
        confidence = np.clip(confidence, 0.2, 0.95)
        
        return confidence
    
    def train_with_time_series_cv(self, df: pd.DataFrame, n_splits: int = 5):
        """Обучение с временной кросс-валидацией"""
        X, targets, metadata = self.prepare_features_and_targets(df)
        
        logger.info("\n🚀 Начало обучения с временной кросс-валидацией")
        
        # Временная кросс-валидация
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Результаты валидации
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"\n--- Fold {fold + 1}/{n_splits} ---")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            
            # Ансамбль стратегий
            ensemble = StrategyEnsemble()
            
            # Подготавливаем целевые переменные для обучения
            train_targets = {}
            for strategy in targets:
                train_targets[strategy] = {
                    'direction': targets[strategy]['direction'].iloc[train_idx],
                    'confidence': targets[strategy]['confidence'].iloc[train_idx]
                }
            
            # Обучаем ансамбль
            ensemble.fit(
                X_train, 
                train_targets, 
                metadata['market_regime'].iloc[train_idx]
            )
            
            # Валидация
            val_results = ensemble.predict(X_val, min_confidence=0.6)
            
            # Метрики
            val_target = metadata['main_target'].iloc[val_idx]
            
            # Overall accuracy
            ensemble_pred = (val_results['ensemble_prediction']['probabilities'] > 0.5).astype(int)
            accuracy = accuracy_score(val_target, ensemble_pred)
            
            # High-confidence accuracy
            high_conf_mask = val_results['ensemble_prediction']['high_confidence_mask']
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = accuracy_score(
                    val_target[high_conf_mask],
                    ensemble_pred[high_conf_mask]
                )
                coverage = high_conf_mask.mean()
            else:
                high_conf_accuracy = 0
                coverage = 0
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(val_target, val_results['ensemble_prediction']['probabilities'])
            except:
                roc_auc = 0.5
            
            fold_results = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'high_confidence_accuracy': high_conf_accuracy,
                'coverage': coverage,
                'roc_auc': roc_auc,
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            }
            
            cv_results.append(fold_results)
            
            # Логируем результаты
            logger.info(f"Accuracy: {accuracy:.3f}")
            logger.info(f"High-conf accuracy: {high_conf_accuracy:.3f}")
            logger.info(f"Coverage: {coverage:.3f}")
            logger.info(f"ROC-AUC: {roc_auc:.3f}")
        
        # Итоговые результаты
        self._log_cv_summary(cv_results)
        
        # Обучаем финальную модель на всех данных
        logger.info("\n🎯 Обучение финальной модели на всех данных...")
        final_ensemble = StrategyEnsemble()
        final_ensemble.fit(X, targets, metadata['market_regime'])
        
        return final_ensemble, cv_results
    
    def _log_cv_summary(self, cv_results: List[Dict]):
        """Логирует сводку по кросс-валидации"""
        df_results = pd.DataFrame(cv_results)
        
        logger.info("\n📊 СВОДКА ПО КРОСС-ВАЛИДАЦИИ:")
        logger.info("-" * 50)
        
        for metric in ['accuracy', 'high_confidence_accuracy', 'coverage', 'roc_auc']:
            mean_val = df_results[metric].mean()
            std_val = df_results[metric].std()
            logger.info(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")
        
        logger.info(f"\nСредний размер валидации: {df_results['val_size'].mean():.0f}")
        logger.info(f"Общее количество fold: {len(cv_results)}")


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Обучение продвинутых моделей")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'],
                       help='Символы для обучения')
    parser.add_argument('--horizon', default='1hour',
                       choices=['15min', '1hour', '4hour', '16hour'],
                       help='Временной горизонт')
    parser.add_argument('--cv-splits', type=int, default=5,
                       help='Количество fold для кросс-валидации')
    
    args = parser.parse_args()
    
    logger.info("""
    ╔══════════════════════════════════════════════════════╗
    ║         Обучение продвинутых ML моделей             ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Создаем систему обучения
    training_system = AdvancedTrainingSystem('config.yaml')
    
    # Загружаем данные
    df = training_system.load_advanced_data(
        symbols=args.symbols,
        target_horizon=args.horizon
    )
    
    if len(df) == 0:
        logger.error("❌ Нет данных для обучения!")
        logger.info("Сначала запустите: python advanced_trading_system.py --test")
        return
    
    # Обучаем модели
    final_model, cv_results = training_system.train_with_time_series_cv(
        df, n_splits=args.cv_splits
    )
    
    # Сохраняем модель
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = Path(f'advanced_models_{timestamp}')
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(final_model, model_dir / 'ensemble_model.pkl')
    
    # Сохраняем результаты CV
    pd.DataFrame(cv_results).to_csv(model_dir / 'cv_results.csv', index=False)
    
    logger.info(f"\n✅ Модель сохранена в {model_dir}")
    logger.info("\n📝 Следующие шаги:")
    logger.info("1. Проанализируйте результаты кросс-валидации")
    logger.info("2. Протестируйте модель на новых данных")
    logger.info("3. Интегрируйте в торговую стратегию")


if __name__ == "__main__":
    main()