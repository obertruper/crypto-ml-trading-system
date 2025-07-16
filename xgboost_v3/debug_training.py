#!/usr/bin/env python3
"""
Отладка обучения XGBoost - минимальный пример
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_xgboost():
    """Тест простой модели XGBoost"""
    
    # 1. Загрузка данных
    logger.info("📥 Загрузка данных...")
    conn = psycopg2.connect(
        host="localhost", port=5555, database="crypto_trading", user="ruslan"
    )
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        # Загружаем небольшой объем данных для теста
        cursor.execute("""
            SELECT * FROM processed_market_data
            WHERE symbol IN ('BTCUSDT', 'ETHUSDT')
            AND buy_expected_return IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 50000
        """)
        data = cursor.fetchall()
    
    conn.close()
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Загружено {len(df)} записей")
    
    # 2. Подготовка признаков (только числовые)
    from config import EXCLUDE_COLUMNS
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    
    # Оставляем только числовые
    numeric_cols = []
    for col in feature_cols:
        try:
            df[col] = pd.to_numeric(df[col])
            numeric_cols.append(col)
        except:
            pass
    
    X = df[numeric_cols].fillna(0)
    y_buy = (df['buy_expected_return'].astype(float) > 1.5).astype(int)
    
    logger.info(f"📊 Признаков: {X.shape[1]}")
    logger.info(f"📊 Баланс классов: {y_buy.mean():.1%} положительных")
    
    # 3. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_buy, test_size=0.2, random_state=42, stratify=y_buy
    )
    
    # 4. Обучение с разными параметрами
    logger.info("\n🔬 Тестирование разных конфигураций XGBoost:")
    
    configs = [
        {
            'name': 'Базовая (без балансировки)',
            'params': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
        },
        {
            'name': 'С scale_pos_weight',
            'params': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum()
            }
        },
        {
            'name': 'С ограниченным scale_pos_weight',
            'params': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'scale_pos_weight': min(3.0, (y_train == 0).sum() / (y_train == 1).sum() * 0.7)
            }
        },
        {
            'name': 'Упрощенная модель',
            'params': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 2,
                'learning_rate': 0.3,
                'n_estimators': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        }
    ]
    
    for config in configs:
        logger.info(f"\n📌 {config['name']}:")
        if 'scale_pos_weight' in config['params']:
            logger.info(f"   scale_pos_weight: {config['params']['scale_pos_weight']:.2f}")
        
        # Обучение
        model = xgb.XGBClassifier(**config['params'], random_state=42)
        model.fit(X_train, y_train, verbose=False)
        
        # Предсказания
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Метрики
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Анализ предсказаний
        pred_stats = {
            'mean': y_pred_proba.mean(),
            'std': y_pred_proba.std(),
            'min': y_pred_proba.min(),
            'max': y_pred_proba.max(),
            'positive_rate': y_pred.mean()
        }
        
        logger.info(f"   ROC-AUC: {auc:.4f}")
        logger.info(f"   Вероятности: mean={pred_stats['mean']:.3f}, std={pred_stats['std']:.3f}, "
                   f"min={pred_stats['min']:.3f}, max={pred_stats['max']:.3f}")
        logger.info(f"   Предсказано класс 1: {pred_stats['positive_rate']:.1%}")
        
        # Топ признаки
        if hasattr(model, 'feature_importances_'):
            top_features_idx = np.argsort(model.feature_importances_)[-5:][::-1]
            logger.info("   Топ-5 признаков:")
            for idx in top_features_idx:
                logger.info(f"      {numeric_cols[idx]}: {model.feature_importances_[idx]:.3f}")

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("ОТЛАДКА ОБУЧЕНИЯ XGBoost")
    logger.info("="*60)
    test_simple_xgboost()