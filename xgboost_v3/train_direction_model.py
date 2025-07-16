#!/usr/bin/env python3
"""
Обучение модели предсказания направления движения цены.
Использует простые бинарные метки и walk-forward анализ.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
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

# Модули проекта
from data.simple_targets import SimpleTargetSystem
from models.xgboost_trainer import XGBoostTrainer
from utils.metrics import MetricsCalculator
from utils.visualization import plot_walk_forward_results, plot_feature_importance
from config import Config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Реализует walk-forward анализ для временных рядов.
    Это золотой стандарт валидации в трейдинге.
    """
    
    def __init__(self,
                 train_window_days: int = 30,
                 test_window_days: int = 7,
                 n_splits: int = 10,
                 gap_hours: int = 1):
        """
        Args:
            train_window_days: Размер окна обучения в днях
            test_window_days: Размер окна тестирования в днях
            n_splits: Количество разбиений
            gap_hours: Зазор между train и test (для избежания утечки)
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.n_splits = n_splits
        self.gap_hours = gap_hours
        
    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Разбивает данные на train/test с помощью walk-forward.
        
        Returns:
            Список кортежей (train_df, test_df)
        """
        df = df.sort_values('_timestamp').copy()
        
        # Определяем временные границы
        start_time = df['_timestamp'].min()
        end_time = df['_timestamp'].max()
        
        # Вычисляем шаг для сдвига окна
        total_days = (end_time - start_time).days
        step_days = (total_days - self.train_window_days - self.test_window_days) // (self.n_splits - 1)
        
        splits = []
        
        for i in range(self.n_splits):
            # Определяем границы train
            train_start = start_time + timedelta(days=i * step_days)
            train_end = train_start + timedelta(days=self.train_window_days)
            
            # Определяем границы test (с зазором)
            test_start = train_end + timedelta(hours=self.gap_hours)
            test_end = test_start + timedelta(days=self.test_window_days)
            
            # Проверяем, что не вышли за границы
            if test_end > end_time:
                break
                
            # Фильтруем данные
            train_df = df[(df['_timestamp'] >= train_start) & (df['_timestamp'] < train_end)]
            test_df = df[(df['_timestamp'] >= test_start) & (df['_timestamp'] < test_end)]
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
                
                logger.info(f"Split {i+1}: Train {train_start.date()} to {train_end.date()} "
                          f"({len(train_df):,} samples), "
                          f"Test {test_start.date()} to {test_end.date()} "
                          f"({len(test_df):,} samples)")
        
        return splits


class DirectionPredictor:
    """Основной класс для обучения модели предсказания направления"""
    
    def __init__(self, config_path: str):
        # Загружаем конфигурацию
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
            
        self.db_config = {
            'host': self.config_dict['database']['host'],
            'port': self.config_dict['database']['port'],
            'database': self.config_dict['database']['database'],
            'user': self.config_dict['database']['user'],
            'password': self.config_dict['database']['password']
        }
        
        # Создаем конфигурацию для XGBoost
        self.config = Config()
        self.config.training.task_type = 'classification_binary'
        
        # Оптимальные параметры для предсказания направления
        self.config.model.max_depth = 6
        self.config.model.learning_rate = 0.05
        self.config.model.n_estimators = 300
        self.config.model.subsample = 0.8
        self.config.model.colsample_bytree = 0.8
        
        # Метрики
        self.metrics_calculator = MetricsCalculator(self.config)
        
        # Результаты
        self.walk_forward_results = []
        
    def load_data(self, 
                  symbols: List[str],
                  target_type: str = 'buy_signal_threshold_1hour',
                  start_date: str = None,
                  end_date: str = None) -> pd.DataFrame:
        """Загружает данные для обучения"""
        
        import psycopg2
        import json
        
        conn = psycopg2.connect(**self.db_config)
        
        try:
            # Загружаем данные с развернутыми техническими индикаторами
            query = """
            SELECT 
                t.timestamp,
                t.symbol,
                t.{} as target,
                p.open,
                p.high,
                p.low,
                p.close,
                p.volume,
                -- Разворачиваем технические индикаторы
                (p.technical_indicators->>'rsi_val')::float as rsi_val,
                (p.technical_indicators->>'macd_val')::float as macd_val,
                (p.technical_indicators->>'macd_signal')::float as macd_signal,
                (p.technical_indicators->>'macd_diff')::float as macd_diff,
                (p.technical_indicators->>'bb_upper')::float as bb_upper,
                (p.technical_indicators->>'bb_middle')::float as bb_middle,
                (p.technical_indicators->>'bb_lower')::float as bb_lower,
                (p.technical_indicators->>'bb_width')::float as bb_width,
                (p.technical_indicators->>'bb_percent')::float as bb_percent,
                (p.technical_indicators->>'atr_val')::float as atr_val,
                (p.technical_indicators->>'adx_val')::float as adx_val,
                (p.technical_indicators->>'adx_plus_di')::float as adx_plus_di,
                (p.technical_indicators->>'adx_minus_di')::float as adx_minus_di,
                (p.technical_indicators->>'stoch_k')::float as stoch_k,
                (p.technical_indicators->>'stoch_d')::float as stoch_d,
                (p.technical_indicators->>'williams_r')::float as williams_r,
                (p.technical_indicators->>'cci_val')::float as cci_val,
                (p.technical_indicators->>'mfi_val')::float as mfi_val,
                (p.technical_indicators->>'obv_val')::float as obv_val,
                (p.technical_indicators->>'ema_9')::float as ema_9,
                (p.technical_indicators->>'ema_21')::float as ema_21,
                (p.technical_indicators->>'sma_50')::float as sma_50,
                (p.technical_indicators->>'sma_200')::float as sma_200,
                (p.technical_indicators->>'vwap')::float as vwap,
                (p.technical_indicators->>'pivot_point')::float as pivot_point,
                (p.technical_indicators->>'resistance_1')::float as resistance_1,
                (p.technical_indicators->>'support_1')::float as support_1,
                (p.technical_indicators->>'hour_sin')::float as hour_sin,
                (p.technical_indicators->>'hour_cos')::float as hour_cos,
                (p.technical_indicators->>'dow_sin')::float as dow_sin,
                (p.technical_indicators->>'dow_cos')::float as dow_cos,
                (p.technical_indicators->>'is_weekend')::float as is_weekend
            FROM simple_targets t
            JOIN processed_market_data p ON EXTRACT(EPOCH FROM t.timestamp) * 1000 = p.timestamp AND t.symbol = p.symbol
            WHERE t.{} IS NOT NULL
            """.format(target_type, target_type)
            
            conditions = []
            params = []
            
            if symbols:
                placeholders = ','.join(['%s'] * len(symbols))
                conditions.append(f"t.symbol IN ({placeholders})")
                params.extend(symbols)
                
            if start_date:
                conditions.append("t.timestamp >= %s")
                params.append(start_date)
                
            if end_date:
                conditions.append("t.timestamp <= %s")
                params.append(end_date)
                
            if conditions:
                query += " AND " + " AND ".join(conditions)
                
            query += " ORDER BY t.timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            logger.info(f"Загружено {len(df):,} записей")
            
            # Проверяем баланс классов
            if 'target' in df.columns:
                class_counts = df['target'].value_counts()
                logger.info(f"\n📊 Распределение классов для {target_type}:")
                for class_val, count in class_counts.items():
                    percent = count / len(df) * 100
                    logger.info(f"  - Класс {class_val}: {count} ({percent:.1f}%)")
            
            return df
            
        finally:
            conn.close()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготавливает признаки и целевую переменную"""
        
        # Целевая переменная
        y = df['target'].astype(int)
        
        # Удаляем ненужные колонки
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'symbol', 'target', 'expected_return_buy', 'expected_return_sell'
        ]]
        
        X = df[feature_cols]
        
        # Добавляем метаданные
        X = X.copy()  # Избегаем SettingWithCopyWarning
        X['_timestamp'] = df['timestamp'].values
        X['_symbol'] = df['symbol'].values
        
        logger.info(f"Подготовлено {X.shape[1]-2} признаков")
        
        return X, y
    
    def select_top_features(self, X: pd.DataFrame, y: pd.Series, top_k: int = 50) -> List[str]:
        """Отбирает топ признаков на основе важности"""
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Удаляем метаданные для обучения
        feature_cols = [col for col in X.columns if not col.startswith('_')]
        X_train = X[feature_cols]
        
        # Быстрая модель для оценки важности
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y)
        
        # Получаем важности
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Отбираем топ признаки
        top_features = importances.head(top_k)['feature'].tolist()
        
        logger.info(f"\nТоп-10 признаков:")
        for i, row in importances.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
        return top_features
    
    def train_walk_forward(self, 
                          df: pd.DataFrame,
                          target_direction: str = 'buy',
                          n_splits: int = 5):
        """Обучение с walk-forward валидацией"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-Forward обучение для {target_direction.upper()}")
        logger.info(f"{'='*60}")
        
        # Подготавливаем данные
        X, y = self.prepare_features(df)
        
        # Walk-forward валидатор
        validator = WalkForwardValidator(
            train_window_days=30,
            test_window_days=7,
            n_splits=n_splits,
            gap_hours=1
        )
        
        # Создаем DataFrame с метаданными
        data_with_meta = X.copy()
        data_with_meta['target'] = y
        
        splits = validator.split(data_with_meta)
        
        # Результаты для каждого split
        split_results = []
        all_predictions = []
        
        for i, (train_df, test_df) in enumerate(splits):
            logger.info(f"\n--- Split {i+1}/{len(splits)} ---")
            
            # Разделяем на X и y
            y_train = train_df['target']
            y_test = test_df['target']
            
            X_train = train_df.drop(columns=['target'])
            X_test = test_df.drop(columns=['target'])
            
            # Отбираем признаки на train данных
            if i == 0:  # Только в первый раз
                metadata_cols = [col for col in X_train.columns if col.startswith('_')]
                feature_cols = [col for col in X_train.columns if not col.startswith('_')]
                
                # Отбираем топ признаки
                selected_features = self.select_top_features(
                    X_train[feature_cols], 
                    y_train, 
                    top_k=50
                )
            
            # Используем только отобранные признаки
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Обучаем модель
            trainer = XGBoostTrainer(self.config, model_name=f"{target_direction}_split_{i+1}")
            
            # Простое разделение на train/val из train данных
            val_size = int(0.2 * len(X_train_selected))
            X_train_model = X_train_selected[:-val_size]
            y_train_model = y_train[:-val_size]
            X_val_model = X_train_selected[-val_size:]
            y_val_model = y_train[-val_size:]
            
            # Обучение
            model = trainer.train(
                X_train_model, y_train_model,
                X_val_model, y_val_model
            )
            
            # Предсказание на тесте
            y_pred_proba = trainer.predict(X_test_selected, return_proba=True)
            
            # Метрики
            metrics = self.metrics_calculator.calculate_classification_metrics(
                y_test.values, y_pred_proba
            )
            
            # Сохраняем результаты
            split_results.append({
                'split': i + 1,
                'train_start': train_df['_timestamp'].min(),
                'train_end': train_df['_timestamp'].max(),
                'test_start': test_df['_timestamp'].min(),
                'test_end': test_df['_timestamp'].max(),
                'train_size': len(train_df),
                'test_size': len(test_df),
                'metrics': metrics,
                'model': model
            })
            
            # Сохраняем предсказания
            predictions_df = test_df[['_timestamp', '_symbol']].copy()
            predictions_df['y_true'] = y_test
            predictions_df['y_pred_proba'] = y_pred_proba
            predictions_df['y_pred'] = (y_pred_proba > metrics['threshold']).astype(int)
            predictions_df['split'] = i + 1
            all_predictions.append(predictions_df)
            
            # Логируем результаты
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
        
        # Объединяем все предсказания
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Общие метрики
        overall_metrics = self.metrics_calculator.calculate_classification_metrics(
            all_predictions_df['y_true'].values,
            all_predictions_df['y_pred_proba'].values
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ОБЩИЕ РЕЗУЛЬТАТЫ для {target_direction.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"ROC-AUC: {overall_metrics['roc_auc']:.4f}")
        logger.info(f"Accuracy: {overall_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {overall_metrics['precision']:.4f}")
        logger.info(f"Recall: {overall_metrics['recall']:.4f}")
        logger.info(f"F1-Score: {overall_metrics['f1']:.4f}")
        
        return {
            'direction': target_direction,
            'splits': split_results,
            'predictions': all_predictions_df,
            'overall_metrics': overall_metrics,
            'selected_features': selected_features
        }
    
    def save_results(self, results: Dict, output_dir: str):
        """Сохраняет результаты обучения"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем метрики
        metrics_data = {
            'overall': results['overall_metrics'],
            'splits': []
        }
        
        for split in results['splits']:
            metrics_data['splits'].append({
                'split': split['split'],
                'train_period': f"{split['train_start']} to {split['train_end']}",
                'test_period': f"{split['test_start']} to {split['test_end']}",
                'metrics': split['metrics']
            })
        
        with open(output_path / f"{results['direction']}_metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        # Сохраняем предсказания
        results['predictions'].to_csv(
            output_path / f"{results['direction']}_predictions.csv",
            index=False
        )
        
        # Сохраняем список признаков
        with open(output_path / f"{results['direction']}_features.txt", 'w') as f:
            for feat in results['selected_features']:
                f.write(f"{feat}\n")
        
        # Сохраняем последнюю модель
        last_model = results['splits'][-1]['model']
        joblib.dump(
            last_model,
            output_path / f"{results['direction']}_model_latest.pkl"
        )
        
        logger.info(f"✅ Результаты сохранены в {output_path}")
    
    def generate_report(self, buy_results: Dict, sell_results: Dict, output_dir: str):
        """Генерирует финальный отчет"""
        
        output_path = Path(output_dir)
        
        report = f"""
============================================================
ОТЧЕТ: Модель предсказания направления движения цены
============================================================

Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

РЕЗУЛЬТАТЫ МОДЕЛИ BUY (предсказание роста цены):
------------------------------------------------------------
ROC-AUC: {buy_results['overall_metrics']['roc_auc']:.4f}
Accuracy: {buy_results['overall_metrics']['accuracy']:.4f}
Precision: {buy_results['overall_metrics']['precision']:.4f}
Recall: {buy_results['overall_metrics']['recall']:.4f}
F1-Score: {buy_results['overall_metrics']['f1']:.4f}

РЕЗУЛЬТАТЫ МОДЕЛИ SELL (предсказание падения цены):
------------------------------------------------------------
ROC-AUC: {sell_results['overall_metrics']['roc_auc']:.4f}
Accuracy: {sell_results['overall_metrics']['accuracy']:.4f}
Precision: {sell_results['overall_metrics']['precision']:.4f}
Recall: {sell_results['overall_metrics']['recall']:.4f}
F1-Score: {sell_results['overall_metrics']['f1']:.4f}

ДЕТАЛИ WALK-FORWARD АНАЛИЗА:
------------------------------------------------------------
Количество splits: {len(buy_results['splits'])}
Размер окна обучения: 30 дней
Размер окна тестирования: 7 дней
Зазор между train/test: 1 час

ТОП-10 ПРИЗНАКОВ (BUY):
------------------------------------------------------------
"""
        for i, feat in enumerate(buy_results['selected_features'][:10], 1):
            report += f"{i:2d}. {feat}\n"
            
        report += """
============================================================
"""
        
        with open(output_path / "final_report.txt", 'w') as f:
            f.write(report)
            
        logger.info(f"✅ Отчет сохранен: {output_path / 'final_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Обучение модели предсказания направления")
    
    parser.add_argument('--symbols', nargs='+', 
                       default=['BTCUSDT', 'ETHUSDT'],
                       help='Список символов для обучения')
    
    parser.add_argument('--target-type', 
                       default='buy_signal_threshold_1hour',
                       help='Тип целевой переменной')
    
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Количество splits для walk-forward')
    
    parser.add_argument('--output-dir', 
                       default='./direction_model_results',
                       help='Директория для результатов')
    
    args = parser.parse_args()
    
    # Создаем предиктор
    predictor = DirectionPredictor('config.yaml')
    
    # Создаем директорию для результатов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}/{timestamp}"
    
    # Загружаем данные
    df = predictor.load_data(
        symbols=args.symbols,
        target_type=args.target_type
    )
    
    # Обучаем модель для покупки
    buy_target = args.target_type.replace('sell', 'buy')
    df_buy = predictor.load_data(
        symbols=args.symbols,
        target_type=buy_target
    )
    
    buy_results = predictor.train_walk_forward(
        df_buy,
        target_direction='buy',
        n_splits=args.n_splits
    )
    
    # Обучаем модель для продажи
    sell_target = args.target_type.replace('buy', 'sell')
    df_sell = predictor.load_data(
        symbols=args.symbols,
        target_type=sell_target
    )
    
    sell_results = predictor.train_walk_forward(
        df_sell,
        target_direction='sell',
        n_splits=args.n_splits
    )
    
    # Сохраняем результаты
    predictor.save_results(buy_results, output_dir)
    predictor.save_results(sell_results, output_dir)
    
    # Генерируем отчет
    predictor.generate_report(buy_results, sell_results, output_dir)
    
    # Визуализация
    try:
        # График walk-forward результатов
        plot_walk_forward_results(
            buy_results['splits'],
            sell_results['splits'],
            output_dir
        )
        
        # График важности признаков
        plot_feature_importance(
            buy_results['selected_features'][:20],
            sell_results['selected_features'][:20],
            output_dir
        )
    except Exception as e:
        logger.warning(f"Не удалось создать графики: {e}")
    
    logger.info(f"\n✅ Обучение завершено! Результаты в {output_dir}")


if __name__ == "__main__":
    main()