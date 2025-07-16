"""
Анализ Bitcoin с использованием обученного ансамбля моделей
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import pickle
import xgboost as xgb
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleAnalyzer:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.connection = None
        self.buy_models = []
        self.sell_models = []
        self.buy_weights = []
        self.sell_weights = []
        self.feature_names = []
        
    def connect_db(self):
        """Подключение к БД"""
        self.connection = psycopg2.connect(
            host='localhost',
            port=5555,
            database='crypto_trading',
            user='ruslan',
            password=''
        )
        logger.info("✅ Подключение к БД установлено")
        
    def load_ensemble(self):
        """Загрузка ансамбля моделей"""
        logger.info(f"\n💾 Загрузка ансамбля из {self.model_path}...")
        
        # Загрузка метаданных ансамбля для buy
        buy_ensemble_path = self.model_path / "buy_models" / "ensemble_metadata.json"
        if buy_ensemble_path.exists():
            with open(buy_ensemble_path, 'r') as f:
                buy_meta = json.load(f)
                self.buy_weights = buy_meta['weights']
                logger.info(f"   ✅ Buy ансамбль: {len(self.buy_weights)} моделей")
                
        # Загрузка метаданных ансамбля для sell
        sell_ensemble_path = self.model_path / "sell_models" / "ensemble_metadata.json"
        if sell_ensemble_path.exists():
            with open(sell_ensemble_path, 'r') as f:
                sell_meta = json.load(f)
                self.sell_weights = sell_meta['weights']
                logger.info(f"   ✅ Sell ансамбль: {len(self.sell_weights)} моделей")
                
        # Загрузка buy моделей
        for i in range(len(self.buy_weights)):
            model_path = self.model_path / "buy_models" / f"classification_binary_model_{i}.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                self.buy_models.append(model)
                
                # Получаем feature names из первой модели
                if i == 0 and hasattr(model, 'feature_names'):
                    self.feature_names = list(model.feature_names)
                    
        # Загрузка sell моделей
        for i in range(len(self.sell_weights)):
            model_path = self.model_path / "sell_models" / f"classification_binary_model_{i}.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                self.sell_models.append(model)
                
        logger.info(f"   ✅ Загружено {len(self.feature_names)} признаков")
        
    def load_bitcoin_data(self):
        """Загрузка данных Bitcoin"""
        logger.info("\n📊 Загрузка данных BTCUSDT...")
        
        query = """
        SELECT 
            timestamp, open, high, low, close, volume,
            technical_indicators,
            buy_expected_return, 
            sell_expected_return
        FROM processed_market_data
        WHERE symbol = 'BTCUSDT'
        ORDER BY timestamp
        """
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            data = cursor.fetchall()
            
        df = pd.DataFrame(data)
        
        # Преобразование типов
        for col in ['open', 'high', 'low', 'close', 'volume', 'buy_expected_return', 'sell_expected_return']:
            df[col] = df[col].astype(float)
            
        # Извлечение технических индикаторов
        logger.info("   Извлечение технических индикаторов...")
        indicators_data = []
        
        for idx, row in df.iterrows():
            if row['technical_indicators']:
                if isinstance(row['technical_indicators'], dict):
                    indicators = row['technical_indicators']
                else:
                    indicators = json.loads(row['technical_indicators'])
                indicators_data.append(indicators)
            else:
                indicators_data.append({})
                
        indicators_df = pd.DataFrame(indicators_data)
        df = pd.concat([df, indicators_df], axis=1)
        df.drop('technical_indicators', axis=1, inplace=True)
        
        # Рассчитываем дополнительные признаки
        from feature_engineering_for_analysis import calculate_additional_features
        df = calculate_additional_features(df)
        
        logger.info(f"✅ Загружено {len(df):,} записей")
        return df
        
    def predict_ensemble(self, features_df: pd.DataFrame):
        """Предсказание с использованием ансамбля"""
        logger.info("\n🤖 Генерация предсказаний ансамбля...")
        
        # Подготовка признаков
        X = features_df[self.feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        dmatrix = xgb.DMatrix(X)
        
        # Buy предсказания
        buy_predictions = np.zeros(len(X))
        for i, (model, weight) in enumerate(zip(self.buy_models, self.buy_weights)):
            pred = model.predict(dmatrix)
            buy_predictions += pred * weight
            logger.info(f"   Buy модель {i+1}: weight={weight:.3f}, mean_pred={pred.mean():.3f}")
            
        # Sell предсказания
        sell_predictions = np.zeros(len(X))
        for i, (model, weight) in enumerate(zip(self.sell_models, self.sell_weights)):
            pred = model.predict(dmatrix)
            sell_predictions += pred * weight
            logger.info(f"   Sell модель {i+1}: weight={weight:.3f}, mean_pred={pred.mean():.3f}")
            
        return buy_predictions, sell_predictions
        
    def analyze_predictions(self, df: pd.DataFrame, buy_preds: np.ndarray, sell_preds: np.ndarray):
        """Анализ предсказаний"""
        logger.info("\n" + "="*60)
        logger.info("📊 АНАЛИЗ ПРЕДСКАЗАНИЙ АНСАМБЛЯ")
        logger.info("="*60)
        
        # Оптимальные пороги из обучения
        buy_threshold = 0.304
        sell_threshold = 0.297
        
        # Генерация сигналов
        buy_signals = buy_preds > buy_threshold
        sell_signals = sell_preds > sell_threshold
        
        logger.info(f"\n🔵 BUY АНАЛИЗ:")
        logger.info(f"   Порог: {buy_threshold:.3f}")
        logger.info(f"   Всего сигналов: {buy_signals.sum():,} ({buy_signals.sum()/len(df)*100:.1f}%)")
        
        buy_df = df[buy_signals]
        if len(buy_df) > 0:
            logger.info(f"   Сумма expected returns: {buy_df['buy_expected_return'].sum():.2f}")
            logger.info(f"   Средний expected return: {buy_df['buy_expected_return'].mean():.4f}%")
            logger.info(f"   Прибыльных (>0): {(buy_df['buy_expected_return'] > 0).sum():,} ({(buy_df['buy_expected_return'] > 0).sum()/len(buy_df)*100:.1f}%)")
            logger.info(f"   Прибыльных (>1.5%): {(buy_df['buy_expected_return'] > 1.5).sum():,} ({(buy_df['buy_expected_return'] > 1.5).sum()/len(buy_df)*100:.1f}%)")
            
            # Распределение вероятностей для сигналов
            signal_probs = buy_preds[buy_signals]
            logger.info(f"   Вероятности: min={signal_probs.min():.3f}, mean={signal_probs.mean():.3f}, max={signal_probs.max():.3f}")
            
        logger.info(f"\n🔴 SELL АНАЛИЗ:")
        logger.info(f"   Порог: {sell_threshold:.3f}")
        logger.info(f"   Всего сигналов: {sell_signals.sum():,} ({sell_signals.sum()/len(df)*100:.1f}%)")
        
        sell_df = df[sell_signals]
        if len(sell_df) > 0:
            logger.info(f"   Сумма expected returns: {sell_df['sell_expected_return'].sum():.2f}")
            logger.info(f"   Средний expected return: {sell_df['sell_expected_return'].mean():.4f}%")
            logger.info(f"   Прибыльных (>0): {(sell_df['sell_expected_return'] > 0).sum():,} ({(sell_df['sell_expected_return'] > 0).sum()/len(sell_df)*100:.1f}%)")
            logger.info(f"   Прибыльных (>1.5%): {(sell_df['sell_expected_return'] > 1.5).sum():,} ({(sell_df['sell_expected_return'] > 1.5).sum()/len(sell_df)*100:.1f}%)")
            
        # Сравнение с идеалом
        logger.info(f"\n🎯 СРАВНЕНИЕ С ИДЕАЛОМ:")
        ideal_buy = df[df['buy_expected_return'] > 1.5]
        ideal_sell = df[df['sell_expected_return'] > 1.5]
        
        logger.info(f"   Идеальных buy: {len(ideal_buy):,} (сумма: {ideal_buy['buy_expected_return'].sum():.2f})")
        logger.info(f"   Модель нашла buy: {buy_signals.sum():,}")
        
        if buy_signals.sum() > 0:
            correct_buy = df[buy_signals & (df['buy_expected_return'] > 1.5)]
            precision = len(correct_buy) / buy_signals.sum() * 100
            recall = len(correct_buy) / len(ideal_buy) * 100 if len(ideal_buy) > 0 else 0
            logger.info(f"   Buy Precision: {precision:.1f}%")
            logger.info(f"   Buy Recall: {recall:.1f}%")
            
        # Анализ по разным порогам
        logger.info(f"\n📈 АНАЛИЗ ПО РАЗНЫМ ПОРОГАМ:")
        thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        
        for thr in thresholds:
            signals = buy_preds > thr
            if signals.sum() > 0:
                selected = df[signals]
                profit_sum = selected['buy_expected_return'].sum()
                profitable_pct = (selected['buy_expected_return'] > 1.5).sum() / len(selected) * 100
                logger.info(f"   Порог {thr:.2f}: {signals.sum():5,} сигналов | Сумма: {profit_sum:8.2f} | Прибыльных: {profitable_pct:5.1f}%")
                
def main():
    model_path = "/Users/ruslan/PycharmProjects/LLM TRANSFORM/logs_from_gpu/xgboost_v3_20250616_070913"
    
    analyzer = EnsembleAnalyzer(model_path)
    
    try:
        # Подключение к БД
        analyzer.connect_db()
        
        # Загрузка ансамбля
        analyzer.load_ensemble()
        
        # Загрузка данных
        df = analyzer.load_bitcoin_data()
        
        # Предсказания
        buy_preds, sell_preds = analyzer.predict_ensemble(df)
        
        # Анализ
        analyzer.analyze_predictions(df, buy_preds, sell_preds)
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if analyzer.connection:
            analyzer.connection.close()
            logger.info("\n✅ Соединение с БД закрыто")

if __name__ == "__main__":
    main()