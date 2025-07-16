"""
Анализ прибыльности модели XGBoost на данных Bitcoin
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import pickle
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BitcoinModelAnalyzer:
    """Анализатор производительности модели на данных Bitcoin"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.connection = None
        self.models = {}
        self.feature_names = []
        self.threshold = 0.40  # Порог для генерации сигналов (повышенный для лучшей точности)
        
    def connect_db(self):
        """Подключение к базе данных"""
        self.connection = psycopg2.connect(
            host='localhost',
            port=5555,
            database='crypto_trading',
            user='ruslan',
            password=''
        )
        logger.info("✅ Подключение к БД установлено")
        
    def analyze_database_expected_returns(self):
        """Анализ expected returns в базе данных"""
        logger.info("\n" + "="*60)
        logger.info("📊 АНАЛИЗ EXPECTED RETURNS В БАЗЕ ДАННЫХ")
        logger.info("="*60)
        
        query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(buy_expected_return) as buy_records,
            COUNT(sell_expected_return) as sell_records,
            SUM(buy_expected_return) as sum_buy_expected,
            SUM(sell_expected_return) as sum_sell_expected,
            AVG(buy_expected_return) as avg_buy_expected,
            AVG(sell_expected_return) as avg_sell_expected,
            MIN(buy_expected_return) as min_buy_expected,
            MAX(buy_expected_return) as max_buy_expected,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY buy_expected_return) as median_buy_expected
        FROM processed_market_data
        WHERE symbol = 'BTCUSDT'
        """
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            stats = cursor.fetchone()
            
        logger.info(f"📈 Общая статистика:")
        logger.info(f"   Всего записей: {stats['total_records']:,}")
        logger.info(f"   Сумма buy_expected_return: {stats['sum_buy_expected']:.2f}")
        logger.info(f"   Сумма sell_expected_return: {stats['sum_sell_expected']:.2f}")
        logger.info(f"   Средний buy_expected_return: {stats['avg_buy_expected']:.4f}%")
        logger.info(f"   Медиана buy_expected_return: {stats['median_buy_expected']:.4f}%")
        logger.info(f"   Мин/Макс buy_expected_return: {stats['min_buy_expected']:.2f}% / {stats['max_buy_expected']:.2f}%")
        
        # Распределение по диапазонам
        ranges_query = """
        WITH ranges AS (
            SELECT 
                CASE 
                    WHEN buy_expected_return < -2 THEN '< -2%'
                    WHEN buy_expected_return >= -2 AND buy_expected_return < -1 THEN '-2% to -1%'
                    WHEN buy_expected_return >= -1 AND buy_expected_return < 0 THEN '-1% to 0%'
                    WHEN buy_expected_return >= 0 AND buy_expected_return < 1 THEN '0% to 1%'
                    WHEN buy_expected_return >= 1 AND buy_expected_return < 2 THEN '1% to 2%'
                    WHEN buy_expected_return >= 2 AND buy_expected_return < 3 THEN '2% to 3%'
                    ELSE '>= 3%'
                END as return_range,
                buy_expected_return,
                CASE 
                    WHEN buy_expected_return < -2 THEN 1
                    WHEN buy_expected_return >= -2 AND buy_expected_return < -1 THEN 2
                    WHEN buy_expected_return >= -1 AND buy_expected_return < 0 THEN 3
                    WHEN buy_expected_return >= 0 AND buy_expected_return < 1 THEN 4
                    WHEN buy_expected_return >= 1 AND buy_expected_return < 2 THEN 5
                    WHEN buy_expected_return >= 2 AND buy_expected_return < 3 THEN 6
                    ELSE 7
                END as sort_order
            FROM processed_market_data
            WHERE symbol = 'BTCUSDT'
        )
        SELECT 
            return_range,
            COUNT(*) as count,
            SUM(buy_expected_return) as sum_returns
        FROM ranges
        GROUP BY return_range, sort_order
        ORDER BY sort_order
        """
        
        logger.info(f"\n📊 Распределение buy_expected_return по диапазонам:")
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(ranges_query)
            ranges = cursor.fetchall()
            
        for r in ranges:
            pct = r['count'] / stats['total_records'] * 100
            logger.info(f"   {r['return_range']:12s}: {r['count']:7,} свечей ({pct:5.1f}%) | Сумма: {r['sum_returns']:10.2f}")
            
        # Проверка топ-10 худших expected returns
        worst_query = """
        SELECT timestamp, buy_expected_return
        FROM processed_market_data
        WHERE symbol = 'BTCUSDT'
        ORDER BY buy_expected_return ASC
        LIMIT 10
        """
        
        logger.info(f"\n⚠️ Топ-10 худших buy_expected_return:")
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(worst_query)
            worst = cursor.fetchall()
            
        for w in worst:
            dt = datetime.fromtimestamp(w['timestamp'] / 1000)
            logger.info(f"   {dt}: {w['buy_expected_return']:.2f}%")
            
        return stats
        
    def load_bitcoin_data(self):
        """Загрузка данных Bitcoin с техническими индикаторами"""
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
                
        # Создаем DataFrame с индикаторами
        indicators_df = pd.DataFrame(indicators_data)
        
        # Объединяем с основными данными
        df = pd.concat([df, indicators_df], axis=1)
        df.drop('technical_indicators', axis=1, inplace=True)
        
        # Проверка наличия всех необходимых признаков
        logger.info(f"✅ Загружено {len(df):,} записей с {len(indicators_df.columns)} индикаторами")
        
        # Рассчитываем дополнительные признаки
        from feature_engineering_for_analysis import calculate_additional_features
        df = calculate_additional_features(df)
        
        return df
        
    def load_models(self):
        """Загрузка обученных моделей"""
        logger.info(f"\n💾 Загрузка моделей из {self.model_path}...")
        
        # Сначала пробуем загрузить из модели (содержит все признаки)
        self.feature_names = []
        
        # Загружаем первую модель для получения списка признаков
        model_file = self.model_path / "buy_models" / "classification_binary_model_0.pkl"
        if model_file.exists():
            with open(model_file, 'rb') as f:
                temp_model = pickle.load(f)
                if hasattr(temp_model, 'feature_names'):
                    self.feature_names = list(temp_model.feature_names)
                    logger.info(f"   ✅ Загружено {len(self.feature_names)} признаков из модели")
        
        # Если не удалось, используем топ-20 из metrics.json
        if not self.feature_names:
            metrics_path = self.model_path / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    self.feature_names = metrics.get('feature_names', [])
                logger.info(f"   ⚠️ Используем топ-{len(self.feature_names)} признаков из metrics.json")
            
        # Загрузка моделей
        model_types = ['buy_profit', 'buy_loss', 'sell_profit', 'sell_loss']
        
        for model_type in model_types:
            if 'buy' in model_type:
                model_dir = self.model_path / "buy_models"
            else:
                model_dir = self.model_path / "sell_models"
                
            # Ищем файлы моделей (обычно classification_binary_model_*.pkl)
            model_files = list(model_dir.glob("classification_binary_model_*.pkl"))
            
            if model_files:
                # Берем первую модель из ансамбля для упрощения
                with open(model_files[0], 'rb') as f:
                    self.models[model_type] = pickle.load(f)
                logger.info(f"   ✅ Загружена модель {model_type}")
            else:
                logger.warning(f"   ⚠️ Модель {model_type} не найдена")
                
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для модели"""
        logger.info("\n🔧 Подготовка признаков...")
        
        # Проверяем наличие всех необходимых признаков
        missing_features = []
        for feature in self.feature_names:
            if feature not in df.columns:
                missing_features.append(feature)
                
        if missing_features:
            logger.warning(f"   ⚠️ Отсутствуют признаки: {missing_features[:10]}...")
            # Заполняем отсутствующие признаки нулями
            for feature in missing_features:
                df[feature] = 0
                
        # Выбираем только нужные признаки
        features_df = df[self.feature_names].copy()
        
        # Заполняем пропуски
        features_df = features_df.fillna(0)
        
        # Заменяем бесконечности
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        logger.info(f"   ✅ Подготовлено {len(features_df.columns)} признаков")
        
        return features_df
        
    def generate_predictions(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Генерация предсказаний модели"""
        logger.info("\n🤖 Генерация предсказаний...")
        
        predictions = {}
        
        # Для XGBoost Booster используем predict напрямую
        import xgboost as xgb
        
        # Для упрощения используем только модели buy_profit и sell_profit
        if 'buy_profit' in self.models:
            # Создаем DMatrix для XGBoost
            dmatrix = xgb.DMatrix(features_df)
            buy_probs = self.models['buy_profit'].predict(dmatrix)
            predictions['buy_signals'] = buy_probs > self.threshold
            predictions['buy_probs'] = buy_probs
            logger.info(f"   ✅ Buy сигналов: {predictions['buy_signals'].sum():,}")
        
        if 'sell_profit' in self.models:
            dmatrix = xgb.DMatrix(features_df)
            sell_probs = self.models['sell_profit'].predict(dmatrix)
            predictions['sell_signals'] = sell_probs > self.threshold
            predictions['sell_probs'] = sell_probs
            logger.info(f"   ✅ Sell сигналов: {predictions['sell_signals'].sum():,}")
            
        return predictions
        
    def analyze_model_performance(self, df: pd.DataFrame, predictions: Dict[str, np.ndarray]):
        """Анализ производительности модели"""
        logger.info("\n" + "="*60)
        logger.info("📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛИ")
        logger.info("="*60)
        
        # Добавляем предсказания к DataFrame
        df['buy_signal'] = predictions.get('buy_signals', False)
        df['sell_signal'] = predictions.get('sell_signals', False)
        df['buy_prob'] = predictions.get('buy_probs', 0)
        df['sell_prob'] = predictions.get('sell_probs', 0)
        
        # 1. Анализ Buy сигналов
        logger.info("\n🔵 BUY СИГНАЛЫ:")
        buy_signals_df = df[df['buy_signal']]
        logger.info(f"   Всего сигналов: {len(buy_signals_df):,} из {len(df):,} ({len(buy_signals_df)/len(df)*100:.1f}%)")
        
        if len(buy_signals_df) > 0:
            buy_sum = buy_signals_df['buy_expected_return'].sum()
            buy_mean = buy_signals_df['buy_expected_return'].mean()
            buy_median = buy_signals_df['buy_expected_return'].median()
            buy_positive = (buy_signals_df['buy_expected_return'] > 0).sum()
            buy_above_threshold = (buy_signals_df['buy_expected_return'] > 1.5).sum()
            
            logger.info(f"   Сумма expected returns: {buy_sum:.2f} (vs -91,568.92 для всех)")
            logger.info(f"   Средний expected return: {buy_mean:.4f}%")
            logger.info(f"   Медиана expected return: {buy_median:.4f}%")
            logger.info(f"   Положительных: {buy_positive:,} ({buy_positive/len(buy_signals_df)*100:.1f}%)")
            logger.info(f"   Выше порога 1.5%: {buy_above_threshold:,} ({buy_above_threshold/len(buy_signals_df)*100:.1f}%)")
            
        # 2. Анализ Sell сигналов
        logger.info("\n🔴 SELL СИГНАЛЫ:")
        sell_signals_df = df[df['sell_signal']]
        logger.info(f"   Всего сигналов: {len(sell_signals_df):,} из {len(df):,} ({len(sell_signals_df)/len(df)*100:.1f}%)")
        
        if len(sell_signals_df) > 0:
            sell_sum = sell_signals_df['sell_expected_return'].sum()
            sell_mean = sell_signals_df['sell_expected_return'].mean()
            sell_median = sell_signals_df['sell_expected_return'].median()
            sell_positive = (sell_signals_df['sell_expected_return'] > 0).sum()
            sell_above_threshold = (sell_signals_df['sell_expected_return'] > 1.5).sum()
            
            logger.info(f"   Сумма expected returns: {sell_sum:.2f}")
            logger.info(f"   Средний expected return: {sell_mean:.4f}%")
            logger.info(f"   Медиана expected return: {sell_median:.4f}%")
            logger.info(f"   Положительных: {sell_positive:,} ({sell_positive/len(sell_signals_df)*100:.1f}%)")
            logger.info(f"   Выше порога 1.5%: {sell_above_threshold:,} ({sell_above_threshold/len(sell_signals_df)*100:.1f}%)")
            
        # 3. Сравнение с идеальной торговлей
        logger.info("\n🎯 СРАВНЕНИЕ С ИДЕАЛЬНОЙ ТОРГОВЛЕЙ:")
        
        # Идеальная стратегия - только прибыльные сделки
        ideal_buy = df[df['buy_expected_return'] > 1.5]
        ideal_sell = df[df['sell_expected_return'] > 1.5]
        
        logger.info(f"   Идеальных buy сигналов: {len(ideal_buy):,}")
        logger.info(f"   Идеальная сумма buy: {ideal_buy['buy_expected_return'].sum():.2f}")
        logger.info(f"   Модель нашла buy: {len(buy_signals_df):,} ({len(buy_signals_df)/len(ideal_buy)*100:.1f}% от идеала)")
        
        # Пересечение - сколько правильных сигналов
        if len(buy_signals_df) > 0:
            correct_buy = df[df['buy_signal'] & (df['buy_expected_return'] > 1.5)]
            precision = len(correct_buy) / len(buy_signals_df) * 100
            recall = len(correct_buy) / len(ideal_buy) * 100
            logger.info(f"   Precision (точность): {precision:.1f}%")
            logger.info(f"   Recall (полнота): {recall:.1f}%")
            
        return df
        
    def save_results(self, df: pd.DataFrame):
        """Сохранение результатов анализа"""
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем все предсказания
        output_file = output_dir / f"bitcoin_predictions_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\n💾 Результаты сохранены в {output_file}")
        
        # Сохраняем только сигналы
        signals_df = df[df['buy_signal'] | df['sell_signal']]
        signals_file = output_dir / f"bitcoin_signals_{timestamp}.csv"
        signals_df.to_csv(signals_file, index=False)
        logger.info(f"💾 Сигналы сохранены в {signals_file}")
        
    def visualize_results(self, df: pd.DataFrame):
        """Визуализация результатов"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Распределение expected returns
        ax = axes[0, 0]
        ax.hist(df['buy_expected_return'], bins=100, alpha=0.5, label='Все свечи', color='blue')
        buy_signals_returns = df[df['buy_signal']]['buy_expected_return']
        if len(buy_signals_returns) > 0:
            ax.hist(buy_signals_returns, bins=50, alpha=0.7, label='Сигналы модели', color='green')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(1.5, color='orange', linestyle='--', alpha=0.5, label='Порог 1.5%')
        ax.set_xlabel('Buy Expected Return (%)')
        ax.set_ylabel('Количество')
        ax.set_title('Распределение Buy Expected Returns')
        ax.legend()
        ax.set_xlim(-5, 10)
        
        # 2. Кумулятивная прибыль
        ax = axes[0, 1]
        df_sorted = df.sort_values('timestamp')
        
        # Все сделки
        cumsum_all = df_sorted['buy_expected_return'].cumsum()
        ax.plot(cumsum_all.values, label='Все сделки', alpha=0.5, color='red')
        
        # Только сигналы модели
        model_returns = df_sorted['buy_expected_return'].where(df_sorted['buy_signal'], 0)
        cumsum_model = model_returns.cumsum()
        ax.plot(cumsum_model.values, label='Сигналы модели', color='green', linewidth=2)
        
        # Идеальная стратегия
        ideal_returns = df_sorted['buy_expected_return'].where(df_sorted['buy_expected_return'] > 1.5, 0)
        cumsum_ideal = ideal_returns.cumsum()
        ax.plot(cumsum_ideal.values, label='Идеальная стратегия', color='blue', linestyle='--')
        
        ax.set_xlabel('Номер свечи')
        ax.set_ylabel('Кумулятивный Return (%)')
        ax.set_title('Кумулятивная прибыль')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Вероятности модели vs Expected Return
        ax = axes[1, 0]
        if 'buy_prob' in df.columns:
            scatter_data = df.sample(min(10000, len(df)))  # Берем выборку для скорости
            colors = ['green' if x > 1.5 else 'red' for x in scatter_data['buy_expected_return']]
            ax.scatter(scatter_data['buy_prob'], scatter_data['buy_expected_return'], 
                      alpha=0.3, c=colors, s=1)
            ax.axhline(1.5, color='orange', linestyle='--', alpha=0.5)
            ax.axvline(self.threshold, color='blue', linestyle='--', alpha=0.5)
            ax.set_xlabel('Вероятность модели')
            ax.set_ylabel('Buy Expected Return (%)')
            ax.set_title('Вероятность vs Expected Return')
            ax.set_ylim(-5, 10)
        
        # 4. Статистика по времени
        ax = axes[1, 1]
        df['hour'] = pd.to_datetime(df['timestamp'], unit='ms').dt.hour
        hourly_stats = df.groupby('hour').agg({
            'buy_expected_return': 'mean',
            'buy_signal': 'sum'
        })
        
        ax2 = ax.twinx()
        ax.bar(hourly_stats.index, hourly_stats['buy_signal'], alpha=0.5, label='Кол-во сигналов')
        ax2.plot(hourly_stats.index, hourly_stats['buy_expected_return'], 
                color='red', marker='o', label='Средний return')
        
        ax.set_xlabel('Час дня')
        ax.set_ylabel('Количество сигналов')
        ax2.set_ylabel('Средний Expected Return (%)')
        ax.set_title('Статистика по часам')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        output_file = "analysis_results/bitcoin_analysis_charts.png"
        plt.savefig(output_file, dpi=300)
        logger.info(f"📊 Графики сохранены в {output_file}")
        plt.close()

def main():
    # Путь к последним результатам модели
    model_path = "/Users/ruslan/PycharmProjects/LLM TRANSFORM/logs_from_gpu/xgboost_v3_20250616_070913"
    
    # Создаем анализатор
    analyzer = BitcoinModelAnalyzer(model_path)
    
    try:
        # Подключаемся к БД
        analyzer.connect_db()
        
        # Анализ expected returns в БД
        db_stats = analyzer.analyze_database_expected_returns()
        
        # Загружаем данные Bitcoin
        df = analyzer.load_bitcoin_data()
        
        # Загружаем модели
        analyzer.load_models()
        
        if analyzer.models and analyzer.feature_names:
            # Подготавливаем признаки
            features_df = analyzer.prepare_features(df)
            
            # Генерируем предсказания
            predictions = analyzer.generate_predictions(features_df)
            
            # Анализируем производительность
            results_df = analyzer.analyze_model_performance(df, predictions)
            
            # Визуализация
            analyzer.visualize_results(results_df)
            
            # Сохраняем результаты
            analyzer.save_results(results_df)
        else:
            logger.error("❌ Не удалось загрузить модели или признаки")
            
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