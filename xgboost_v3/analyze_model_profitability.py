"""
Анализ прибыльности модели XGBoost v3
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import pickle
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfitabilityAnalyzer:
    """Анализатор прибыльности торговых сигналов"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
        
    def connect(self):
        """Подключение к БД"""
        self.connection = psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config.get('password', '')
        )
        logger.info("✅ Подключение к БД установлено")
        
    def load_btc_data(self, start_date=None, end_date=None):
        """Загрузка данных BTCUSDT"""
        query = """
        SELECT 
            timestamp, 
            open, high, low, close, volume,
            buy_expected_return, 
            sell_expected_return
        FROM processed_market_data
        WHERE symbol = 'BTCUSDT'
        """
        
        if start_date:
            query += f" AND timestamp >= {int(start_date.timestamp() * 1000)}"
        if end_date:
            query += f" AND timestamp <= {int(end_date.timestamp() * 1000)}"
            
        query += " ORDER BY timestamp"
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            data = cursor.fetchall()
            
        df = pd.DataFrame(data)
        
        # Преобразование типов
        for col in ['open', 'high', 'low', 'close', 'volume', 'buy_expected_return', 'sell_expected_return']:
            df[col] = df[col].astype(float)
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ Загружено {len(df)} записей BTCUSDT")
        return df
        
    def calculate_ideal_profit(self, df, threshold=1.5):
        """Расчет идеальной прибыли при пороге"""
        # Подсчет идеальных сделок
        buy_signals = df['buy_expected_return'] > threshold
        sell_signals = df['sell_expected_return'] > threshold
        
        # Статистика
        total_bars = len(df)
        buy_count = buy_signals.sum()
        sell_count = sell_signals.sum()
        
        # Средняя прибыль
        avg_buy_profit = df.loc[buy_signals, 'buy_expected_return'].mean() if buy_count > 0 else 0
        avg_sell_profit = df.loc[sell_signals, 'sell_expected_return'].mean() if sell_count > 0 else 0
        
        # Общая прибыль (в %)
        total_buy_profit = df.loc[buy_signals, 'buy_expected_return'].sum()
        total_sell_profit = df.loc[sell_signals, 'sell_expected_return'].sum()
        
        logger.info(f"\n📊 ИДЕАЛЬНАЯ ТОРГОВЛЯ (порог {threshold}%):")
        logger.info(f"   Всего баров: {total_bars:,}")
        logger.info(f"   Buy сигналов: {buy_count:,} ({buy_count/total_bars*100:.1f}%)")
        logger.info(f"   Sell сигналов: {sell_count:,} ({sell_count/total_bars*100:.1f}%)")
        logger.info(f"   Средняя прибыль Buy: {avg_buy_profit:.2f}%")
        logger.info(f"   Средняя прибыль Sell: {avg_sell_profit:.2f}%")
        logger.info(f"   Общая прибыль Buy: {total_buy_profit:.0f}%")
        logger.info(f"   Общая прибыль Sell: {total_sell_profit:.0f}%")
        logger.info(f"   ИТОГО идеальная прибыль: {total_buy_profit + total_sell_profit:.0f}%")
        
        return {
            'total_bars': total_bars,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'avg_buy_profit': avg_buy_profit,
            'avg_sell_profit': avg_sell_profit,
            'total_buy_profit': total_buy_profit,
            'total_sell_profit': total_sell_profit,
            'total_profit': total_buy_profit + total_sell_profit
        }
        
    def simulate_model_trading(self, df, precision=0.36, recall=0.72):
        """Симуляция торговли с учетом точности модели"""
        # Для упрощения используем средние метрики модели
        # precision = 36% означает, что из всех сигналов модели только 36% прибыльные
        # recall = 72% означает, что модель находит 72% от всех прибыльных сделок
        
        # Реальные прибыльные сделки (expected_return > 1.5%)
        real_buy_profitable = (df['buy_expected_return'] > 1.5).sum()
        real_sell_profitable = (df['sell_expected_return'] > 1.5).sum()
        
        # Модель находит recall% от реальных прибыльных сделок
        model_found_buy = int(real_buy_profitable * recall)
        model_found_sell = int(real_sell_profitable * recall)
        
        # Но из-за низкой точности, общее количество сигналов больше
        total_buy_signals = int(model_found_buy / precision)
        total_sell_signals = int(model_found_sell / precision)
        
        # Ложные сигналы
        false_buy_signals = total_buy_signals - model_found_buy
        false_sell_signals = total_sell_signals - model_found_sell
        
        # Средние потери на ложных сигналах (примерно -0.5% на сделку с учетом комиссий)
        avg_loss_per_false_signal = -0.5
        
        # Прибыль от правильных сигналов
        profitable_buy_mask = df['buy_expected_return'] > 1.5
        profitable_sell_mask = df['sell_expected_return'] > 1.5
        
        # Берем случайную выборку прибыльных сделок (recall%)
        buy_indices = df[profitable_buy_mask].sample(n=min(model_found_buy, profitable_buy_mask.sum())).index
        sell_indices = df[profitable_sell_mask].sample(n=min(model_found_sell, profitable_sell_mask.sum())).index
        
        profit_from_buy = df.loc[buy_indices, 'buy_expected_return'].sum()
        profit_from_sell = df.loc[sell_indices, 'sell_expected_return'].sum()
        
        # Убытки от ложных сигналов
        loss_from_false_buy = false_buy_signals * avg_loss_per_false_signal
        loss_from_false_sell = false_sell_signals * avg_loss_per_false_signal
        
        # Итоговая прибыль
        total_profit = profit_from_buy + profit_from_sell + loss_from_false_buy + loss_from_false_sell
        
        logger.info(f"\n🤖 ТОРГОВЛЯ С МОДЕЛЬЮ (precision={precision:.0%}, recall={recall:.0%}):")
        logger.info(f"   Buy сигналов всего: {total_buy_signals:,}")
        logger.info(f"   - Прибыльных: {model_found_buy:,}")
        logger.info(f"   - Ложных: {false_buy_signals:,}")
        logger.info(f"   Sell сигналов всего: {total_sell_signals:,}")
        logger.info(f"   - Прибыльных: {model_found_sell:,}")
        logger.info(f"   - Ложных: {false_sell_signals:,}")
        logger.info(f"\n   Прибыль от Buy: {profit_from_buy:.0f}%")
        logger.info(f"   Прибыль от Sell: {profit_from_sell:.0f}%")
        logger.info(f"   Убытки от ложных Buy: {loss_from_false_buy:.0f}%")
        logger.info(f"   Убытки от ложных Sell: {loss_from_false_sell:.0f}%")
        logger.info(f"   ИТОГО прибыль модели: {total_profit:.0f}%")
        
        return {
            'total_buy_signals': total_buy_signals,
            'total_sell_signals': total_sell_signals,
            'profitable_buy': model_found_buy,
            'profitable_sell': model_found_sell,
            'false_buy': false_buy_signals,
            'false_sell': false_sell_signals,
            'profit_from_buy': profit_from_buy,
            'profit_from_sell': profit_from_sell,
            'loss_from_false': loss_from_false_buy + loss_from_false_sell,
            'total_profit': total_profit
        }
        
    def analyze_threshold_impact(self, df):
        """Анализ влияния порога на прибыльность"""
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        results = []
        
        for threshold in thresholds:
            ideal = self.calculate_ideal_profit(df, threshold)
            
            # Примерная точность модели в зависимости от порога
            # Чем выше порог, тем выше точность
            estimated_precision = 0.25 + (threshold / 10)  # от 30% до 55%
            estimated_recall = 0.80 - (threshold / 20)     # от 77% до 65%
            
            model = self.simulate_model_trading(df, estimated_precision, estimated_recall)
            
            results.append({
                'threshold': threshold,
                'ideal_profit': ideal['total_profit'],
                'model_profit': model['total_profit'],
                'efficiency': model['total_profit'] / ideal['total_profit'] * 100 if ideal['total_profit'] > 0 else 0,
                'signal_count': model['total_buy_signals'] + model['total_sell_signals'],
                'precision': estimated_precision,
                'recall': estimated_recall
            })
            
        results_df = pd.DataFrame(results)
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График прибыли
        ax = axes[0, 0]
        ax.plot(results_df['threshold'], results_df['ideal_profit'], 'b-', label='Идеальная прибыль', linewidth=2)
        ax.plot(results_df['threshold'], results_df['model_profit'], 'r--', label='Прибыль модели', linewidth=2)
        ax.set_xlabel('Порог (%)')
        ax.set_ylabel('Прибыль (%)')
        ax.set_title('Прибыль vs Порог')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # График эффективности
        ax = axes[0, 1]
        ax.plot(results_df['threshold'], results_df['efficiency'], 'g-', linewidth=2)
        ax.set_xlabel('Порог (%)')
        ax.set_ylabel('Эффективность (%)')
        ax.set_title('Эффективность модели')
        ax.grid(True, alpha=0.3)
        
        # График количества сигналов
        ax = axes[1, 0]
        ax.bar(results_df['threshold'], results_df['signal_count'])
        ax.set_xlabel('Порог (%)')
        ax.set_ylabel('Количество сигналов')
        ax.set_title('Частота торговли')
        
        # График precision/recall
        ax = axes[1, 1]
        ax.plot(results_df['threshold'], results_df['precision'] * 100, 'b-', label='Precision', linewidth=2)
        ax.plot(results_df['threshold'], results_df['recall'] * 100, 'r-', label='Recall', linewidth=2)
        ax.set_xlabel('Порог (%)')
        ax.set_ylabel('Метрика (%)')
        ax.set_title('Precision vs Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_analysis.png', dpi=300)
        plt.close()
        
        logger.info(f"\n📊 АНАЛИЗ ПОРОГОВ:")
        logger.info(results_df.to_string(index=False))
        
        return results_df
        
    def analyze_real_performance(self, df, model_predictions=None):
        """Анализ реальной производительности с учетом комиссий и проскальзывания"""
        # Параметры
        commission = 0.1  # 0.1% комиссия за сделку
        slippage = 0.05   # 0.05% проскальзывание
        
        # Если нет реальных предсказаний модели, симулируем
        if model_predictions is None:
            # Симуляция на основе метрик
            buy_signals = (df['buy_expected_return'] > 1.5) & (np.random.random(len(df)) < 0.72)  # recall
            # Добавляем ложные сигналы
            false_buy = (df['buy_expected_return'] <= 1.5) & (np.random.random(len(df)) < 0.15)
            buy_signals = buy_signals | false_buy
            
            sell_signals = (df['sell_expected_return'] > 1.5) & (np.random.random(len(df)) < 0.72)
            false_sell = (df['sell_expected_return'] <= 1.5) & (np.random.random(len(df)) < 0.15)
            sell_signals = sell_signals | false_sell
        else:
            buy_signals = model_predictions['buy']
            sell_signals = model_predictions['sell']
            
        # Расчет прибыли с учетом реальных условий
        results = []
        capital = 100  # Начальный капитал
        
        for idx, row in df.iterrows():
            if buy_signals.get(idx, False):
                # Buy сигнал
                entry_price = row['close'] * (1 + slippage/100)
                exit_price = row['close'] * (1 + row['buy_expected_return']/100)
                profit_pct = ((exit_price - entry_price) / entry_price - commission/100) * 100
                capital *= (1 + profit_pct/100)
                
                results.append({
                    'timestamp': idx,
                    'type': 'buy',
                    'expected_return': row['buy_expected_return'],
                    'real_profit': profit_pct,
                    'capital': capital
                })
                
            elif sell_signals.get(idx, False):
                # Sell сигнал
                entry_price = row['close'] * (1 - slippage/100)
                exit_price = row['close'] * (1 - row['sell_expected_return']/100)
                profit_pct = ((entry_price - exit_price) / entry_price - commission/100) * 100
                capital *= (1 + profit_pct/100)
                
                results.append({
                    'timestamp': idx,
                    'type': 'sell',
                    'expected_return': row['sell_expected_return'],
                    'real_profit': profit_pct,
                    'capital': capital
                })
                
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Статистика
            total_trades = len(results_df)
            profitable_trades = (results_df['real_profit'] > 0).sum()
            win_rate = profitable_trades / total_trades * 100
            
            avg_win = results_df[results_df['real_profit'] > 0]['real_profit'].mean()
            avg_loss = results_df[results_df['real_profit'] <= 0]['real_profit'].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            final_capital = capital
            total_return = (final_capital - 100)
            
            logger.info(f"\n💰 РЕАЛЬНАЯ ПРОИЗВОДИТЕЛЬНОСТЬ:")
            logger.info(f"   Всего сделок: {total_trades}")
            logger.info(f"   Прибыльных: {profitable_trades} ({win_rate:.1f}%)")
            logger.info(f"   Средняя прибыль: {avg_win:.2f}%")
            logger.info(f"   Средний убыток: {avg_loss:.2f}%")
            logger.info(f"   Profit Factor: {profit_factor:.2f}")
            logger.info(f"   Итоговый капитал: {final_capital:.2f}")
            logger.info(f"   Общая доходность: {total_return:.1f}%")
            
            # График капитала
            plt.figure(figsize=(12, 6))
            plt.plot(results_df['timestamp'], results_df['capital'], linewidth=2)
            plt.axhline(y=100, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Время')
            plt.ylabel('Капитал')
            plt.title('Динамика капитала')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('capital_dynamics.png', dpi=300)
            plt.close()
            
        return results_df

def main():
    # Конфигурация БД
    db_config = {
        'host': 'localhost',
        'port': 5555,
        'database': 'crypto_trading',
        'user': 'ruslan',
        'password': ''
    }
    
    # Создаем анализатор
    analyzer = ProfitabilityAnalyzer(db_config)
    analyzer.connect()
    
    # Загружаем данные
    logger.info("📊 Загрузка данных BTCUSDT...")
    df = analyzer.load_btc_data()
    
    # Анализ идеальной прибыли
    logger.info("\n" + "="*60)
    ideal_stats = analyzer.calculate_ideal_profit(df, threshold=1.5)
    
    # Симуляция торговли с моделью
    logger.info("\n" + "="*60)
    model_stats = analyzer.simulate_model_trading(df, precision=0.36, recall=0.72)
    
    # Сравнение
    efficiency = model_stats['total_profit'] / ideal_stats['total_profit'] * 100 if ideal_stats['total_profit'] > 0 else 0
    logger.info(f"\n📈 СРАВНЕНИЕ:")
    logger.info(f"   Идеальная прибыль: {ideal_stats['total_profit']:.0f}%")
    logger.info(f"   Прибыль модели: {model_stats['total_profit']:.0f}%")
    logger.info(f"   Эффективность: {efficiency:.1f}%")
    
    # Анализ порогов
    logger.info("\n" + "="*60)
    threshold_analysis = analyzer.analyze_threshold_impact(df)
    
    # Реальная производительность
    logger.info("\n" + "="*60)
    real_performance = analyzer.analyze_real_performance(df)
    
    logger.info("\n✅ Анализ завершен!")
    logger.info("📊 Графики сохранены: threshold_analysis.png, capital_dynamics.png")

if __name__ == "__main__":
    main()