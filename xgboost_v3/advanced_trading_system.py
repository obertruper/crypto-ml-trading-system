#!/usr/bin/env python3
"""
Продвинутая система ML трейдинга с множественными целевыми переменными
и адаптивным обучением.

Решает проблему ROC-AUC 0.5 через:
1. Множественные временные горизонты (5мин, 15мин, 1ч, 4ч)
2. Адаптивные пороги на основе волатильности
3. Confidence-based предсказания
4. Ансамбль моделей с разными стратегиями
5. Учет рыночного режима (тренд/флет/волатильность)
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
from dataclasses import dataclass
from enum import Enum

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Рыночные режимы"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class PredictionStrategy(Enum):
    """Стратегии предсказания"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"


@dataclass
class TradingSignal:
    """Торговый сигнал"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 - 1.0
    strategy: PredictionStrategy
    target_price: float
    stop_loss: float
    take_profit: float
    expected_return: float
    risk_reward_ratio: float
    market_regime: MarketRegime


class AdvancedTargetGenerator:
    """
    Генератор продвинутых целевых переменных
    """
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        
    def generate_adaptive_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает адаптивные целевые переменные на основе:
        - Волатильности (ATR)
        - Рыночного режима
        - Времени дня/недели
        """
        result_df = df.copy()
        
        # 1. Рассчитываем ATR для адаптивных порогов
        result_df['atr_14'] = self._calculate_atr(result_df, 14)
        result_df['volatility_percentile'] = result_df['atr_14'].rolling(100).rank(pct=True)
        
        # 2. Определяем рыночный режим
        result_df['market_regime'] = self._classify_market_regime(result_df)
        
        # 3. Адаптивные пороги на основе волатильности
        result_df['adaptive_threshold'] = self._calculate_adaptive_threshold(result_df)
        
        # 4. Множественные целевые переменные
        horizons = [1, 4, 16, 64]  # 15мин, 1ч, 4ч, 16ч
        
        for horizon in horizons:
            horizon_name = self._get_horizon_name(horizon)
            
            # Будущая цена
            future_price = result_df['close'].shift(-horizon)
            price_change_pct = ((future_price - result_df['close']) / result_df['close']) * 100
            
            # Адаптивные метки
            threshold = result_df['adaptive_threshold']
            
            # Бинарные метки с адаптивным порогом
            result_df[f'buy_adaptive_{horizon_name}'] = price_change_pct > threshold
            result_df[f'sell_adaptive_{horizon_name}'] = price_change_pct < -threshold
            
            # Confidence метки (сильные сигналы)
            strong_threshold = threshold * 2
            result_df[f'buy_strong_{horizon_name}'] = price_change_pct > strong_threshold
            result_df[f'sell_strong_{horizon_name}'] = price_change_pct < -strong_threshold
            
            # Регрессионные таргеты (нормализованные)
            result_df[f'return_normalized_{horizon_name}'] = price_change_pct / threshold
            
            # Risk-adjusted returns
            max_drawdown = self._calculate_max_drawdown_forward(result_df, horizon)
            result_df[f'risk_adjusted_return_{horizon_name}'] = price_change_pct / (max_drawdown + 0.1)
            
        # 5. Мультистратегийные таргеты
        result_df = self._generate_strategy_targets(result_df)
        
        return result_df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Рассчитывает Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _classify_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Классифицирует рыночный режим"""
        # EMA для трендов
        ema_short = df['close'].ewm(span=20).mean()
        ema_long = df['close'].ewm(span=50).mean()
        
        # Волатильность
        volatility = df['close'].rolling(20).std()
        vol_ma = volatility.rolling(50).mean()
        
        # Классификация
        conditions = [
            (ema_short > ema_long * 1.005) & (volatility < vol_ma),  # Тренд вверх
            (ema_short < ema_long * 0.995) & (volatility < vol_ma),  # Тренд вниз
            (volatility > vol_ma * 1.5),  # Высокая волатильность
            (volatility < vol_ma * 0.5),  # Низкая волатильность
        ]
        
        choices = [
            MarketRegime.TRENDING_UP.value,
            MarketRegime.TRENDING_DOWN.value,
            MarketRegime.HIGH_VOLATILITY.value,
            MarketRegime.LOW_VOLATILITY.value
        ]
        
        return pd.Series(
            np.select(conditions, choices, default=MarketRegime.SIDEWAYS.value),
            index=df.index
        )
    
    def _calculate_adaptive_threshold(self, df: pd.DataFrame) -> pd.Series:
        """Рассчитывает адаптивный порог на основе волатильности"""
        base_threshold = 0.5  # Увеличиваем базовый порог до 0.5%
        
        # Масштабируем на основе ATR
        atr_factor = df['atr_14'] / df['close'] * 100  # ATR в процентах
        atr_factor = atr_factor.fillna(base_threshold)
        
        # Ограничиваем диапазон (увеличиваем минимум)
        adaptive_threshold = np.clip(atr_factor * 0.8, 0.3, 3.0)
        
        return adaptive_threshold
    
    def _get_horizon_name(self, horizon: int) -> str:
        """Преобразует номер горизонта в название"""
        names = {1: '15min', 4: '1hour', 16: '4hour', 64: '16hour'}
        return names.get(horizon, f'{horizon}bars')
    
    def _calculate_max_drawdown_forward(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """Рассчитывает максимальную просадку вперед"""
        try:
            # Сброс индекса для избежания проблем с типами
            df_reset = df.reset_index(drop=True)
            
            future_prices = []
            for i in range(1, min(horizon + 1, 25)):  # Максимум 25 баров
                future_prices.append(df_reset['close'].shift(-i))
            
            if not future_prices:
                return pd.Series(0.1, index=df.index)
            
            # Создаем DataFrame с одинаковыми индексами
            future_df = pd.concat(future_prices, axis=1)
            current_price = df_reset['close']
            
            # Убеждаемся, что индексы совпадают
            future_df.index = current_price.index
            
            # Максимальная просадка от текущей цены
            price_changes = []
            for col in future_df.columns:
                change = (future_df[col] - current_price) / current_price * 100
                price_changes.append(change)
            
            if price_changes:
                combined_changes = pd.concat(price_changes, axis=1)
                max_drawdown = combined_changes.min(axis=1).abs()
            else:
                max_drawdown = pd.Series(0.1, index=current_price.index)
            
            # Восстанавливаем оригинальный индекс
            max_drawdown.index = df.index
            return max_drawdown.fillna(0.1)
            
        except Exception as e:
            logger.warning(f"Ошибка в расчете max_drawdown: {e}, используем константу")
            return pd.Series(0.1, index=df.index)
    
    def _generate_strategy_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерирует таргеты для разных стратегий"""
        # Trend Following
        df['trend_signal'] = self._generate_trend_following_signal(df)
        
        # Mean Reversion
        df['reversion_signal'] = self._generate_mean_reversion_signal(df)
        
        # Breakout
        df['breakout_signal'] = self._generate_breakout_signal(df)
        
        # Momentum
        df['momentum_signal'] = self._generate_momentum_signal(df)
        
        return df
    
    def _generate_trend_following_signal(self, df: pd.DataFrame) -> pd.Series:
        """Сигналы для трендовой стратегии"""
        # EMA пересечения
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        
        return ((ema_fast > ema_slow) & (ema_fast.shift() <= ema_slow.shift())).astype(int)
    
    def _generate_mean_reversion_signal(self, df: pd.DataFrame) -> pd.Series:
        """Сигналы для стратегии возврата к среднему"""
        # RSI oversold/overbought
        rsi = self._calculate_rsi(df['close'], 14)
        
        return ((rsi < 30) | (rsi > 70)).astype(int)
    
    def _generate_breakout_signal(self, df: pd.DataFrame) -> pd.Series:
        """Сигналы для стратегии пробоя"""
        # Bollinger Bands пробой
        bb_upper = df['close'].rolling(20).mean() + df['close'].rolling(20).std() * 2
        bb_lower = df['close'].rolling(20).mean() - df['close'].rolling(20).std() * 2
        
        return ((df['close'] > bb_upper) | (df['close'] < bb_lower)).astype(int)
    
    def _generate_momentum_signal(self, df: pd.DataFrame) -> pd.Series:
        """Сигналы для моментум стратегии"""
        # Price momentum
        momentum = (df['close'] / df['close'].shift(10) - 1) * 100
        
        return (momentum.abs() > 2).astype(int)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Рассчитывает RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class MultiHorizonModel:
    """
    Модель с множественными временными горизонтами
    """
    
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
        
        self.models = {}  # Модели для разных горизонтов и стратегий
        self.target_generator = AdvancedTargetGenerator(self.db_config)
        
    def create_advanced_targets_table(self):
        """Создает таблицу для продвинутых целевых переменных"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        try:
            cur.execute("DROP TABLE IF EXISTS advanced_targets CASCADE")
            
            create_query = """
            CREATE TABLE advanced_targets (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                
                -- Базовые данные
                close_price DECIMAL(20, 8) NOT NULL,
                atr_14 DECIMAL(10, 6),
                volatility_percentile DECIMAL(5, 4),
                market_regime VARCHAR(20),
                adaptive_threshold DECIMAL(10, 6),
                
                -- Адаптивные бинарные таргеты
                buy_adaptive_15min BOOLEAN,
                sell_adaptive_15min BOOLEAN,
                buy_adaptive_1hour BOOLEAN,
                sell_adaptive_1hour BOOLEAN,
                buy_adaptive_4hour BOOLEAN,
                sell_adaptive_4hour BOOLEAN,
                buy_adaptive_16hour BOOLEAN,
                sell_adaptive_16hour BOOLEAN,
                
                -- Сильные сигналы (confidence)
                buy_strong_15min BOOLEAN,
                sell_strong_15min BOOLEAN,
                buy_strong_1hour BOOLEAN,
                sell_strong_1hour BOOLEAN,
                buy_strong_4hour BOOLEAN,
                sell_strong_4hour BOOLEAN,
                
                -- Нормализованные returns
                return_normalized_15min DECIMAL(10, 4),
                return_normalized_1hour DECIMAL(10, 4),
                return_normalized_4hour DECIMAL(10, 4),
                return_normalized_16hour DECIMAL(10, 4),
                
                -- Risk-adjusted returns
                risk_adjusted_return_15min DECIMAL(10, 4),
                risk_adjusted_return_1hour DECIMAL(10, 4),
                risk_adjusted_return_4hour DECIMAL(10, 4),
                
                -- Стратегические сигналы
                trend_signal SMALLINT,
                reversion_signal SMALLINT,
                breakout_signal SMALLINT,
                momentum_signal SMALLINT,
                
                -- Метаданные
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(timestamp, symbol)
            );
            
            -- Индексы
            CREATE INDEX idx_advanced_targets_timestamp ON advanced_targets(timestamp);
            CREATE INDEX idx_advanced_targets_symbol ON advanced_targets(symbol);
            CREATE INDEX idx_advanced_targets_regime ON advanced_targets(market_regime);
            CREATE INDEX idx_advanced_targets_buy_1h ON advanced_targets(buy_adaptive_1hour);
            CREATE INDEX idx_advanced_targets_strong_1h ON advanced_targets(buy_strong_1hour);
            """
            
            cur.execute(create_query)
            conn.commit()
            
            logger.info("✅ Таблица advanced_targets создана")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Ошибка при создании таблицы: {e}")
            raise
        finally:
            cur.close()
            conn.close()
    
    def generate_and_save_targets(self, symbols: List[str] = None, limit: int = None):
        """Генерирует и сохраняет продвинутые таргеты"""
        conn = psycopg2.connect(**self.db_config)
        
        try:
            # Загружаем сырые данные
            query = """
            SELECT timestamp, symbol, open, high, low, close, volume
            FROM raw_market_data
            WHERE 1=1
            """
            
            params = []
            if symbols:
                placeholders = ','.join(['%s'] * len(symbols))
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
            
            query += " ORDER BY symbol, timestamp"
            
            if limit:
                query += f" LIMIT {limit}"
            
            logger.info("Загрузка сырых данных...")
            df = pd.read_sql_query(query, conn, params=params)
            logger.info(f"Загружено {len(df)} записей")
            
            # Конвертируем timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Обрабатываем по символам
            all_results = []
            
            for symbol, symbol_df in df.groupby('symbol'):
                logger.info(f"Обработка {symbol}: {len(symbol_df)} записей")
                
                # Генерируем продвинутые таргеты
                targets_df = self.target_generator.generate_adaptive_targets(symbol_df)
                all_results.append(targets_df)
            
            # Объединяем результаты
            final_df = pd.concat(all_results, ignore_index=True)
            
            # Сохраняем в БД
            self._save_targets_to_db(final_df)
            
            logger.info("✅ Продвинутые таргеты созданы и сохранены")
            
        finally:
            conn.close()
    
    def _save_targets_to_db(self, df: pd.DataFrame, batch_size: int = 10000):
        """Сохраняет таргеты в БД"""
        from psycopg2.extras import execute_batch
        
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        try:
            # Подготавливаем данные
            columns = [
                'timestamp', 'symbol', 'close_price', 'atr_14', 'volatility_percentile',
                'market_regime', 'adaptive_threshold',
                'buy_adaptive_15min', 'sell_adaptive_15min',
                'buy_adaptive_1hour', 'sell_adaptive_1hour',
                'buy_adaptive_4hour', 'sell_adaptive_4hour',
                'buy_adaptive_16hour', 'sell_adaptive_16hour',
                'buy_strong_15min', 'sell_strong_15min',
                'buy_strong_1hour', 'sell_strong_1hour',
                'buy_strong_4hour', 'sell_strong_4hour',
                'return_normalized_15min', 'return_normalized_1hour',
                'return_normalized_4hour', 'return_normalized_16hour',
                'risk_adjusted_return_15min', 'risk_adjusted_return_1hour',
                'risk_adjusted_return_4hour',
                'trend_signal', 'reversion_signal', 'breakout_signal', 'momentum_signal'
            ]
            
            # Маппинг колонок DataFrame -> БД
            column_mapping = {
                'close_price': 'close'  # В БД нужна close_price, в DF есть close
            }
            
            data = []
            for _, row in df.iterrows():
                values = []
                for col in columns:
                    # Проверяем маппинг
                    df_col = column_mapping.get(col, col)
                    
                    if df_col in row.index:
                        value = row[df_col]
                        if pd.isna(value):
                            values.append(None)
                        elif isinstance(value, (bool, np.bool_)):
                            values.append(bool(value))
                        elif isinstance(value, (np.integer, np.floating)):
                            values.append(float(value))
                        else:
                            values.append(value)
                    else:
                        values.append(None)
                
                data.append(tuple(values))
            
            # Batch insert
            query = f"""
            INSERT INTO advanced_targets ({', '.join(columns)})
            VALUES ({', '.join(['%s'] * len(columns))})
            ON CONFLICT (timestamp, symbol) DO UPDATE SET
                {', '.join([f'{col} = EXCLUDED.{col}' for col in columns if col not in ['timestamp', 'symbol']])}
            """
            
            total_rows = len(data)
            for i in range(0, total_rows, batch_size):
                batch = data[i:i + batch_size]
                execute_batch(cur, query, batch, page_size=batch_size)
                
                if (i + len(batch)) % 50000 == 0:
                    logger.info(f"  Сохранено {i + len(batch)}/{total_rows} записей...")
                    conn.commit()
            
            conn.commit()
            logger.info(f"✅ Сохранено {total_rows} записей в advanced_targets")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Ошибка при сохранении: {e}")
            raise
        finally:
            cur.close()
            conn.close()


def main():
    """Главная функция для запуска продвинутой системы"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Продвинутая система ML трейдинга")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Символы для обработки')
    parser.add_argument('--limit', type=int, help='Лимит записей')
    parser.add_argument('--test', action='store_true', help='Тестовый режим')
    
    args = parser.parse_args()
    
    if args.test:
        symbols = ['BTCUSDT']
        limit = 50000
        logger.info("🧪 ТЕСТОВЫЙ РЕЖИМ")
    else:
        symbols = args.symbols
        limit = args.limit
    
    logger.info("""
    ╔══════════════════════════════════════════════════════╗
    ║          Продвинутая система ML трейдинга           ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Создаем систему
    system = MultiHorizonModel('config.yaml')
    
    # Создаем таблицу
    system.create_advanced_targets_table()
    
    # Генерируем таргеты
    system.generate_and_save_targets(symbols=symbols, limit=limit)
    
    logger.info("\n📝 Следующие шаги:")
    logger.info("1. Проверьте данные: SELECT * FROM advanced_targets LIMIT 10;")
    logger.info("2. Запустите обучение улучшенных моделей")
    logger.info("3. Проанализируйте performance по режимам рынка")


if __name__ == "__main__":
    main()