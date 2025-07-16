#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Подготовка датасета с реалистичными точками входа в рынок
Создает последовательные неперекрывающиеся сделки для LONG и SHORT позиций
"""

import pandas as pd
import numpy as np
import ta
import json
from datetime import datetime
import warnings
import time
import pickle
import os
import random

warnings.filterwarnings('ignore')
import psycopg2
from psycopg2.extras import execute_values, Json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """
    Менеджер для работы с PostgreSQL
    """

    def __init__(self, db_config: dict):
        self.db_config = db_config.copy()
        # Удаляем пустой пароль
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
        self.connection = None

    def connect(self):
        """Создает подключение к БД"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = True
            logger.info("✅ Подключение к PostgreSQL установлено")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
            raise

    def disconnect(self):
        """Закрывает подключение к БД"""
        if self.connection:
            self.connection.close()
            logger.info("📤 Подключение к PostgreSQL закрыто")

    def execute_query(self, query: str, params=None, fetch=False):
        """Выполняет SQL запрос"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса: {e}")
            raise

    def fetch_dataframe(self, query: str, params=None) -> pd.DataFrame:
        """
        Выполняет запрос и возвращает результат как DataFrame
        """
        try:
            return pd.read_sql_query(query, self.connection, params=params)
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса DataFrame: {e}")
            raise


class RealisticMarketDatasetPreparator:
    """
    Класс для подготовки датасета с реалистичными точками входа
    """

    def __init__(self, db_manager: PostgreSQLManager, risk_profile: dict):
        """
        Инициализация с риск-профилем

        Args:
            db_manager: Менеджер PostgreSQL
            risk_profile: Словарь с параметрами риска из config.yaml
        """
        self.db = db_manager
        self.risk_profile = risk_profile
        self.feature_columns = []
        
        # Параметры для генерации точек входа
        self.entry_probability = 0.02  # 2% вероятность входа
        self.min_bars_between_trades = 5  # Минимум баров между сделками
        self.bad_entry_probability = 0.15  # 15% входов будут "плохими"
        self.good_entry_probability = 0.15  # 15% входов будут "хорошими"
        # Остальные 70% - случайные

    def get_available_symbols(self) -> list:
        """
        Получает список доступных символов из БД
        """
        query = """
        SELECT DISTINCT symbol, COUNT(*) as record_count
        FROM raw_market_data 
        WHERE interval_minutes = 15 AND market_type = 'futures'
        GROUP BY symbol
        HAVING COUNT(*) >= 1000
        ORDER BY symbol
        """

        results = self.db.execute_query(query, fetch=True)
        symbols = [row[0] for row in results]

        logger.info(f"📊 Найдено {len(symbols)} символов с достаточным количеством данных")
        return symbols

    def load_raw_data(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """
        Загружает сырые данные для символа из PostgreSQL
        """
        query = """
        SELECT id, symbol, timestamp, datetime, open, high, low, close, volume, turnover
        FROM raw_market_data 
        WHERE symbol = %s AND interval_minutes = 15 AND market_type = 'futures'
        ORDER BY timestamp
        """

        if limit:
            query += f" LIMIT {limit}"

        df = self.db.fetch_dataframe(query, (symbol,))

        if len(df) == 0:
            logger.warning(f"⚠️ Нет данных для символа {symbol}")
            return pd.DataFrame()

        # Конвертируем типы данных
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"✅ Загружено {len(df)} записей для {symbol}")
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> dict:
        """
        Рассчитывает все технические индикаторы
        """
        if len(df) < 100:  # Минимум данных для индикаторов
            logger.warning("⚠️ Недостаточно данных для расчета индикаторов")
            return {}

        logger.info("📈 Расчет технических индикаторов...")

        indicators = {}

        try:
            # === БАЗОВЫЕ ИНДИКАТОРЫ ===
            # EMA 15
            indicators['ema_15'] = ta.trend.EMAIndicator(df['close'], window=15).ema_indicator().tolist()

            # ADX (Average Directional Index)
            adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            indicators['adx_val'] = adx_indicator.adx().tolist()
            indicators['adx_plus_di'] = adx_indicator.adx_pos().tolist()
            indicators['adx_minus_di'] = adx_indicator.adx_neg().tolist()

            # RSI (Relative Strength Index)
            indicators['rsi_val'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().tolist()

            # MACD
            macd_indicator = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            indicators['macd_val'] = macd_indicator.macd().tolist()
            indicators['macd_signal'] = macd_indicator.macd_signal().tolist()
            indicators['macd_hist'] = macd_indicator.macd_diff().tolist()

            # ATR (Average True Range)
            indicators['atr_val'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'],
                                                                   window=10).average_true_range().tolist()

            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            indicators['bb_upper'] = bb_indicator.bollinger_hband().tolist()
            indicators['bb_lower'] = bb_indicator.bollinger_lband().tolist()
            indicators['bb_basis'] = bb_indicator.bollinger_mavg().tolist()

            # Stochastic
            stoch_indicator = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
            indicators['stoch_k'] = stoch_indicator.stoch().tolist()
            indicators['stoch_d'] = stoch_indicator.stoch_signal().tolist()

            # CCI (Commodity Channel Index)
            indicators['cci_val'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci().tolist()

            # Williams %R
            indicators['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'],
                                                                      lbp=14).williams_r().tolist()

            # OBV (On Balance Volume)
            indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'],
                                                                   df['volume']).on_balance_volume().tolist()

            # CMF (Chaikin Money Flow)
            indicators['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'],
                                                                    window=20).chaikin_money_flow().tolist()

            # MFI (Money Flow Index)
            indicators['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'],
                                                       window=14).money_flow_index().tolist()

            # Ichimoku
            ichimoku_indicator = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26)
            indicators['ichimoku_conv'] = ichimoku_indicator.ichimoku_conversion_line().tolist()
            indicators['ichimoku_base'] = ichimoku_indicator.ichimoku_base_line().tolist()

            # Parabolic SAR
            indicators['sar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar().tolist()

            # Donchian Channel
            donchian_indicator = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=20)
            indicators['donchian_upper'] = donchian_indicator.donchian_channel_hband().tolist()
            indicators['donchian_lower'] = donchian_indicator.donchian_channel_lband().tolist()

            # ROC (Rate of Change)
            indicators['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc().tolist()

            # Aroon
            aroon_indicator = ta.trend.AroonIndicator(df['high'], df['low'], window=14)
            indicators['aroon_up'] = aroon_indicator.aroon_up().tolist()
            indicators['aroon_down'] = aroon_indicator.aroon_down().tolist()

            # DPO (Detrended Price Oscillator)
            indicators['dpo'] = ta.trend.DPOIndicator(df['close'], window=20).dpo().tolist()

            # Ultimate Oscillator
            indicators['ult_osc'] = ta.momentum.UltimateOscillator(
                df['high'], df['low'], df['close'],
                window1=7, window2=14, window3=28
            ).ultimate_oscillator().tolist()

            # Vortex Indicator
            vortex_indicator = ta.trend.VortexIndicator(df['high'], df['low'], df['close'], window=14)
            indicators['vortex_vip'] = vortex_indicator.vortex_indicator_pos().tolist()
            indicators['vortex_vin'] = vortex_indicator.vortex_indicator_neg().tolist()

            # === ПРОИЗВОДНЫЕ ИНДИКАТОРЫ ===
            # Создаем временные серии для расчетов
            macd_val = pd.Series(indicators['macd_val'])
            macd_signal = pd.Series(indicators['macd_signal'])
            adx_plus_di = pd.Series(indicators['adx_plus_di'])
            adx_minus_di = pd.Series(indicators['adx_minus_di'])
            bb_upper = pd.Series(indicators['bb_upper'])
            bb_lower = pd.Series(indicators['bb_lower'])
            rsi_val = pd.Series(indicators['rsi_val'])
            stoch_k = pd.Series(indicators['stoch_k'])
            stoch_d = pd.Series(indicators['stoch_d'])
            vortex_vip = pd.Series(indicators['vortex_vip'])
            vortex_vin = pd.Series(indicators['vortex_vin'])
            ichimoku_conv = pd.Series(indicators['ichimoku_conv'])
            ichimoku_base = pd.Series(indicators['ichimoku_base'])
            atr_val = pd.Series(indicators['atr_val'])

            # MACD сигнальное соотношение
            indicators['macd_signal_ratio'] = (macd_val / (macd_signal + 1e-8)).tolist()

            # ADX разности
            indicators['adx_diff'] = (adx_plus_di - adx_minus_di).tolist()

            # Bollinger позиция
            band_width = bb_upper - bb_lower
            indicators['bb_position'] = ((df['close'] - bb_lower) / (band_width + 1e-8)).tolist()

            # RSI отклонение от центра
            indicators['rsi_dist_from_mid'] = (np.abs(rsi_val - 50) / 50.0).tolist()

            # Stochastic конвергенция
            indicators['stoch_diff'] = (stoch_k - stoch_d).tolist()

            # Vortex соотношение
            indicators['vortex_ratio'] = (vortex_vip / (vortex_vin + 1e-8)).tolist()

            # Ichimoku облако
            indicators['ichimoku_diff'] = (ichimoku_conv - ichimoku_base).tolist()

            # ATR нормализованный
            indicators['atr_norm'] = (atr_val / (df['close'] + 1e-8)).tolist()

            # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
            indicators['hour'] = df['datetime'].dt.hour.tolist()
            indicators['day_of_week'] = df['datetime'].dt.dayofweek.tolist()
            indicators['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int).tolist()

            # === ЦЕНОВЫЕ ПАТТЕРНЫ ===
            indicators['price_change_1'] = df['close'].pct_change(1).tolist()
            indicators['price_change_4'] = df['close'].pct_change(4).tolist()  # 1 час
            indicators['price_change_16'] = df['close'].pct_change(16).tolist()  # 4 часа

            # Волатильность
            indicators['volatility_4'] = df['close'].rolling(4).std().tolist()
            indicators['volatility_16'] = df['close'].rolling(16).std().tolist()

            # Объемные индикаторы
            volume_sma = df['volume'].rolling(20).mean()
            indicators['volume_sma'] = volume_sma.tolist()
            indicators['volume_ratio'] = (df['volume'] / (volume_sma + 1e-8)).tolist()

            # Заменяем NaN и inf на None для JSON сериализации
            for key, values in indicators.items():
                indicators[key] = [None if pd.isna(x) or np.isinf(x) else float(x) for x in values]

            logger.info(f"✅ Рассчитано {len(indicators)} групп индикаторов")

        except Exception as e:
            logger.error(f"❌ Ошибка расчета индикаторов: {e}")
            raise

        return indicators

    def _determine_entry_quality(self, indicators_at_bar: dict, direction: str) -> tuple:
        """
        Определяет качество точки входа на основе индикаторов
        
        Returns:
            tuple: (entry_type, confidence)
        """
        # Извлекаем текущие значения индикаторов
        rsi = indicators_at_bar.get('rsi_val', 50)
        adx = indicators_at_bar.get('adx_val', 0)
        bb_position = indicators_at_bar.get('bb_position', 0.5)
        macd_hist = indicators_at_bar.get('macd_hist', 0)
        volume_ratio = indicators_at_bar.get('volume_ratio', 1.0)
        atr_norm = indicators_at_bar.get('atr_norm', 0.01)
        
        # Считаем "плохие" и "хорошие" сигналы
        bad_signals = 0
        good_signals = 0
        
        if direction == 'long':
            # Плохие сигналы для LONG
            if rsi and rsi > 75:  # Перекупленность
                bad_signals += 2
            if bb_position and bb_position > 0.9:  # У верхней границы
                bad_signals += 1
            if macd_hist and macd_hist < -0.5:  # Сильный медвежий momentum
                bad_signals += 1
            if atr_norm and atr_norm > 0.03:  # Высокая волатильность
                bad_signals += 1
                
            # Хорошие сигналы для LONG
            if rsi and 30 < rsi < 50:  # Умеренная перепроданность
                good_signals += 1
            if bb_position and bb_position < 0.3:  # У нижней границы
                good_signals += 1
            if macd_hist and macd_hist > 0.5:  # Бычий momentum
                good_signals += 2
            if adx and adx > 25:  # Сильный тренд
                good_signals += 1
            if volume_ratio and volume_ratio > 1.5:  # Повышенный объем
                good_signals += 1
                
        else:  # SHORT
            # Плохие сигналы для SHORT
            if rsi and rsi < 25:  # Перепроданность
                bad_signals += 2
            if bb_position and bb_position < 0.1:  # У нижней границы
                bad_signals += 1
            if macd_hist and macd_hist > 0.5:  # Сильный бычий momentum
                bad_signals += 1
            if atr_norm and atr_norm > 0.03:  # Высокая волатильность
                bad_signals += 1
                
            # Хорошие сигналы для SHORT
            if rsi and 50 < rsi < 70:  # Умеренная перекупленность
                good_signals += 1
            if bb_position and bb_position > 0.7:  # У верхней границы
                good_signals += 1
            if macd_hist and macd_hist < -0.5:  # Медвежий momentum
                good_signals += 2
            if adx and adx > 25:  # Сильный тренд
                good_signals += 1
            if volume_ratio and volume_ratio > 1.5:  # Повышенный объем
                good_signals += 1
        
        # Определяем тип входа и уверенность
        if bad_signals >= 3:
            entry_type = 'bad'
            confidence = 0.2 + (bad_signals - 3) * 0.1  # 0.2-0.5
        elif good_signals >= 3:
            entry_type = 'good'
            confidence = 0.7 + (good_signals - 3) * 0.05  # 0.7-0.95
        else:
            entry_type = 'random'
            confidence = 0.4 + random.uniform(-0.1, 0.1)  # 0.3-0.5
            
        return entry_type, min(max(confidence, 0.1), 0.95)

    def _calculate_enhanced_result(self, entry_price, future_bars, direction,
                                 sl_pct, tp_pct, partial_levels, protection):
        """
        Рассчитывает результат торговли с учетом частичных закрытий и защиты прибыли
        (Копия из оригинального prepare_dataset.py)
        """
        # Инициализация
        position_size = 1.0
        realized_pnl = 0.0
        executed_level_indices = []
        exit_reason = None
        exit_bar = None
        
        # Расчет уровней цен
        if direction == 'buy':
            sl_price = entry_price * sl_pct
            tp_price = entry_price * tp_pct
            current_sl = sl_price
        else:  # sell
            sl_price = entry_price * sl_pct
            tp_price = entry_price * tp_pct
            current_sl = sl_price
        
        # Проходим по будущим барам
        for bar_idx, bar in enumerate(future_bars):
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])
            
            # === РЕАЛИСТИЧНЫЙ ПОДХОД ===
            
            if direction == 'buy':
                # Открытие с гэпом вниз - сразу стоп
                if bar['open'] <= current_sl and position_size > 0:
                    loss_pct = ((bar['open'] - entry_price) / entry_price) * 100
                    remaining_loss = loss_pct * position_size
                    exit_reason = f"Stop Loss at open {bar['open']:.4f} (bar {bar_idx+1})"
                    exit_bar = bar_idx + 1
                    return {
                        'final_return': realized_pnl + remaining_loss,
                        'realized_pnl': realized_pnl,
                        'unrealized_pnl': remaining_loss,
                        'exit_reason': exit_reason,
                        'exit_bar': exit_bar,
                        'final_position_size': 0
                    }
                
                # Проверяем частичные TP
                tp_hit_in_this_bar = False
                
                for idx, level in enumerate(partial_levels):
                    level_price = entry_price * (1 + level['percent'] / 100)
                    
                    if high >= level_price and idx not in executed_level_indices and position_size > 0:
                        # Проверяем, что TP достигнут до SL
                        if low <= current_sl:
                            open_to_tp = abs(bar['open'] - level_price)
                            open_to_sl = abs(bar['open'] - current_sl)
                            if open_to_sl < open_to_tp:
                                continue
                        
                        # Закрываем часть позиции
                        close_ratio = min(level['close_ratio'], position_size)
                        execution_price = max(level_price, bar['open'])
                        actual_profit_pct = ((execution_price - entry_price) / entry_price) * 100
                        
                        profit_at_level = actual_profit_pct * close_ratio
                        realized_pnl += profit_at_level
                        position_size -= close_ratio
                        executed_level_indices.append(idx)
                        tp_hit_in_this_bar = True
                        
                        # Обновляем trailing stop
                        if level['percent'] >= protection['breakeven_percent']:
                            new_sl = entry_price * (1 + protection['breakeven_offset'] / 100)
                            current_sl = max(current_sl, new_sl)
                        
                        for lock in protection['lock_levels']:
                            if level['percent'] >= lock['trigger']:
                                new_sl = entry_price * (1 + lock['lock'] / 100)
                                current_sl = max(current_sl, new_sl)
                
                # Проверка стоп-лосса
                if low <= current_sl and position_size > 0 and not tp_hit_in_this_bar:
                    loss_pct = ((current_sl - entry_price) / entry_price) * 100
                    remaining_loss = loss_pct * position_size
                    exit_reason = f"Stop Loss at {current_sl:.4f} (bar {bar_idx+1})"
                    exit_bar = bar_idx + 1
                    return {
                        'final_return': realized_pnl + remaining_loss,
                        'realized_pnl': realized_pnl,
                        'unrealized_pnl': remaining_loss,
                        'exit_reason': exit_reason,
                        'exit_bar': exit_bar,
                        'final_position_size': 0
                    }
                
                # Проверка финального тейк-профита
                if high >= tp_price and position_size > 0:
                    final_profit_pct = ((tp_price - entry_price) / entry_price) * 100
                    remaining_profit = final_profit_pct * position_size
                    exit_reason = f"Take Profit at {tp_price:.4f} (bar {bar_idx+1})"
                    exit_bar = bar_idx + 1
                    return {
                        'final_return': realized_pnl + remaining_profit,
                        'realized_pnl': realized_pnl,
                        'unrealized_pnl': remaining_profit,
                        'exit_reason': exit_reason,
                        'exit_bar': exit_bar,
                        'final_position_size': 0
                    }
                    
            else:  # SELL позиция
                # Аналогичная логика для SHORT
                if bar['open'] >= current_sl and position_size > 0:
                    loss_pct = -((bar['open'] - entry_price) / entry_price) * 100
                    remaining_loss = loss_pct * position_size
                    exit_reason = f"Stop Loss at open {bar['open']:.4f} (bar {bar_idx+1})"
                    exit_bar = bar_idx + 1
                    return {
                        'final_return': realized_pnl + remaining_loss,
                        'realized_pnl': realized_pnl,
                        'unrealized_pnl': remaining_loss,
                        'exit_reason': exit_reason,
                        'exit_bar': exit_bar,
                        'final_position_size': 0
                    }
                
                tp_hit_in_this_bar = False
                
                for idx, level in enumerate(partial_levels):
                    level_price = entry_price * (1 - level['percent'] / 100)
                    
                    if low <= level_price and idx not in executed_level_indices and position_size > 0:
                        if high >= current_sl:
                            open_to_tp = abs(bar['open'] - level_price)
                            open_to_sl = abs(bar['open'] - current_sl)
                            if open_to_sl < open_to_tp:
                                continue
                        
                        close_ratio = min(level['close_ratio'], position_size)
                        execution_price = min(level_price, bar['open'])
                        actual_profit_pct = ((entry_price - execution_price) / entry_price) * 100
                        
                        profit_at_level = actual_profit_pct * close_ratio
                        realized_pnl += profit_at_level
                        position_size -= close_ratio
                        executed_level_indices.append(idx)
                        tp_hit_in_this_bar = True
                        
                        if level['percent'] >= protection['breakeven_percent']:
                            new_sl = entry_price * (1 - protection['breakeven_offset'] / 100)
                            current_sl = min(current_sl, new_sl)
                        
                        for lock in protection['lock_levels']:
                            if level['percent'] >= lock['trigger']:
                                new_sl = entry_price * (1 - lock['lock'] / 100)
                                current_sl = min(current_sl, new_sl)
                
                if high >= current_sl and position_size > 0 and not tp_hit_in_this_bar:
                    loss_pct = -((current_sl - entry_price) / entry_price) * 100
                    remaining_loss = loss_pct * position_size
                    exit_reason = f"Stop Loss at {current_sl:.4f} (bar {bar_idx+1})"
                    exit_bar = bar_idx + 1
                    return {
                        'final_return': realized_pnl + remaining_loss,
                        'realized_pnl': realized_pnl,
                        'unrealized_pnl': remaining_loss,
                        'exit_reason': exit_reason,
                        'exit_bar': exit_bar,
                        'final_position_size': 0
                    }
                
                if low <= tp_price and position_size > 0:
                    final_profit_pct = ((entry_price - tp_price) / entry_price) * 100
                    remaining_profit = final_profit_pct * position_size
                    exit_reason = f"Take Profit at {tp_price:.4f} (bar {bar_idx+1})"
                    exit_bar = bar_idx + 1
                    return {
                        'final_return': realized_pnl + remaining_profit,
                        'realized_pnl': realized_pnl,
                        'unrealized_pnl': remaining_profit,
                        'exit_reason': exit_reason,
                        'exit_bar': exit_bar,
                        'final_position_size': 0
                    }
        
        # Таймаут
        if position_size > 0 and future_bars:
            last_close = float(future_bars[-1]['close'])
            if direction == 'buy':
                unrealized_pct = ((last_close - entry_price) / entry_price) * 100
            else:
                unrealized_pct = ((entry_price - last_close) / entry_price) * 100
            
            variation = len(executed_level_indices) * 0.01
            unrealized_pnl = unrealized_pct * position_size + variation
            
            exit_reason = f"Timeout at bar {len(future_bars)} (price: {last_close:.4f})"
            exit_bar = len(future_bars)
            
            return {
                'final_return': realized_pnl + unrealized_pnl,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'exit_reason': exit_reason,
                'exit_bar': exit_bar,
                'final_position_size': position_size
            }
        
        # Нет данных
        return {
            'final_return': realized_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': 0.0,
            'exit_reason': 'No data',
            'exit_bar': 0,
            'final_position_size': position_size
        }

    def create_realistic_trades(self, df: pd.DataFrame, indicators: dict, symbol: str) -> dict:
        """
        Создает реалистичные последовательные сделки для LONG и SHORT
        """
        logger.info(f"🎯 Создание реалистичных точек входа для {symbol}...")
        
        # Риск-профиль
        buy_sl_pct = self.risk_profile.get('stop_loss_pct_buy', 0.989)
        buy_tp_pct = self.risk_profile.get('take_profit_pct_buy', 1.058)
        sell_sl_pct = self.risk_profile.get('stop_loss_pct_sell', 1.011)
        sell_tp_pct = self.risk_profile.get('take_profit_pct_sell', 0.942)
        
        partial_tp_levels = [
            {'percent': 1.2, 'close_ratio': 0.20},
            {'percent': 2.4, 'close_ratio': 0.30},
            {'percent': 3.5, 'close_ratio': 0.30}
        ]
        
        profit_protection = {
            'breakeven_percent': 1.2,
            'breakeven_offset': 0.3,
            'lock_levels': [
                {'trigger': 2.4, 'lock': 1.2},
                {'trigger': 3.5, 'lock': 2.4},
                {'trigger': 4.6, 'lock': 3.5}
            ]
        }
        
        total_bars = len(df)
        lookahead_limit = 100
        
        # Инициализация результатов
        results = {
            'is_long_entry': [False] * total_bars,
            'is_short_entry': [False] * total_bars,
            'long_entry_type': [None] * total_bars,
            'short_entry_type': [None] * total_bars,
            'long_entry_confidence': [None] * total_bars,
            'short_entry_confidence': [None] * total_bars,
            'buy_expected_return': [0.0] * total_bars,
            'sell_expected_return': [0.0] * total_bars,
            'buy_profit_target': [0] * total_bars,
            'buy_loss_target': [0] * total_bars,
            'sell_profit_target': [0] * total_bars,
            'sell_loss_target': [0] * total_bars
        }
        
        # Счетчики для статистики
        long_trades = []
        short_trades = []
        
        # Отслеживание текущих позиций
        current_long_position = None
        current_short_position = None
        
        # Статистика по типам входов
        entry_stats = {
            'long': {'good': 0, 'bad': 0, 'random': 0},
            'short': {'good': 0, 'bad': 0, 'random': 0}
        }
        
        # Проходим по барам
        for i in tqdm(range(total_bars - lookahead_limit), desc=f"Генерация сделок {symbol}"):
            
            # === LONG ПОЗИЦИИ ===
            if current_long_position is None:
                # Можем открыть новую позицию
                if random.random() < self.entry_probability:
                    # Определяем качество входа
                    indicators_at_bar = {key: values[i] for key, values in indicators.items() if i < len(values)}
                    entry_type, confidence = self._determine_entry_quality(indicators_at_bar, 'long')
                    
                    # Принудительно меняем тип для разнообразия
                    rand_val = random.random()
                    if rand_val < self.bad_entry_probability:
                        entry_type = 'bad'
                        confidence = random.uniform(0.1, 0.3)
                    elif rand_val < self.bad_entry_probability + self.good_entry_probability:
                        entry_type = 'good'
                        confidence = random.uniform(0.7, 0.9)
                    
                    # Открываем позицию
                    current_long_position = {
                        'entry_bar': i,
                        'entry_price': df.iloc[i]['close'],
                        'entry_type': entry_type,
                        'confidence': confidence
                    }
                    
                    results['is_long_entry'][i] = True
                    results['long_entry_type'][i] = entry_type
                    results['long_entry_confidence'][i] = confidence
                    entry_stats['long'][entry_type] += 1
                    
            else:
                # Позиция уже открыта, проверяем не закрылась ли
                entry_bar = current_long_position['entry_bar']
                bars_in_position = i - entry_bar
                
                if bars_in_position >= self.min_bars_between_trades:
                    # Достаточно времени прошло, можем рассчитать результат
                    
                    # Получаем будущие бары от точки входа
                    future_bars = []
                    for j in range(entry_bar + 1, min(entry_bar + lookahead_limit + 1, total_bars)):
                        future_bars.append({
                            'open': df.iloc[j]['open'],
                            'high': df.iloc[j]['high'],
                            'low': df.iloc[j]['low'],
                            'close': df.iloc[j]['close']
                        })
                    
                    # Рассчитываем результат
                    buy_result = self._calculate_enhanced_result(
                        current_long_position['entry_price'], 
                        future_bars, 
                        'buy',
                        buy_sl_pct, buy_tp_pct, partial_tp_levels, profit_protection
                    )
                    
                    # Сохраняем результат только для точки входа
                    results['buy_expected_return'][entry_bar] = buy_result['final_return']
                    
                    # Бинарные метки
                    if buy_result['final_return'] > 0.5:
                        results['buy_profit_target'][entry_bar] = 1
                    elif buy_result['final_return'] < -0.5:
                        results['buy_loss_target'][entry_bar] = 1
                    
                    # Определяем бар выхода
                    exit_bar = entry_bar + buy_result['exit_bar']
                    
                    # Сохраняем сделку
                    long_trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': exit_bar,
                        'entry_type': current_long_position['entry_type'],
                        'return': buy_result['final_return'],
                        'exit_reason': buy_result['exit_reason']
                    })
                    
                    # Если позиция закрылась, освобождаем
                    if exit_bar <= i:
                        current_long_position = None
            
            # === SHORT ПОЗИЦИИ (аналогично, но независимо) ===
            if current_short_position is None:
                if random.random() < self.entry_probability:
                    indicators_at_bar = {key: values[i] for key, values in indicators.items() if i < len(values)}
                    entry_type, confidence = self._determine_entry_quality(indicators_at_bar, 'short')
                    
                    # Принудительно меняем тип для разнообразия
                    rand_val = random.random()
                    if rand_val < self.bad_entry_probability:
                        entry_type = 'bad'
                        confidence = random.uniform(0.1, 0.3)
                    elif rand_val < self.bad_entry_probability + self.good_entry_probability:
                        entry_type = 'good'
                        confidence = random.uniform(0.7, 0.9)
                    
                    current_short_position = {
                        'entry_bar': i,
                        'entry_price': df.iloc[i]['close'],
                        'entry_type': entry_type,
                        'confidence': confidence
                    }
                    
                    results['is_short_entry'][i] = True
                    results['short_entry_type'][i] = entry_type
                    results['short_entry_confidence'][i] = confidence
                    entry_stats['short'][entry_type] += 1
                    
            else:
                entry_bar = current_short_position['entry_bar']
                bars_in_position = i - entry_bar
                
                if bars_in_position >= self.min_bars_between_trades:
                    future_bars = []
                    for j in range(entry_bar + 1, min(entry_bar + lookahead_limit + 1, total_bars)):
                        future_bars.append({
                            'open': df.iloc[j]['open'],
                            'high': df.iloc[j]['high'],
                            'low': df.iloc[j]['low'],
                            'close': df.iloc[j]['close']
                        })
                    
                    sell_result = self._calculate_enhanced_result(
                        current_short_position['entry_price'], 
                        future_bars, 
                        'sell',
                        sell_sl_pct, sell_tp_pct, partial_tp_levels, profit_protection
                    )
                    
                    results['sell_expected_return'][entry_bar] = sell_result['final_return']
                    
                    if sell_result['final_return'] > 0.5:
                        results['sell_profit_target'][entry_bar] = 1
                    elif sell_result['final_return'] < -0.5:
                        results['sell_loss_target'][entry_bar] = 1
                    
                    exit_bar = entry_bar + sell_result['exit_bar']
                    
                    short_trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': exit_bar,
                        'entry_type': current_short_position['entry_type'],
                        'return': sell_result['final_return'],
                        'exit_reason': sell_result['exit_reason']
                    })
                    
                    if exit_bar <= i:
                        current_short_position = None
        
        # Статистика
        logger.info(f"\n📊 Статистика сделок для {symbol}:")
        logger.info(f"   LONG сделок: {len(long_trades)}")
        logger.info(f"   SHORT сделок: {len(short_trades)}")
        
        # Анализ по типам входов
        for direction in ['long', 'short']:
            trades = long_trades if direction == 'long' else short_trades
            if trades:
                logger.info(f"\n   {direction.upper()} позиции:")
                for entry_type in ['good', 'bad', 'random']:
                    type_trades = [t for t in trades if t['entry_type'] == entry_type]
                    if type_trades:
                        avg_return = np.mean([t['return'] for t in type_trades])
                        win_rate = len([t for t in type_trades if t['return'] > 0]) / len(type_trades) * 100
                        logger.info(f"      {entry_type}: {len(type_trades)} сделок, "
                                  f"средний результат: {avg_return:.2f}%, win rate: {win_rate:.1f}%")
        
        # Процент баров с входами
        long_entry_pct = sum(results['is_long_entry']) / len(results['is_long_entry']) * 100
        short_entry_pct = sum(results['is_short_entry']) / len(results['is_short_entry']) * 100
        logger.info(f"\n   Процент баров с входами:")
        logger.info(f"      LONG: {long_entry_pct:.2f}%")
        logger.info(f"      SHORT: {short_entry_pct:.2f}%")
        
        return results

    def save_processed_data(self, symbol: str, df: pd.DataFrame, indicators: dict, trade_results: dict):
        """
        Сохраняет обработанные данные с реалистичными точками входа в PostgreSQL
        """
        logger.info(f"💾 Сохранение обработанных данных для {symbol}...")

        # Очищаем существующие данные для этого символа
        delete_query = "DELETE FROM processed_market_data WHERE symbol = %s"
        self.db.execute_query(delete_query, (symbol,))

        # Подготавливаем данные для вставки
        values_to_insert = []

        for i in range(len(df)):
            # Создаем JSON объект с индикаторами для этого бара
            bar_indicators = {}
            for indicator_name, values_list in indicators.items():
                if i < len(values_list):
                    bar_indicators[indicator_name] = values_list[i]
                else:
                    bar_indicators[indicator_name] = None
            
            # Добавляем expected returns в JSON
            bar_indicators['buy_expected_return'] = trade_results['buy_expected_return'][i]
            bar_indicators['sell_expected_return'] = trade_results['sell_expected_return'][i]

            # Подготавливаем запись
            record = (
                int(df.iloc[i]['id']),  # raw_data_id
                symbol,
                int(df.iloc[i]['timestamp']),
                df.iloc[i]['datetime'],
                float(df.iloc[i]['open']),
                float(df.iloc[i]['high']),
                float(df.iloc[i]['low']),
                float(df.iloc[i]['close']),
                float(df.iloc[i]['volume']),
                Json(bar_indicators),  # JSONB с индикаторами
                int(trade_results['buy_profit_target'][i]),
                int(trade_results['buy_loss_target'][i]),
                int(trade_results['sell_profit_target'][i]),
                int(trade_results['sell_loss_target'][i]),
                float(trade_results['buy_expected_return'][i]),
                float(trade_results['sell_expected_return'][i]),
                bool(trade_results['is_long_entry'][i]),
                bool(trade_results['is_short_entry'][i]),
                trade_results['long_entry_type'][i],
                trade_results['short_entry_type'][i],
                float(trade_results['long_entry_confidence'][i]) if trade_results['long_entry_confidence'][i] else None,
                float(trade_results['short_entry_confidence'][i]) if trade_results['short_entry_confidence'][i] else None
            )

            values_to_insert.append(record)

        # SQL запрос для вставки с новыми полями
        insert_query = """
        INSERT INTO processed_market_data 
        (raw_data_id, symbol, timestamp, datetime, open, high, low, close, volume, 
         technical_indicators, buy_profit_target, buy_loss_target, 
         sell_profit_target, sell_loss_target, buy_expected_return, sell_expected_return,
         is_long_entry, is_short_entry, long_entry_type, short_entry_type,
         long_entry_confidence, short_entry_confidence)
        VALUES %s
        """

        try:
            with self.db.connection.cursor() as cursor:
                execute_values(cursor, insert_query, values_to_insert, page_size=1000)

            logger.info(f"✅ Сохранено {len(values_to_insert)} записей для {symbol}")

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения данных для {symbol}: {e}")
            raise

    def process_single_symbol(self, symbol: str, limit: int = None) -> dict:
        """
        Обрабатывает один символ с реалистичными точками входа
        """
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Обработка данных для {symbol}")
        logger.info(f"{'=' * 50}")

        try:
            # 1. Загружаем сырые данные
            df = self.load_raw_data(symbol, limit)

            if len(df) == 0:
                return {'success': False, 'error': 'Нет данных'}

            # 2. Расчет технических индикаторов
            indicators = self.calculate_technical_indicators(df)

            if not indicators:
                return {'success': False, 'error': 'Ошибка расчета индикаторов'}

            # 3. Создание реалистичных точек входа
            trade_results = self.create_realistic_trades(df, indicators, symbol)

            # 4. Сохранение в БД
            self.save_processed_data(symbol, df, indicators, trade_results)

            # 5. Статистика
            long_entries = sum(trade_results['is_long_entry'])
            short_entries = sum(trade_results['is_short_entry'])
            
            stats = {
                'success': True,
                'symbol': symbol,
                'total_records': len(df),
                'indicators_count': len(indicators),
                'long_entries': long_entries,
                'short_entries': short_entries,
                'long_entry_rate': long_entries / len(df) * 100,
                'short_entry_rate': short_entries / len(df) * 100
            }

            return stats

        except Exception as e:
            logger.error(f"❌ Ошибка обработки {symbol}: {e}")
            return {'success': False, 'error': str(e)}

    def process_all_symbols(self, symbols: list = None, limit: int = None) -> dict:
        """
        Обрабатывает все символы
        """
        if symbols is None:
            symbols = self.get_available_symbols()

        logger.info(f"🚀 Начинаем обработку {len(symbols)} символов с реалистичными точками входа...")

        results = {}
        successful = 0
        failed = 0
        start_time = time.time()

        for idx, symbol in enumerate(symbols):
            logger.info(f"\n⏱️ Прогресс: {idx+1}/{len(symbols)} ({(idx+1)/len(symbols)*100:.1f}%)")
            
            result = self.process_single_symbol(symbol, limit)
            results[symbol] = result

            if result['success']:
                successful += 1
            else:
                failed += 1

        # Итоговая статистика
        total_time = time.time() - start_time
        logger.info(f"\n{'=' * 50}")
        logger.info(f"📊 ИТОГИ ОБРАБОТКИ")
        logger.info(f"{'=' * 50}")
        logger.info(f"✅ Успешно обработано: {successful}")
        logger.info(f"❌ Ошибок: {failed}")
        logger.info(f"⏱️ Общее время: {total_time/60:.1f} минут")

        return results


def main():
    """Основная функция для подготовки датасета с реалистичными точками входа"""

    # Загружаем конфигурацию
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    risk_profile = config['risk_profile']

    # Инициализируем менеджер БД
    db_manager = PostgreSQLManager(db_config)

    try:
        # Подключаемся к БД
        db_manager.connect()

        # Инициализируем препаратор датасета
        preparator = RealisticMarketDatasetPreparator(db_manager, risk_profile)

        # Проверяем аргументы командной строки
        import sys
        if len(sys.argv) > 1:
            symbol = sys.argv[1]
            logger.info(f"🎯 Обработка одного символа: {symbol}")
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
            results = {symbol: preparator.process_single_symbol(symbol, limit)}
        else:
            # Обрабатываем все символы
            logger.info("🚀 Обработка всех символов")
            results = preparator.process_all_symbols()

        # Показываем итоговую статистику
        if results:
            logger.info(f"\n🎉 ОБРАБОТКА ДАННЫХ ЗАВЕРШЕНА!")
            logger.info(f"💾 Данные сохранены в таблице processed_market_data")
            logger.info(f"🎯 Готово для обучения модели с реалистичными точками входа!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Обработка прервана пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()