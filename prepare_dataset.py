#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Подготовка датасета для обучения модели в PostgreSQL
Создает признаки (features) и метки (labels) на основе вашего риск-профиля
Рассчитывает expected_return для ВСЕХ баров + случайные точки входа для статистики
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
    Менеджер для работы с PostgreSQL (тот же что в предыдущем скрипте)
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

        Args:
            query: SQL запрос
            params: Параметры запроса

        Returns:
            pd.DataFrame: Результат запроса
        """
        try:
            return pd.read_sql_query(query, self.connection, params=params)
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса DataFrame: {e}")
            raise


class MarketDatasetPreparator:
    """
    Класс для подготовки датасета из PostgreSQL
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

    def get_available_symbols(self) -> list:
        """
        Получает список доступных символов из БД

        Returns:
            list: Список символов
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
        for symbol, count in results:
            logger.info(f"   {symbol}: {count:,} записей")

        return symbols
    
    def get_unprocessed_symbols(self) -> list:
        """
        Получает список необработанных символов
        
        Returns:
            list: Список символов, которые еще не обработаны
        """
        # Получаем все доступные символы
        all_symbols = set(self.get_available_symbols())
        
        # Получаем уже обработанные символы
        query = """
        SELECT DISTINCT symbol 
        FROM processed_market_data
        """
        results = self.db.execute_query(query, fetch=True)
        processed_symbols = {row[0] for row in results} if results else set()
        
        # Находим необработанные
        unprocessed = all_symbols - processed_symbols
        
        logger.info(f"📊 Статистика обработки:")
        logger.info(f"   Всего символов: {len(all_symbols)}")
        logger.info(f"   ✅ Обработано: {len(processed_symbols)}")
        logger.info(f"   ⏳ Осталось обработать: {len(unprocessed)}")
        
        if unprocessed:
            logger.info(f"   Необработанные символы: {', '.join(sorted(unprocessed))}")
        
        return list(unprocessed)
    
    def verify_processed_data(self, symbol: str) -> bool:
        """
        Проверяет корректность обработанных данных для символа
        
        Args:
            symbol: Символ для проверки
            
        Returns:
            bool: True если данные корректны
        """
        query = """
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN buy_expected_return < -1.2 OR buy_expected_return > 6 THEN 1 END) as bad_buy,
            COUNT(CASE WHEN sell_expected_return < -1.2 OR sell_expected_return > 6 THEN 1 END) as bad_sell,
            MIN(buy_expected_return) as min_buy,
            MAX(buy_expected_return) as max_buy,
            MIN(sell_expected_return) as min_sell,
            MAX(sell_expected_return) as max_sell
        FROM processed_market_data
        WHERE symbol = %s
        """
        
        result = self.db.execute_query(query, (symbol,), fetch=True)
        
        if not result or result[0][0] == 0:
            return False
        
        total, bad_buy, bad_sell, min_buy, max_buy, min_sell, max_sell = result[0]
        
        # Проверяем корректность данных
        if bad_buy > 0 or bad_sell > 0:
            logger.warning(f"⚠️ {symbol}: найдены некорректные expected_returns")
            logger.warning(f"   Buy range: [{min_buy:.2f}%, {max_buy:.2f}%]")
            logger.warning(f"   Sell range: [{min_sell:.2f}%, {max_sell:.2f}%]")
            return False
        
        # Проверяем, что есть разнообразие в данных
        if max_buy == min_buy or max_sell == min_sell:
            logger.warning(f"⚠️ {symbol}: все expected_returns одинаковые")
            return False
        
        return True

    def load_raw_data(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """
        Загружает сырые данные для символа из PostgreSQL

        Args:
            symbol: Символ для загрузки
            limit: Ограничение количества записей (для тестирования)

        Returns:
            pd.DataFrame: Данные OHLCV
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

        Args:
            df: DataFrame с OHLCV данными

        Returns:
            dict: Словарь с рассчитанными индикаторами
        """

        if len(df) < 100:  # Минимум данных для индикаторов
            logger.warning("⚠️ Недостаточно данных для расчета индикаторов")
            return {}

        logger.info("📈 Расчет технических индикаторов...")

        indicators = {}

        try:
            # === БАЗОВЫЕ ИНДИКАТОРЫ ===

            # EMA 15 (как в Pine Script)
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

    def create_labels_based_on_risk_profile(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> tuple:
        """
        Создает метки (labels) на основе риск-профиля с учетом частичных закрытий
        Рассчитывает expected_return для КАЖДОГО бара + случайные точки входа для статистики

        Args:
            df: DataFrame с данными
            symbol: Название символа для логирования

        Returns:
            tuple: (buy_profit_targets, buy_loss_targets, sell_profit_targets, sell_loss_targets,
                   buy_expected_returns, sell_expected_returns, is_long_entry, is_short_entry)
        """

        logger.info(f"🎯 Создание меток для {symbol} - расчет для ВСЕХ баров...")

        # Базовый риск-профиль
        buy_sl_pct = self.risk_profile.get('stop_loss_pct_buy', 0.989)  # -1.1%
        buy_tp_pct = self.risk_profile.get('take_profit_pct_buy', 1.058)  # +5.8%
        sell_sl_pct = self.risk_profile.get('stop_loss_pct_sell', 1.011)  # +1.1%
        sell_tp_pct = self.risk_profile.get('take_profit_pct_sell', 0.942)  # -5.8%
        
        # Вероятность случайного входа
        random_entry_probability = 0.15  # 15% баров будут помечены как входы

        # Параметры частичных закрытий (из old/config.yaml)
        partial_tp_levels = [
            {'percent': 1.2, 'close_ratio': 0.20},
            {'percent': 2.4, 'close_ratio': 0.30},
            {'percent': 3.5, 'close_ratio': 0.30}
        ]
        
        # Параметры защиты прибыли
        profit_protection = {
            'breakeven_percent': 1.2,
            'breakeven_offset': 0.3,
            'lock_levels': [
                {'trigger': 2.4, 'lock': 1.2},
                {'trigger': 3.5, 'lock': 2.4},
                {'trigger': 4.6, 'lock': 3.5}
            ]
        }

        logger.info(f"📊 Риск-профиль: BUY SL={buy_sl_pct:.3f}, TP={buy_tp_pct:.3f}")
        logger.info(f"📊 Риск-профиль: SELL SL={sell_sl_pct:.3f}, TP={sell_tp_pct:.3f}")
        logger.info(f"🔄 Частичные закрытия: {', '.join([f'{l['percent']}%' for l in partial_tp_levels])}")

        total_bars = len(df)
        lookahead_limit = 100  # Максимум 100 баров вперед (25 часов для 15m)

        # Инициализируем массивы меток
        buy_profit_targets = [0] * total_bars
        buy_loss_targets = [0] * total_bars
        sell_profit_targets = [0] * total_bars
        sell_loss_targets = [0] * total_bars
        
        # Новые массивы для ожидаемых результатов
        buy_expected_returns = [0.0] * total_bars
        sell_expected_returns = [0.0] * total_bars
        
        # Массивы для случайных точек входа (для статистики)
        is_long_entry = [False] * total_bars
        is_short_entry = [False] * total_bars

        buy_profits = 0
        buy_losses = 0
        sell_profits = 0
        sell_losses = 0
        
        # Статистика по новому подходу
        total_buy_return = 0.0
        total_sell_return = 0.0
        
        # Статистика для случайных входов
        random_long_entries = 0
        random_short_entries = 0
        random_long_return = 0.0
        random_short_return = 0.0
        
        # Статистика по типам выходов
        exit_stats = {
            'buy': {
                'stop_loss': 0,           # Прямой SL без частичных
                'take_profit': 0,          # Прямой TP без частичных
                'timeout': 0,              # Таймаут без частичных
                'partial_then_sl': 0,      # Частичные, затем SL
                'partial_then_tp': 0,      # Частичные, затем TP
                'partial_then_timeout': 0, # Частичные, затем таймаут
                'total_with_partials': 0   # Всего с частичными
            },
            'sell': {
                'stop_loss': 0,           # Прямой SL без частичных
                'take_profit': 0,          # Прямой TP без частичных
                'timeout': 0,              # Таймаут без частичных
                'partial_then_sl': 0,      # Частичные, затем SL
                'partial_then_tp': 0,      # Частичные, затем TP
                'partial_then_timeout': 0, # Частичные, затем таймаут
                'total_with_partials': 0   # Всего с частичными
            }
        }

        logger.info(f"🔄 Анализ {total_bars} баров для создания меток...")

        # Проходим по каждому бару для создания меток
        for i in tqdm(range(total_bars - lookahead_limit), desc=f"Создание меток {symbol}"):
            entry_price = df.iloc[i]['close']
            
            # Получаем будущие бары с high/low/open
            future_bars = []
            for j in range(i + 1, min(i + lookahead_limit + 1, total_bars)):
                future_bars.append({
                    'open': df.iloc[j]['open'],
                    'high': df.iloc[j]['high'],
                    'low': df.iloc[j]['low'],
                    'close': df.iloc[j]['close']
                })
            
            # === РАСЧЕТ ДЛЯ BUY ПОЗИЦИИ ===
            buy_result = self._calculate_enhanced_result(
                entry_price, future_bars, 'buy',
                buy_sl_pct, buy_tp_pct, partial_tp_levels, profit_protection
            )
            
            # === РАСЧЕТ ДЛЯ SELL ПОЗИЦИИ ===
            sell_result = self._calculate_enhanced_result(
                entry_price, future_bars, 'sell',
                sell_sl_pct, sell_tp_pct, partial_tp_levels, profit_protection
            )
            
            # Сохраняем результаты для ВСЕХ баров
            buy_expected_returns[i] = buy_result['final_return']
            sell_expected_returns[i] = sell_result['final_return']
            
            # Случайно помечаем некоторые бары как точки входа (для статистики)
            if random.random() < random_entry_probability:
                is_long_entry[i] = True
                random_long_entries += 1
                random_long_return += buy_result['final_return']
                
            if random.random() < random_entry_probability:
                is_short_entry[i] = True
                random_short_entries += 1
                random_short_return += sell_result['final_return']
            
            # Обновляем статистику по типам выходов с учетом частичных закрытий
            # BUY
            if buy_result.get('had_partials', False):
                exit_stats['buy']['total_with_partials'] += 1
                if 'Stop Loss' in buy_result['exit_reason']:
                    exit_stats['buy']['partial_then_sl'] += 1
                elif 'Take Profit' in buy_result['exit_reason']:
                    exit_stats['buy']['partial_then_tp'] += 1
                elif 'Timeout' in buy_result['exit_reason']:
                    exit_stats['buy']['partial_then_timeout'] += 1
            else:
                if 'Stop Loss' in buy_result['exit_reason']:
                    exit_stats['buy']['stop_loss'] += 1
                elif 'Take Profit' in buy_result['exit_reason']:
                    exit_stats['buy']['take_profit'] += 1
                elif 'Timeout' in buy_result['exit_reason']:
                    exit_stats['buy']['timeout'] += 1
                    
            # SELL
            if sell_result.get('had_partials', False):
                exit_stats['sell']['total_with_partials'] += 1
                if 'Stop Loss' in sell_result['exit_reason']:
                    exit_stats['sell']['partial_then_sl'] += 1
                elif 'Take Profit' in sell_result['exit_reason']:
                    exit_stats['sell']['partial_then_tp'] += 1
                elif 'Timeout' in sell_result['exit_reason']:
                    exit_stats['sell']['partial_then_timeout'] += 1
            else:
                if 'Stop Loss' in sell_result['exit_reason']:
                    exit_stats['sell']['stop_loss'] += 1
                elif 'Take Profit' in sell_result['exit_reason']:
                    exit_stats['sell']['take_profit'] += 1
                elif 'Timeout' in sell_result['exit_reason']:
                    exit_stats['sell']['timeout'] += 1
            
            # Старые бинарные метки для совместимости
            if buy_result['final_return'] > 0.5:
                buy_profit_targets[i] = 1
                buy_profits += 1
            elif buy_result['final_return'] < -0.5:
                buy_loss_targets[i] = 1
                buy_losses += 1
                
            if sell_result['final_return'] > 0.5:
                sell_profit_targets[i] = 1
                sell_profits += 1
            elif sell_result['final_return'] < -0.5:
                sell_loss_targets[i] = 1
                sell_losses += 1
            
            total_buy_return += buy_result['final_return']
            total_sell_return += sell_result['final_return']

        # Статистика меток
        total_buy_signals = buy_profits + buy_losses
        total_sell_signals = sell_profits + sell_losses

        buy_win_rate = (buy_profits / total_buy_signals * 100) if total_buy_signals > 0 else 0
        sell_win_rate = (sell_profits / total_sell_signals * 100) if total_sell_signals > 0 else 0
        
        avg_buy_return = total_buy_return / (total_bars - lookahead_limit)
        avg_sell_return = total_sell_return / (total_bars - lookahead_limit)

        logger.info(f"📊 Статистика меток для {symbol}:")
        logger.info(f"   🟢 BUY: {buy_profits} прибыльных, {buy_losses} убыточных (Win Rate: {buy_win_rate:.1f}%)")
        logger.info(f"   🔴 SELL: {sell_profits} прибыльных, {sell_losses} убыточных (Win Rate: {sell_win_rate:.1f}%)")
        logger.info(f"📈 Средние ожидаемые результаты (все бары):")
        logger.info(f"   BUY: {avg_buy_return:.2f}%")
        logger.info(f"   SELL: {avg_sell_return:.2f}%")
        
        # Статистика случайных входов
        if random_long_entries > 0:
            avg_random_long = random_long_return / random_long_entries
            random_long_win_rate = sum(1 for i in range(total_bars - lookahead_limit) 
                                      if is_long_entry[i] and buy_expected_returns[i] > 0) / random_long_entries * 100
        else:
            avg_random_long = 0
            random_long_win_rate = 0
            
        if random_short_entries > 0:
            avg_random_short = random_short_return / random_short_entries
            random_short_win_rate = sum(1 for i in range(total_bars - lookahead_limit) 
                                       if is_short_entry[i] and sell_expected_returns[i] > 0) / random_short_entries * 100
        else:
            avg_random_short = 0
            random_short_win_rate = 0
            
        logger.info(f"\n🎲 Статистика СЛУЧАЙНЫХ входов:")
        logger.info(f"   LONG: {random_long_entries} входов ({random_long_entries/(total_bars-lookahead_limit)*100:.1f}%), "
                   f"средний return: {avg_random_long:.2f}%, win rate: {random_long_win_rate:.1f}%")
        logger.info(f"   SHORT: {random_short_entries} входов ({random_short_entries/(total_bars-lookahead_limit)*100:.1f}%), "
                   f"средний return: {avg_random_short:.2f}%, win rate: {random_short_win_rate:.1f}%")
        
        # Статистика по типам выходов
        logger.info(f"📊 Типы выходов из позиций:")
        total_buy_exits = total_bars - lookahead_limit
        total_sell_exits = total_bars - lookahead_limit
        
        if total_buy_exits > 0:
            logger.info(f"   🟢 BUY выходы:")
            logger.info(f"   БЕЗ частичных закрытий:")
            logger.info(f"      Stop Loss: {exit_stats['buy']['stop_loss']} ({exit_stats['buy']['stop_loss']/total_buy_exits*100:.1f}%)")
            logger.info(f"      Take Profit: {exit_stats['buy']['take_profit']} ({exit_stats['buy']['take_profit']/total_buy_exits*100:.1f}%)")
            logger.info(f"      Таймаут: {exit_stats['buy']['timeout']} ({exit_stats['buy']['timeout']/total_buy_exits*100:.1f}%)")
            
            logger.info(f"   С частичными закрытиями: {exit_stats['buy']['total_with_partials']} ({exit_stats['buy']['total_with_partials']/total_buy_exits*100:.1f}%)")
            if exit_stats['buy']['total_with_partials'] > 0:
                logger.info(f"      → Stop Loss: {exit_stats['buy']['partial_then_sl']} ({exit_stats['buy']['partial_then_sl']/total_buy_exits*100:.1f}%)")
                logger.info(f"      → Take Profit: {exit_stats['buy']['partial_then_tp']} ({exit_stats['buy']['partial_then_tp']/total_buy_exits*100:.1f}%)")
                logger.info(f"      → Таймаут: {exit_stats['buy']['partial_then_timeout']} ({exit_stats['buy']['partial_then_timeout']/total_buy_exits*100:.1f}%)")
            
        if total_sell_exits > 0:
            logger.info(f"   🔴 SELL выходы:")
            logger.info(f"   БЕЗ частичных закрытий:")
            logger.info(f"      Stop Loss: {exit_stats['sell']['stop_loss']} ({exit_stats['sell']['stop_loss']/total_sell_exits*100:.1f}%)")
            logger.info(f"      Take Profit: {exit_stats['sell']['take_profit']} ({exit_stats['sell']['take_profit']/total_sell_exits*100:.1f}%)")
            logger.info(f"      Таймаут: {exit_stats['sell']['timeout']} ({exit_stats['sell']['timeout']/total_sell_exits*100:.1f}%)")
            
            logger.info(f"   С частичными закрытиями: {exit_stats['sell']['total_with_partials']} ({exit_stats['sell']['total_with_partials']/total_sell_exits*100:.1f}%)")
            if exit_stats['sell']['total_with_partials'] > 0:
                logger.info(f"      → Stop Loss: {exit_stats['sell']['partial_then_sl']} ({exit_stats['sell']['partial_then_sl']/total_sell_exits*100:.1f}%)")
                logger.info(f"      → Take Profit: {exit_stats['sell']['partial_then_tp']} ({exit_stats['sell']['partial_then_tp']/total_sell_exits*100:.1f}%)")
                logger.info(f"      → Таймаут: {exit_stats['sell']['partial_then_timeout']} ({exit_stats['sell']['partial_then_timeout']/total_sell_exits*100:.1f}%)")

        return (buy_profit_targets, buy_loss_targets, sell_profit_targets, sell_loss_targets,
                buy_expected_returns, sell_expected_returns, is_long_entry, is_short_entry)
    
    def _calculate_enhanced_result(self, entry_price, future_bars, direction,
                                 sl_pct, tp_pct, partial_levels, protection):
        """
        Рассчитывает результат торговли с учетом частичных закрытий и защиты прибыли
        Использует консервативный подход: сначала проверяет стоп-лосс, затем профиты
        """
        # Инициализация
        position_size = 1.0
        realized_pnl = 0.0
        executed_level_indices = []  # Используем индексы вместо float значений
        exit_reason = None
        exit_bar = None
        
        # Расчет уровней цен
        if direction == 'buy':
            sl_price = entry_price * sl_pct  # sl_pct = 0.989 означает -1.1%
            tp_price = entry_price * tp_pct  # tp_pct = 1.058 означает +5.8%
            current_sl = sl_price
        else:  # sell
            sl_price = entry_price * sl_pct  # sl_pct = 1.011 означает +1.1% (для шорта это стоп)
            tp_price = entry_price * tp_pct  # tp_pct = 0.942 означает -5.8% (для шорта это профит)
            current_sl = sl_price
        
        # Проходим по будущим барам
        for bar_idx, bar in enumerate(future_bars):
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])
            
            # === РЕАЛИСТИЧНЫЙ ПОДХОД: учитываем последовательность событий ===
            
            if direction == 'buy':
                # Определяем, что произошло раньше - TP или SL
                # Если открытие ниже SL - сразу стоп
                if bar['open'] <= current_sl and position_size > 0:
                    # Открытие с гэпом вниз - сразу стоп
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
                        'final_position_size': 0,
                        'had_partials': len(executed_level_indices) > 0,
                        'partials_count': len(executed_level_indices)
                    }
                
                # Проверяем частичные TP перед проверкой SL
                # Это более реалистично, так как TP могут исполниться до достижения SL
                tp_hit_in_this_bar = False
                
                # 2. Проверка частичных тейк-профитов по уровням цен
                for idx, level in enumerate(partial_levels):
                    level_price = entry_price * (1 + level['percent'] / 100)
                    
                    if high >= level_price and idx not in executed_level_indices and position_size > 0:
                        # Проверяем, что TP достигнут до SL (если оба в одном баре)
                        if low <= current_sl:
                            # Оба уровня в одном баре - определяем что было раньше
                            # Простое правило: если открытие ближе к TP - сначала TP
                            open_to_tp = abs(bar['open'] - level_price)
                            open_to_sl = abs(bar['open'] - current_sl)
                            if open_to_sl < open_to_tp:
                                # SL был ближе - пропускаем TP
                                continue
                        
                        # Закрываем часть позиции по этому уровню
                        close_ratio = min(level['close_ratio'], position_size)
                        
                        # Рассчитываем точную цену исполнения (может быть выше level_price при гэпах)
                        execution_price = max(level_price, bar['open'])
                        actual_profit_pct = ((execution_price - entry_price) / entry_price) * 100
                        
                        profit_at_level = actual_profit_pct * close_ratio
                        realized_pnl += profit_at_level
                        position_size -= close_ratio
                        executed_level_indices.append(idx)
                        tp_hit_in_this_bar = True
                        
                        # Обновляем trailing stop после достижения уровня
                        if level['percent'] >= protection['breakeven_percent']:
                            # Переносим стоп в безубыток + небольшой профит
                            new_sl = entry_price * (1 + protection['breakeven_offset'] / 100)
                            current_sl = max(current_sl, new_sl)
                        
                        # Дополнительная защита прибыли на более высоких уровнях
                        for lock in protection['lock_levels']:
                            if level['percent'] >= lock['trigger']:
                                new_sl = entry_price * (1 + lock['lock'] / 100)
                                current_sl = max(current_sl, new_sl)
                
                # Проверка стоп-лосса ПОСЛЕ частичных TP
                if low <= current_sl and position_size > 0 and not tp_hit_in_this_bar:
                    # Стоп-лосс срабатывает только если не было TP в этом баре
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
                        'final_position_size': 0,
                        'had_partials': len(executed_level_indices) > 0,
                        'partials_count': len(executed_level_indices)
                    }
                
                # 3. Проверка финального тейк-профита
                if high >= tp_price and position_size > 0:
                    # Закрываем оставшуюся позицию по финальному TP
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
                        'final_position_size': 0,
                        'had_partials': len(executed_level_indices) > 0,
                        'partials_count': len(executed_level_indices)
                    }
                    
            else:  # SELL позиция
                # Определяем, что произошло раньше - TP или SL
                # Если открытие выше SL - сразу стоп
                if bar['open'] >= current_sl and position_size > 0:
                    # Открытие с гэпом вверх - сразу стоп
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
                        'final_position_size': 0,
                        'had_partials': len(executed_level_indices) > 0,
                        'partials_count': len(executed_level_indices)
                    }
                
                # Проверяем частичные TP перед проверкой SL
                tp_hit_in_this_bar = False
                
                # 2. Проверка частичных тейк-профитов (для шорта профит при падении цены)
                for idx, level in enumerate(partial_levels):
                    level_price = entry_price * (1 - level['percent'] / 100)
                    
                    if low <= level_price and idx not in executed_level_indices and position_size > 0:
                        # Проверяем, что TP достигнут до SL (если оба в одном баре)
                        if high >= current_sl:
                            # Оба уровня в одном баре - определяем что было раньше
                            open_to_tp = abs(bar['open'] - level_price)
                            open_to_sl = abs(bar['open'] - current_sl)
                            if open_to_sl < open_to_tp:
                                # SL был ближе - пропускаем TP
                                continue
                        
                        # Закрываем часть позиции по этому уровню
                        close_ratio = min(level['close_ratio'], position_size)
                        
                        # Рассчитываем точную цену исполнения (может быть ниже level_price при гэпах)
                        execution_price = min(level_price, bar['open'])
                        actual_profit_pct = ((entry_price - execution_price) / entry_price) * 100
                        
                        profit_at_level = actual_profit_pct * close_ratio
                        realized_pnl += profit_at_level
                        position_size -= close_ratio
                        executed_level_indices.append(idx)
                        tp_hit_in_this_bar = True
                        
                        # Обновляем trailing stop
                        if level['percent'] >= protection['breakeven_percent']:
                            new_sl = entry_price * (1 - protection['breakeven_offset'] / 100)
                            current_sl = min(current_sl, new_sl)
                        
                        for lock in protection['lock_levels']:
                            if level['percent'] >= lock['trigger']:
                                new_sl = entry_price * (1 - lock['lock'] / 100)
                                current_sl = min(current_sl, new_sl)
                
                # Проверка стоп-лосса ПОСЛЕ частичных TP
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
                        'final_position_size': 0,
                        'had_partials': len(executed_level_indices) > 0,
                        'partials_count': len(executed_level_indices)
                    }
                
                # 2. Проверка частичных тейк-профитов (для шорта профит при падении цены)
                for idx, level in enumerate(partial_levels):
                    level_price = entry_price * (1 - level['percent'] / 100)
                    
                    if low <= level_price and idx not in executed_level_indices and position_size > 0:
                        close_ratio = min(level['close_ratio'], position_size)
                        
                        # Рассчитываем точную цену исполнения (может быть ниже level_price при гэпах)
                        execution_price = min(level_price, bar['open'])
                        actual_profit_pct = ((entry_price - execution_price) / entry_price) * 100
                        
                        profit_at_level = actual_profit_pct * close_ratio
                        realized_pnl += profit_at_level
                        position_size -= close_ratio
                        executed_level_indices.append(idx)
                        
                        # Обновляем trailing stop
                        if level['percent'] >= protection['breakeven_percent']:
                            new_sl = entry_price * (1 - protection['breakeven_offset'] / 100)
                            current_sl = min(current_sl, new_sl)
                        
                        for lock in protection['lock_levels']:
                            if level['percent'] >= lock['trigger']:
                                new_sl = entry_price * (1 - lock['lock'] / 100)
                                current_sl = min(current_sl, new_sl)
                
                # 3. Проверка финального тейк-профита
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
                        'final_position_size': 0,
                        'had_partials': len(executed_level_indices) > 0,
                        'partials_count': len(executed_level_indices)
                    }
        
        # Если позиция не закрылась за отведенное время
        if position_size > 0 and future_bars:
            last_close = float(future_bars[-1]['close'])
            if direction == 'buy':
                unrealized_pct = ((last_close - entry_price) / entry_price) * 100
            else:
                unrealized_pct = ((entry_price - last_close) / entry_price) * 100
            
            # Добавляем небольшую вариацию для таймаута в зависимости от оставшейся позиции
            # и количества исполненных уровней
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
                'final_position_size': position_size,
                'had_partials': len(executed_level_indices) > 0,
                'partials_count': len(executed_level_indices)
            }
        
        # Если нет данных
        return {
            'final_return': realized_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': 0.0,
            'exit_reason': 'No data',
            'exit_bar': 0,
            'final_position_size': position_size,
            'had_partials': len(executed_level_indices) > 0,
            'partials_count': len(executed_level_indices)
        }

    def save_processed_data(self, symbol: str, df: pd.DataFrame, indicators: dict, labels: tuple):
        """
        Сохраняет обработанные данные в PostgreSQL

        Args:
            symbol: Символ
            df: Исходный DataFrame
            indicators: Технические индикаторы
            labels: Кортеж с метками (buy_profit, buy_loss, sell_profit, sell_loss, 
                    buy_expected_returns, sell_expected_returns, is_long_entry, is_short_entry)
        """

        logger.info(f"💾 Сохранение обработанных данных для {symbol}...")

        # Распаковываем метки
        if len(labels) == 8:
            (buy_profit_targets, buy_loss_targets, sell_profit_targets, sell_loss_targets,
             buy_expected_returns, sell_expected_returns, is_long_entry, is_short_entry) = labels
        elif len(labels) == 6:
            (buy_profit_targets, buy_loss_targets, sell_profit_targets, sell_loss_targets,
             buy_expected_returns, sell_expected_returns) = labels
            is_long_entry = [False] * len(df)
            is_short_entry = [False] * len(df)
        else:
            # Для обратной совместимости
            buy_profit_targets, buy_loss_targets, sell_profit_targets, sell_loss_targets = labels
            buy_expected_returns = [0.0] * len(df)
            sell_expected_returns = [0.0] * len(df)
            is_long_entry = [False] * len(df)
            is_short_entry = [False] * len(df)

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
            
            # ВАЖНО: НЕ добавляем expected_returns в technical_indicators!
            # Это приведет к утечке данных при обучении модели
            # Целевые переменные должны храниться ТОЛЬКО в отдельных колонках БД

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
                int(buy_profit_targets[i]),
                int(buy_loss_targets[i]),
                int(sell_profit_targets[i]),
                int(sell_loss_targets[i]),
                float(buy_expected_returns[i]),  # Добавляем expected returns
                float(sell_expected_returns[i]),
                bool(is_long_entry[i]),  # Флаг случайного LONG входа
                bool(is_short_entry[i]),  # Флаг случайного SHORT входа
                None,  # long_entry_type (не используется для случайных входов)
                None,  # short_entry_type (не используется для случайных входов)
                None,  # long_entry_confidence
                None   # short_entry_confidence
            )

            values_to_insert.append(record)

        # SQL запрос для вставки с expected returns и флагами случайных входов
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
        Обрабатывает один символ полностью

        Args:
            symbol: Название символа
            limit: Ограничение количества записей (для тестирования)

        Returns:
            dict: Статистика обработки
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

            # 3. Создание меток на основе риск-профиля
            labels = self.create_labels_based_on_risk_profile(df, symbol)

            # 4. Сохранение в БД
            self.save_processed_data(symbol, df, indicators, labels)

            # 5. Статистика
            buy_profit_count = sum(labels[0])
            buy_loss_count = sum(labels[1])
            sell_profit_count = sum(labels[2])
            sell_loss_count = sum(labels[3])

            stats = {
                'success': True,
                'symbol': symbol,
                'total_records': len(df),
                'indicators_count': len(indicators),
                'buy_profit_signals': buy_profit_count,
                'buy_loss_signals': buy_loss_count,
                'sell_profit_signals': sell_profit_count,
                'sell_loss_signals': sell_loss_count,
                'buy_win_rate': buy_profit_count / (buy_profit_count + buy_loss_count) * 100 if (
                                                                                                            buy_profit_count + buy_loss_count) > 0 else 0,
                'sell_win_rate': sell_profit_count / (sell_profit_count + sell_loss_count) * 100 if (
                                                                                                                sell_profit_count + sell_loss_count) > 0 else 0
            }

            return stats

        except Exception as e:
            logger.error(f"❌ Ошибка обработки {symbol}: {e}")
            return {'success': False, 'error': str(e)}

    def process_single_symbol_with_retry(self, symbol: str, limit: int = None, max_retries: int = 3) -> dict:
        """
        Обрабатывает символ с повторными попытками при разрыве соединения
        
        Args:
            symbol: Символ для обработки
            limit: Ограничение записей
            max_retries: Максимальное количество попыток
            
        Returns:
            dict: Результат обработки
        """
        for attempt in range(max_retries):
            try:
                return self.process_single_symbol(symbol, limit)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                logger.warning(f"⚠️ Разрыв соединения при обработке {symbol}, попытка {attempt + 1}/{max_retries}")
                logger.warning(f"   Ошибка: {e}")
                
                if attempt < max_retries - 1:
                    # Пытаемся переподключиться
                    try:
                        self.db.disconnect()
                        time.sleep(5)  # Пауза перед переподключением
                        self.db.connect()
                        logger.info("✅ Соединение восстановлено")
                    except Exception as conn_error:
                        logger.error(f"❌ Не удалось восстановить соединение: {conn_error}")
                else:
                    # Последняя попытка не удалась
                    return {'success': False, 'error': f'Не удалось обработать после {max_retries} попыток'}
            except Exception as e:
                # Другие ошибки не требуют повторных попыток
                logger.error(f"❌ Ошибка обработки {symbol}: {e}")
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Неизвестная ошибка'}

    def save_processing_checkpoint(self, processed_symbols: list, failed_symbols: list):
        """
        Сохраняет чекпоинт обработки для возможности продолжения
        
        Args:
            processed_symbols: Список успешно обработанных символов
            failed_symbols: Список символов с ошибками
        """
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'processed': processed_symbols,
            'failed': failed_symbols
        }
        
        checkpoint_file = 'prepare_dataset_checkpoint.pkl'
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            logger.info(f"💾 Чекпоинт сохранен: {len(processed_symbols)} обработано, {len(failed_symbols)} с ошибками")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения чекпоинта: {e}")
    
    def load_processing_checkpoint(self) -> dict:
        """
        Загружает чекпоинт обработки
        
        Returns:
            dict: Данные чекпоинта или пустой словарь
        """
        checkpoint_file = 'prepare_dataset_checkpoint.pkl'
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                logger.info(f"📂 Загружен чекпоинт от {checkpoint['timestamp']}")
                logger.info(f"   Обработано: {len(checkpoint['processed'])} символов")
                logger.info(f"   С ошибками: {len(checkpoint['failed'])} символов")
                return checkpoint
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки чекпоинта: {e}")
        return {'processed': [], 'failed': []}

    def process_all_symbols_with_resume(self, symbols: list = None, limit: int = None, use_checkpoint: bool = True) -> dict:
        """
        Обрабатывает все символы с возможностью продолжения после прерывания
        
        Args:
            symbols: Список символов для обработки (если None - только необработанные)
            limit: Ограничение записей для тестирования
            use_checkpoint: Использовать ли чекпоинт для продолжения
            
        Returns:
            dict: Результаты обработки всех символов
        """
        # Определяем символы для обработки
        if symbols is None:
            symbols = self.get_unprocessed_symbols()
            if not symbols:
                logger.info("✅ Все доступные символы уже обработаны!")
                return {}
        
        # Загружаем чекпоинт если нужно
        processed_symbols = []
        failed_symbols = []
        
        if use_checkpoint:
            checkpoint = self.load_processing_checkpoint()
            processed_symbols = checkpoint.get('processed', [])
            failed_symbols = checkpoint.get('failed', [])
            
            # Фильтруем символы, которые уже обработаны
            symbols_to_process = [s for s in symbols if s not in processed_symbols]
            
            if len(symbols_to_process) < len(symbols):
                logger.info(f"📋 Пропускаем {len(symbols) - len(symbols_to_process)} уже обработанных символов")
        else:
            symbols_to_process = symbols
        
        logger.info(f"🚀 Начинаем обработку {len(symbols_to_process)} символов...")
        
        results = {}
        successful = 0
        failed = 0
        start_time = time.time()
        
        # Обрабатываем каждый символ
        for idx, symbol in enumerate(symbols_to_process):
            # Показываем прогресс
            elapsed = time.time() - start_time
            if idx > 0:
                avg_time = elapsed / idx
                remaining = avg_time * (len(symbols_to_process) - idx)
                logger.info(f"\n⏱️ Прогресс: {idx}/{len(symbols_to_process)} " +
                          f"({idx/len(symbols_to_process)*100:.1f}%) " +
                          f"Осталось примерно: {remaining/60:.1f} мин")
            
            # Обрабатываем символ с повторными попытками
            result = self.process_single_symbol_with_retry(symbol, limit)
            results[symbol] = result
            
            if result['success']:
                successful += 1
                processed_symbols.append(symbol)
                
                # Проверяем корректность данных
                if self.verify_processed_data(symbol):
                    logger.info(f"✅ {symbol}: данные проверены и корректны")
                else:
                    logger.warning(f"⚠️ {symbol}: данные требуют проверки")
            else:
                failed += 1
                failed_symbols.append(symbol)
            
            # Сохраняем чекпоинт каждые 5 символов
            if (idx + 1) % 5 == 0:
                self.save_processing_checkpoint(processed_symbols, failed_symbols)
        
        # Финальное сохранение чекпоинта
        self.save_processing_checkpoint(processed_symbols, failed_symbols)
        
        # Сохраняем метаданные о признаках
        if successful > 0:
            self.save_feature_columns_metadata()
        
        # Итоговая статистика
        total_time = time.time() - start_time
        logger.info(f"\n{'=' * 50}")
        logger.info(f"📊 ИТОГИ ОБРАБОТКИ")
        logger.info(f"{'=' * 50}")
        logger.info(f"✅ Успешно обработано: {successful}")
        logger.info(f"❌ Ошибок: {failed}")
        logger.info(f"⏱️ Общее время: {total_time/60:.1f} минут")
        
        if successful > 0:
            logger.info(f"⚡ Среднее время на символ: {total_time/successful:.1f} сек")
        
        # Детальная статистика
        if successful > 0:
            total_records = sum(r.get('total_records', 0) for r in results.values() if r['success'])
            total_buy_profits = sum(r.get('buy_profit_signals', 0) for r in results.values() if r['success'])
            total_buy_losses = sum(r.get('buy_loss_signals', 0) for r in results.values() if r['success'])
            total_sell_profits = sum(r.get('sell_profit_signals', 0) for r in results.values() if r['success'])
            total_sell_losses = sum(r.get('sell_loss_signals', 0) for r in results.values() if r['success'])

            overall_buy_wr = total_buy_profits / (total_buy_profits + total_buy_losses) * 100 if (
                                                                                                             total_buy_profits + total_buy_losses) > 0 else 0
            overall_sell_wr = total_sell_profits / (total_sell_profits + total_sell_losses) * 100 if (
                                                                                                                 total_sell_profits + total_sell_losses) > 0 else 0

            logger.info(f"\n📊 Общая статистика:")
            logger.info(f"   📈 Всего записей обработано: {total_records:,}")
            logger.info(f"   🟢 BUY Win Rate: {overall_buy_wr:.1f}%")
            logger.info(f"   🔴 SELL Win Rate: {overall_sell_wr:.1f}%")
        
        if failed_symbols:
            logger.warning(f"\n⚠️ Символы с ошибками: {', '.join(failed_symbols)}")
        
        return results

    def process_all_symbols(self, symbols: list = None, limit: int = None) -> dict:
        """
        Обрабатывает все символы (устаревший метод, использует новый с резюме)

        Args:
            symbols: Список символов для обработки (если None - все доступные)
            limit: Ограничение записей для тестирования

        Returns:
            dict: Результаты обработки всех символов
        """
        return self.process_all_symbols_with_resume(symbols, limit, use_checkpoint=False)

    def save_feature_columns_metadata(self):
        """Сохраняет метаданные о признаках в БД"""

        # Получаем пример индикаторов из БД
        query = """
        SELECT technical_indicators 
        FROM processed_market_data 
        WHERE technical_indicators IS NOT NULL 
        LIMIT 1
        """

        result = self.db.execute_query(query, fetch=True)

        if result and result[0][0]:
            indicators_sample = result[0][0]
            feature_columns = list(indicators_sample.keys())

            # Сохраняем в таблицу метаданных
            metadata_query = """
            INSERT INTO model_metadata (model_name, model_type, version, feature_columns, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT DO NOTHING
            """

            self.db.execute_query(metadata_query, (
                'feature_extraction',
                'preprocessing',
                '1.0',
                Json(feature_columns)
            ))

            logger.info(f"💾 Сохранено {len(feature_columns)} признаков в метаданные")

    def get_processing_statistics(self) -> dict:
        """Возвращает статистику обработанных данных"""

        stats_query = """
        SELECT 
            symbol,
            COUNT(*) as total_records,
            SUM(buy_profit_target) as buy_profits,
            SUM(buy_loss_target) as buy_losses,
            SUM(sell_profit_target) as sell_profits,
            SUM(sell_loss_target) as sell_losses,
            MIN(datetime) as start_date,
            MAX(datetime) as end_date
        FROM processed_market_data 
        GROUP BY symbol
        ORDER BY symbol
        """

        results = self.db.execute_query(stats_query, fetch=True)

        stats = {}
        for row in results:
            symbol_stats = {
                'total_records': row[1],
                'buy_profits': row[2],
                'buy_losses': row[3],
                'sell_profits': row[4],
                'sell_losses': row[5],
                'start_date': row[6],
                'end_date': row[7],
                'buy_win_rate': (row[2] / (row[2] + row[3]) * 100) if (row[2] + row[3]) > 0 else 0,
                'sell_win_rate': (row[4] / (row[4] + row[5]) * 100) if (row[4] + row[5]) > 0 else 0
            }
            stats[row[0]] = symbol_stats

        return stats


def main():
    """Основная функция для подготовки датасета"""

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
        preparator = MarketDatasetPreparator(db_manager, risk_profile)

        # Проверяем аргументы командной строки
        import sys
        if len(sys.argv) > 1:
            if sys.argv[1] == '--resume':
                logger.info("🔄 Режим продолжения обработки")
                results = preparator.process_all_symbols_with_resume(use_checkpoint=True)
            elif sys.argv[1] == '--verify':
                logger.info("🔍 Режим проверки данных")
                # Проверяем все обработанные символы
                processed_query = "SELECT DISTINCT symbol FROM processed_market_data"
                processed = db_manager.execute_query(processed_query, fetch=True)
                
                for symbol, in processed:
                    if preparator.verify_processed_data(symbol):
                        logger.info(f"✅ {symbol}: данные корректны")
                    else:
                        logger.warning(f"❌ {symbol}: найдены проблемы в данных")
                return
            else:
                symbol = sys.argv[1]
                logger.info(f"🎯 Обработка одного символа: {symbol}")
                limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
                results = {symbol: preparator.process_single_symbol_with_retry(symbol, limit)}
        else:
            # Обрабатываем только необработанные символы
            logger.info("🚀 Обработка всех необработанных символов")
            results = preparator.process_all_symbols_with_resume(use_checkpoint=False)

        # Показываем итоговую статистику
        if results:
            stats = preparator.get_processing_statistics()
            logger.info(f"\n🎉 ОБРАБОТКА ДАННЫХ ЗАВЕРШЕНА!")
            logger.info(f"💾 Данные сохранены в таблице processed_market_data")
            logger.info(f"🎯 Готово для обучения модели!")
            
            # Рекомендации
            unprocessed = preparator.get_unprocessed_symbols()
            if unprocessed:
                logger.info(f"\n💡 Остались необработанные символы: {len(unprocessed)}")
                logger.info(f"   Запустите скрипт снова для их обработки")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Обработка прервана пользователем")
        logger.info("💡 Используйте 'python prepare_dataset.py --resume' для продолжения")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()