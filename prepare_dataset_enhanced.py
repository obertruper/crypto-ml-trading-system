#!/usr/bin/env python3
"""
Улучшенная версия подготовки датасета с учетом частичных закрытий и защиты прибыли
"""

import numpy as np
import pandas as pd
import psycopg2
import json
import yaml
import talib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDatasetPreparer:
    """Подготовка датасета с учетом реальной логики торгового бота"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Базовые параметры риска
        self.risk_profile = self.config['risk_profile']
        
        # Параметры улучшенной системы (из old/config.yaml)
        self.enhanced_config = {
            'partial_tp': {
                'enabled': True,
                'levels': [
                    {'percent': 1.2, 'close_ratio': 0.20},
                    {'percent': 2.4, 'close_ratio': 0.30},
                    {'percent': 3.5, 'close_ratio': 0.30}
                ]
            },
            'profit_protection': {
                'enabled': True,
                'breakeven_percent': 1.2,
                'breakeven_offset': 0.3,
                'lock_levels': [
                    {'trigger': 2.4, 'lock': 1.2},
                    {'trigger': 3.5, 'lock': 2.4},
                    {'trigger': 4.6, 'lock': 3.5}
                ]
            }
        }
        
        self.db_config = self.config['database'].copy()
        if not self.db_config.get('password'):
            self.db_config.pop('password', None)
    
    def calculate_enhanced_targets(self, entry_price: float, future_prices: np.array, 
                                 direction: str = 'buy') -> Dict[str, float]:
        """
        Рассчитывает результат торговли с учетом частичных закрытий и защиты прибыли
        
        Returns:
            Dict с метриками:
            - final_result: итоговый результат в процентах
            - max_profit: максимальная достигнутая прибыль
            - realized_profit: реализованная прибыль от частичных закрытий
            - remaining_position: оставшийся размер позиции
            - exit_reason: причина выхода (tp/sl/timeout/partial)
        """
        
        # Начальные параметры
        if direction == 'buy':
            sl_price = entry_price * self.risk_profile['stop_loss_pct_buy']
            tp_price = entry_price * self.risk_profile['take_profit_pct_buy']
        else:
            sl_price = entry_price * self.risk_profile['stop_loss_pct_sell'] 
            tp_price = entry_price * self.risk_profile['take_profit_pct_sell']
        
        # Инициализация переменных
        position_size = 1.0  # Начальный размер позиции
        realized_profit = 0.0  # Реализованная прибыль
        executed_levels = []  # Исполненные уровни частичного закрытия
        current_sl = sl_price  # Текущий стоп-лосс
        max_profit = 0.0
        
        # Проходим по будущим ценам
        for i, current_price in enumerate(future_prices):
            # Рассчитываем текущую прибыль
            if direction == 'buy':
                profit_percent = ((current_price - entry_price) / entry_price) * 100
                
                # Проверка стоп-лосса
                if current_price <= current_sl:
                    remaining_loss = profit_percent * position_size
                    total_result = realized_profit + remaining_loss
                    return {
                        'final_result': total_result,
                        'max_profit': max_profit,
                        'realized_profit': realized_profit,
                        'remaining_position': position_size,
                        'exit_reason': 'sl',
                        'exit_bar': i
                    }
                
                # Проверка тейк-профита
                if current_price >= tp_price:
                    remaining_profit = ((tp_price - entry_price) / entry_price) * 100 * position_size
                    total_result = realized_profit + remaining_profit
                    return {
                        'final_result': total_result,
                        'max_profit': max_profit,
                        'realized_profit': realized_profit,
                        'remaining_position': 0,
                        'exit_reason': 'tp',
                        'exit_bar': i
                    }
            else:  # sell
                profit_percent = ((entry_price - current_price) / entry_price) * 100
                
                # Проверка стоп-лосса
                if current_price >= current_sl:
                    remaining_loss = profit_percent * position_size
                    total_result = realized_profit + remaining_loss
                    return {
                        'final_result': total_result,
                        'max_profit': max_profit,
                        'realized_profit': realized_profit,
                        'remaining_position': position_size,
                        'exit_reason': 'sl',
                        'exit_bar': i
                    }
                
                # Проверка тейк-профита
                if current_price <= tp_price:
                    remaining_profit = ((entry_price - tp_price) / entry_price) * 100 * position_size
                    total_result = realized_profit + remaining_profit
                    return {
                        'final_result': total_result,
                        'max_profit': max_profit,
                        'realized_profit': realized_profit,
                        'remaining_position': 0,
                        'exit_reason': 'tp',
                        'exit_bar': i
                    }
            
            # Обновляем максимальную прибыль
            if profit_percent > max_profit:
                max_profit = profit_percent
            
            # Проверка частичных закрытий
            if self.enhanced_config['partial_tp']['enabled'] and position_size > 0:
                for level in reversed(self.enhanced_config['partial_tp']['levels']):
                    if (profit_percent >= level['percent'] and 
                        level['percent'] not in executed_levels):
                        
                        # Закрываем часть позиции
                        close_amount = level['close_ratio']  # От изначального размера
                        
                        # Проверяем, не закроем ли больше чем есть
                        if close_amount > position_size:
                            close_amount = position_size
                        
                        # Реализуем прибыль
                        realized_profit += profit_percent * close_amount
                        position_size -= close_amount
                        executed_levels.append(level['percent'])
                        
                        # Обновляем стоп-лосс в безубыток
                        if direction == 'buy':
                            new_sl = entry_price * 1.001  # +0.1% от входа
                            current_sl = max(current_sl, new_sl)
                        else:
                            new_sl = entry_price * 0.999  # -0.1% от входа
                            current_sl = min(current_sl, new_sl)
                        
                        break  # Только один уровень за раз
            
            # Проверка защиты прибыли
            if self.enhanced_config['profit_protection']['enabled']:
                # Безубыток
                if (profit_percent >= self.enhanced_config['profit_protection']['breakeven_percent'] and
                    position_size > 0):
                    if direction == 'buy':
                        offset = self.enhanced_config['profit_protection']['breakeven_offset']
                        new_sl = entry_price * (1 + offset / 100)
                        current_sl = max(current_sl, new_sl)
                    else:
                        offset = self.enhanced_config['profit_protection']['breakeven_offset']
                        new_sl = entry_price * (1 - offset / 100)
                        current_sl = min(current_sl, new_sl)
                
                # Фиксация прибыли
                for lock_level in self.enhanced_config['profit_protection']['lock_levels']:
                    if profit_percent >= lock_level['trigger']:
                        if direction == 'buy':
                            new_sl = entry_price * (1 + lock_level['lock'] / 100)
                            current_sl = max(current_sl, new_sl)
                        else:
                            new_sl = entry_price * (1 - lock_level['lock'] / 100)
                            current_sl = min(current_sl, new_sl)
        
        # Если дошли до конца периода
        if position_size > 0:
            # Считаем текущую нереализованную прибыль
            last_price = future_prices[-1]
            if direction == 'buy':
                unrealized = ((last_price - entry_price) / entry_price) * 100 * position_size
            else:
                unrealized = ((entry_price - last_price) / entry_price) * 100 * position_size
            
            total_result = realized_profit + unrealized
        else:
            total_result = realized_profit
        
        return {
            'final_result': total_result,
            'max_profit': max_profit,
            'realized_profit': realized_profit,
            'remaining_position': position_size,
            'exit_reason': 'timeout' if position_size > 0 else 'partial_only',
            'exit_bar': len(future_prices) - 1
        }
    
    def prepare_enhanced_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка улучшенных меток с учетом реальной логики торговли"""
        
        logger.info("Подготовка улучшенных меток...")
        
        # Инициализация новых колонок
        df['buy_profit_enhanced'] = 0.0
        df['buy_max_profit'] = 0.0
        df['buy_realized_profit'] = 0.0
        df['buy_exit_reason'] = ''
        df['buy_exit_bar'] = 0
        df['buy_is_profitable'] = 0
        
        df['sell_profit_enhanced'] = 0.0
        df['sell_max_profit'] = 0.0
        df['sell_realized_profit'] = 0.0
        df['sell_exit_reason'] = ''
        df['sell_exit_bar'] = 0
        df['sell_is_profitable'] = 0
        
        symbols = df['symbol'].unique()
        
        for symbol in symbols:
            logger.info(f"Обработка {symbol}...")
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            
            for i in range(len(symbol_data) - self.config['model']['prediction_horizon']):
                entry_price = symbol_data.iloc[i]['close']
                future_prices = symbol_data.iloc[i+1:i+1+self.config['model']['prediction_horizon']]['close'].values
                
                # Рассчитываем результаты для BUY
                buy_result = self.calculate_enhanced_targets(entry_price, future_prices, 'buy')
                
                # Рассчитываем результаты для SELL
                sell_result = self.calculate_enhanced_targets(entry_price, future_prices, 'sell')
                
                # Сохраняем результаты
                idx = symbol_data.index[i]
                
                df.loc[idx, 'buy_profit_enhanced'] = buy_result['final_result']
                df.loc[idx, 'buy_max_profit'] = buy_result['max_profit']
                df.loc[idx, 'buy_realized_profit'] = buy_result['realized_profit']
                df.loc[idx, 'buy_exit_reason'] = buy_result['exit_reason']
                df.loc[idx, 'buy_exit_bar'] = buy_result['exit_bar']
                df.loc[idx, 'buy_is_profitable'] = 1 if buy_result['final_result'] > 0.5 else 0
                
                df.loc[idx, 'sell_profit_enhanced'] = sell_result['final_result']
                df.loc[idx, 'sell_max_profit'] = sell_result['max_profit']
                df.loc[idx, 'sell_realized_profit'] = sell_result['realized_profit']
                df.loc[idx, 'sell_exit_reason'] = sell_result['exit_reason']
                df.loc[idx, 'sell_exit_bar'] = sell_result['exit_bar']
                df.loc[idx, 'sell_is_profitable'] = 1 if sell_result['final_result'] > 0.5 else 0
        
        return df
    
    def analyze_enhanced_results(self, df: pd.DataFrame):
        """Анализ результатов с учетом улучшенной логики"""
        
        print("\n" + "="*80)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ С УЧЕТОМ ЧАСТИЧНЫХ ЗАКРЫТИЙ")
        print("="*80)
        
        # Общая статистика
        total = len(df)
        buy_profitable = df['buy_is_profitable'].sum()
        sell_profitable = df['sell_is_profitable'].sum()
        
        print(f"\n📊 Общая статистика:")
        print(f"Всего точек входа: {total:,}")
        print(f"Прибыльных BUY: {buy_profitable:,} ({buy_profitable/total*100:.1f}%)")
        print(f"Прибыльных SELL: {sell_profitable:,} ({sell_profitable/total*100:.1f}%)")
        
        # Средние показатели
        print(f"\n💰 Средние показатели прибыли:")
        print(f"BUY - средняя прибыль: {df['buy_profit_enhanced'].mean():.2f}%")
        print(f"BUY - средняя макс. прибыль: {df['buy_max_profit'].mean():.2f}%")
        print(f"BUY - средняя реализованная: {df['buy_realized_profit'].mean():.2f}%")
        
        print(f"\nSELL - средняя прибыль: {df['sell_profit_enhanced'].mean():.2f}%")
        print(f"SELL - средняя макс. прибыль: {df['sell_max_profit'].mean():.2f}%")
        print(f"SELL - средняя реализованная: {df['sell_realized_profit'].mean():.2f}%")
        
        # Причины выхода
        print(f"\n🎯 Распределение по причинам выхода (BUY):")
        buy_exits = df['buy_exit_reason'].value_counts()
        for reason, count in buy_exits.items():
            print(f"  {reason}: {count:,} ({count/total*100:.1f}%)")
        
        print(f"\n🎯 Распределение по причинам выхода (SELL):")
        sell_exits = df['sell_exit_reason'].value_counts()
        for reason, count in sell_exits.items():
            print(f"  {reason}: {count:,} ({count/total*100:.1f}%)")
        
        # Распределение прибыли
        print(f"\n📈 Распределение финальной прибыли:")
        
        profit_ranges = [
            (-100, -5, "Убыток > 5%"),
            (-5, -2, "Убыток 2-5%"),
            (-2, 0, "Убыток < 2%"),
            (0, 1, "Прибыль 0-1%"),
            (1, 2, "Прибыль 1-2%"),
            (2, 3, "Прибыль 2-3%"),
            (3, 5, "Прибыль 3-5%"),
            (5, 100, "Прибыль > 5%")
        ]
        
        print("\nBUY позиции:")
        for min_val, max_val, label in profit_ranges:
            count = df[(df['buy_profit_enhanced'] > min_val) & 
                      (df['buy_profit_enhanced'] <= max_val)].shape[0]
            print(f"  {label}: {count:,} ({count/total*100:.1f}%)")
        
        print("\nSELL позиции:")
        for min_val, max_val, label in profit_ranges:
            count = df[(df['sell_profit_enhanced'] > min_val) & 
                      (df['sell_profit_enhanced'] <= max_val)].shape[0]
            print(f"  {label}: {count:,} ({count/total*100:.1f}%)")
        
        # Влияние частичных закрытий
        partial_only_buy = df[df['buy_exit_reason'] == 'partial_only'].shape[0]
        partial_only_sell = df[df['sell_exit_reason'] == 'partial_only'].shape[0]
        
        print(f"\n🔄 Влияние частичных закрытий:")
        print(f"BUY - только частичные закрытия: {partial_only_buy:,} ({partial_only_buy/total*100:.1f}%)")
        print(f"SELL - только частичные закрытия: {partial_only_sell:,} ({partial_only_sell/total*100:.1f}%)")
        
        # Средняя реализованная прибыль для позиций с частичными закрытиями
        partial_buy = df[df['buy_realized_profit'] > 0]
        partial_sell = df[df['sell_realized_profit'] > 0]
        
        if len(partial_buy) > 0:
            print(f"\nBUY - средняя реализованная (где были частичные): {partial_buy['buy_realized_profit'].mean():.2f}%")
        if len(partial_sell) > 0:
            print(f"SELL - средняя реализованная (где были частичные): {partial_sell['sell_realized_profit'].mean():.2f}%")


def main():
    """Основная функция"""
    
    preparer = EnhancedDatasetPreparer()
    
    # Подключение к БД
    conn = psycopg2.connect(**preparer.db_config)
    
    # Загрузка данных
    query = """
    SELECT 
        r.symbol, r.timestamp, r.open, r.high, r.low, r.close, r.volume,
        p.datetime
    FROM raw_market_data r
    JOIN processed_market_data p ON r.id = p.raw_data_id
    WHERE r.market_type = 'futures'
    ORDER BY r.symbol, r.timestamp
    LIMIT 10000  -- Для теста
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Загружено {len(df)} записей")
    
    # Подготовка улучшенных меток
    df_enhanced = preparer.prepare_enhanced_labels(df)
    
    # Анализ результатов
    preparer.analyze_enhanced_results(df_enhanced)
    
    # Пример расчета для одной позиции
    print("\n" + "="*80)
    print("ПРИМЕР РАСЧЕТА ДЛЯ ОДНОЙ ПОЗИЦИИ")
    print("="*80)
    
    example = df_enhanced[df_enhanced['buy_realized_profit'] > 0].iloc[0]
    
    print(f"\nСимвол: {example['symbol']}")
    print(f"Цена входа: ${example['close']:.2f}")
    print(f"Направление: BUY")
    print(f"\nРезультаты:")
    print(f"- Финальная прибыль: {example['buy_profit_enhanced']:.2f}%")
    print(f"- Максимальная прибыль: {example['buy_max_profit']:.2f}%")
    print(f"- Реализованная прибыль: {example['buy_realized_profit']:.2f}%")
    print(f"- Причина выхода: {example['buy_exit_reason']}")
    print(f"- Выход через: {example['buy_exit_bar']} баров")


if __name__ == "__main__":
    main()