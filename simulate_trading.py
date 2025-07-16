#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Симуляция торговли для проверки expected returns и расчета реального PnL
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingSimulator:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.connection = None
        
        # Комиссии Bybit
        self.maker_fee = 0.0005  # 0.05%
        self.taker_fee = 0.00075  # 0.075%
        
    def connect(self):
        """Подключение к БД"""
        self.connection = psycopg2.connect(**self.db_config)
        logger.info("✅ Подключение к PostgreSQL установлено")
        
    def disconnect(self):
        """Отключение от БД"""
        if self.connection:
            self.connection.close()
            logger.info("📤 Подключение к PostgreSQL закрыто")
    
    def simulate_symbol(self, symbol: str, capital: float = 10000):
        """
        Симуляция торговли для одного символа
        
        Args:
            symbol: Торговый символ
            capital: Начальный капитал
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Симуляция торговли для {symbol}")
        logger.info(f"💰 Начальный капитал: ${capital:,.2f}")
        
        # Загружаем данные
        query = """
        SELECT 
            datetime,
            close,
            buy_expected_return,
            sell_expected_return,
            (technical_indicators->>'buy_expected_return')::float as json_buy_return,
            (technical_indicators->>'sell_expected_return')::float as json_sell_return
        FROM processed_market_data
        WHERE symbol = %s
        ORDER BY timestamp
        """
        
        with self.connection.cursor() as cur:
            cur.execute(query, (symbol,))
            data = cur.fetchall()
        
        if not data:
            logger.warning(f"⚠️ Нет данных для {symbol}")
            return None
            
        logger.info(f"📊 Загружено {len(data)} записей")
        
        # Инициализация
        trades = []
        balance = capital
        position = None
        
        # Стратегия: входим когда expected return > 1%
        min_expected_return = 1.0
        
        for i, row in enumerate(tqdm(data, desc="Симуляция")):
            datetime_val, close, buy_ret, sell_ret, json_buy, json_sell = row
            close = float(close)
            
            # Используем данные из колонок, если есть, иначе из JSON
            buy_return = buy_ret if buy_ret != 0 else (json_buy or 0)
            sell_return = sell_ret if sell_ret != 0 else (json_sell or 0)
            
            # Если есть позиция, проверяем выход
            if position:
                # Симулируем закрытие через 100 баров
                if i >= position['exit_bar']:
                    pnl_pct = 0
                    if position['type'] == 'BUY':
                        pnl_pct = ((close - position['entry_price']) / position['entry_price']) * 100
                    else:  # SELL
                        pnl_pct = ((position['entry_price'] - close) / position['entry_price']) * 100
                    
                    # Вычитаем комиссии (вход + выход)
                    total_commission = (self.taker_fee + self.taker_fee) * 100
                    net_pnl_pct = pnl_pct - total_commission
                    
                    # Обновляем баланс
                    pnl_amount = balance * (net_pnl_pct / 100)
                    balance += pnl_amount
                    
                    trades.append({
                        'symbol': symbol,
                        'type': position['type'],
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_time': datetime_val,
                        'exit_price': close,
                        'expected_return': position['expected_return'],
                        'actual_return': pnl_pct,
                        'net_return': net_pnl_pct,
                        'pnl_amount': pnl_amount,
                        'balance': balance
                    })
                    
                    position = None
            
            # Если нет позиции, проверяем вход
            if not position and i < len(data) - 100:
                if buy_return > min_expected_return:
                    position = {
                        'type': 'BUY',
                        'entry_time': datetime_val,
                        'entry_price': close,
                        'expected_return': buy_return,
                        'exit_bar': i + 100
                    }
                elif sell_return > min_expected_return:
                    position = {
                        'type': 'SELL',
                        'entry_time': datetime_val,
                        'entry_price': close,
                        'expected_return': sell_return,
                        'exit_bar': i + 100
                    }
        
        # Статистика
        if trades:
            df_trades = pd.DataFrame(trades)
            
            total_trades = len(trades)
            winning_trades = len(df_trades[df_trades['net_return'] > 0])
            losing_trades = len(df_trades[df_trades['net_return'] < 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = df_trades[df_trades['net_return'] > 0]['net_return'].mean() if winning_trades > 0 else 0
            avg_loss = df_trades[df_trades['net_return'] < 0]['net_return'].mean() if losing_trades > 0 else 0
            
            total_return = ((balance - capital) / capital) * 100
            
            logger.info(f"\n📊 РЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
            logger.info(f"{'='*60}")
            logger.info(f"📈 Всего сделок: {total_trades}")
            logger.info(f"✅ Прибыльных: {winning_trades} ({win_rate:.1f}%)")
            logger.info(f"❌ Убыточных: {losing_trades}")
            logger.info(f"💰 Средняя прибыль: {avg_win:.2f}%")
            logger.info(f"💸 Средний убыток: {avg_loss:.2f}%")
            logger.info(f"📊 Итоговый баланс: ${balance:,.2f}")
            logger.info(f"🎯 Общая доходность: {total_return:.2f}%")
            
            # Проверка точности expected returns
            df_trades['prediction_error'] = df_trades['expected_return'] - df_trades['actual_return']
            mae = df_trades['prediction_error'].abs().mean()
            
            logger.info(f"\n📏 ТОЧНОСТЬ ПРОГНОЗОВ:")
            logger.info(f"MAE: {mae:.2f}%")
            
            return {
                'symbol': symbol,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'final_balance': balance,
                'trades': df_trades
            }
        else:
            logger.warning(f"⚠️ Нет сделок для {symbol} с порогом {min_expected_return}%")
            return None
    
    def simulate_all_symbols(self):
        """Симуляция для всех символов"""
        # Получаем список символов
        query = "SELECT DISTINCT symbol FROM processed_market_data ORDER BY symbol"
        
        with self.connection.cursor() as cur:
            cur.execute(query)
            symbols = [row[0] for row in cur.fetchall()]
        
        logger.info(f"🚀 Начинаем симуляцию для {len(symbols)} символов")
        
        results = {}
        for symbol in symbols:
            result = self.simulate_symbol(symbol)
            if result:
                results[symbol] = result
        
        # Общая статистика
        if results:
            total_trades = sum(r['total_trades'] for r in results.values())
            avg_win_rate = np.mean([r['win_rate'] for r in results.values()])
            avg_return = np.mean([r['total_return'] for r in results.values()])
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 ОБЩАЯ СТАТИСТИКА:")
            logger.info(f"{'='*60}")
            logger.info(f"📈 Всего сделок: {total_trades}")
            logger.info(f"✅ Средний Win Rate: {avg_win_rate:.1f}%")
            logger.info(f"💰 Средняя доходность: {avg_return:.2f}%")
        
        return results


def main():
    """Основная функция"""
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    db_config.pop('password', None)  # Убираем пустой пароль
    
    simulator = TradingSimulator(db_config)
    
    try:
        simulator.connect()
        results = simulator.simulate_all_symbols()
        
        # Сохраняем результаты
        if results:
            import json
            with open('trading_simulation_results.json', 'w') as f:
                json.dump(
                    {k: {kk: vv for kk, vv in v.items() if kk != 'trades'} 
                     for k, v in results.items()},
                    f, indent=2
                )
            logger.info("\n✅ Результаты сохранены в trading_simulation_results.json")
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulator.disconnect()


if __name__ == "__main__":
    main()