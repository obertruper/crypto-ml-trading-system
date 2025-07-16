#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Детальная проверка фьючерсных контрактов на Bybit
"""

from pybit.unified_trading import HTTP
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_futures_detailed():
    """Детальная проверка фьючерсов"""
    session = HTTP(testnet=False)
    
    logger.info("🔍 Детальная проверка фьючерсных контрактов Bybit\n")
    
    # Получаем ВСЕ инструменты
    try:
        # Проверяем linear futures (USDT perpetual)
        linear_response = session.get_instruments_info(
            category="linear",
            limit=1000  # Максимум записей
        )
        
        if linear_response['retCode'] == 0:
            instruments = linear_response['result']['list']
            logger.info(f"✅ Найдено {len(instruments)} линейных фьючерсных контрактов\n")
            
            # Ищем конкретные символы
            target_symbols = ['XRP', 'TRX', 'TON', 'UNI', 'TIA', 'TWT', 'WIF']
            
            for target in target_symbols:
                logger.info(f"\n🔍 Поиск {target}:")
                found_symbols = []
                
                for inst in instruments:
                    symbol = inst.get('symbol', '')
                    base_coin = inst.get('baseCoin', '')
                    quote_coin = inst.get('quoteCoin', '')
                    status = inst.get('status', '')
                    
                    # Проверяем разные варианты
                    if (target in symbol or 
                        target == base_coin or 
                        target in base_coin):
                        
                        found_symbols.append({
                            'symbol': symbol,
                            'base': base_coin,
                            'quote': quote_coin,
                            'status': status
                        })
                
                if found_symbols:
                    for fs in found_symbols:
                        logger.info(f"  ✅ {fs['symbol']} (База: {fs['base']}, Котировка: {fs['quote']}, Статус: {fs['status']})")
                else:
                    logger.info(f"  ❌ Не найдено")
            
            # Показываем все USDT фьючерсы для проверки
            logger.info("\n\n📊 Примеры доступных USDT фьючерсов:")
            usdt_futures = [inst['symbol'] for inst in instruments 
                          if inst.get('quoteCoin') == 'USDT' 
                          and inst.get('status') == 'Trading']
            
            # Показываем первые 20
            for i, symbol in enumerate(sorted(usdt_futures)[:20]):
                logger.info(f"  {i+1}. {symbol}")
            
            logger.info(f"\n  ... и еще {len(usdt_futures) - 20} контрактов")
            
            # Проверяем конкретно XRPUSDT разными способами
            logger.info("\n\n🎯 Прямая проверка XRPUSDT:")
            
            # Способ 1: Поиск в списке
            xrp_in_list = any(inst['symbol'] == 'XRPUSDT' for inst in instruments)
            logger.info(f"  Поиск в списке: {'✅ Найден' if xrp_in_list else '❌ Не найден'}")
            
            # Способ 2: Прямой запрос
            try:
                ticker_response = session.get_tickers(
                    category="linear",
                    symbol="XRPUSDT"
                )
                if ticker_response['retCode'] == 0:
                    logger.info(f"  Прямой запрос тикера: ✅ Существует")
                    ticker_data = ticker_response['result']['list'][0] if ticker_response['result']['list'] else None
                    if ticker_data:
                        logger.info(f"    Цена: {ticker_data.get('lastPrice')}")
                        logger.info(f"    Объем 24ч: {ticker_data.get('volume24h')}")
                else:
                    logger.info(f"  Прямой запрос тикера: ❌ Ошибка - {ticker_response['retMsg']}")
            except Exception as e:
                logger.info(f"  Прямой запрос тикера: ❌ Исключение - {str(e)}")
            
        else:
            logger.error(f"Ошибка получения данных: {linear_response['retMsg']}")
            
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_futures_detailed()