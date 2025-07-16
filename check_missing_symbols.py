#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Проверка недостающих символов
"""

from pybit.unified_trading import HTTP
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_missing_symbols():
    """Проверяет недостающие символы"""
    session = HTTP(testnet=False)
    
    # Недостающие символы
    missing_symbols = ['TAOUSDT', 'TRBUSDT', 'TRUMPUSDT', 'ZEREBROUSDT']
    
    logger.info("🔍 Проверка недостающих символов на Bybit\n")
    
    # Получаем список фьючерсов
    response = session.get_instruments_info(category="linear", limit=1000)
    
    if response['retCode'] == 0:
        futures_symbols = {inst['symbol'] for inst in response['result']['list'] 
                         if inst['status'] == 'Trading'}
        
        logger.info(f"📊 Всего активных фьючерсных контрактов: {len(futures_symbols)}\n")
        
        for symbol in missing_symbols:
            if symbol in futures_symbols:
                logger.info(f"✅ {symbol} - ДОСТУПЕН на фьючерсах!")
                
                # Получаем детали
                try:
                    ticker = session.get_tickers(category="linear", symbol=symbol)
                    if ticker['retCode'] == 0 and ticker['result']['list']:
                        data = ticker['result']['list'][0]
                        logger.info(f"   Цена: ${data.get('lastPrice')}")
                        logger.info(f"   Объем 24ч: ${float(data.get('volume24h', 0)):,.0f}\n")
                except:
                    pass
            else:
                logger.info(f"❌ {symbol} - НЕ доступен на фьючерсах")
                
                # Проверяем альтернативные названия
                base = symbol.replace('USDT', '')
                alternatives = [s for s in futures_symbols if base in s and 'USDT' in s]
                if alternatives:
                    logger.info(f"   💡 Альтернативы: {', '.join(alternatives)}\n")
                else:
                    logger.info("")
        
        # Проверяем спот для недоступных
        logger.info("\n📈 Проверка на споте:")
        spot_response = session.get_instruments_info(category="spot", limit=1000)
        
        if spot_response['retCode'] == 0:
            spot_symbols = {inst['symbol'] for inst in spot_response['result']['list']}
            
            for symbol in missing_symbols:
                if symbol not in futures_symbols and symbol in spot_symbols:
                    logger.info(f"  {symbol} - доступен только на СПОТЕ")

if __name__ == "__main__":
    check_missing_symbols()