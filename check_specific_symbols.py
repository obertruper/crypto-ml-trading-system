#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Проверка конкретных символов на доступность в фьючерсах
"""

from pybit.unified_trading import HTTP
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_symbols_availability():
    """Проверяет доступность символов на разных рынках"""
    session = HTTP(testnet=False)
    
    # Список символов для проверки
    symbols_to_check = [
        'TIAUSDT', 'TONUSDT', 'TRXUSDT', 'TWTUSDT', 
        'UNIUSDT', 'WIFUSDT', 'XRPUSDT'
    ]
    
    logger.info("🔍 Проверка доступности популярных символов на Bybit\n")
    
    # Получаем список фьючерсов
    futures_response = session.get_instruments_info(category="linear")
    futures_symbols = set()
    if futures_response['retCode'] == 0:
        for inst in futures_response['result']['list']:
            futures_symbols.add(inst['symbol'])
    
    # Получаем список спотовых пар
    spot_response = session.get_instruments_info(category="spot")
    spot_symbols = set()
    if spot_response['retCode'] == 0:
        for inst in spot_response['result']['list']:
            spot_symbols.add(inst['symbol'])
    
    # Проверяем каждый символ
    for symbol in symbols_to_check:
        in_futures = symbol in futures_symbols
        in_spot = symbol in spot_symbols
        
        logger.info(f"{symbol}:")
        logger.info(f"  📈 Спот: {'✅ Доступен' if in_spot else '❌ Не доступен'}")
        logger.info(f"  🔮 Фьючерсы: {'✅ Доступен' if in_futures else '❌ Не доступен'}")
        
        # Ищем альтернативные названия для фьючерсов
        if not in_futures:
            base = symbol.replace('USDT', '')
            alternatives = []
            for f_symbol in futures_symbols:
                if base in f_symbol and 'USDT' in f_symbol:
                    alternatives.append(f_symbol)
            
            if alternatives:
                logger.info(f"  💡 Альтернативы на фьючерсах: {', '.join(alternatives)}")
        
        logger.info("")
    
    # Дополнительная информация
    logger.info("📊 Общая статистика:")
    logger.info(f"  - Всего фьючерсных контрактов: {len(futures_symbols)}")
    logger.info(f"  - Всего спотовых пар: {len(spot_symbols)}")
    
    # Проверяем XRP более детально (как популярную монету)
    logger.info("\n🔍 Детальная проверка XRP:")
    xrp_futures = [s for s in futures_symbols if 'XRP' in s]
    if xrp_futures:
        logger.info(f"  Фьючерсные контракты с XRP: {', '.join(sorted(xrp_futures))}")
    else:
        logger.info("  ❌ XRP не найден на фьючерсах")
    
    # Показываем примеры доступных фьючерсов
    logger.info("\n📈 Примеры популярных монет на фьючерсах:")
    popular = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'ADAUSDT']
    for symbol in popular:
        if symbol in futures_symbols:
            logger.info(f"  ✅ {symbol}")
    
    # Проверяем возможные вариации названий
    logger.info("\n🔍 Проверка вариаций названий:")
    for symbol in symbols_to_check:
        base = symbol.replace('USDT', '')
        # Проверяем с префиксом 1000
        variant_1000 = f"1000{symbol}"
        variant_10000 = f"10000{symbol}"
        
        if variant_1000 in futures_symbols:
            logger.info(f"  💡 {symbol} -> {variant_1000} (на фьючерсах)")
        elif variant_10000 in futures_symbols:
            logger.info(f"  💡 {symbol} -> {variant_10000} (на фьючерсах)")
    
    # Дебаг: покажем все фьючерсы с определенными токенами
    logger.info("\n🔍 Поиск всех вариаций на фьючерсах:")
    search_tokens = ['XRP', 'TRX', 'TON', 'UNI', 'TIA', 'TWT', 'WIF']
    
    for token in search_tokens:
        found = [s for s in futures_symbols if token in s and 'USDT' in s]
        if found:
            logger.info(f"  {token}: {', '.join(sorted(found))}")
        else:
            logger.info(f"  {token}: ❌ Не найдено")

if __name__ == "__main__":
    check_symbols_availability()