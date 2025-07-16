#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Валидация и фильтрация символов для фьючерсного рынка
"""

import yaml
import logging
from pybit.unified_trading import HTTP
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_active_futures():
    """Получает список активных фьючерсных контрактов"""
    session = HTTP(testnet=False)
    
    try:
        response = session.get_instruments_info(category="linear", limit=1000)
        
        if response['retCode'] == 0:
            active_futures = []
            for instrument in response['result']['list']:
                if instrument['symbol'].endswith('USDT') and instrument['status'] == 'Trading':
                    active_futures.append(instrument['symbol'])
            logger.info(f"🔍 Получено {len(response['result']['list'])} инструментов из API")
            return set(active_futures)
        else:
            logger.error(f"Ошибка API: {response['retMsg']}")
            return set()
    except Exception as e:
        logger.error(f"Ошибка получения фьючерсов: {e}")
        return set()


def main():
    """Основная функция"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Наш список из 50 символов (без TESTUSDT)
    our_symbols = [s for s in config['data_download']['symbols'] if 'TEST' not in s]
    
    logger.info(f"📊 Проверка {len(our_symbols)} символов на фьючерсном рынке...")
    
    # Получаем список активных фьючерсов
    active_futures = get_active_futures()
    logger.info(f"✅ Найдено {len(active_futures)} активных фьючерсных контрактов")
    
    # Проверяем какие из наших символов доступны
    available = []
    not_available = []
    
    for symbol in our_symbols:
        if symbol in active_futures:
            available.append(symbol)
        else:
            not_available.append(symbol)
    
    # Результаты
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Доступно на фьючерсах: {len(available)}/{len(our_symbols)}")
    
    if available:
        logger.info(f"\n✅ Доступные фьючерсные символы ({len(available)}):")
        for i, symbol in enumerate(sorted(available), 1):
            logger.info(f"   {i}. {symbol}")
    
    if not_available:
        logger.info(f"\n❌ Недоступные на фьючерсах ({len(not_available)}):")
        for symbol in sorted(not_available):
            logger.info(f"   - {symbol}")
            
            # Предлагаем альтернативы
            base = symbol.replace('USDT', '')
            if base == '1000PEPE':
                alt = 'PEPEUSDT'
                if alt in active_futures:
                    logger.info(f"     💡 Альтернатива: {alt}")
    
    # Обновляем конфиг
    if available:
        config['data_download']['symbols'] = sorted(available)
        config['data_download']['futures_validated'] = True
        config['data_download']['validation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open('config_futures_validated.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"\n💾 Создан config_futures_validated.yaml с {len(available)} проверенными фьючерсными символами")
    
    # Статистика по типам
    logger.info(f"\n📊 Анализ недоступных символов:")
    meme_coins = ['PEPE', 'PNUT', 'POPCAT', 'WIF', 'FARTCOIN', 'GRIFFAIN', 'MELANIA']
    new_projects = ['ZEREBRO', 'TRUMP', 'TAO']
    
    for symbol in not_available:
        base = symbol.replace('USDT', '').replace('1000', '')
        if any(meme in base.upper() for meme in meme_coins):
            logger.info(f"   {symbol} - мем-коин (обычно только спот)")
        elif any(proj in base.upper() for proj in new_projects):
            logger.info(f"   {symbol} - новый проект")
        else:
            logger.info(f"   {symbol} - проверьте альтернативное название")
    
    logger.info(f"\n✅ Готово к загрузке: {len(available)} фьючерсных символов")


if __name__ == "__main__":
    main()