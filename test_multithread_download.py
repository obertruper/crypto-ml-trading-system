#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тестовый скрипт для проверки улучшенной многопоточной загрузки
"""

import yaml
import logging
from download_data import PostgreSQLManager, BybitDataDownloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Тест загрузки 3 символов в 3 потока"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    
    # Тестовые символы
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Инициализация с пулом соединений
    db_manager = PostgreSQLManager(db_config, max_connections=5)
    downloader = BybitDataDownloader(db_manager, market_type='futures')
    
    try:
        db_manager.connect()
        
        logger.info(f"🧪 ТЕСТ: Загрузка {len(test_symbols)} символов в 3 потока")
        logger.info(f"📊 Символы: {', '.join(test_symbols)}")
        
        # Загружаем данные за последние 7 дней для теста
        results = downloader.download_multiple_symbols(
            symbols=test_symbols,
            interval='15',
            days=7,
            max_workers=3
        )
        
        # Результаты
        success_count = sum(1 for r in results.values() if r.get('success', False))
        logger.info(f"\n✅ Тест завершен: {success_count}/{len(test_symbols)} символов загружены успешно")
        
        # Детали по каждому символу
        for symbol, result in results.items():
            if result.get('success'):
                stats = result.get('stats', {})
                logger.info(f"   {symbol}: {stats.get('newly_inserted', 0)} новых записей")
            else:
                logger.error(f"   {symbol}: {result.get('error', 'Unknown error')}")
                
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Тест прерван пользователем")
    except Exception as e:
        logger.error(f"❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db_manager.disconnect()
        logger.info("🔚 Тест завершен")


if __name__ == "__main__":
    main()