#!/usr/bin/env python3
"""
Быстрая подготовка ограниченного датасета для тестирования
"""

import yaml
from prepare_dataset import PostgreSQLManager, MarketDatasetPreparator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем конфигурацию
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database']
risk_profile = config['risk_profile']

# Подключаемся к БД
db_manager = PostgreSQLManager(db_config)
db_manager.connect()

# Создаем препаратор
preparator = MarketDatasetPreparator(db_manager, risk_profile)

# Обрабатываем символы с ограничением
symbols_to_process = ['1000PEPEUSDT', '1INCHUSDT']
limit_per_symbol = 20000  # 20k записей на символ для быстрого теста

logger.info(f"🚀 Начинаем быструю подготовку датасета...")
logger.info(f"   Символы: {', '.join(symbols_to_process)}")
logger.info(f"   Лимит на символ: {limit_per_symbol:,}")

results = {}
for symbol in symbols_to_process:
    logger.info(f"\n📊 Обработка {symbol}...")
    result = preparator.process_single_symbol(symbol, limit=limit_per_symbol)
    results[symbol] = result
    
    if result['success']:
        logger.info(f"✅ {symbol}: обработано {result['total_records']} записей")
    else:
        logger.error(f"❌ {symbol}: {result['error']}")

# Сохраняем метаданные признаков
if any(r['success'] for r in results.values()):
    preparator.save_feature_columns_metadata()

# Итоговая статистика
stats = preparator.get_processing_statistics()
logger.info(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
for symbol, stat in stats.items():
    if stat['total_records'] > 0:
        logger.info(f"\n{symbol}:")
        logger.info(f"   Записей: {stat['total_records']:,}")
        logger.info(f"   BUY win rate: {stat['buy_win_rate']:.1f}%")
        logger.info(f"   SELL win rate: {stat['sell_win_rate']:.1f}%")

db_manager.disconnect()
logger.info(f"\n✅ Быстрая подготовка завершена!")