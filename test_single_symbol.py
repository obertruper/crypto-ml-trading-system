#!/usr/bin/env python3
"""
Тестовая обработка одного символа с ограниченным количеством данных
"""

import yaml
import psycopg2
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

# Обрабатываем один символ с ограничением
logger.info("🚀 Начинаем тестовую обработку...")
result = preparator.process_single_symbol('1000PEPEUSDT', limit=5000)

if result['success']:
    logger.info(f"✅ Успешно обработано!")
    logger.info(f"   Записей: {result['total_records']}")
    logger.info(f"   BUY win rate: {result['buy_win_rate']:.1f}%")
    logger.info(f"   SELL win rate: {result['sell_win_rate']:.1f}%")
else:
    logger.error(f"❌ Ошибка: {result['error']}")

db_manager.disconnect()