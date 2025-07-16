#!/usr/bin/env python3
"""
Обработка данных только для 1000PEPEUSDT
"""

import yaml
from prepare_dataset import MarketDatasetPreparator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Инициализируем процессор с риск-профилем
    db_config = config['database']
    risk_profile = config['risk_profile']
    processor = MarketDatasetPreparator(db_config, risk_profile)
    
    # Обрабатываем только PEPE
    logger.info("🚀 Начинаем обработку 1000PEPEUSDT...")
    
    stats = processor.process_single_symbol('1000PEPEUSDT')
    
    if stats['success']:
        logger.info(f"\n✅ Обработка завершена успешно!")
        logger.info(f"📊 Статистика:")
        logger.info(f"   Всего записей: {stats['total_records']:,}")
        logger.info(f"   Индикаторов: {stats['indicators_count']}")
        logger.info(f"   🟢 BUY Win Rate: {stats['buy_win_rate']:.2f}%")
        logger.info(f"   🔴 SELL Win Rate: {stats['sell_win_rate']:.2f}%")
    else:
        logger.error(f"❌ Ошибка: {stats.get('error')}")

if __name__ == "__main__":
    main()