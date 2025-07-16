#!/usr/bin/env python3
"""
Инициализация простой системы целевых переменных.
Создает новую таблицу и заполняет её данными.
"""

import sys
import logging
import yaml
import argparse
from datetime import datetime

# Импортируем нашу систему
from data.simple_targets import create_simple_targets

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Инициализация простой системы целевых переменных"
    )
    
    parser.add_argument(
        '--symbols', 
        nargs='+',
        help='Список символов для обработки (по умолчанию все)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Лимит записей для обработки (для тестирования)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Минимальный порог движения цены в % (по умолчанию 0.1)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Тестовый режим - только 2 символа и 100k записей'
    )
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = {
        'host': config['database']['host'],
        'port': config['database']['port'],
        'database': config['database']['database'],
        'user': config['database']['user'],
        'password': config['database']['password']
    }
    
    logger.info("""
    ╔══════════════════════════════════════════════════════╗
    ║     Инициализация простой системы целевых меток     ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Определяем параметры
    if args.test:
        symbols = ['BTCUSDT', 'ETHUSDT']
        limit = 100000
        logger.info("🧪 ТЕСТОВЫЙ РЕЖИМ")
    else:
        symbols = args.symbols
        limit = args.limit
        
    logger.info(f"Параметры:")
    logger.info(f"  - Символы: {symbols if symbols else 'ВСЕ'}")
    logger.info(f"  - Лимит: {limit if limit else 'БЕЗ ЛИМИТА'}")
    logger.info(f"  - Порог движения: {args.threshold}%")
    
    start_time = datetime.now()
    
    try:
        # Создаем целевые переменные
        create_simple_targets(
            db_config=db_config,
            symbols=symbols,
            limit=limit,
            min_movement_threshold=args.threshold
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✅ Инициализация завершена за {elapsed:.1f} секунд")
        
        # Рекомендации
        logger.info("\n📝 Следующие шаги:")
        logger.info("1. Проверьте данные: SELECT * FROM simple_targets LIMIT 10;")
        logger.info("2. Запустите обучение: python train_direction_model.py")
        
        if args.test:
            logger.info("\n⚠️ Это был тестовый запуск! Для полной инициализации:")
            logger.info("   python init_simple_targets.py")
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()