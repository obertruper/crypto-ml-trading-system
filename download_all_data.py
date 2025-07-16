#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для загрузки данных всех криптовалют из config.yaml
"""

import yaml
import logging
from download_data import PostgreSQLManager, BybitDataDownloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Загрузка данных для всех символов с многопоточностью"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    data_config = config['data_download']
    
    # Инициализация с пулом соединений
    max_workers = data_config.get('max_workers', 5)
    db_manager = PostgreSQLManager(db_config, max_connections=max_workers + 2)
    market_type = data_config.get('market_type', 'futures')
    downloader = BybitDataDownloader(db_manager, market_type=market_type)
    
    try:
        db_manager.connect()
        
        # Получаем список символов и параметры
        symbols = data_config['symbols']
        # Исключаем тестовые символы
        symbols = [s for s in symbols if 'TEST' not in s]
        interval = data_config['interval']
        days = data_config['days']
        max_workers = data_config.get('max_workers', 5)
        
        logger.info(f"🚀 Начинаем многопоточную загрузку данных для {len(symbols)} символов")
        logger.info(f"📊 Параметры: интервал={interval}m, период={days} дней")
        logger.info(f"🔧 Потоков: {max_workers}")
        logger.info(f"📈 Тип рынка: {market_type.upper()}")
        
        # Загружаем данные для всех символов с многопоточностью
        results = downloader.download_multiple_symbols(symbols, interval, days, max_workers=max_workers)
        
        # Статистика
        success_count = sum(1 for r in results.values() if r.get('success', False))
        skipped_count = sum(1 for r in results.values() if r.get('success') and r.get('stats', {}).get('skipped', False))
        new_count = success_count - skipped_count
        
        logger.info(f"\n📊 ИТОГИ:")
        logger.info(f"✅ Успешно обработано: {success_count}/{len(symbols)} символов")
        logger.info(f"   📥 Новых загрузок: {new_count}")
        logger.info(f"   ⏭️ Пропущено (актуальные): {skipped_count}")
        
        # Показываем общую статистику
        stats, total_records = downloader.get_database_stats()
        
        logger.info("\n📊 СТАТИСТИКА БД:")
        for symbol, stat in sorted(stats.items()):
            logger.info(f"{symbol}: {stat['total_records']:,} записей, "
                       f"период: {stat['start_date']} - {stat['end_date']}")
        
        logger.info(f"\n🎯 ИТОГО: {total_records:,} записей для {len(stats)} символов")
        
        # Неудачные загрузки
        failed = [s for s, r in results.items() if not r.get('success', False)]
        if failed:
            logger.warning(f"\n⚠️ Не удалось загрузить ({len(failed)} символов):")
            for symbol in failed:
                error = results[symbol].get('error', 'Unknown error')
                logger.warning(f"   {symbol}: {error}")
                
        # Проверяем, была ли остановка
        if hasattr(downloader, 'shutdown_flag') and downloader.shutdown_flag.is_set():
            logger.warning("\n⚠️ Загрузка была прервана пользователем")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Получен сигнал прерывания. Завершаем работу...")
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("\n🔚 Завершение работы...")
        db_manager.disconnect()


if __name__ == "__main__":
    main()