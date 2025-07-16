#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Полный пайплайн для работы с ФЬЮЧЕРСНЫМИ данными
1. Проверка символов на фьючерсном рынке
2. Загрузка данных
3. Подготовка датасета
4. Обучение модели
"""

import yaml
import logging
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Основная функция пайплайна"""
    
    logger.info("🚀 ЗАПУСК ПАЙПЛАЙНА ДЛЯ ФЬЮЧЕРСНЫХ ДАННЫХ")
    logger.info("="*60)
    
    # Шаг 1: Валидация фьючерсных символов
    logger.info("\n📊 ШАГ 1: Валидация фьючерсных символов")
    logger.info("-"*40)
    
    os.system("python validate_futures_symbols.py")
    
    # Используем валидированный конфиг
    config_path = 'config_futures_validated.yaml'
    if not os.path.exists(config_path):
        logger.error("❌ Не найден файл config_futures_validated.yaml")
        logger.info("💡 Запустите сначала: python validate_futures_symbols.py")
        return
    
    # Загружаем валидированный конфиг
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['data_download']['symbols']
    logger.info(f"✅ Будет обработано {len(symbols)} фьючерсных символов")
    
    # Шаг 2: Инициализация БД
    logger.info("\n📊 ШАГ 2: Инициализация базы данных")
    logger.info("-"*40)
    
    os.system("python init_database.py")
    
    # Шаг 3: Загрузка фьючерсных данных
    logger.info("\n📊 ШАГ 3: Загрузка фьючерсных данных")
    logger.info("-"*40)
    
    # Сохраняем временный конфиг с market_type = futures
    config['data_download']['market_type'] = 'futures'
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    os.system("python download_data.py")
    
    # Шаг 4: Подготовка датасета
    logger.info("\n📊 ШАГ 4: Подготовка датасета с индикаторами")
    logger.info("-"*40)
    
    os.system("python prepare_dataset.py")
    
    # Шаг 5: Обучение модели
    logger.info("\n📊 ШАГ 5: Обучение модели")
    logger.info("-"*40)
    
    # Проверяем что есть обработанные данные
    import psycopg2
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Проверяем количество обработанных данных
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol) as symbols, COUNT(*) as total_records
            FROM processed_market_data p
            JOIN raw_market_data r ON p.raw_data_id = r.id
            WHERE r.market_type = 'futures'
        """)
        
        result = cursor.fetchone()
        symbols_count, records_count = result
        
        logger.info(f"✅ Готово к обучению: {symbols_count} символов, {records_count:,} записей")
        
        cursor.close()
        conn.close()
        
        if records_count > 1000:
            logger.info("\n🚀 Запуск обучения продвинутой модели...")
            os.system("python train_advanced.py")
        else:
            logger.error("❌ Недостаточно данных для обучения!")
            
    except Exception as e:
        logger.error(f"❌ Ошибка проверки данных: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ ПАЙПЛАЙН ЗАВЕРШЕН!")
    logger.info("="*60)
    
    # Итоговая информация
    logger.info("\n📊 РЕЗУЛЬТАТЫ:")
    logger.info(f"✅ Валидировано символов: {len(symbols)}")
    logger.info(f"💾 Модели сохранены в: trained_model/")
    logger.info(f"📈 Графики в: plots/")
    logger.info(f"📝 Логи в: logs/")


if __name__ == "__main__":
    main()