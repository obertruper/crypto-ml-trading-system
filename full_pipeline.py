#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Полный пайплайн: загрузка данных, подготовка датасета и обучение модели
"""

import yaml
import logging
import time
from datetime import datetime
from download_data import PostgreSQLManager, BybitDataDownloader
from prepare_dataset import MarketDatasetPreparator
from train_model_postgres import MarketMovementPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_symbol_support(downloader, symbols):
    """Проверяет какие символы поддерживаются API"""
    supported = []
    unsupported = []
    
    logger.info("🔍 Проверка поддержки символов...")
    
    for symbol in symbols:
        try:
            # Пробуем загрузить 1 свечу
            klines = downloader.get_klines(symbol, '15', 
                                          int(time.time() * 1000) - 900000,  # 15 минут назад
                                          int(time.time() * 1000))
            if klines:
                supported.append(symbol)
                logger.info(f"✅ {symbol} - поддерживается")
            else:
                unsupported.append(symbol)
                logger.warning(f"❌ {symbol} - не поддерживается")
        except Exception as e:
            unsupported.append(symbol)
            logger.warning(f"❌ {symbol} - ошибка: {str(e)}")
        
        time.sleep(0.1)  # Пауза между запросами
    
    return supported, unsupported


def main():
    """Основной пайплайн"""
    
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    data_config = config['data_download']
    model_config = config['model']
    risk_profile = config['risk_profile']
    
    # Инициализация
    db_manager = PostgreSQLManager(db_config)
    
    try:
        db_manager.connect()
        
        # ШАГ 1: ЗАГРУЗКА ДАННЫХ
        logger.info("\n" + "="*60)
        logger.info("ШАГ 1: ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ")
        logger.info("="*60)
        
        downloader = BybitDataDownloader(db_manager)
        
        # Проверяем поддерживаемые символы
        all_symbols = data_config['symbols']
        supported_symbols, unsupported_symbols = check_symbol_support(downloader, all_symbols)
        
        logger.info(f"\n📊 Поддерживается: {len(supported_symbols)}/{len(all_symbols)} символов")
        
        if unsupported_symbols:
            logger.warning(f"⚠️ Не поддерживаются: {', '.join(unsupported_symbols)}")
            
            # Обновляем config.yaml с поддерживаемыми символами
            config['data_download']['symbols'] = supported_symbols
            config['data_download']['unsupported_symbols'] = unsupported_symbols
            
            with open('config_updated.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info("💾 Сохранен обновленный конфиг: config_updated.yaml")
        
        # Загружаем данные только для поддерживаемых символов
        if supported_symbols:
            interval = data_config['interval']
            days = data_config['days']
            
            logger.info(f"\n🚀 Загрузка данных для {len(supported_symbols)} символов")
            logger.info(f"📊 Параметры: интервал={interval}m, период={days} дней")
            
            results = downloader.download_multiple_symbols(supported_symbols, interval, days)
            
            # Статистика загрузки
            success_count = sum(1 for r in results.values() if r.get('success', False))
            logger.info(f"\n✅ Успешно загружено: {success_count}/{len(supported_symbols)} символов")
        
        # ШАГ 2: ПОДГОТОВКА ДАТАСЕТА
        logger.info("\n" + "="*60)
        logger.info("ШАГ 2: ПОДГОТОВКА ДАТАСЕТА")
        logger.info("="*60)
        
        preparator = MarketDatasetPreparator(db_manager, risk_profile)
        
        # Получаем список символов с данными
        symbols_with_data = preparator.get_available_symbols()
        
        if not symbols_with_data:
            logger.error("❌ Нет символов с достаточным количеством данных!")
            return
        
        logger.info(f"📊 Найдено {len(symbols_with_data)} символов для обработки")
        
        # Обрабатываем каждый символ
        total_processed = 0
        for symbol in symbols_with_data:
            try:
                logger.info(f"\n🔄 Обработка {symbol}...")
                
                # Загружаем сырые данные
                df = preparator.load_raw_data(symbol)
                
                if len(df) < 100:
                    logger.warning(f"⚠️ Недостаточно данных для {symbol}")
                    continue
                
                # Рассчитываем индикаторы
                indicators = preparator.calculate_technical_indicators(df)
                
                # Создаем метки
                labels = preparator.create_labels_based_on_risk_profile(df, symbol)
                
                # Сохраняем в БД
                preparator.save_processed_data(symbol, df, indicators, labels)
                
                total_processed += 1
                
                # Сохраняем метаданные признаков
                if total_processed == 1:
                    preparator.save_feature_columns_metadata(list(indicators.keys()))
                
            except Exception as e:
                logger.error(f"❌ Ошибка обработки {symbol}: {e}")
                continue
        
        logger.info(f"\n✅ Обработано {total_processed} символов")
        
        # ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ
        logger.info("\n" + "="*60)
        logger.info("ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ")
        logger.info("="*60)
        
        if total_processed == 0:
            logger.error("❌ Нет обработанных данных для обучения!")
            return
        
        # Инициализируем предиктор
        predictor = MarketMovementPredictor(
            db_manager=db_manager,
            sequence_length=model_config['sequence_length'],
            prediction_horizon=model_config['prediction_horizon']
        )
        
        # Загружаем список признаков
        predictor.load_feature_columns()
        
        if not predictor.feature_columns:
            logger.error("❌ Не удалось загрузить список признаков!")
            return
        
        logger.info(f"✅ Загружено {len(predictor.feature_columns)} признаков")
        
        # Создаем последовательности из БД
        logger.info("\n🔄 Создание последовательностей для обучения...")
        training_data = predictor.create_sequences_from_db()
        
        if not training_data:
            logger.error("❌ Не удалось создать последовательности!")
            return
        
        logger.info(f"✅ Создано {len(training_data['X_sequences'])} последовательностей")
        
        # Обучаем все модели
        logger.info("\n🚀 Начинаем обучение моделей...")
        training_results = predictor.train_all_models(training_data)
        
        # Оцениваем производительность
        performance_summary = predictor.evaluate_model_performance(training_results)
        
        # Сохраняем модель
        predictor.save_complete_model(training_results, training_data)
        
        logger.info("\n" + "="*60)
        logger.info("🎉 ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        logger.info("="*60)
        logger.info(f"✅ Загружено символов: {success_count}")
        logger.info(f"✅ Обработано символов: {total_processed}")
        logger.info(f"✅ Обучено моделей: 4")
        logger.info(f"💾 Модели сохранены в: trained_model/")
        logger.info(f"📊 Графики в: plots/")
        
        # Финальная статистика БД
        stats = downloader.get_database_stats()
        total_records = sum(s['total_records'] for s in stats.values())
        logger.info(f"\n📊 ИТОГО В БД: {total_records:,} записей для {len(stats)} символов")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        raise
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()