#!/usr/bin/env python3
"""
Скрипт для кэширования всех данных на сервере
Загружает данные из БД и сохраняет в локальный кеш для автономной работы
"""

import argparse
import logging
import time
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# Настройка окружения
warnings.filterwarnings('ignore')

# Импорты из модулей
from config import Config, EXCLUDE_SYMBOLS
from data import DataLoader, DataPreprocessor, SequenceCreator, CacheManager
from utils import LoggingManager

logger = logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Cache All Data for Transformer v3")
    
    parser.add_argument('--all-symbols', action='store_true',
                       help='Загрузить данные по всем символам')
    
    parser.add_argument('--test-symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Символы для тестового режима')
    
    parser.add_argument('--force-refresh', action='store_true',
                       help='Принудительно обновить кеш')
    
    parser.add_argument('--skip-sequences', action='store_true',
                       help='Не создавать последовательности (только сырые данные)')
    
    return parser.parse_args()


def cache_raw_data_by_symbols(data_loader: DataLoader, cacher: CacheManager, 
                             all_symbols: bool = False, test_symbols: list = None,
                             incremental: bool = True) -> pd.DataFrame:
    """Кэширование сырых данных по символам с прогрессом и инкрементальным обновлением"""
    logger.info("📥 Кэширование сырых данных из PostgreSQL...")
    
    start_time = time.time()
    
    # Проверяем существующий кэш для инкрементального обновления
    existing_df = None
    existing_symbols = set()
    existing_timestamps = {}
    
    if incremental:
        # Сначала проверяем основной кэш
        existing_df = cacher.load_processed_data('raw')
        if existing_df is not None:
            logger.info(f"📂 Найден существующий кэш с {len(existing_df):,} записями")
        else:
            # Проверяем промежуточный кэш
            existing_df = cacher.load_processed_data('raw_temp')
            if existing_df is not None:
                logger.info(f"📂 Найден промежуточный кэш с {len(existing_df):,} записями")
                logger.info("🔄 Продолжение с места остановки...")
                
        if existing_df is not None:
            existing_symbols = set(existing_df['symbol'].unique())
            # Сохраняем последние timestamp для каждого символа
            for symbol in existing_symbols:
                symbol_data = existing_df[existing_df['symbol'] == symbol]
                # Конвертируем numpy.int64 в обычный Python int
                existing_timestamps[symbol] = int(symbol_data['timestamp'].max())
            logger.info(f"📊 В кэше {len(existing_symbols)} символов: {', '.join(sorted(existing_symbols))}")
    
    if all_symbols:
        logger.info("🌍 Загрузка данных по ВСЕМ доступным символам")
        # Получаем список всех символов
        try:
            symbols_to_load = data_loader.load_symbols_list()
            logger.info(f"📊 Найдено {len(symbols_to_load)} доступных символов")
            
            if incremental and existing_symbols:
                new_symbols = set(symbols_to_load) - existing_symbols
                if new_symbols:
                    logger.info(f"🆕 Найдено {len(new_symbols)} новых символов: {', '.join(list(new_symbols)[:5])}...")
                else:
                    logger.info("✅ Все символы уже есть в кэше")
                    
            logger.info(f"📋 Символы: {', '.join(symbols_to_load[:10])}{'...' if len(symbols_to_load) > 10 else ''}")
        except Exception as e:
            logger.error(f"❌ Не удалось получить список символов: {e}")
            raise
    else:
        logger.info(f"🧪 Загрузка данных по тестовым символам: {test_symbols}")
        symbols_to_load = test_symbols
    
    # Загружаем данные по символам с прогрессом
    all_dataframes = []
    if existing_df is not None and incremental:
        all_dataframes.append(existing_df)  # Добавляем существующие данные
        
    total_records = 0
    updated_symbols = 0
    new_records = 0
    
    for i, symbol in enumerate(symbols_to_load):
        symbol_start = time.time()
        
        # Пропускаем уже полностью загруженные символы
        if incremental and symbol in existing_symbols and symbol not in existing_timestamps:
            logger.info(f"⏩ Пропуск {i+1}/{len(symbols_to_load)}: {symbol} (уже в кэше)")
            continue
            
        # Проверяем нужно ли обновление для этого символа
        if incremental and symbol in existing_timestamps:
            last_timestamp = existing_timestamps[symbol]
            logger.info(f"⏳ Проверка обновлений {i+1}/{len(symbols_to_load)}: {symbol} (после {pd.to_datetime(last_timestamp, unit='ms')})")
            
            # Загружаем только новые данные
            symbol_df = data_loader.load_symbol_updates(symbol, after_timestamp=last_timestamp)
            
            if len(symbol_df) > 0:
                # Удаляем старые данные этого символа из existing_df
                if existing_df is not None:
                    mask = existing_df['symbol'] != symbol
                    all_dataframes[0] = existing_df[mask]
                    
                # Добавляем все данные символа (старые + новые)
                full_symbol_df = data_loader.load_symbol_data(symbol)
                all_dataframes.append(full_symbol_df)
                
                new_records += len(symbol_df)
                updated_symbols += 1
                logger.info(f"🔄 {symbol}: +{len(symbol_df):,} новых записей (всего {len(full_symbol_df):,})")
            else:
                logger.info(f"✅ {symbol}: данные актуальны")
        else:
            logger.info(f"⏳ Загрузка {i+1}/{len(symbols_to_load)}: {symbol}")
            
            try:
                symbol_df = data_loader.load_symbol_data(symbol)
                symbol_time = time.time() - symbol_start
                
                if len(symbol_df) > 0:
                    all_dataframes.append(symbol_df)
                    total_records += len(symbol_df)
                    logger.info(f"✅ {symbol}: {len(symbol_df):,} записей за {symbol_time:.2f} сек")
                else:
                    logger.warning(f"⚠️ {symbol}: нет данных")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: ошибка загрузки - {e}")
                continue
            
        # Промежуточная статистика каждые 10 символов
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = (len(symbols_to_load) - i - 1) * avg_time
            logger.info(f"📊 Прогресс: {i+1}/{len(symbols_to_load)} ({(i+1)/len(symbols_to_load)*100:.1f}%), "
                       f"записей: {total_records:,}, осталось: ~{remaining/60:.1f} мин")
            
        # Сохраняем промежуточные результаты каждые 5 символов
        if (i + 1) % 5 == 0 and len(all_dataframes) > 0:
            logger.info("💾 Сохранение промежуточных результатов...")
            temp_df = pd.concat(all_dataframes, ignore_index=True)
            temp_df = temp_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
            cacher.save_processed_data(temp_df, 'raw_temp')
            logger.info(f"✅ Промежуточный кэш сохранен: {len(temp_df):,} записей")
    
    # Объединяем все данные
    if not all_dataframes:
        raise ValueError("Нет данных для кэширования!")
        
    logger.info("📋 Объединение данных от всех символов...")
    df = pd.concat(all_dataframes, ignore_index=True)
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    query_time = time.time() - start_time
    logger.info(f"📊 Загрузка завершена за {query_time:.2f} сек")
    logger.info(f"📊 Всего в кэше: {len(df):,} записей по {df['symbol'].nunique()} символам")
    
    if incremental and (updated_symbols > 0 or new_records > 0):
        logger.info(f"🔄 Обновлено символов: {updated_symbols}")
        logger.info(f"🆕 Новых записей: {new_records:,}")
    
    # Статистика по данным
    date_range = df['datetime'].max() - df['datetime'].min()
    logger.info(f"📅 Период данных: {df['datetime'].min():%Y-%m-%d} - {df['datetime'].max():%Y-%m-%d} ({date_range.days} дней)")
    
    # Топ символов по количеству записей
    top_symbols = df['symbol'].value_counts().head(5)
    logger.info("🔝 Топ-5 символов по записям:")
    for symbol, count in top_symbols.items():
        logger.info(f"   {symbol}: {count:,} записей")
    
    # Сохраняем в кеш
    logger.info("💾 Сохранение в кеш...")
    cache_start = time.time()
    
    cacher.save_processed_data(df, 'raw')
    
    # Удаляем временный кэш если он существует
    temp_cache_path = cacher.cache_dir / "data_raw_temp.parquet"
    if temp_cache_path.exists():
        temp_cache_path.unlink()
        logger.info("🗑️ Временный кэш удален")
    
    cache_time = time.time() - cache_start
    total_time = time.time() - start_time
    
    logger.info(f"✅ Сохранение в кеш: {cache_time:.2f} сек")
    logger.info(f"✅ Общее время загрузки: {total_time:.2f} сек")
    logger.info(f"📈 Скорость обработки: {len(df)/total_time:,.0f} записей/сек")
    
    return df


def cache_raw_data(data_loader: DataLoader, cacher: CacheManager, 
                   all_symbols: bool = False, test_symbols: list = None,
                   force_refresh: bool = False) -> pd.DataFrame:
    """Кэширование сырых данных из БД (с выбором метода)"""
    
    if all_symbols:
        # Для всех символов ВСЕГДА используем загрузку по частям
        return cache_raw_data_by_symbols(data_loader, cacher, all_symbols, test_symbols, 
                                       incremental=not force_refresh)
    else:
        # Для тестовых символов - обычная загрузка
        logger.info("📥 Кэширование сырых данных из PostgreSQL...")
        start_time = time.time()
        
        logger.info(f"🧪 Загрузка данных по тестовым символам: {test_symbols}")
        
        logger.info("⏳ Выполняется запрос к PostgreSQL...")
        df = data_loader.load_data(symbols=test_symbols)
        
        logger.info("💾 Сохранение в кеш...")
        cacher.save_processed_data(df, 'raw')
        
        total_time = time.time() - start_time
        logger.info(f"✅ Загружено {len(df):,} записей за {total_time:.2f} сек")
        
        return df


def cache_processed_features(df: pd.DataFrame, preprocessor: DataPreprocessor, 
                           cacher: CacheManager) -> pd.DataFrame:
    """Кэширование обработанных признаков"""
    logger.info("🔧 Кэширование извлеченных признаков...")
    
    start_time = time.time()
    logger.info(f"📊 Исходные данные: {len(df):,} записей, {len(df.columns)} колонок")
    
    # Извлекаем признаки
    logger.info("⏳ Извлечение технических индикаторов...")
    extraction_start = time.time()
    
    features_df = preprocessor.extract_features(df)
    
    extraction_time = time.time() - extraction_start
    logger.info(f"📊 Извлечение признаков: {extraction_time:.2f} сек")
    logger.info(f"📊 Получено признаков: {len(preprocessor.feature_columns)} колонок")
    logger.info(f"📊 Размер данных: {len(features_df):,} записей")
    
    # Сохраняем в кеш
    logger.info("💾 Сохранение признаков в кеш...")
    cache_start = time.time()
    
    cacher.save_processed_data(features_df, 'features')
    
    cache_time = time.time() - cache_start
    total_time = time.time() - start_time
    
    logger.info(f"✅ Сохранение в кеш: {cache_time:.2f} сек")
    logger.info(f"✅ Общее время обработки: {total_time:.2f} сек")
    logger.info(f"📈 Скорость: {len(features_df)/total_time:,.0f} записей/сек")
    
    return features_df


def cache_normalized_data(features_df: pd.DataFrame, preprocessor: DataPreprocessor,
                         cacher: CacheManager) -> tuple:
    """Кэширование нормализованных данных"""
    logger.info("📏 Кэширование нормализованных данных...")
    
    start_time = time.time()
    logger.info(f"📊 Входные данные: {len(features_df):,} записей")
    
    # Разделяем данные
    logger.info("⏳ Временное разделение данных...")
    split_start = time.time()
    
    data_splits = preprocessor.split_data_temporal(features_df)
    train_df = data_splits['train']
    val_df = data_splits['val']
    test_df = data_splits['test']
    
    split_time = time.time() - split_start
    logger.info(f"📊 Разделение данных: {split_time:.2f} сек")
    logger.info(f"📊 Обучающая выборка: {len(train_df):,} записей ({len(train_df)/len(features_df)*100:.1f}%)")
    logger.info(f"📊 Валидационная выборка: {len(val_df):,} записей ({len(val_df)/len(features_df)*100:.1f}%)")
    logger.info(f"📊 Тестовая выборка: {len(test_df):,} записей ({len(test_df)/len(features_df)*100:.1f}%)")
    
    # Нормализуем
    logger.info("⏳ Нормализация признаков...")
    norm_start = time.time()
    
    train_df_norm, val_df_norm, test_df_norm = preprocessor.normalize_features(
        train_df, val_df, test_df
    )
    
    norm_time = time.time() - norm_start
    logger.info(f"📊 Нормализация: {norm_time:.2f} сек")
    
    # Сохраняем нормализованные данные
    logger.info("💾 Сохранение нормализованных данных в кеш...")
    cache_start = time.time()
    
    normalized_data = {
        'train': train_df_norm,
        'val': val_df_norm, 
        'test': test_df_norm,
        'scaler': preprocessor.scaler,
        'feature_columns': preprocessor.feature_columns
    }
    
    cacher.save_data(normalized_data, 'normalized_splits')
    
    cache_time = time.time() - cache_start
    total_time = time.time() - start_time
    
    logger.info(f"✅ Сохранение в кеш: {cache_time:.2f} сек")
    logger.info(f"✅ Общее время нормализации: {total_time:.2f} сек")
    
    return train_df_norm, val_df_norm, test_df_norm, preprocessor.scaler


def cache_sequences(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   feature_columns: list, sequence_creator: SequenceCreator, 
                   cacher: CacheManager):
    """Кэширование последовательностей для обеих задач"""
    logger.info("🔄 Кэширование последовательностей...")
    
    start_time = time.time()
    logger.info(f"📊 Параметры последовательностей:")
    logger.info(f"   Длина последовательности: {sequence_creator.config.model.sequence_length}")
    logger.info(f"   Количество признаков: {len(feature_columns)}")
    
    # Последовательности для BUY
    logger.info("📈 Создание последовательностей для BUY...")
    buy_start = time.time()
    
    buy_sequences = sequence_creator.create_sequences_for_splits(
        train_df, val_df, test_df, feature_columns, target_type='buy'
    )
    
    buy_time = time.time() - buy_start
    logger.info(f"📊 BUY последовательности созданы за {buy_time:.2f} сек")
    logger.info(f"   Train: {len(buy_sequences['train']['X'])} последовательностей")
    logger.info(f"   Val: {len(buy_sequences['val']['X'])} последовательностей") 
    logger.info(f"   Test: {len(buy_sequences['test']['X'])} последовательностей")
    
    cacher.save_sequences(buy_sequences, 'buy', 'regression')
    
    # Последовательности для SELL
    logger.info("📉 Создание последовательностей для SELL...")
    sell_start = time.time()
    
    sell_sequences = sequence_creator.create_sequences_for_splits(
        train_df, val_df, test_df, feature_columns, target_type='sell'
    )
    
    sell_time = time.time() - sell_start
    logger.info(f"📊 SELL последовательности созданы за {sell_time:.2f} сек")
    logger.info(f"   Train: {len(sell_sequences['train']['X'])} последовательностей")
    logger.info(f"   Val: {len(sell_sequences['val']['X'])} последовательностей")
    logger.info(f"   Test: {len(sell_sequences['test']['X'])} последовательностей")
    
    cacher.save_sequences(sell_sequences, 'sell', 'regression')
    
    total_time = time.time() - start_time
    total_sequences = (len(buy_sequences['train']['X']) + len(buy_sequences['val']['X']) + 
                      len(buy_sequences['test']['X']) + len(sell_sequences['train']['X']) + 
                      len(sell_sequences['val']['X']) + len(sell_sequences['test']['X']))
    
    logger.info(f"✅ Все последовательности закэшированы за {total_time:.2f} сек")
    logger.info(f"📈 Создано {total_sequences:,} последовательностей")
    logger.info(f"📈 Скорость: {total_sequences/total_time:,.0f} последовательностей/сек")


def cache_database_metadata(data_loader: DataLoader, cacher: CacheManager):
    """Кэширование метаданных БД"""
    logger.info("📊 Кэширование метаданных БД...")
    
    try:
        # Загружаем список символов
        symbols_list = data_loader.load_symbols_list()
        
        # Метаданные
        metadata = {
            'available_symbols': symbols_list,
            'excluded_symbols': EXCLUDE_SYMBOLS,
            'total_symbols': len(symbols_list),
            'cache_timestamp': pd.Timestamp.now().isoformat()
        }
        
        cacher.save_data(metadata, 'database_metadata')
        
        logger.info(f"✅ Метаданные закэшированы: {len(symbols_list)} символов")
        
    except Exception as e:
        logger.error(f"❌ Ошибка кэширования метаданных: {e}")


def main():
    """Главная функция кэширования"""
    args = parse_args()
    
    # Конфигурация
    config = Config()
    
    # Настройка логирования
    log_dir = Path("logs/cache_data")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging_manager = LoggingManager(log_dir)
    logging_manager.setup_logging()
    
    logger.info("""
    ╔══════════════════════════════════════════╗
    ║      Data Caching for Transformer v3     ║
    ╚══════════════════════════════════════════╝
    """)
    
    # Инициализация компонентов
    cacher = CacheManager(config)
    preprocessor = DataPreprocessor(config)
    sequence_creator = SequenceCreator(config)
    
    try:
        # Проверка существующего кеша
        if not args.force_refresh:
            existing_raw = cacher.load_processed_data('raw')
            if existing_raw is not None:
                logger.info("📂 Найдены кэшированные сырые данные")
                if not args.all_symbols:
                    logger.info("✅ Кеш актуален, используем существующие данные")
                    df = existing_raw
                else:
                    logger.info("🔄 Требуется обновление для всех символов")
                    df = None
            else:
                df = None
        else:
            logger.info("🔄 Принудительное обновление кеша")
            df = None
        
        # 1. Кэширование сырых данных
        if df is None:
            with DataLoader(config) as data_loader:
                # Сначала кэшируем метаданные
                cache_database_metadata(data_loader, cacher)
                
                # Затем данные
                df = cache_raw_data(
                    data_loader, cacher, 
                    all_symbols=args.all_symbols,
                    test_symbols=args.test_symbols,
                    force_refresh=args.force_refresh
                )
        
        # 2. Кэширование признаков
        logger.info("\n" + "="*60)
        logger.info("🔧 ЭТАП 2: КЭШИРОВАНИЕ ПРИЗНАКОВ")
        logger.info("="*60)
        
        features_df = cache_processed_features(df, preprocessor, cacher)
        
        # 3. Кэширование нормализованных данных
        logger.info("\n" + "="*60)
        logger.info("📏 ЭТАП 3: КЭШИРОВАНИЕ НОРМАЛИЗАЦИИ")
        logger.info("="*60)
        
        train_df, val_df, test_df, scaler = cache_normalized_data(
            features_df, preprocessor, cacher
        )
        
        # 4. Кэширование последовательностей
        if not args.skip_sequences:
            logger.info("\n" + "="*60)
            logger.info("🔄 ЭТАП 4: КЭШИРОВАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
            logger.info("="*60)
            
            cache_sequences(
                train_df, val_df, test_df, 
                preprocessor.feature_columns,
                sequence_creator, cacher
            )
        
        # Информация о кеше
        logger.info("\n" + "="*60)
        logger.info("📊 ИНФОРМАЦИЯ О КЕШЕ")
        logger.info("="*60)
        
        cache_info = cacher.get_cache_info()
        logger.info(f"📁 Директория кеша: {cache_info['cache_dir']}")
        logger.info(f"📄 Количество файлов: {cache_info['n_files']}")
        logger.info(f"💾 Общий размер: {cache_info['total_size_mb']:.1f} MB")
        logger.info("📋 Файлы кеша:")
        for file_name in sorted(cache_info['files']):
            logger.info(f"   - {file_name}")
        
        logger.info("\n🎉 Кэширование завершено успешно!")
        logger.info("🚀 Теперь обучение может работать автономно (без интернета)")
        
        # Создаем файл-флаг готовности кеша
        cache_ready_file = cacher.cache_dir / "CACHE_READY"
        with open(cache_ready_file, 'w') as f:
            f.write(f"Cache created: {pd.Timestamp.now().isoformat()}\n")
            f.write(f"Symbols: {'ALL' if args.all_symbols else str(args.test_symbols)}\n")
            f.write(f"Total size: {cache_info['total_size_mb']:.1f} MB\n")
        
    except Exception as e:
        logger.error(f"❌ Ошибка кэширования: {e}")
        raise


if __name__ == "__main__":
    main()