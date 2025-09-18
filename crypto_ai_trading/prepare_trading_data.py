#!/usr/bin/env python3
"""
Полный скрипт для подготовки данных и запуска обучения торговой модели
Версия 2.0 - с версионированием кэша и оптимизацией
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
from multiprocessing import RLock
from functools import partial
import concurrent.futures
import psutil
import time
import hashlib
import gc

from data.data_loader import CryptoDataLoader
from data.feature_engineering import FeatureEngineer
from utils.logger import get_logger
from tqdm import tqdm

# ВАЖНО: Увеличивать при изменении логики создания признаков
FEATURE_VERSION = "4.1"  # Обновленная версия БЕЗ УТЕЧЕК ДАННЫХ - 20 целевых переменных


def check_database_connection(config: dict, logger):
    """Проверка подключения к БД и наличия данных"""
    logger.info("🔍 Проверка подключения к PostgreSQL...")
    
    try:
        data_loader = CryptoDataLoader(config)
        stats = data_loader.get_data_stats()
        
        logger.info(f"✅ Подключение успешно!")
        logger.info(f"📊 Статистика БД: {stats['total_records']:,} записей, {stats['unique_symbols']} символов")
        
        return True, data_loader
        
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к БД: {e}")
        logger.error("Проверьте настройки в config.yaml и запустите PostgreSQL")
        return False, None


def process_symbol_features(symbol: str, symbol_data: pd.DataFrame, config: dict,
                           logger_name: str, use_cache: bool = True,
                           position: int = None, disable_progress: bool = False,
                           raise_on_error: bool = False) -> pd.DataFrame:
    """Обработка признаков для одного символа (для параллельной обработки)"""
    logger = get_logger(f"{logger_name}_{symbol}", is_subprocess=True)
    
    # Проверяем кеш с учетом версии
    cache_dir = Path("cache/features")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Включаем версию в хэш для инвалидации при изменении кода
    cache_key = f"{symbol}_{len(symbol_data)}_{symbol_data.index[0]}_{symbol_data.index[-1]}_{FEATURE_VERSION}"
    data_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = cache_dir / f"{symbol}_{data_hash}.parquet"
    
    if use_cache and cache_file.exists():
        try:
            logger.info(f"📦 Загрузка {symbol} из кеша (v{FEATURE_VERSION})...")
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning(f"⚠️ Ошибка чтения кеша для {symbol}: {e}")
    
    try:
        # Создаем отдельный экземпляр FeatureEngineer для каждого процесса
        feature_engineer = FeatureEngineer(config)
        feature_engineer.process_position = position
        feature_engineer.disable_progress = disable_progress
        
        # Сортируем данные
        symbol_data = symbol_data.sort_values('datetime')
        
        # Оптимизация типов данных
        symbol_data = optimize_memory_usage(symbol_data, logger)
        
        # ИСПРАВЛЕНО: Используем create_features вместо отдельных методов
        # Это обеспечивает правильную последовательность обработки
        symbol_data = feature_engineer.create_features(symbol_data)
        
        # Финальная оптимизация памяти
        symbol_data = optimize_memory_usage(symbol_data, logger)
        
        logger.info(f"✅ {symbol}: обработано {len(symbol_data)} записей")
        
        # Сохраняем в кеш
        if use_cache:
            try:
                symbol_data.to_parquet(cache_file, compression='snappy')
                logger.debug(f"💾 {symbol} сохранен в кеш v{FEATURE_VERSION}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось сохранить кеш для {symbol}: {e}")
        
        # Явная очистка памяти
        gc.collect()
        
        return symbol_data
    
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке {symbol}: {e}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        # Сохраняем подробную информацию об ошибке
        try:
            with open(f'/tmp/{symbol}_error.log', 'w') as f:
                f.write(f"Error processing {symbol}:\n")
                f.write(f"Error: {str(e)}\n\n")
                f.write(f"Traceback:\n{tb}\n")
        except Exception:
            pass
        if raise_on_error:
            raise
        return pd.DataFrame()


def optimize_memory_usage(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Оптимизация типов данных для экономии памяти"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    # Оптимизация числовых типов
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0 and df[col].max() <= 255:
            df[col] = df[col].astype('uint8')
        elif df[col].min() >= -32768 and df[col].max() <= 32767:
            df[col] = df[col].astype('int16')
        elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
            df[col] = df[col].astype('int32')
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if end_mem < start_mem:
        logger.debug(f"💾 Память: {start_mem:.1f}MB → {end_mem:.1f}MB")
    
    return df


def prepare_features_for_trading(config: dict, logger, force_recreate: bool = False,
                                 workers: int | None = None,
                                 sequential: bool = False,
                                 no_cache: bool = False):
    """Подготовка признаков для торговой модели"""
    
    logger.info("\n" + "="*80)
    logger.info("🚀 ПОДГОТОВКА ДАННЫХ ДЛЯ ТОРГОВОЙ МОДЕЛИ")
    logger.info(f"📦 Версия признаков: {FEATURE_VERSION}")
    if force_recreate:
        logger.info("🔄 Принудительное пересоздание кэша включено")
    logger.info("="*80)
    
    # Проверка БД
    success, data_loader = check_database_connection(config, logger)
    if not success:
        return None
    
    # Загрузка сырых данных
    logger.info("\n📥 Загрузка данных из БД...")
    
    # Определяем символы для загрузки
    symbols_to_load = config['data']['symbols']
    if symbols_to_load == 'all':
        available = data_loader.get_available_symbols()
        symbols_to_load = available[:20]  # Топ 20 для начала
        logger.info(f"📊 Загружаем топ-20 символов")
    
    # Загружаем данные
    raw_data = data_loader.load_data(
        symbols=symbols_to_load,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    if raw_data is None or raw_data.empty:
        logger.error('❌ В базе данных нет записей по указанным символам/датам.')
        logger.error('   Заполните БД историческими данными:')
        logger.error('   python download_data.py    # загрузка данных из Bybit/...')
        return None

    
    logger.info(f"✅ Загружено {len(raw_data):,} записей")
    
    # Проверка качества данных
    logger.info("\n🔍 Проверка качества данных...")
    quality_report = data_loader.validate_data_quality(raw_data)
    
    total_issues = 0
    for symbol, report in quality_report.items():
        missing_count = sum(report['missing_values'].values())
        anomalies_count = len(report['anomalies'])
        
        if missing_count > 0 or anomalies_count > 0:
            logger.warning(f"⚠️ {symbol}: пропущено {missing_count}, аномалий {anomalies_count}")
            total_issues += missing_count + anomalies_count
    
    if total_issues == 0:
        logger.info("✅ Качество данных отличное!")
    
    # Создание признаков с параллельной обработкой
    logger.info("\n🛠️ Создание признаков...")
    
    # Простой расчет количества процессов
    cpu_count = mp.cpu_count()
    n_processes = 1 if sequential else min(cpu_count - 1, 8)
    if workers is not None:
        n_processes = max(1, int(workers))
    logger.info(f"⚡ Используем {n_processes} процессов")
    
    # Группируем данные по символам
    symbols = raw_data['symbol'].unique()
    logger.info(f"📊 Обработка {len(symbols)} символов...")
    
    start_time = time.time()
    
    # Устанавливаем глобальную блокировку для tqdm
    tqdm.set_lock(RLock())
    
    # Параллельная обработка символов
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes, 
                                                initializer=tqdm.set_lock, 
                                                initargs=(tqdm.get_lock(),)) as executor:
        # Создаем функцию с частичными параметрами
        process_func = partial(process_symbol_features, 
                              config=config, 
                              logger_name="FeatureEngineering",
                              use_cache=not (force_recreate or no_cache),
                              raise_on_error=sequential or (workers == 1))  # В отладочном режиме пробрасываем исключения
        
        # Создаем задачи для каждого символа
        future_to_symbol = {}
        for idx, symbol in enumerate(symbols):
            future = executor.submit(
                process_func, 
                symbol, 
                raw_data[raw_data['symbol'] == symbol].copy(),
                position=idx * 2,
                disable_progress=True
            )
            future_to_symbol[future] = symbol
        
        # Собираем результаты с прогресс-баром
        featured_dfs = []
        
        with tqdm(total=len(symbols), desc="🚀 Обработка символов", unit="символ") as pbar:
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if not result.empty:
                        featured_dfs.append(result)
                        pbar.set_postfix({'Символ': symbol})
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"❌ Ошибка при обработке {symbol}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    pbar.update(1)
    
    # Объединяем результаты
    logger.info("\n📊 Объединение результатов...")
    if not featured_dfs:
        logger.warning("⚠️ Параллельная обработка вернула пустой результат. Пытаемся обработать последовательно...")
        featured_dfs = []
        for symbol in symbols:
            try:
                result = process_symbol_features(
                    symbol,
                    raw_data[raw_data['symbol'] == symbol].copy(),
                    config=config,
                    logger_name="FeatureEngineeringFallback",
                    use_cache=not (force_recreate or no_cache),
                    position=0,
                    disable_progress=True,
                    raise_on_error=True
                )
                if not result.empty:
                    featured_dfs.append(result)
                    logger.info(f"✅ {symbol}: обработано {len(result)} записей (fallback)")
            except Exception as e:
                logger.error(f"❌ Ошибка при последовательной обработке {symbol}: {e}")
        # Подсказка: где искать причины
        try:
            import glob
            err_logs = glob.glob('/tmp/*_error.log')
            if err_logs:
                logger.warning(f"⚠️ Найдены логи ошибок subprocess: {err_logs[:5]} ...")
        except Exception:
            pass

        if not featured_dfs:
            raise RuntimeError("Пустой результат feature engineering: ни один символ не был обработан. Проверьте логи /tmp/*_error.log")
    
    processed_data = pd.concat(featured_dfs, ignore_index=True)
    
    # ИСПРАВЛЕНО: Удаляем дубликаты после параллельной обработки
    initial_count = len(processed_data)
    processed_data = processed_data.drop_duplicates(subset=['datetime', 'symbol'], keep='first')
    duplicates_removed = initial_count - len(processed_data)
    
    if duplicates_removed > 0:
        logger.warning(f"⚠️ Удалено {duplicates_removed} дубликатов по (datetime, symbol)")
    else:
        logger.info("✅ Дубликаты не обнаружены")
    
    elapsed_time = time.time() - start_time
    logger.info(f"⏱️ Обработка заняла {elapsed_time:.1f} секунд")
    
    # ИСПРАВЛЕНО: cross-asset features требуют ВСЕ символы вместе
    # В process_symbol_features обрабатывается каждый символ отдельно
    # Поэтому нужно выполнить cross-asset features здесь
    feature_engineer = FeatureEngineer(config)
    
    logger.info("🔄 Создание cross-asset признаков...")
    processed_data = feature_engineer._create_cross_asset_features(processed_data)

    # Удаляем строки с NaN в целевых переменных (хвосты рядов без достаточного будущего горизонта)
    required_targets = [
        'future_return_15m','future_return_1h','future_return_4h','future_return_12h',
        'direction_15m','direction_1h','direction_4h','direction_12h',
        'long_will_reach_1pct_4h','long_will_reach_2pct_4h','long_will_reach_3pct_12h','long_will_reach_5pct_12h',
        'short_will_reach_1pct_4h','short_will_reach_2pct_4h','short_will_reach_3pct_12h','short_will_reach_5pct_12h',
        'max_drawdown_1h','max_rally_1h','max_drawdown_4h','max_rally_4h'
    ]
    if any(col in processed_data.columns for col in required_targets):
        before = len(processed_data)
        processed_data = processed_data.dropna(subset=[c for c in required_targets if c in processed_data.columns])
        after = len(processed_data)
        dropped = before - after
        if dropped > 0:
            logger.info(f'Dropped rows with NaN in targets: {dropped}')


    # Пост-проверка наличия ключевых целевых колонок
    required_targets = [
        'future_return_15m','future_return_1h','future_return_4h','future_return_12h',
        'direction_15m','direction_1h','direction_4h','direction_12h',
        'long_will_reach_1pct_4h','long_will_reach_2pct_4h','long_will_reach_3pct_12h','long_will_reach_5pct_12h',
        'short_will_reach_1pct_4h','short_will_reach_2pct_4h','short_will_reach_3pct_12h','short_will_reach_5pct_12h',
        'max_drawdown_1h','max_rally_1h','max_drawdown_4h','max_rally_4h'
    ]
    missing = [c for c in required_targets if c not in processed_data.columns]
    if missing:
        logger.warning(f"⚠️ Отсутствуют ожидаемые целевые колонки: {missing}")
    
    # Разделение на train/val/test с временным gap
    logger.info("📊 Разделение на выборки с временным gap...")
    
    # Используем функцию из preprocessor с gap
    from data.preprocessor import create_train_val_test_split
    
    train_data, val_data, test_data = create_train_val_test_split(
        processed_data,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        time_column='datetime',
        gap_days=2  # 2 дня gap между выборками для предотвращения утечек
    )
    
    # ИСПРАВЛЕНО: Обучаем препроцессор ТОЛЬКО на train данных
    logger.info("📏 Настройка препроцессора (нормализация ТОЛЬКО на train)...")
    
    # Импортируем препроцессор
    from data.preprocessor import DataPreprocessor
    import pickle
    
    # Создаем и обучаем препроцессор ТОЛЬКО на train данных
    preprocessor = DataPreprocessor(config)
    preprocessor.fit(train_data, exclude_targets=True)  # Исключаем целевые из нормализации
    
    # Сохраняем препроцессор
    preprocessor_path = Path('models_saved/preprocessor_v4.pkl')
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    logger.info(f"✅ Препроцессор обучен на TRAIN и сохранен в {preprocessor_path}")
    
    # Статистика по данным
    logger.info("\n📊 СТАТИСТИКА ПОДГОТОВЛЕННЫХ ДАННЫХ:")
    logger.info(f"   - Обучающая выборка: {len(train_data):,} записей")
    logger.info(f"   - Валидационная: {len(val_data):,} записей")
    logger.info(f"   - Тестовая: {len(test_data):,} записей")
    logger.info(f"   - Всего признаков: {len(train_data.columns)}")
    
    # Сохранение данных
    logger.info("\n💾 Сохранение подготовленных данных...")
    
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = data_dir / "train_data.parquet"
    val_path = data_dir / "val_data.parquet"
    test_path = data_dir / "test_data.parquet"
    
    train_data.to_parquet(train_path, compression='snappy')
    val_data.to_parquet(val_path, compression='snappy')
    test_data.to_parquet(test_path, compression='snappy')
    
    logger.info(f"✅ Данные сохранены в {data_dir}")
    
    # ВАЖНО: Проверка на утечки данных
    logger.info("\n🔍 Проверка на утечки данных...")
    
    # Проверяем что нет future колонок в признаках
    feature_cols = [col for col in train_data.columns 
                   if col not in ['datetime', 'symbol'] and not any(
                       keyword in col for keyword in [
                           'future_', 'direction_', 'will_reach_', 'max_drawdown_', 'max_rally_',
                           'long_tp', 'short_tp', 'long_sl', 'short_sl',
                           '_reached', '_hit', '_time', 'expected_value', 'best_direction',
                           'target_return', 'long_optimal_entry', 'short_optimal_entry'
                       ]
                   )]
    # Удалено: 'best_action', 'risk_reward', 'optimal_hold' - больше не целевые
    # signal_strength теперь признак (не целевая)
    
    # Определяем целевые колонки
    target_cols = [col for col in train_data.columns if col.startswith((
        'future_return_',      # 4 переменные: 15m, 1h, 4h, 12h
        'direction_',          # 4 переменные: направления
        'long_will_reach_',    # 4 переменные: достижение целей LONG
        'short_will_reach_',   # 4 переменные: достижение целей SHORT
        'max_drawdown_',       # 2 переменные: 1h, 4h
        'max_rally_'           # 2 переменные: 1h, 4h
    ))]
    
    # Сохраняем списки колонок в текстовые файлы для staged training
    feature_cols_path = data_dir / "feature_cols.txt"
    target_cols_path = data_dir / "target_cols.txt"
    
    with open(feature_cols_path, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    with open(target_cols_path, 'w') as f:
        for col in target_cols:
            f.write(f"{col}\n")
    
    logger.info(f"✅ Сохранены списки колонок:")
    logger.info(f"   - Признаки ({len(feature_cols)}): {feature_cols_path}")
    logger.info(f"   - Целевые ({len(target_cols)}): {target_cols_path}")
    
    # Более точная проверка на подозрительные колонки
    # optimal_leverage и safe_leverage - это рекомендации на основе исторической волатильности, не утечка
    suspicious_cols = [col for col in feature_cols 
                      if any(word in col.lower() for word in ['future', 'target']) and 
                      col not in ['optimal_leverage', 'safe_leverage']]
    
    if suspicious_cols:
        logger.warning(f"⚠️ Найдены подозрительные колонки: {suspicious_cols}")
    else:
        logger.info("✅ Утечек данных не обнаружено!")
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'feature_count': len(train_data.columns),
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'symbols': symbols_to_load
    }


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Подготовка данных для торговой модели')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Путь к конфигурации')
    parser.add_argument('--test', action='store_true',
                       help='Тестовый режим с 3 монетами (BTC, ETH, SOL)')
    parser.add_argument('--force-recreate', action='store_true',
                       help='Принудительно пересоздать кэш (игнорировать существующий)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Только анализ без сохранения')
    parser.add_argument('--workers', type=int, default=None,
                       help='Число воркеров для параллельной обработки (по умолчанию auto)')
    parser.add_argument('--sequential', action='store_true',
                       help='Последовательная обработка без multiprocessing (отладка)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Игнорировать feature cache и пересчитывать признаки')
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger = get_logger("DataPreparation")
    
    # В тестовом режиме используем только 3 монеты
    if args.test:
        config['data']['symbols'] = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        logger.info("🧪 ТЕСТОВЫЙ РЕЖИМ: используются только 3 монеты")
    
    # Подготавливаем данные
    result = prepare_features_for_trading(
        config, logger,
        force_recreate=args.force_recreate,
        workers=args.workers,
        sequential=args.sequential,
        no_cache=args.no_cache
    )
    
    if result and not args.analyze_only:
        logger.info("\n" + "="*80)
        logger.info("✅ ДАННЫЕ ГОТОВЫ К ОБУЧЕНИЮ!")
        logger.info("="*80)
        logger.info("\n🚀 Запустите обучение:")
        logger.info("   python main.py --mode train")


if __name__ == "__main__":
    main()
