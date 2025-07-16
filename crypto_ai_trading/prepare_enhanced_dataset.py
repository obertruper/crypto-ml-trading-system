"""
Скрипт для подготовки датасета с расширенными признаками
Применяет enhanced features к существующим данным для улучшения direction prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data.data_loader import CryptoDataLoader
from data.feature_engineering import FeatureEngineer
from data.enhanced_features import EnhancedFeatureEngineer
from utils.logger import get_logger, setup_logging
from utils.config import load_config
from utils.db_utils import DatabaseManager


class EnhancedDatasetPreparer:
    """Подготовка датасета с расширенными признаками"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("EnhancedDatasetPreparer")
        self.db_manager = DatabaseManager(config)
        self.feature_engineer = FeatureEngineer(config)
        self.enhanced_engineer = EnhancedFeatureEngineer()
        
    def prepare_enhanced_dataset(self, 
                               symbols: Optional[List[str]] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               save_to_db: bool = True) -> pd.DataFrame:
        """
        Подготовка датасета с enhanced features
        
        Args:
            symbols: список символов (None = все из конфига)
            start_date: начальная дата
            end_date: конечная дата
            save_to_db: сохранить в БД
            
        Returns:
            DataFrame с расширенными признаками
        """
        self.logger.info("🚀 Начинаем подготовку enhanced датасета...")
        
        # Загрузка базовых данных
        data_loader = CryptoDataLoader(self.config)
        
        if symbols is None:
            symbols = self.config['data']['symbols']
        
        self.logger.info(f"📊 Загрузка данных для {len(symbols)} символов...")
        raw_data = data_loader.load_raw_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        if raw_data.empty:
            self.logger.error("❌ Нет данных для обработки!")
            return pd.DataFrame()
        
        self.logger.info(f"✅ Загружено {len(raw_data)} строк данных")
        
        # Базовые технические индикаторы
        self.logger.info("📈 Создание базовых технических индикаторов...")
        featured_data = self.feature_engineer.create_features(raw_data)
        
        # Группировка данных по символам для cross-asset features
        all_symbols_data = {}
        for symbol in symbols:
            symbol_data = featured_data[featured_data['symbol'] == symbol].copy()
            all_symbols_data[symbol] = symbol_data
        
        # Enhanced features для каждого символа
        enhanced_dfs = []
        
        for symbol in tqdm(symbols, desc="Добавление enhanced features"):
            symbol_data = all_symbols_data[symbol].copy()
            
            # Применяем enhanced features
            enhanced_data = self.enhanced_engineer.create_enhanced_features(
                symbol_data,
                all_symbols_data  # Передаем все символы для cross-asset
            )
            
            enhanced_dfs.append(enhanced_data)
        
        # Объединяем все символы
        final_dataset = pd.concat(enhanced_dfs, ignore_index=True)
        
        # Финальная обработка
        final_dataset = self._post_process_features(final_dataset)
        
        # Статистика по признакам
        self._log_feature_statistics(final_dataset)
        
        # Сохранение в БД
        if save_to_db:
            self._save_to_database(final_dataset)
        
        # Сохранение в файл
        self._save_to_file(final_dataset)
        
        self.logger.info(f"✅ Enhanced датасет готов: {final_dataset.shape}")
        
        return final_dataset
    
    def _post_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Постобработка признаков"""
        self.logger.info("🔧 Постобработка признаков...")
        
        # Удаляем временные колонки
        temp_columns = [col for col in df.columns if col.startswith('_temp_')]
        if temp_columns:
            df = df.drop(columns=temp_columns)
        
        # Проверка на inf значения
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        inf_columns = []
        
        for col in numeric_columns:
            if np.isinf(df[col]).any():
                inf_columns.append(col)
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        if inf_columns:
            self.logger.warning(f"⚠️ Заменены inf значения в колонках: {inf_columns[:5]}...")
        
        # Заполнение пропусков
        # Для большинства признаков используем forward fill
        feature_columns = [col for col in df.columns 
                          if col not in ['id', 'symbol', 'datetime', 'timestamp']
                          and not col.startswith(('target_', 'future_', 'direction_'))]
        
        df[feature_columns] = df.groupby('symbol')[feature_columns].fillna(method='ffill')
        
        # Для оставшихся NaN используем 0
        df[feature_columns] = df[feature_columns].fillna(0)
        
        # Сортировка
        df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        
        return df
    
    def _log_feature_statistics(self, df: pd.DataFrame):
        """Логирование статистики по признакам"""
        self.logger.info("\n📊 СТАТИСТИКА ENHANCED FEATURES:")
        
        # Подсчет признаков по категориям
        feature_categories = {
            'price_based': [],
            'volume_based': [],
            'technical': [],
            'market_regime': [],
            'microstructure': [],
            'cross_asset': [],
            'sentiment': [],
            'other': []
        }
        
        feature_columns = [col for col in df.columns 
                          if col not in ['id', 'symbol', 'datetime', 'timestamp']
                          and not col.startswith(('target_', 'future_', 'direction_'))]
        
        for col in feature_columns:
            if any(x in col for x in ['open', 'high', 'low', 'close', 'vwap', 'price']):
                feature_categories['price_based'].append(col)
            elif any(x in col for x in ['volume', 'turnover', 'trade']):
                feature_categories['volume_based'].append(col)
            elif any(x in col for x in ['rsi', 'macd', 'ema', 'sma', 'bb_', 'atr']):
                feature_categories['technical'].append(col)
            elif any(x in col for x in ['regime', 'wyckoff', 'trend_strength']):
                feature_categories['market_regime'].append(col)
            elif any(x in col for x in ['ofi', 'tick', 'aggressive', 'imbalance']):
                feature_categories['microstructure'].append(col)
            elif any(x in col for x in ['btc_', 'sector_', 'beta_', 'correlation']):
                feature_categories['cross_asset'].append(col)
            elif any(x in col for x in ['fear_greed', 'panic', 'euphoria', 'sentiment']):
                feature_categories['sentiment'].append(col)
            else:
                feature_categories['other'].append(col)
        
        # Вывод статистики
        self.logger.info(f"Всего признаков: {len(feature_columns)}")
        for category, features in feature_categories.items():
            if features:
                self.logger.info(f"  - {category}: {len(features)} признаков")
                # Показываем примеры
                examples = features[:3]
                self.logger.info(f"    Примеры: {', '.join(examples)}")
        
        # Проверка целевых переменных
        target_columns = [col for col in df.columns if col.startswith(('direction_', 'future_return_'))]
        self.logger.info(f"\nЦелевые переменные: {len(target_columns)}")
        for target in target_columns[:5]:
            if target.startswith('direction_'):
                value_counts = df[target].value_counts()
                self.logger.info(f"  - {target}: UP={value_counts.get(0, 0)}, "
                               f"DOWN={value_counts.get(1, 0)}, FLAT={value_counts.get(2, 0)}")
    
    def _save_to_database(self, df: pd.DataFrame):
        """Сохранение в базу данных"""
        self.logger.info("💾 Сохранение в базу данных...")
        
        try:
            # Создаем таблицу если не существует
            create_table_query = """
            CREATE TABLE IF NOT EXISTS enhanced_market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                datetime TIMESTAMP NOT NULL,
                data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, datetime)
            );
            
            CREATE INDEX IF NOT EXISTS idx_enhanced_symbol_datetime 
            ON enhanced_market_data(symbol, datetime);
            """
            
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_query)
                    conn.commit()
            
            # Сохраняем данные порциями
            batch_size = 1000
            total_saved = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                # Конвертируем в JSON для хранения
                records = []
                for _, row in batch.iterrows():
                    record_data = row.to_dict()
                    # Убираем служебные поля
                    for field in ['id', 'symbol', 'datetime']:
                        record_data.pop(field, None)
                    
                    records.append((
                        row['symbol'],
                        row['datetime'],
                        json.dumps(record_data, default=str)
                    ))
                
                # Вставка с обновлением при конфликте
                insert_query = """
                INSERT INTO enhanced_market_data (symbol, datetime, data)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol, datetime) 
                DO UPDATE SET data = EXCLUDED.data;
                """
                
                with self.db_manager.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.executemany(insert_query, records)
                        conn.commit()
                
                total_saved += len(records)
                
            self.logger.info(f"✅ Сохранено {total_saved} записей в БД")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения в БД: {e}")
    
    def _save_to_file(self, df: pd.DataFrame):
        """Сохранение в файл для быстрой загрузки"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем директорию если не существует
        data_dir = Path("data/enhanced_datasets")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем в разных форматах
        # Parquet для эффективного хранения
        parquet_path = data_dir / f"enhanced_dataset_{timestamp}.parquet"
        df.to_parquet(parquet_path, index=False, compression='snappy')
        self.logger.info(f"💾 Сохранено в Parquet: {parquet_path}")
        
        # Pickle для быстрой загрузки с сохранением типов
        pickle_path = data_dir / f"enhanced_dataset_{timestamp}.pkl"
        df.to_pickle(pickle_path)
        self.logger.info(f"💾 Сохранено в Pickle: {pickle_path}")
        
        # Метаинформация
        meta_info = {
            'timestamp': timestamp,
            'shape': df.shape,
            'symbols': df['symbol'].unique().tolist(),
            'date_range': {
                'start': str(df['datetime'].min()),
                'end': str(df['datetime'].max())
            },
            'features': {
                'total': len([col for col in df.columns 
                            if col not in ['id', 'symbol', 'datetime', 'timestamp']]),
                'enhanced': len([col for col in df.columns 
                               if any(x in col for x in ['regime', 'ofi', 'btc_', 'sentiment'])])
            }
        }
        
        meta_path = data_dir / f"enhanced_dataset_{timestamp}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2)
        self.logger.info(f"📝 Метаинформация сохранена: {meta_path}")
    
    def load_latest_enhanced_dataset(self) -> Optional[pd.DataFrame]:
        """Загрузка последнего enhanced датасета"""
        data_dir = Path("data/enhanced_datasets")
        
        if not data_dir.exists():
            return None
        
        # Ищем последний pickle файл
        pickle_files = list(data_dir.glob("enhanced_dataset_*.pkl"))
        
        if not pickle_files:
            return None
        
        # Сортируем по времени создания
        latest_file = max(pickle_files, key=lambda x: x.stat().st_mtime)
        
        self.logger.info(f"📂 Загрузка enhanced датасета: {latest_file}")
        
        try:
            df = pd.read_pickle(latest_file)
            self.logger.info(f"✅ Загружено {df.shape[0]} строк с {df.shape[1]} колонками")
            return df
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Prepare Enhanced Dataset')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Symbols to process (default: all from config)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-save-db', action='store_true',
                       help='Do not save to database')
    parser.add_argument('--load-latest', action='store_true',
                       help='Load latest enhanced dataset instead of creating new')
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    
    # Логирование
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/enhanced_dataset_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(log_dir, "enhanced_dataset")
    logger = get_logger("PrepareEnhancedDataset")
    
    preparer = EnhancedDatasetPreparer(config)
    
    if args.load_latest:
        # Загрузка существующего датасета
        logger.info("📂 Загрузка последнего enhanced датасета...")
        df = preparer.load_latest_enhanced_dataset()
        
        if df is not None:
            logger.info("✅ Датасет успешно загружен!")
            # Можно добавить дополнительную обработку
        else:
            logger.error("❌ Не удалось загрузить датасет")
    else:
        # Создание нового enhanced датасета
        logger.info("🚀 Создание нового enhanced датасета...")
        
        # Если не указаны символы, используем топ из конфига
        if args.symbols is None and config.get('data', {}).get('max_symbols'):
            # Используем ограниченный набор для теста
            args.symbols = config['data']['symbols'][:config['data']['max_symbols']]
        
        df = preparer.prepare_enhanced_dataset(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            save_to_db=not args.no_save_db
        )
        
        if not df.empty:
            logger.info("✅ Enhanced датасет успешно создан!")
            logger.info(f"📊 Размер: {df.shape}")
            logger.info(f"📅 Период: {df['datetime'].min()} - {df['datetime'].max()}")
        else:
            logger.error("❌ Не удалось создать датасет")


if __name__ == "__main__":
    main()