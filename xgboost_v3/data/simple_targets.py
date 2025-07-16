"""
Простая система целевых переменных для предсказания направления движения цены.
Основано на исследованиях показывающих, что бинарная классификация направления
более эффективна для трейдинга, чем предсказание точной доходности.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_batch

logger = logging.getLogger(__name__)


class SimpleTargetSystem:
    """
    Простая система расчета целевых переменных для трейдинга.
    
    Основные принципы:
    1. Предсказываем направление движения цены (вверх/вниз)
    2. Используем несколько временных горизонтов (5мин, 15мин, 1час, 4часа)
    3. Добавляем минимальный порог движения для фильтрации шума
    4. Раздельные модели для покупки и продажи
    """
    
    def __init__(self, 
                 db_config: Dict[str, str],
                 min_movement_threshold: float = 0.1,  # Минимальное движение 0.1%
                 table_name: str = "simple_targets"):
        """
        Инициализация системы целевых переменных.
        
        Args:
            db_config: Конфигурация подключения к БД
            min_movement_threshold: Минимальный порог движения цены в %
            table_name: Название таблицы для сохранения
        """
        self.db_config = db_config
        self.min_movement_threshold = min_movement_threshold
        self.table_name = table_name
        
        # Временные горизонты (в барах по 15 минут)
        self.horizons = {
            '5min': 0.33,    # ~5 минут (частичный бар)
            '15min': 1,      # 15 минут (1 бар)
            '1hour': 4,      # 1 час (4 бара)
            '4hours': 16     # 4 часа (16 баров)
        }
        
        logger.info(f"Инициализирована SimpleTargetSystem:")
        logger.info(f"  - Минимальный порог движения: {min_movement_threshold}%")
        logger.info(f"  - Временные горизонты: {list(self.horizons.keys())}")
    
    def create_table(self):
        """Создание таблицы для хранения простых целевых переменных"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        try:
            # Удаляем старую таблицу если существует
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE")
            
            # Создаем новую таблицу
            create_query = f"""
            CREATE TABLE {self.table_name} (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                
                -- Текущая цена
                close_price DECIMAL(20, 8) NOT NULL,
                
                -- Будущие цены для разных горизонтов
                price_5min DECIMAL(20, 8),
                price_15min DECIMAL(20, 8),
                price_1hour DECIMAL(20, 8),
                price_4hours DECIMAL(20, 8),
                
                -- Процентные изменения
                change_5min DECIMAL(10, 4),
                change_15min DECIMAL(10, 4),
                change_1hour DECIMAL(10, 4),
                change_4hours DECIMAL(10, 4),
                
                -- Бинарные метки для покупки (цена выросла)
                buy_signal_5min BOOLEAN,
                buy_signal_15min BOOLEAN,
                buy_signal_1hour BOOLEAN,
                buy_signal_4hours BOOLEAN,
                
                -- Бинарные метки для продажи (цена упала)
                sell_signal_5min BOOLEAN,
                sell_signal_15min BOOLEAN,
                sell_signal_1hour BOOLEAN,
                sell_signal_4hours BOOLEAN,
                
                -- Метки с порогом (движение больше min_threshold)
                buy_signal_threshold_5min BOOLEAN,
                buy_signal_threshold_15min BOOLEAN,
                buy_signal_threshold_1hour BOOLEAN,
                buy_signal_threshold_4hours BOOLEAN,
                
                sell_signal_threshold_5min BOOLEAN,
                sell_signal_threshold_15min BOOLEAN,
                sell_signal_threshold_1hour BOOLEAN,
                sell_signal_threshold_4hours BOOLEAN,
                
                -- Мультиклассовые метки (сильное/слабое движение)
                direction_class_5min SMALLINT,
                direction_class_15min SMALLINT,
                direction_class_1hour SMALLINT,
                direction_class_4hours SMALLINT,
                
                -- Метаданные
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Индексы
                UNIQUE(timestamp, symbol)
            );
            
            -- Создаем индексы для быстрого поиска
            CREATE INDEX idx_{self.table_name}_timestamp ON {self.table_name}(timestamp);
            CREATE INDEX idx_{self.table_name}_symbol ON {self.table_name}(symbol);
            CREATE INDEX idx_{self.table_name}_symbol_timestamp ON {self.table_name}(symbol, timestamp);
            
            -- Индексы для целевых переменных
            CREATE INDEX idx_{self.table_name}_buy_1h ON {self.table_name}(buy_signal_1hour);
            CREATE INDEX idx_{self.table_name}_sell_1h ON {self.table_name}(sell_signal_1hour);
            CREATE INDEX idx_{self.table_name}_buy_thresh_1h ON {self.table_name}(buy_signal_threshold_1hour);
            CREATE INDEX idx_{self.table_name}_sell_thresh_1h ON {self.table_name}(sell_signal_threshold_1hour);
            """
            
            cur.execute(create_query)
            conn.commit()
            
            logger.info(f"✅ Таблица {self.table_name} успешно создана")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Ошибка при создании таблицы: {e}")
            raise
            
        finally:
            cur.close()
            conn.close()
    
    def calculate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает все целевые переменные для датафрейма.
        
        Args:
            df: DataFrame с колонками timestamp, symbol, close
            
        Returns:
            DataFrame с добавленными целевыми переменными
        """
        logger.info(f"Расчет целевых переменных для {len(df)} записей")
        
        # Сортируем по символу и времени
        df = df.sort_values(['symbol', 'timestamp']).copy()
        
        # Группируем по символу для корректного расчета
        results = []
        
        for symbol, symbol_df in df.groupby('symbol'):
            logger.info(f"  Обработка {symbol}: {len(symbol_df)} записей")
            
            # Копируем данные
            result_df = symbol_df.copy()
            result_df['close_price'] = result_df['close']
            
            # Рассчитываем для каждого горизонта
            for horizon_name, n_bars in self.horizons.items():
                if horizon_name == '5min':
                    # Для 5 минут используем интерполяцию
                    # Это упрощение - в реальности нужны тиковые данные
                    shift_bars = 1
                    weight = 0.33
                    next_close = result_df['close'].shift(-1)
                    future_price = result_df['close'] * (1 - weight) + next_close * weight
                else:
                    shift_bars = int(n_bars)
                    future_price = result_df['close'].shift(-shift_bars)
                
                # Будущая цена
                result_df[f'price_{horizon_name}'] = future_price
                
                # Процентное изменение
                price_change = ((future_price - result_df['close']) / result_df['close']) * 100
                result_df[f'change_{horizon_name}'] = price_change
                
                # Простые бинарные сигналы (вверх/вниз)
                result_df[f'buy_signal_{horizon_name}'] = price_change > 0
                result_df[f'sell_signal_{horizon_name}'] = price_change < 0
                
                # Сигналы с порогом
                result_df[f'buy_signal_threshold_{horizon_name}'] = price_change > self.min_movement_threshold
                result_df[f'sell_signal_threshold_{horizon_name}'] = price_change < -self.min_movement_threshold
                
                # Мультиклассовая классификация
                # 0: сильное падение (<-1%), 1: слабое падение (-1% до -0.1%)
                # 2: нейтрально (-0.1% до 0.1%), 3: слабый рост (0.1% до 1%)
                # 4: сильный рост (>1%)
                conditions = [
                    price_change < -1.0,
                    (price_change >= -1.0) & (price_change < -self.min_movement_threshold),
                    (price_change >= -self.min_movement_threshold) & (price_change <= self.min_movement_threshold),
                    (price_change > self.min_movement_threshold) & (price_change <= 1.0),
                    price_change > 1.0
                ]
                choices = [0, 1, 2, 3, 4]
                result_df[f'direction_class_{horizon_name}'] = np.select(conditions, choices, default=2)
            
            results.append(result_df)
        
        # Объединяем результаты
        final_df = pd.concat(results, ignore_index=True)
        
        # Логируем статистику
        self._log_statistics(final_df)
        
        return final_df
    
    def _log_statistics(self, df: pd.DataFrame):
        """Логирует статистику по рассчитанным целевым переменным"""
        logger.info("\n=== Статистика целевых переменных ===")
        
        # Убираем записи с NaN (последние строки без будущих данных)
        valid_df = df.dropna(subset=['change_1hour'])
        
        for horizon in ['15min', '1hour', '4hours']:
            logger.info(f"\n📊 Горизонт {horizon}:")
            
            # Процентные изменения
            changes = valid_df[f'change_{horizon}']
            logger.info(f"  Изменение цены:")
            logger.info(f"    - Среднее: {changes.mean():.3f}%")
            logger.info(f"    - Медиана: {changes.median():.3f}%")
            logger.info(f"    - Std: {changes.std():.3f}%")
            logger.info(f"    - Min/Max: {changes.min():.2f}% / {changes.max():.2f}%")
            
            # Простые сигналы
            buy_ratio = valid_df[f'buy_signal_{horizon}'].sum() / len(valid_df) * 100
            sell_ratio = valid_df[f'sell_signal_{horizon}'].sum() / len(valid_df) * 100
            logger.info(f"  Простые сигналы:")
            logger.info(f"    - Покупка (рост): {buy_ratio:.1f}%")
            logger.info(f"    - Продажа (падение): {sell_ratio:.1f}%")
            
            # Сигналы с порогом
            buy_thresh = valid_df[f'buy_signal_threshold_{horizon}'].sum() / len(valid_df) * 100
            sell_thresh = valid_df[f'sell_signal_threshold_{horizon}'].sum() / len(valid_df) * 100
            logger.info(f"  Сигналы с порогом {self.min_movement_threshold}%:")
            logger.info(f"    - Покупка: {buy_thresh:.1f}%")
            logger.info(f"    - Продажа: {sell_thresh:.1f}%")
            
            # Мультикласс
            class_dist = valid_df[f'direction_class_{horizon}'].value_counts().sort_index()
            logger.info(f"  Распределение классов:")
            class_names = ['Сильное падение', 'Слабое падение', 'Нейтрально', 'Слабый рост', 'Сильный рост']
            for i, name in enumerate(class_names):
                count = class_dist.get(i, 0)
                percent = count / len(valid_df) * 100
                logger.info(f"    - {name}: {percent:.1f}%")
    
    def save_to_database(self, df: pd.DataFrame, batch_size: int = 10000):
        """
        Сохраняет рассчитанные целевые переменные в базу данных.
        
        Args:
            df: DataFrame с рассчитанными целевыми переменными
            batch_size: Размер батча для вставки
        """
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        try:
            # Подготавливаем данные для вставки
            columns = [
                'timestamp', 'symbol', 'close_price',
                'price_5min', 'price_15min', 'price_1hour', 'price_4hours',
                'change_5min', 'change_15min', 'change_1hour', 'change_4hours',
                'buy_signal_5min', 'buy_signal_15min', 'buy_signal_1hour', 'buy_signal_4hours',
                'sell_signal_5min', 'sell_signal_15min', 'sell_signal_1hour', 'sell_signal_4hours',
                'buy_signal_threshold_5min', 'buy_signal_threshold_15min', 
                'buy_signal_threshold_1hour', 'buy_signal_threshold_4hours',
                'sell_signal_threshold_5min', 'sell_signal_threshold_15min',
                'sell_signal_threshold_1hour', 'sell_signal_threshold_4hours',
                'direction_class_5min', 'direction_class_15min',
                'direction_class_1hour', 'direction_class_4hours'
            ]
            
            # Преобразуем DataFrame в список кортежей
            data = []
            for _, row in df.iterrows():
                values = []
                for col in columns:
                    value = row.get(col)
                    # Обрабатываем NaN значения
                    if pd.isna(value):
                        values.append(None)
                    elif col == 'timestamp':
                        # Преобразуем timestamp правильно
                        if isinstance(value, (int, np.integer)):
                            # Если это unix timestamp в миллисекундах
                            values.append(pd.Timestamp(value, unit='ms'))
                        else:
                            values.append(pd.Timestamp(value))
                    elif isinstance(value, (bool, np.bool_)):
                        values.append(bool(value))
                    elif isinstance(value, (np.integer, np.floating)):
                        values.append(float(value) if col.startswith(('price_', 'change_', 'close_')) else int(value))
                    else:
                        values.append(value)
                data.append(tuple(values))
            
            # Вставляем данные батчами
            query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(['%s'] * len(columns))})
            ON CONFLICT (timestamp, symbol) DO UPDATE SET
                {', '.join([f'{col} = EXCLUDED.{col}' for col in columns if col not in ['timestamp', 'symbol']])}
            """
            
            # Вставляем батчами
            total_rows = len(data)
            inserted = 0
            
            for i in range(0, total_rows, batch_size):
                batch = data[i:i + batch_size]
                execute_batch(cur, query, batch, page_size=batch_size)
                inserted += len(batch)
                
                if inserted % 50000 == 0:
                    logger.info(f"  Вставлено {inserted}/{total_rows} записей...")
                    conn.commit()
            
            conn.commit()
            logger.info(f"✅ Успешно сохранено {total_rows} записей в {self.table_name}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Ошибка при сохранении в БД: {e}")
            raise
            
        finally:
            cur.close()
            conn.close()
    
    def get_training_data(self, 
                         symbols: list = None,
                         target_type: str = 'buy_signal_threshold_1hour',
                         start_date: str = None,
                         end_date: str = None,
                         min_samples_per_class: int = 1000) -> pd.DataFrame:
        """
        Получает данные для обучения из базы данных.
        
        Args:
            symbols: Список символов (None = все)
            target_type: Тип целевой переменной
            start_date: Начальная дата
            end_date: Конечная дата
            min_samples_per_class: Минимальное количество примеров на класс
            
        Returns:
            DataFrame с данными для обучения
        """
        conn = psycopg2.connect(**self.db_config)
        
        try:
            # Формируем запрос - исключаем дублирующиеся колонки
            query = f"""
            SELECT 
                t.timestamp,
                t.symbol,
                t.close_price,
                t.{target_type} as target,
                p.open,
                p.high,
                p.low,
                p.close,
                p.volume,
                p.technical_indicators,
                p.expected_return_buy,
                p.expected_return_sell
            FROM {self.table_name} t
            JOIN processed_market_data p ON EXTRACT(EPOCH FROM t.timestamp) * 1000 = p.timestamp AND t.symbol = p.symbol
            WHERE t.{target_type} IS NOT NULL
            """
            
            conditions = []
            params = []
            
            if symbols:
                placeholders = ','.join(['%s'] * len(symbols))
                conditions.append(f"t.symbol IN ({placeholders})")
                params.extend(symbols)
            
            if start_date:
                conditions.append("t.timestamp >= %s")
                params.append(start_date)
            
            if end_date:
                conditions.append("t.timestamp <= %s")
                params.append(end_date)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY t.timestamp"
            
            # Загружаем данные
            df = pd.read_sql_query(query, conn, params=params)
            
            # Проверяем баланс классов
            if 'target' in df.columns:
                class_counts = df['target'].value_counts()
                logger.info(f"\n📊 Распределение классов для {target_type}:")
                for class_val, count in class_counts.items():
                    percent = count / len(df) * 100
                    logger.info(f"  - Класс {class_val}: {count} ({percent:.1f}%)")
                
                # Проверяем минимальное количество
                min_class_count = class_counts.min()
                if min_class_count < min_samples_per_class:
                    logger.warning(f"⚠️ Недостаточно примеров в классе: {min_class_count} < {min_samples_per_class}")
            
            return df
            
        finally:
            conn.close()
    
    def analyze_symbol_performance(self, symbols: list = None) -> pd.DataFrame:
        """
        Анализирует производительность целевых переменных по символам.
        
        Args:
            symbols: Список символов для анализа
            
        Returns:
            DataFrame со статистикой по символам
        """
        conn = psycopg2.connect(**self.db_config)
        
        try:
            query = f"""
            WITH symbol_stats AS (
                SELECT 
                    symbol,
                    COUNT(*) as total_samples,
                    
                    -- Статистика по 1-часовым сигналам
                    AVG(CASE WHEN buy_signal_1hour THEN 1 ELSE 0 END) * 100 as buy_signal_1h_pct,
                    AVG(CASE WHEN sell_signal_1hour THEN 1 ELSE 0 END) * 100 as sell_signal_1h_pct,
                    AVG(CASE WHEN buy_signal_threshold_1hour THEN 1 ELSE 0 END) * 100 as buy_thresh_1h_pct,
                    AVG(CASE WHEN sell_signal_threshold_1hour THEN 1 ELSE 0 END) * 100 as sell_thresh_1h_pct,
                    
                    -- Статистика изменений
                    AVG(change_1hour) as avg_change_1h,
                    STDDEV(change_1hour) as std_change_1h,
                    MIN(change_1hour) as min_change_1h,
                    MAX(change_1hour) as max_change_1h,
                    
                    -- Распределение классов
                    AVG(CASE WHEN direction_class_1hour = 0 THEN 1 ELSE 0 END) * 100 as strong_down_pct,
                    AVG(CASE WHEN direction_class_1hour = 1 THEN 1 ELSE 0 END) * 100 as weak_down_pct,
                    AVG(CASE WHEN direction_class_1hour = 2 THEN 1 ELSE 0 END) * 100 as neutral_pct,
                    AVG(CASE WHEN direction_class_1hour = 3 THEN 1 ELSE 0 END) * 100 as weak_up_pct,
                    AVG(CASE WHEN direction_class_1hour = 4 THEN 1 ELSE 0 END) * 100 as strong_up_pct
                    
                FROM {self.table_name}
                WHERE change_1hour IS NOT NULL
                {f"AND symbol IN ({','.join(['%s'] * len(symbols))})" if symbols else ""}
                GROUP BY symbol
            )
            SELECT * FROM symbol_stats
            ORDER BY total_samples DESC
            """
            
            params = symbols if symbols else []
            df = pd.read_sql_query(query, conn, params=params)
            
            # Форматируем для удобного вывода
            logger.info("\n=== Анализ производительности по символам ===")
            for _, row in df.iterrows():
                logger.info(f"\n📊 {row['symbol']}:")
                logger.info(f"  Всего записей: {row['total_samples']:,}")
                logger.info(f"  Среднее изменение за 1ч: {row['avg_change_1h']:.3f}% ± {row['std_change_1h']:.3f}%")
                logger.info(f"  Диапазон: [{row['min_change_1h']:.2f}%, {row['max_change_1h']:.2f}%]")
                logger.info(f"  Простые сигналы: ↑{row['buy_signal_1h_pct']:.1f}% ↓{row['sell_signal_1h_pct']:.1f}%")
                logger.info(f"  Сигналы с порогом: ↑{row['buy_thresh_1h_pct']:.1f}% ↓{row['sell_thresh_1h_pct']:.1f}%")
            
            return df
            
        finally:
            conn.close()


# Функция для быстрого запуска
def create_simple_targets(db_config: Dict[str, str], 
                         symbols: list = None,
                         limit: int = None,
                         min_movement_threshold: float = 0.1):
    """
    Создает простые целевые переменные для обучения.
    
    Args:
        db_config: Конфигурация БД
        symbols: Список символов (None = все)
        limit: Лимит записей для обработки
        min_movement_threshold: Минимальный порог движения
    """
    # Создаем систему
    target_system = SimpleTargetSystem(db_config, min_movement_threshold)
    
    # Создаем таблицу
    target_system.create_table()
    
    # Загружаем данные
    conn = psycopg2.connect(**db_config)
    
    try:
        query = """
        SELECT timestamp, symbol, close
        FROM raw_market_data
        WHERE 1=1
        """
        
        params = []
        if symbols:
            placeholders = ','.join(['%s'] * len(symbols))
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        
        query += " ORDER BY symbol, timestamp"
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"Загрузка данных из raw_market_data...")
        df = pd.read_sql_query(query, conn, params=params)
        logger.info(f"Загружено {len(df)} записей")
        
        # Рассчитываем целевые переменные
        targets_df = target_system.calculate_targets(df)
        
        # Сохраняем в БД
        target_system.save_to_database(targets_df)
        
        # Анализируем результаты
        if symbols and len(symbols) <= 10:
            target_system.analyze_symbol_performance(symbols)
        
        logger.info("\n✅ Простые целевые переменные успешно созданы!")
        
    finally:
        conn.close()


if __name__ == "__main__":
    # Пример использования
    import yaml
    
    # Загружаем конфигурацию
    with open('/Users/ruslan/PycharmProjects/LLM TRANSFORM/xgboost_v3/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = {
        'host': config['database']['host'],
        'port': config['database']['port'],
        'database': config['database']['database'],
        'user': config['database']['user'],
        'password': config['database']['password']
    }
    
    # Создаем целевые переменные для тестовых символов
    create_simple_targets(
        db_config=db_config,
        symbols=['BTCUSDT', 'ETHUSDT'],
        limit=100000,  # Для теста
        min_movement_threshold=0.1  # 0.1% минимальное движение
    )