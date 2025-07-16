#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для инициализации базы данных PostgreSQL
Создает все необходимые таблицы для проекта
"""

import psycopg2
import yaml
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Загружает конфигурацию из YAML файла"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        raise


def create_database(db_config: dict):
    """Создает базу данных если она не существует"""
    
    # Подключаемся к postgres для создания БД
    conn_params = db_config.copy()
    database_name = conn_params.pop('dbname')
    conn_params['dbname'] = 'postgres'
    
    # Удаляем пустой пароль
    if not conn_params.get('password'):
        conn_params.pop('password', None)
    
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Проверяем существует ли БД
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (database_name,)
        )
        
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {database_name}")
            logger.info(f"✅ База данных '{database_name}' создана")
        else:
            logger.info(f"ℹ️ База данных '{database_name}' уже существует")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Ошибка создания базы данных: {e}")
        raise


def init_tables(db_config: dict):
    """Создает все необходимые таблицы"""
    
    # Удаляем пустой пароль
    conn_params = db_config.copy()
    if not conn_params.get('password'):
        conn_params.pop('password', None)
    
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cursor = conn.cursor()
        
        logger.info("🔧 Создание таблиц в PostgreSQL...")
        
        # Таблица для сырых рыночных данных
        create_raw_data_table = """
        CREATE TABLE IF NOT EXISTS raw_market_data (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp BIGINT NOT NULL,
            datetime TIMESTAMP NOT NULL,
            open DECIMAL(20, 8) NOT NULL,
            high DECIMAL(20, 8) NOT NULL,
            low DECIMAL(20, 8) NOT NULL,
            close DECIMAL(20, 8) NOT NULL,
            volume DECIMAL(20, 8) NOT NULL,
            turnover DECIMAL(20, 8) DEFAULT 0,
            interval_minutes INTEGER NOT NULL DEFAULT 15,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(symbol, timestamp, interval_minutes)
        );
        """
        
        # Индексы для оптимизации
        create_indexes = """
        CREATE INDEX IF NOT EXISTS idx_raw_market_data_symbol_timestamp 
        ON raw_market_data(symbol, timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_raw_market_data_datetime 
        ON raw_market_data(datetime);
        
        CREATE INDEX IF NOT EXISTS idx_raw_market_data_symbol_datetime 
        ON raw_market_data(symbol, datetime);
        """
        
        # Таблица для обработанных данных с индикаторами
        create_processed_data_table = """
        CREATE TABLE IF NOT EXISTS processed_market_data (
            id BIGSERIAL PRIMARY KEY,
            raw_data_id BIGINT REFERENCES raw_market_data(id),
            symbol VARCHAR(20) NOT NULL,
            timestamp BIGINT NOT NULL,
            datetime TIMESTAMP NOT NULL,
            
            -- Базовые OHLCV
            open DECIMAL(20, 8) NOT NULL,
            high DECIMAL(20, 8) NOT NULL,
            low DECIMAL(20, 8) NOT NULL,
            close DECIMAL(20, 8) NOT NULL,
            volume DECIMAL(20, 8) NOT NULL,
            
            -- Технические индикаторы (JSON для гибкости)
            technical_indicators JSONB,
            
            -- Целевые переменные (метки)
            buy_profit_target INTEGER DEFAULT 0,
            buy_loss_target INTEGER DEFAULT 0,
            sell_profit_target INTEGER DEFAULT 0,
            sell_loss_target INTEGER DEFAULT 0,
            
            -- Метаданные
            processing_version VARCHAR(10) DEFAULT '1.0',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            
            UNIQUE(symbol, timestamp)
        );
        """
        
        # Индексы для processed_data
        create_processed_indexes = """
        CREATE INDEX IF NOT EXISTS idx_processed_market_data_symbol_timestamp 
        ON processed_market_data(symbol, timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_processed_market_data_targets 
        ON processed_market_data(buy_profit_target, buy_loss_target, sell_profit_target, sell_loss_target);
        
        CREATE INDEX IF NOT EXISTS idx_processed_technical_indicators 
        ON processed_market_data USING GIN (technical_indicators);
        """
        
        # Таблица для метаданных модели
        create_model_metadata_table = """
        CREATE TABLE IF NOT EXISTS model_metadata (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            version VARCHAR(20) NOT NULL,
            feature_columns JSONB,
            training_config JSONB,
            performance_metrics JSONB,
            file_path VARCHAR(500),
            created_at TIMESTAMP DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE
        );
        """
        
        # Таблица для хранения последовательностей
        create_sequences_table = """
        CREATE TABLE IF NOT EXISTS training_sequences (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            sequence_start_timestamp BIGINT NOT NULL,
            sequence_end_timestamp BIGINT NOT NULL,
            sequence_length INTEGER NOT NULL,
            features JSONB NOT NULL,
            buy_profit_target INTEGER DEFAULT 0,
            buy_loss_target INTEGER DEFAULT 0,
            sell_profit_target INTEGER DEFAULT 0,
            sell_loss_target INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # Таблица для логирования предсказаний
        create_predictions_table = """
        CREATE TABLE IF NOT EXISTS model_predictions (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp BIGINT NOT NULL,
            datetime TIMESTAMP NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            
            -- Предсказания вероятностей
            buy_profit_probability DECIMAL(5, 4),
            buy_loss_probability DECIMAL(5, 4),
            sell_profit_probability DECIMAL(5, 4),
            sell_loss_probability DECIMAL(5, 4),
            
            -- Рекомендация
            recommendation VARCHAR(20),  -- 'BUY', 'SELL', 'HOLD'
            confidence DECIMAL(5, 4),
            
            -- Фактический результат (для анализа)
            actual_outcome VARCHAR(20),
            actual_pnl DECIMAL(10, 4),
            
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        # Индексы для predictions
        create_predictions_indexes = """
        CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timestamp 
        ON model_predictions(symbol, timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_predictions_datetime 
        ON model_predictions(datetime);
        """
        
        # Выполняем создание таблиц
        queries = [
            create_raw_data_table,
            create_indexes,
            create_processed_data_table,
            create_processed_indexes,
            create_model_metadata_table,
            create_sequences_table,
            create_predictions_table,
            create_predictions_indexes
        ]
        
        for i, query in enumerate(queries, 1):
            cursor.execute(query)
            logger.info(f"✅ Выполнен запрос {i}/{len(queries)}")
        
        # Создаем расширение для работы с JSONB если нужно
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        # Добавляем колонку market_type если её еще нет
        logger.info("🔧 Добавление колонки market_type...")
        cursor.execute("""
            ALTER TABLE raw_market_data 
            ADD COLUMN IF NOT EXISTS market_type VARCHAR(20) DEFAULT 'spot';
        """)
        
        # Создаем индекс для market_type
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_raw_market_data_market_type 
            ON raw_market_data(market_type);
        """)
        
        logger.info("✅ Все таблицы созданы успешно!")
        
        # Показываем созданные таблицы
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        logger.info("\n📋 Созданные таблицы:")
        for table in tables:
            logger.info(f"   - {table[0]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Ошибка создания таблиц: {e}")
        raise


def main():
    """Основная функция"""
    
    logger.info("🚀 Инициализация базы данных для ML системы криптотрейдинга")
    
    # Загружаем конфигурацию
    config = load_config()
    db_config = config['database']
    
    try:
        # Создаем БД если не существует
        create_database(db_config)
        
        # Создаем таблицы
        init_tables(db_config)
        
        logger.info("\n✅ База данных успешно инициализирована!")
        logger.info("📊 Теперь можно запускать скрипты:")
        logger.info("   1. python download_data.py - для загрузки данных")
        logger.info("   2. python prepare_dataset.py - для подготовки датасета")
        logger.info("   3. python train_model_postgres.py - для обучения модели")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()