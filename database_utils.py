#!/usr/bin/env python3
"""
Утилиты для работы с PostgreSQL базой данных
"""

import psycopg2
import pandas as pd
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """Менеджер для работы с PostgreSQL"""
    
    def __init__(self, dbname, user, password, host='localhost', port=5432):
        """Инициализация подключения к PostgreSQL"""
        try:
            self.connection = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )
            self.connection.autocommit = True
            logger.info("✅ Подключение к PostgreSQL установлено")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
            raise
    
    def disconnect(self):
        """Закрывает подключение к БД"""
        if self.connection:
            self.connection.close()
            logger.info("📤 Подключение к PostgreSQL закрыто")
    
    def execute_query(self, query: str, params=None, fetch=False):
        """
        Выполняет SQL запрос
        
        Args:
            query: SQL запрос
            params: Параметры запроса
            fetch: Нужно ли возвращать результат
            
        Returns:
            Результат запроса или None
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса: {e}")
            logger.error(f"Запрос: {query}")
            raise
    
    def fetch_dataframe(self, query: str, params=None):
        """
        Выполняет запрос и возвращает результат как pandas DataFrame
        
        Args:
            query: SQL запрос
            params: Параметры запроса
            
        Returns:
            pandas DataFrame с результатами
        """
        try:
            return pd.read_sql_query(query, self.connection, params=params)
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения запроса: {e}")
            raise
    
    def __enter__(self):
        """Поддержка контекстного менеджера"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрытие подключения при выходе из контекста"""
        self.disconnect()