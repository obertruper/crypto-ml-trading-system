#!/usr/bin/env python3
"""
Простой тест исправления VWAP
"""

import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine

# Загрузка конфигурации
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Подключение к БД
db_config = config['database']
engine = create_engine(
    f"postgresql://{db_config['user']}:{db_config['password']}@"
    f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
)

print("🔍 Тестирование исправления VWAP...")

# Загрузка данных
query = """
SELECT symbol, datetime, open, high, low, close, volume, turnover
FROM raw_market_data
WHERE symbol = 'BTCUSDT'
ORDER BY datetime DESC
LIMIT 100
"""

df = pd.read_sql(query, engine)
print(f"\n📊 Загружено {len(df)} записей")

# Применяем feature engineering
from data.feature_engineering import FeatureEngineer
engineer = FeatureEngineer(config)

# Тестируем базовые признаки
df_featured = engineer._create_basic_features(df.copy())

# Проверяем результаты
print("\n📈 Результаты VWAP:")
print(f"VWAP min: {df_featured['vwap'].min():.2f}")
print(f"VWAP max: {df_featured['vwap'].max():.2f}")
print(f"VWAP mean: {df_featured['vwap'].mean():.2f}")

print("\n📊 Результаты close_vwap_ratio:")
print(f"Min: {df_featured['close_vwap_ratio'].min():.4f}")
print(f"Max: {df_featured['close_vwap_ratio'].max():.4f}")
print(f"Mean: {df_featured['close_vwap_ratio'].mean():.4f}")
print(f"Std: {df_featured['close_vwap_ratio'].std():.4f}")

# Проверка на проблемные значения
problematic = df_featured[df_featured['close_vwap_ratio'] > 2.0]
print(f"\n⚠️ Проблемных значений (>2.0): {len(problematic)}")

if len(problematic) == 0:
    print("✅ VWAP исправлен успешно! Все значения в норме.")
else:
    print("❌ Все еще есть проблемы с VWAP")
    print(problematic[['datetime', 'close', 'vwap', 'close_vwap_ratio']].head())