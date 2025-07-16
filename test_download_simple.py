#!/usr/bin/env python3
print("Script started!")

# Простой тест без логирования
import yaml

print("Loading config...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Database: {config['database']['dbname']}")
print(f"Symbols to download: {len(config['data_download']['symbols'])}")
print(f"Max workers: {config['data_download']['max_workers']}")

# Теперь попробуем импортировать download_data
print("\nTrying to run download_data.main()...")

# Временно перенаправим логирование на консоль
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

# Импортируем после настройки логирования
import download_data
download_data.main()