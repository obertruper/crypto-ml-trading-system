"""
XGBoost v3.0 - Модульная система для криптотрейдинга
====================================================

Чистая архитектура с разделением ответственности:
- config: Конфигурация и настройки
- data: Загрузка и подготовка данных
- models: Модели и обучение
- utils: Вспомогательные утилиты
"""

__version__ = "3.0.0"
__author__ = "Ruslan"

from config import Config
from data import DataLoader, FeatureEngineer
from models import XGBoostTrainer
from utils import MetricsCalculator, Visualizer

__all__ = [
    "Config",
    "DataLoader", 
    "FeatureEngineer",
    "XGBoostTrainer",
    "MetricsCalculator",
    "Visualizer"
]