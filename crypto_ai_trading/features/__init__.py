"""
Features модуль - Расширенные признаки для модели прогнозирования
Этапная реализация с защитой от утечек данных
"""

from .temporal import TemporalEmbeddings
from .market_context import MarketContextFeatures  
from .normalization import AdaptiveNormalization

__all__ = [
    'TemporalEmbeddings',
    'MarketContextFeatures', 
    'AdaptiveNormalization'
]