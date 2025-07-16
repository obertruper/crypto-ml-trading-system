"""
Расчет упрощенных таргетов для обучения XGBoost модели.
Заменяет сложный расчет expected returns на более простые и предсказуемые таргеты.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TargetType(Enum):
    """Типы таргетов для модели"""
    SIMPLE_BINARY = "simple_binary"
    THRESHOLD_BINARY = "threshold_binary"
    DIRECTION_MULTICLASS = "direction_multiclass"
    SIMPLE_REGRESSION = "simple_regression"


class TargetCalculator:
    """Калькулятор упрощенных таргетов для ML модели"""
    
    def __init__(self, 
                 lookahead_bars: int = 4,  # 4 бара = 1 час для 15-минутных данных
                 price_threshold: float = 0.5,  # 0.5% для threshold_binary
                 multiclass_thresholds: Dict[str, float] = None):
        """
        Инициализация калькулятора таргетов.
        
        Args:
            lookahead_bars: количество баров вперед для расчета таргета
            price_threshold: процентный порог для threshold_binary
            multiclass_thresholds: пороги для мультиклассовой классификации
        """
        self.lookahead_bars = lookahead_bars
        self.price_threshold = price_threshold
        
        # Пороги для мультиклассовой классификации
        self.multiclass_thresholds = multiclass_thresholds or {
            'strong_up': 1.0,      # > 1%
            'weak_up': 0.3,        # 0.3% - 1%
            'neutral': -0.3,       # -0.3% - 0.3%
            'weak_down': -1.0,     # -1% - -0.3%
            'strong_down': -999    # < -1%
        }
        
        logger.info(f"Инициализирован TargetCalculator с параметрами:")
        logger.info(f"  - lookahead_bars: {self.lookahead_bars}")
        logger.info(f"  - price_threshold: {self.price_threshold}%")
        logger.info(f"  - multiclass_thresholds: {self.multiclass_thresholds}")
    
    def calculate_all_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает все типы таргетов для датафрейма.
        
        Args:
            df: исходный датафрейм с OHLCV данными
            
        Returns:
            DataFrame с добавленными колонками таргетов
        """
        logger.info(f"Расчет всех таргетов для {len(df)} записей")
        
        # Создаем копию для работы
        result_df = df.copy()
        
        # Рассчитываем процентное изменение цены
        price_change = self._calculate_price_change(result_df)
        
        # 1. Simple Binary - простое направление
        result_df['target_simple_binary'] = self.calculate_simple_binary(price_change)
        
        # 2. Threshold Binary - с порогом
        result_df['target_threshold_binary'] = self.calculate_threshold_binary(
            price_change, self.price_threshold
        )
        
        # 3. Direction Multiclass - мультиклассовая
        result_df['target_direction_multiclass'] = self.calculate_direction_multiclass(
            price_change
        )
        
        # 4. Simple Regression - процентное изменение
        result_df['target_simple_regression'] = self.calculate_simple_regression(
            price_change
        )
        
        # Добавляем вспомогательные колонки
        result_df['price_change_percent'] = price_change
        result_df['future_close'] = result_df['close'].shift(-self.lookahead_bars)
        
        # Статистика по таргетам
        self._log_target_statistics(result_df)
        
        return result_df
    
    def _calculate_price_change(self, df: pd.DataFrame) -> pd.Series:
        """
        Рассчитывает процентное изменение цены через N баров.
        
        Args:
            df: датафрейм с колонкой 'close'
            
        Returns:
            Series с процентными изменениями
        """
        current_price = df['close']
        future_price = df['close'].shift(-self.lookahead_bars)
        
        # Процентное изменение
        price_change = ((future_price - current_price) / current_price) * 100
        
        return price_change
    
    def calculate_simple_binary(self, price_change: pd.Series) -> pd.Series:
        """
        Простая бинарная классификация: рост (1) или падение (0).
        
        Args:
            price_change: процентное изменение цены
            
        Returns:
            Series с бинарными метками
        """
        return (price_change > 0).astype(int)
    
    def calculate_threshold_binary(self, price_change: pd.Series, 
                                 threshold: float) -> pd.Series:
        """
        Бинарная классификация с порогом: значительный рост (1) или нет (0).
        
        Args:
            price_change: процентное изменение цены
            threshold: минимальный процент для положительного класса
            
        Returns:
            Series с бинарными метками
        """
        return (price_change > threshold).astype(int)
    
    def calculate_direction_multiclass(self, price_change: pd.Series) -> pd.Series:
        """
        Мультиклассовая классификация направления и силы движения.
        
        Классы:
        0 - сильное падение (< -1%)
        1 - слабое падение (-1% до -0.3%)
        2 - нейтрально (-0.3% до 0.3%)
        3 - слабый рост (0.3% до 1%)
        4 - сильный рост (> 1%)
        
        Args:
            price_change: процентное изменение цены
            
        Returns:
            Series с метками классов
        """
        conditions = [
            price_change < self.multiclass_thresholds['weak_down'],      # сильное падение
            price_change < self.multiclass_thresholds['neutral'],        # слабое падение
            price_change < self.multiclass_thresholds['weak_up'],         # нейтрально
            price_change < self.multiclass_thresholds['strong_up'],       # слабый рост
            price_change >= self.multiclass_thresholds['strong_up']       # сильный рост
        ]
        
        choices = [0, 1, 2, 3, 4]
        
        return pd.Series(np.select(conditions, choices), index=price_change.index)
    
    def calculate_simple_regression(self, price_change: pd.Series) -> pd.Series:
        """
        Простая регрессия на процентное изменение цены.
        
        Args:
            price_change: процентное изменение цены
            
        Returns:
            Series с процентными изменениями (может быть нормализована)
        """
        # Возвращаем как есть, можно добавить нормализацию если нужно
        return price_change
    
    def calculate_custom_target(self, df: pd.DataFrame, 
                              target_function: callable) -> pd.Series:
        """
        Рассчитывает кастомный таргет с помощью переданной функции.
        
        Args:
            df: исходный датафрейм
            target_function: функция для расчета таргета
            
        Returns:
            Series с кастомными таргетами
        """
        return target_function(df)
    
    def _log_target_statistics(self, df: pd.DataFrame):
        """Логирует статистику по рассчитанным таргетам"""
        
        # Убираем NaN значения для статистики
        valid_mask = ~df['price_change_percent'].isna()
        
        logger.info("\n=== Статистика таргетов ===")
        
        # Simple Binary
        if 'target_simple_binary' in df.columns:
            binary_stats = df.loc[valid_mask, 'target_simple_binary'].value_counts()
            logger.info(f"\nSimple Binary (рост/падение):")
            logger.info(f"  - Падение (0): {binary_stats.get(0, 0)} ({binary_stats.get(0, 0)/len(df[valid_mask])*100:.1f}%)")
            logger.info(f"  - Рост (1): {binary_stats.get(1, 0)} ({binary_stats.get(1, 0)/len(df[valid_mask])*100:.1f}%)")
        
        # Threshold Binary
        if 'target_threshold_binary' in df.columns:
            threshold_stats = df.loc[valid_mask, 'target_threshold_binary'].value_counts()
            logger.info(f"\nThreshold Binary (рост > {self.price_threshold}%):")
            logger.info(f"  - Нет (0): {threshold_stats.get(0, 0)} ({threshold_stats.get(0, 0)/len(df[valid_mask])*100:.1f}%)")
            logger.info(f"  - Да (1): {threshold_stats.get(1, 0)} ({threshold_stats.get(1, 0)/len(df[valid_mask])*100:.1f}%)")
        
        # Direction Multiclass
        if 'target_direction_multiclass' in df.columns:
            multiclass_stats = df.loc[valid_mask, 'target_direction_multiclass'].value_counts().sort_index()
            logger.info(f"\nDirection Multiclass:")
            class_names = ['Сильное падение', 'Слабое падение', 'Нейтрально', 'Слабый рост', 'Сильный рост']
            for i, name in enumerate(class_names):
                count = multiclass_stats.get(i, 0)
                logger.info(f"  - {name} ({i}): {count} ({count/len(df[valid_mask])*100:.1f}%)")
        
        # Simple Regression
        if 'target_simple_regression' in df.columns:
            regression_data = df.loc[valid_mask, 'target_simple_regression']
            logger.info(f"\nSimple Regression (процентное изменение):")
            logger.info(f"  - Среднее: {regression_data.mean():.3f}%")
            logger.info(f"  - Медиана: {regression_data.median():.3f}%")
            logger.info(f"  - Std: {regression_data.std():.3f}%")
            logger.info(f"  - Min: {regression_data.min():.3f}%")
            logger.info(f"  - Max: {regression_data.max():.3f}%")
            logger.info(f"  - 25%: {regression_data.quantile(0.25):.3f}%")
            logger.info(f"  - 75%: {regression_data.quantile(0.75):.3f}%")
    
    def get_balanced_sample(self, df: pd.DataFrame, 
                          target_column: str,
                          sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Получает сбалансированную выборку для обучения.
        
        Args:
            df: исходный датафрейм
            target_column: название колонки с таргетом
            sample_size: размер выборки для каждого класса
            
        Returns:
            Сбалансированный датафрейм
        """
        # Убираем NaN значения
        valid_df = df.dropna(subset=[target_column])
        
        # Группируем по классам
        grouped = valid_df.groupby(target_column)
        
        # Определяем размер выборки
        if sample_size is None:
            sample_size = grouped.size().min()
        
        # Берем равное количество примеров из каждого класса
        balanced_dfs = []
        for target_value, group in grouped:
            if len(group) >= sample_size:
                balanced_dfs.append(group.sample(n=sample_size, random_state=42))
            else:
                # Если примеров меньше, берем все
                balanced_dfs.append(group)
        
        # Объединяем и перемешиваем
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Логируем статистику
        logger.info(f"\nСбалансированная выборка для '{target_column}':")
        logger.info(f"  - Исходный размер: {len(valid_df)}")
        logger.info(f"  - Сбалансированный размер: {len(balanced_df)}")
        logger.info(f"  - Распределение классов:")
        for value, count in balanced_df[target_column].value_counts().sort_index().items():
            logger.info(f"    * Класс {value}: {count} ({count/len(balanced_df)*100:.1f}%)")
        
        return balanced_df


def compare_targets(df: pd.DataFrame, target_columns: list) -> pd.DataFrame:
    """
    Сравнивает корреляцию между различными типами таргетов.
    
    Args:
        df: датафрейм с таргетами
        target_columns: список колонок с таргетами для сравнения
        
    Returns:
        DataFrame с корреляциями
    """
    # Фильтруем только существующие колонки
    existing_columns = [col for col in target_columns if col in df.columns]
    
    # Убираем NaN значения
    valid_df = df[existing_columns].dropna()
    
    # Рассчитываем корреляции
    correlation_matrix = valid_df.corr()
    
    logger.info("\n=== Корреляции между таргетами ===")
    logger.info(correlation_matrix.to_string())
    
    return correlation_matrix


# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Создаем тестовые данные
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    
    # Симулируем ценовые данные с трендом и шумом
    trend = np.linspace(100, 110, 1000)
    noise = np.random.normal(0, 2, 1000)
    prices = trend + noise
    
    test_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.normal(0, 0.5, 1000),
        'high': prices + np.abs(np.random.normal(0.5, 0.5, 1000)),
        'low': prices - np.abs(np.random.normal(0.5, 0.5, 1000)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Создаем калькулятор
    calculator = TargetCalculator(
        lookahead_bars=4,  # 1 час для 15-минутных данных
        price_threshold=0.5,  # 0.5% порог
    )
    
    # Рассчитываем таргеты
    df_with_targets = calculator.calculate_all_targets(test_df)
    
    # Сравниваем таргеты
    target_cols = [
        'target_simple_binary',
        'target_threshold_binary',
        'target_direction_multiclass',
        'target_simple_regression'
    ]
    
    correlations = compare_targets(df_with_targets, target_cols)
    
    # Получаем сбалансированную выборку для бинарной классификации
    balanced_df = calculator.get_balanced_sample(
        df_with_targets, 
        'target_simple_binary'
    )
    
    logger.info("\n=== Готово! ===")
    logger.info(f"Рассчитано таргетов для {len(df_with_targets)} записей")
    logger.info(f"Последние строки с таргетами удалены (NaN): {len(test_df) - len(df_with_targets.dropna())}")