"""
Препроцессор данных для XGBoost v3.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from config import Config, FEATURE_GROUPS, EXCLUDE_COLUMNS, TECHNICAL_INDICATORS_BOUNDS
from config.constants import FILLNA_STRATEGIES, VALIDATION_PARAMS, DATA_LEAKAGE_PARAMS

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.feature_names = None
        self.binary_features = FEATURE_GROUPS['binary_features']
        
        # Группы монет для адаптивного отбора признаков
        self.coin_groups = {
            'majors': ['BTC', 'ETH'],
            'top_alts': ['BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'MATIC', 'LINK'],
            'memecoins': ['DOGE', 'SHIB', 'PEPE', 'BONK', 'WIF', 'FLOKI'],
            'defi': ['UNI', 'AAVE', 'MKR', 'COMP', 'CRV', 'SUSHI', 'LDO']
        }
        
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Основной метод предобработки данных
        
        Returns:
            X: признаки
            y_buy: целевая переменная для buy
            y_sell: целевая переменная для sell
        """
        logger.info("🔧 Начало предобработки данных...")
        
        # Оптимизация памяти
        df = self._optimize_memory(df)
        
        # Обработка пропусков
        df = self._handle_missing_values(df)
        
        # Обработка выбросов
        df = self._handle_outliers(df)
        
        # Подготовка признаков и целевых переменных
        X, y_buy, y_sell = self._prepare_features_and_targets(df)
        
        # Валидация
        self._validate_preprocessed_data(X, y_buy, y_sell)
        
        logger.info(f"✅ Предобработка завершена. Размер: {X.shape}")
        
        return X, y_buy, y_sell
        
    def split_data(self, X: pd.DataFrame, y_buy: pd.Series, y_sell: pd.Series) -> Dict:
        """Разделение данных на train/test с использованием ВРЕМЕННОГО разделения"""
        logger.info(f"📊 Разделение данных (test_size={self.config.training.validation_split})...")
        logger.info("⏰ ИСПОЛЬЗУЕМ ВРЕМЕННОЕ РАЗДЕЛЕНИЕ для борьбы с переобучением!")
        
        # Убеждаемся, что индексы синхронизированы
        X = X.reset_index(drop=True)
        y_buy = y_buy.reset_index(drop=True)
        y_sell = y_sell.reset_index(drop=True)
        
        # ВРЕМЕННОЕ РАЗДЕЛЕНИЕ - последние 20% данных для теста
        n_samples = len(X)
        test_size = self.config.training.validation_split
        train_size = int(n_samples * (1 - test_size))
        
        logger.info(f"   Всего образцов: {n_samples:,}")
        logger.info(f"   Train: первые {train_size:,} образцов ({(1-test_size)*100:.0f}%)")
        logger.info(f"   Test: последние {n_samples - train_size:,} образцов ({test_size*100:.0f}%)")
        
        # Разделяем по времени
        X_train = X.iloc[:train_size].copy()
        X_test = X.iloc[train_size:].copy()
        
        y_train_buy = y_buy.iloc[:train_size].copy()
        y_test_buy = y_buy.iloc[train_size:].copy()
        
        y_train_sell = y_sell.iloc[:train_size].copy()
        y_test_sell = y_sell.iloc[train_size:].copy()
        
        # Логируем статистику для проверки
        logger.info("\n📊 СТАТИСТИКА РАЗДЕЛЕНИЯ:")
        
        if self.config.training.task_type != "regression":
            # Для классификации показываем баланс классов
            y_buy_binary_train = (y_train_buy > self.config.training.classification_threshold).astype(int)
            y_buy_binary_test = (y_test_buy > self.config.training.classification_threshold).astype(int)
            
            logger.info(f"   Buy Train: {y_buy_binary_train.sum():,} положительных ({y_buy_binary_train.mean()*100:.1f}%)")
            logger.info(f"   Buy Test: {y_buy_binary_test.sum():,} положительных ({y_buy_binary_test.mean()*100:.1f}%)")
        else:
            # Для регрессии показываем распределение
            logger.info(f"   Buy Train: mean={y_train_buy.mean():.2f}, std={y_train_buy.std():.2f}")
            logger.info(f"   Buy Test: mean={y_test_buy.mean():.2f}, std={y_test_buy.std():.2f}")
            
        return {
            'buy': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train_buy,
                'y_test': y_test_buy
            },
            'sell': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train_sell,
                'y_test': y_test_sell
            }
        }
        
    def normalize_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Нормализация признаков"""
        logger.info("🔄 Нормализация признаков...")
        
        # Разделяем бинарные и непрерывные признаки
        binary_cols = [col for col in self.binary_features if col in X_train.columns]
        continuous_cols = [col for col in X_train.columns if col not in binary_cols]
        
        logger.info(f"   📊 Бинарных признаков: {len(binary_cols)}")
        logger.info(f"   📊 Непрерывных признаков: {len(continuous_cols)}")
        
        # Нормализуем только непрерывные признаки
        if continuous_cols:
            self.scaler = RobustScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[continuous_cols] = self.scaler.fit_transform(X_train[continuous_cols])
            X_test_scaled[continuous_cols] = self.scaler.transform(X_test[continuous_cols])
            
            return X_train_scaled, X_test_scaled
        else:
            return X_train, X_test
            
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизация использования памяти"""
        logger.info("💾 Оптимизация памяти...")
        
        start_mem = df.memory_usage().sum() / 1024**2
        
        # Оптимизация числовых типов
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        # Категориальные переменные
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype('category')
            
        end_mem = df.memory_usage().sum() / 1024**2
        
        logger.info(f"   Память уменьшена с {start_mem:.1f} MB до {end_mem:.1f} MB ({(start_mem-end_mem)/start_mem*100:.1f}% экономии)")
        
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений"""
        missing_counts = df.isnull().sum()
        
        if missing_counts.sum() > 0:
            logger.info(f"🔧 Обработка {missing_counts.sum()} пропущенных значений...")
            
            # Используем стратегии заполнения из констант
            fill_strategies = FILLNA_STRATEGIES
            
            # Заполняем пропуски в технических индикаторах
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().any():
                    if col in fill_strategies:
                        # Используем стратегию из констант
                        df[col] = df[col].fillna(fill_strategies[col])
                    else:
                        # Для остальных - медиана или 0
                        median_val = df[col].median()
                        if pd.notna(median_val):
                            df[col] = df[col].fillna(median_val)
                        else:
                            df[col] = df[col].fillna(0)
                            
            # Проверка после заполнения
            remaining_nans = df.isnull().sum().sum()
            if remaining_nans > 0:
                logger.warning(f"⚠️ Остались NaN после обработки: {remaining_nans}")
                # Финальное заполнение - разные стратегии для разных типов
                for col in df.columns:
                    if df[col].isnull().any():
                        if df[col].dtype == 'category':
                            # Для категориальных - заполняем модой или 'unknown'
                            mode = df[col].mode()
                            if len(mode) > 0:
                                df[col] = df[col].fillna(mode[0])
                            else:
                                df[col] = df[col].cat.add_categories(['unknown']).fillna('unknown')
                        elif df[col].dtype in ['object']:
                            # Для строковых - заполняем пустой строкой
                            df[col] = df[col].fillna('')
                        else:
                            # Для числовых - заполняем нулями
                            df[col] = df[col].fillna(0)
                        
        return df
        
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка выбросов"""
        logger.info("📊 Обработка выбросов...")
        
        # Ограничиваем технические индикаторы их допустимыми диапазонами
        for indicator, (min_val, max_val) in TECHNICAL_INDICATORS_BOUNDS.items():
            if indicator in df.columns:
                original_outliers = ((df[indicator] < min_val) | (df[indicator] > max_val)).sum()
                if original_outliers > 0:
                    df[indicator] = df[indicator].clip(min_val, max_val)
                    logger.info(f"   📊 {indicator}: ограничено {original_outliers} значений в диапазон [{min_val}, {max_val}]")
                    
        return df
        
    def _prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Подготовка признаков и целевых переменных"""
        logger.info("📊 Подготовка признаков и целевых переменных...")
        
        # Целевые переменные
        if 'buy_expected_return' not in df.columns or 'sell_expected_return' not in df.columns:
            raise ValueError("Отсутствуют целевые переменные")
            
        y_buy = df['buy_expected_return'].copy()
        y_sell = df['sell_expected_return'].copy()
        
        # Статистика целевых переменных
        logger.info("📊 ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ:")
        logger.info(f"   buy_expected_return: min={y_buy.min():.2f}, max={y_buy.max():.2f}, mean={y_buy.mean():.2f}, std={y_buy.std():.2f}")
        logger.info(f"   sell_expected_return: min={y_sell.min():.2f}, max={y_sell.max():.2f}, mean={y_sell.mean():.2f}, std={y_sell.std():.2f}")
        
        # Признаки
        feature_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
        X = df[feature_columns].copy()
        
        # Удаляем константные признаки
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
                
        if constant_features:
            logger.warning(f"⚠️ Обнаружено {len(constant_features)} константных признаков, удаляем: {constant_features[:5]}...")
            X = X.drop(columns=constant_features)
        
        # Сохраняем имена признаков
        self.feature_names = list(X.columns)
        
        logger.info(f"✅ Подготовлено {len(X.columns)} признаков")
        
        return X, y_buy, y_sell
        
    def _validate_preprocessed_data(self, X: pd.DataFrame, y_buy: pd.Series, y_sell: pd.Series):
        """Валидация предобработанных данных"""
        logger.info("🔍 Валидация предобработанных данных...")
        
        # Проверка размерностей
        assert len(X) == len(y_buy) == len(y_sell), "Несоответствие размерностей"
        
        # Проверка на NaN с диагностикой
        nan_cols = X.columns[X.isnull().any()].tolist()
        if nan_cols:
            logger.error(f"❌ Обнаружены NaN в признаках: {nan_cols}")
            for col in nan_cols[:5]:  # Показываем первые 5
                nan_count = X[col].isnull().sum()
                logger.error(f"   {col}: {nan_count} NaN значений")
        
        assert not X.isnull().any().any(), "Обнаружены NaN в признаках"
        assert not y_buy.isnull().any(), "Обнаружены NaN в y_buy"
        assert not y_sell.isnull().any(), "Обнаружены NaN в y_sell"
        
        # Проверка на бесконечности
        assert not np.isinf(X.select_dtypes(include=[np.number])).any().any(), "Обнаружены бесконечности в признаках"
        
        # Проверка на утечку данных
        self._check_data_leakage(X, y_buy, y_sell)
        
        logger.info("✅ Валидация пройдена успешно")
        
    def _check_data_leakage(self, X: pd.DataFrame, y_buy: pd.Series, y_sell: pd.Series):
        """Проверка на утечку данных"""
        logger.info("🔍 ПРОВЕРКА НА УТЕЧКУ ДАННЫХ:")
        
        # Проверяем корреляцию признаков с целевыми переменными
        high_corr_features = []
        max_correlation = VALIDATION_PARAMS['max_feature_target_correlation']
        
        # Определяем количество признаков для проверки
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        if DATA_LEAKAGE_PARAMS['check_all_features']:
            columns_to_check = numeric_columns
        else:
            n_features = min(DATA_LEAKAGE_PARAMS['check_n_features'], len(numeric_columns))
            if DATA_LEAKAGE_PARAMS['random_sample'] and len(numeric_columns) > n_features:
                # Случайная выборка
                np.random.seed(42)
                columns_to_check = np.random.choice(numeric_columns, n_features, replace=False)
            else:
                # Первые n признаков
                columns_to_check = numeric_columns[:n_features]
        
        logger.info(f"   📋 Проверяем {len(columns_to_check)} из {len(numeric_columns)} признаков")
        
        for col in columns_to_check:
            corr_buy = X[col].corr(y_buy)
            corr_sell = X[col].corr(y_sell)
            
            if abs(corr_buy) > max_correlation or abs(corr_sell) > max_correlation:
                high_corr_features.append(col)
                logger.warning(f"   ⚠️ Высокая корреляция для {col}: buy={corr_buy:.3f}, sell={corr_sell:.3f}")
            else:
                logger.info(f"   ✅ Признак {col}: corr_buy={corr_buy:.3f}, corr_sell={corr_sell:.3f}")
                
        if high_corr_features:
            logger.warning(f"⚠️ Обнаружены признаки с высокой корреляцией: {high_corr_features}")
            
    def transform_to_classification_labels(self, y_buy: pd.Series, y_sell: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Преобразование в метки для классификации"""
        if self.config.training.task_type == "classification_binary":
            threshold = self.config.training.classification_threshold
            logger.info(f"🔄 Преобразование в метки классификации (порог > {threshold}%)...")
            
            y_buy_binary = (y_buy > threshold).astype(int)
            y_sell_binary = (y_sell > threshold).astype(int)
            
            # Статистика
            logger.info(f"📊 Статистика бинарных меток (порог > {threshold}%):")
            logger.info(f"   Buy - Класс 1 (входить): {y_buy_binary.mean()*100:.1f}%")
            logger.info(f"   Sell - Класс 1 (входить): {y_sell_binary.mean()*100:.1f}%")
            
            # Детальная статистика
            logger.info(f"\n📈 Детальная статистика expected_return:")
            logger.info(f"   Buy > 0%: {(y_buy > 0).mean()*100:.1f}%, Buy > 0.5%: {(y_buy > 0.5).mean()*100:.1f}%, Buy > 1%: {(y_buy > 1).mean()*100:.1f}%")
            logger.info(f"   Sell > 0%: {(y_sell > 0).mean()*100:.1f}%, Sell > 0.5%: {(y_sell > 0.5).mean()*100:.1f}%, Sell > 1%: {(y_sell > 1).mean()*100:.1f}%")
            
            return y_buy_binary, y_sell_binary
            
        elif self.config.training.task_type == "classification_multi":
            # Мультиклассовая классификация
            thresholds = self.config.training.multiclass_thresholds
            logger.info(f"🔄 Преобразование в мультиклассовые метки (пороги: {thresholds})...")
            
            y_buy_multi = pd.cut(y_buy, bins=[-np.inf] + thresholds + [np.inf], labels=False)
            y_sell_multi = pd.cut(y_sell, bins=[-np.inf] + thresholds + [np.inf], labels=False)
            
            return y_buy_multi, y_sell_multi
            
        else:
            # Регрессия - возвращаем как есть
            return y_buy, y_sell
    
    def group_symbols(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Группирует данные по типам монет для адаптивного отбора признаков
        
        Returns:
            Словарь с DataFrame для каждой группы
        """
        grouped_data = {}
        
        # Получаем все уникальные символы
        all_symbols = df['symbol'].unique() if 'symbol' in df.columns else []
        
        # Заполняем группу 'other' остальными символами
        used_symbols = set()
        for group_symbols in self.coin_groups.values():
            used_symbols.update(group_symbols)
        
        self.coin_groups['other'] = [s for s in all_symbols if s not in used_symbols]
        
        # Создаем DataFrame для каждой группы
        for group_name, symbols in self.coin_groups.items():
            if symbols:  # Если есть символы в группе
                group_df = df[df['symbol'].isin(symbols)] if 'symbol' in df.columns else pd.DataFrame()
                if not group_df.empty:
                    grouped_data[group_name] = group_df
                    logger.info(f"   Группа '{group_name}': {len(symbols)} монет, {len(group_df)} записей")
        
        return grouped_data
    
    def get_group_weights(self, group_name: str) -> Dict[str, float]:
        """
        Возвращает веса категорий признаков для конкретной группы монет
        """
        # Адаптивные веса для разных групп
        group_weights = {
            'majors': {
                'technical': 0.70,    # Больше технических для стабильных
                'temporal': 0.15,
                'btc_related': 0.05,  # Меньше BTC для самого BTC
                'other': 0.10
            },
            'top_alts': {
                'technical': 0.60,    # Стандартное распределение
                'temporal': 0.20,
                'btc_related': 0.10,
                'other': 0.10
            },
            'memecoins': {
                'technical': 0.50,    # Меньше технических
                'temporal': 0.20,
                'btc_related': 0.20,  # Больше корреляции с BTC
                'other': 0.10
            },
            'defi': {
                'technical': 0.65,    # Больше технических
                'temporal': 0.15,
                'btc_related': 0.10,
                'other': 0.10
            },
            'other': {
                'technical': 0.60,    # Стандартное распределение
                'temporal': 0.20,
                'btc_related': 0.10,
                'other': 0.10
            }
        }
        
        return group_weights.get(group_name, group_weights['other'])