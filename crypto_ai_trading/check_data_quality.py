#!/usr/bin/env python3
"""
Проверка качества данных для выявления проблем
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger('DataQuality')

class DataQualityChecker:
    """Проверка качества данных"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.issues = []
        self.stats = {}
        
    def load_data(self):
        """Загрузка данных"""
        logger.info("📥 Загрузка данных...")
        
        self.train_data = pd.read_parquet(self.data_dir / "train_data.parquet")
        self.val_data = pd.read_parquet(self.data_dir / "val_data.parquet")
        self.test_data = pd.read_parquet(self.data_dir / "test_data.parquet")
        
        logger.info(f"✅ Загружено: train={len(self.train_data):,}, val={len(self.val_data):,}, test={len(self.test_data):,}")
        
    def check_missing_values(self):
        """Проверка пропущенных значений"""
        logger.info("🔍 Проверка пропущенных значений...")
        
        for name, data in [("train", self.train_data), ("val", self.val_data), ("test", self.test_data)]:
            missing = data.isnull().sum()
            missing_pct = (missing / len(data)) * 100
            
            if missing.any():
                logger.warning(f"⚠️ {name}: найдено {missing.sum()} пропущенных значений")
                top_missing = missing[missing > 0].sort_values(ascending=False).head(10)
                for col, count in top_missing.items():
                    logger.warning(f"    {col}: {count} ({missing_pct[col]:.2f}%)")
                    
                self.issues.append(f"{name}_missing_values")
            else:
                logger.info(f"✅ {name}: пропущенных значений нет")
                
    def check_infinite_values(self):
        """Проверка бесконечных значений"""
        logger.info("🔍 Проверка бесконечных значений...")
        
        for name, data in [("train", self.train_data), ("val", self.val_data), ("test", self.test_data)]:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            inf_mask = np.isinf(data[numeric_cols])
            inf_count = inf_mask.sum()
            
            if inf_count.any():
                logger.warning(f"⚠️ {name}: найдено {inf_count.sum()} бесконечных значений")
                top_inf = inf_count[inf_count > 0].sort_values(ascending=False).head(10)
                for col, count in top_inf.items():
                    logger.warning(f"    {col}: {count}")
                    
                self.issues.append(f"{name}_infinite_values")
            else:
                logger.info(f"✅ {name}: бесконечных значений нет")
                
    def check_data_distributions(self):
        """Проверка распределений данных"""
        logger.info("🔍 Проверка распределений данных...")
        
        # Определение групп признаков
        target_cols = [col for col in self.train_data.columns if any(
            x in col for x in ['future_return', 'direction_', 'will_reach', 'tp_time', 'sl_time']
        )]
        
        feature_cols = [col for col in self.train_data.columns if col not in target_cols and 
                       col not in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        # Статистика по признакам
        for name, data in [("train", self.train_data), ("val", self.val_data), ("test", self.test_data)]:
            numeric_data = data[feature_cols].select_dtypes(include=[np.number])
            
            stats = {
                'mean': numeric_data.mean(),
                'std': numeric_data.std(),
                'min': numeric_data.min(),
                'max': numeric_data.max(),
                'skew': numeric_data.skew(),
                'kurtosis': numeric_data.kurtosis()
            }
            
            self.stats[name] = stats
            
            # Проверка на экстремальные значения
            extreme_skew = stats['skew'][abs(stats['skew']) > 3]
            if len(extreme_skew) > 0:
                logger.warning(f"⚠️ {name}: {len(extreme_skew)} признаков с экстремальной асимметрией")
                for col in extreme_skew.index[:5]:
                    logger.warning(f"    {col}: skew={stats['skew'][col]:.2f}")
                    
            extreme_kurt = stats['kurtosis'][abs(stats['kurtosis']) > 10]
            if len(extreme_kurt) > 0:
                logger.warning(f"⚠️ {name}: {len(extreme_kurt)} признаков с экстремальным эксцессом")
                
    def check_target_balance(self):
        """Проверка баланса целевых переменных"""
        logger.info("🔍 Проверка баланса целевых переменных...")
        
        # Определение целевых переменных v4.0
        regression_targets = ['future_return_15m', 'future_return_1h', 
                            'future_return_4h', 'future_return_12h']
        
        categorical_targets = ['direction_15m', 'direction_1h', 
                             'direction_4h', 'direction_12h']
        
        binary_targets = [col for col in self.train_data.columns if 'will_reach' in col]
        
        # Проверка регрессионных таргетов
        logger.info("📊 Статистика регрессионных таргетов:")
        for target in regression_targets:
            if target in self.train_data.columns:
                train_mean = self.train_data[target].mean()
                val_mean = self.val_data[target].mean()
                test_mean = self.test_data[target].mean()
                
                logger.info(f"  {target}: train={train_mean:.6f}, val={val_mean:.6f}, test={test_mean:.6f}")
                
                # Проверка на сдвиг распределения
                if abs(train_mean - val_mean) > 0.001 or abs(train_mean - test_mean) > 0.001:
                    logger.warning(f"    ⚠️ Обнаружен сдвиг в распределении!")
                    self.issues.append(f"distribution_shift_{target}")
                    
        # Проверка категориальных таргетов
        logger.info("📊 Баланс категориальных таргетов:")
        for target in categorical_targets:
            if target in self.train_data.columns:
                train_dist = self.train_data[target].value_counts(normalize=True)
                logger.info(f"  {target}:")
                for cat, pct in train_dist.items():
                    logger.info(f"    {cat}: {pct:.2%}")
                    
        # Проверка бинарных таргетов
        logger.info("📊 Баланс бинарных таргетов:")
        for target in binary_targets[:5]:  # Первые 5 для примера
            if target in self.train_data.columns:
                positive_pct = self.train_data[target].mean()
                logger.info(f"  {target}: {positive_pct:.2%} positive")
                
                if positive_pct < 0.05 or positive_pct > 0.95:
                    logger.warning(f"    ⚠️ Сильный дисбаланс классов!")
                    self.issues.append(f"class_imbalance_{target}")
                    
    def check_feature_correlations(self):
        """Проверка корреляций между признаками"""
        logger.info("🔍 Проверка корреляций между признаками...")
        
        # Выбираем числовые признаки
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not any(
            x in col for x in ['future_return', 'direction_', 'will_reach', 'datetime']
        )]
        
        # Вычисляем корреляционную матрицу
        corr_matrix = self.train_data[feature_cols[:50]].corr()  # Первые 50 для скорости
        
        # Находим высокие корреляции
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.95:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
                    
        if high_corr:
            logger.warning(f"⚠️ Найдено {len(high_corr)} пар признаков с корреляцией > 0.95:")
            for col1, col2, corr in high_corr[:10]:
                logger.warning(f"    {col1} <-> {col2}: {corr:.3f}")
            self.issues.append("high_feature_correlation")
        else:
            logger.info("✅ Высоких корреляций между признаками не обнаружено")
            
    def check_temporal_consistency(self):
        """Проверка временной консистентности"""
        logger.info("🔍 Проверка временной консистентности...")
        
        # Проверка порядка дат
        for name, data in [("train", self.train_data), ("val", self.val_data), ("test", self.test_data)]:
            if 'datetime' in data.columns:
                # Конвертируем в datetime если нужно
                if data['datetime'].dtype == 'object':
                    data['datetime'] = pd.to_datetime(data['datetime'])
                    
                # Проверка монотонности
                is_sorted = data['datetime'].is_monotonic_increasing
                if not is_sorted:
                    logger.warning(f"⚠️ {name}: данные не отсортированы по времени!")
                    self.issues.append(f"{name}_not_sorted")
                    
                # Проверка пропусков во времени
                time_diffs = data.groupby('symbol')['datetime'].diff()
                expected_diff = pd.Timedelta(minutes=15)  # Для 15-минутных данных
                
                gaps = time_diffs[time_diffs > expected_diff * 2]
                if len(gaps) > 0:
                    logger.warning(f"⚠️ {name}: найдено {len(gaps)} временных пропусков")
                    
    def check_data_leakage(self):
        """Проверка на утечки данных между выборками"""
        logger.info("🔍 Проверка на утечки данных...")
        
        # Проверка пересечения по времени
        if 'datetime' in self.train_data.columns:
            train_max = self.train_data['datetime'].max()
            val_min = self.val_data['datetime'].min()
            val_max = self.val_data['datetime'].max()
            test_min = self.test_data['datetime'].min()
            
            if train_max >= val_min:
                logger.error("❌ Обнаружено пересечение train и val по времени!")
                self.issues.append("train_val_overlap")
                
            if val_max >= test_min:
                logger.error("❌ Обнаружено пересечение val и test по времени!")
                self.issues.append("val_test_overlap")
                
            # Проверка временного gap
            train_val_gap = (val_min - train_max).days
            val_test_gap = (test_min - val_max).days
            
            logger.info(f"📅 Временные промежутки:")
            logger.info(f"  Train → Val: {train_val_gap} дней")
            logger.info(f"  Val → Test: {val_test_gap} дней")
            
            if train_val_gap < 1 or val_test_gap < 1:
                logger.warning("⚠️ Слишком маленький временной gap между выборками!")
                
    def generate_report(self, save_path: str = "data_quality_report.txt"):
        """Генерация отчета"""
        logger.info("📄 Генерация отчета...")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"📊 Размеры датасетов:\n")
            f.write(f"  Train: {len(self.train_data):,} записей\n")
            f.write(f"  Val: {len(self.val_data):,} записей\n")
            f.write(f"  Test: {len(self.test_data):,} записей\n\n")
            
            if self.issues:
                f.write(f"⚠️ Обнаружено проблем: {len(self.issues)}\n")
                for issue in self.issues:
                    f.write(f"  - {issue}\n")
            else:
                f.write("✅ Критических проблем не обнаружено\n")
                
            f.write("\n" + "=" * 80 + "\n")
            f.write("РЕКОМЕНДАЦИИ:\n")
            f.write("=" * 80 + "\n")
            
            if "high_feature_correlation" in self.issues:
                f.write("• Удалить высоко коррелированные признаки\n")
            if any("class_imbalance" in issue for issue in self.issues):
                f.write("• Использовать взвешенную loss функцию для несбалансированных классов\n")
            if any("distribution_shift" in issue for issue in self.issues):
                f.write("• Проверить процесс разделения данных на выборки\n")
            if any("missing_values" in issue for issue in self.issues):
                f.write("• Обработать пропущенные значения\n")
                
        logger.info(f"✅ Отчет сохранен в {save_path}")
        

def main():
    """Главная функция"""
    checker = DataQualityChecker()
    
    # Загрузка данных
    checker.load_data()
    
    # Проверки
    checker.check_missing_values()
    checker.check_infinite_values()
    checker.check_data_distributions()
    checker.check_target_balance()
    checker.check_feature_correlations()
    checker.check_temporal_consistency()
    checker.check_data_leakage()
    
    # Генерация отчета
    checker.generate_report()
    
    logger.info("✅ Проверка качества данных завершена!")
    

if __name__ == "__main__":
    main()