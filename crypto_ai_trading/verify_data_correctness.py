#!/usr/bin/env python3
"""
Проверка корректности данных версии 4.0 - с фокусом на утечки и переобучение
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Определяем константы локально для версии 4.0 БЕЗ УТЕЧЕК
TRADING_TARGET_VARIABLES = [
    # A. Базовые возвраты (4)
    'future_return_15m',   # через 1 свечу (15 минут)
    'future_return_1h',    # через 4 свечи (1 час) 
    'future_return_4h',    # через 16 свечей (4 часа)
    'future_return_12h',   # через 48 свечей (12 часов)
    
    # B. Направление движения (4)
    'direction_15m',       # UP/DOWN/FLAT
    'direction_1h',        
    'direction_4h',        
    'direction_12h',       
    
    # C. Достижение уровней прибыли LONG (4)
    'long_will_reach_1pct_4h',   
    'long_will_reach_2pct_4h',   
    'long_will_reach_3pct_12h',  
    'long_will_reach_5pct_12h',  
    
    # D. Достижение уровней прибыли SHORT (4)
    'short_will_reach_1pct_4h',   
    'short_will_reach_2pct_4h',   
    'short_will_reach_3pct_12h',  
    'short_will_reach_5pct_12h',  
    
    # E. Риск-метрики (4)
    'max_drawdown_1h',     
    'max_rally_1h',        
    'max_drawdown_4h',     
    'max_rally_4h'        
    
    # УДАЛЕНО: best_action, signal_strength, risk_reward_ratio, optimal_hold_time
    # Эти переменные содержали утечки данных
]

# Разделение целевых переменных по типам
REGRESSION_TARGETS = [
    'future_return_15m', 'future_return_1h', 'future_return_4h', 'future_return_12h',
    'max_drawdown_1h', 'max_rally_1h', 'max_drawdown_4h', 'max_rally_4h'
]

BINARY_TARGETS = [
    'long_will_reach_1pct_4h', 'long_will_reach_2pct_4h', 
    'long_will_reach_3pct_12h', 'long_will_reach_5pct_12h',
    'short_will_reach_1pct_4h', 'short_will_reach_2pct_4h',
    'short_will_reach_3pct_12h', 'short_will_reach_5pct_12h'
]

MULTICLASS_TARGETS = [
    'direction_15m', 'direction_1h', 'direction_4h', 'direction_12h'
]

SERVICE_COLUMNS = ['datetime', 'symbol', 'id', 'timestamp']

TARGET_GROUPS = {
    'returns': ['future_return_15m', 'future_return_1h', 'future_return_4h', 'future_return_12h'],
    'directions': ['direction_15m', 'direction_1h', 'direction_4h', 'direction_12h'],
    'long_profits': ['long_will_reach_1pct_4h', 'long_will_reach_2pct_4h', 
                     'long_will_reach_3pct_12h', 'long_will_reach_5pct_12h'],
    'short_profits': ['short_will_reach_1pct_4h', 'short_will_reach_2pct_4h',
                      'short_will_reach_3pct_12h', 'short_will_reach_5pct_12h'],
    'risk_metrics': ['max_drawdown_1h', 'max_rally_1h', 'max_drawdown_4h', 'max_rally_4h']
}

# Цветной вывод
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKBLUE}ℹ️  {text}{Colors.ENDC}")


class DataCorrectnessVerifier:
    """Класс для комплексной проверки корректности данных v4.0"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self):
        """Загрузка всех датасетов"""
        files = {
            'train': 'data/processed/train_data.parquet',
            'val': 'data/processed/val_data.parquet', 
            'test': 'data/processed/test_data.parquet'
        }
        
        for name, path in files.items():
            file_path = Path(path)
            if not file_path.exists():
                print_error(f"Файл {path} не найден!")
                self.issues.append(f"{name}_file_missing")
                continue
                
            data = pd.read_parquet(file_path)
            setattr(self, f"{name}_data", data)
            print_success(f"Загружен {name}: {len(data):,} записей")
            
    def check_data_leakage(self):
        """Проверка на утечки данных"""
        print_header("🔍 ПРОВЕРКА НА УТЕЧКИ ДАННЫХ (Data Leakage)")
        
        if self.train_data is None:
            print_error("Данные не загружены!")
            return
            
        # 1. Проверка временных утечек (look-ahead bias)
        print_info("\n1. Проверка look-ahead bias в признаках...")
        
        # Проверяем что в признаках нет будущих данных
        # Исключения для индикаторов, которые не являются утечками
        false_positives = ['williams_r']  # Williams %R - технический индикатор, не утечка
        
        future_cols = [col for col in self.train_data.columns 
                      if any(word in col.lower() for word in ['future', 'forward', 'next', 'will'])
                      and col not in TRADING_TARGET_VARIABLES
                      and col not in false_positives]
        
        if future_cols:
            print_error(f"Найдены признаки с возможной утечкой: {len(future_cols)}")
            for col in future_cols[:10]:
                print_error(f"   - {col}")
            self.issues.append("future_data_in_features")
        else:
            print_success("Look-ahead bias не обнаружен в признаках")
            
        # 2. Проверка использования абсолютных цен в целевых
        print_info("\n2. Проверка абсолютных цен в целевых переменных...")
        
        price_like_targets = []
        for col in TRADING_TARGET_VARIABLES:
            if col in self.train_data.columns:
                # Проверяем тип данных - категориальные колонки пропускаем
                if col in MULTICLASS_TARGETS or col in BINARY_TARGETS:
                    continue  # Пропускаем категориальные и бинарные переменные
                    
                # Для числовых колонок проверяем максимальное значение
                if self.train_data[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                    max_val = self.train_data[col].abs().max()
                    if max_val > 1000:  # Подозрение на абсолютные цены
                        price_like_targets.append((col, max_val))
                    
        if price_like_targets:
            print_error(f"Целевые переменные с подозрением на абсолютные цены: {len(price_like_targets)}")
            for col, val in price_like_targets:
                print_error(f"   - {col}: max={val:.2f}")
            self.issues.append("absolute_prices_in_targets")
        else:
            print_success("Целевые переменные не содержат абсолютных цен")
            
        # 3. Проверка временного gap между выборками
        print_info("\n3. Проверка временного разделения выборок...")
        
        if self.train_data is not None and self.val_data is not None:
            train_end = pd.to_datetime(self.train_data['datetime']).max()
            val_start = pd.to_datetime(self.val_data['datetime']).min()
            gap_days = (val_start - train_end).days
            
            if gap_days < 1:
                print_error(f"Недостаточный gap между train и val: {gap_days} дней")
                self.issues.append("insufficient_temporal_gap")
            else:
                print_success(f"Временной gap train→val: {gap_days} дней ✓")
                
        if self.val_data is not None and self.test_data is not None:
            val_end = pd.to_datetime(self.val_data['datetime']).max()
            test_start = pd.to_datetime(self.test_data['datetime']).min()
            gap_days = (test_start - val_end).days
            
            if gap_days < 1:
                print_error(f"Недостаточный gap между val и test: {gap_days} дней")
                self.issues.append("insufficient_temporal_gap")
            else:
                print_success(f"Временной gap val→test: {gap_days} дней ✓")
                
        # 4. Проверка корреляций между признаками и будущими данными
        print_info("\n4. Проверка корреляций с целевыми переменными...")
        
        # Берем случайную выборку для ускорения
        sample_size = min(10000, len(self.train_data))
        sample_data = self.train_data.sample(n=sample_size)
        
        # Исключаем служебные и целевые колонки
        feature_cols = [col for col in sample_data.columns 
                       if col not in SERVICE_COLUMNS 
                       and col not in TRADING_TARGET_VARIABLES]
        
        # Проверяем корреляцию с future_return_4h
        if 'future_return_4h' in sample_data.columns:
            suspicious_features = []
            
            for col in feature_cols[:50]:  # Проверяем первые 50 признаков
                if sample_data[col].dtype in ['float32', 'float64']:
                    corr = sample_data[col].corr(sample_data['future_return_4h'])
                    if abs(corr) > 0.5:  # Подозрительно высокая корреляция
                        suspicious_features.append((col, corr))
                        
            if suspicious_features:
                print_warning(f"Признаки с высокой корреляцией с будущими данными:")
                for col, corr in sorted(suspicious_features, key=lambda x: abs(x[1]), reverse=True)[:5]:
                    print_warning(f"   - {col}: {corr:.3f}")
                self.warnings.append("high_correlation_with_targets")
            else:
                print_success("Нет подозрительных корреляций с целевыми")
                
    def check_overfitting_signs(self):
        """Проверка на признаки переобучения"""
        print_header("🔍 ПРОВЕРКА НА ПЕРЕОБУЧЕНИЕ (Overfitting)")
        
        if not all([self.train_data is not None, self.val_data is not None, self.test_data is not None]):
            print_error("Не все датасеты загружены!")
            return
            
        # 1. Сравнение распределений между выборками
        print_info("\n1. Сравнение распределений train/val/test...")
        
        # Проверяем ключевые статистики
        feature_cols = [col for col in self.train_data.columns 
                       if col not in SERVICE_COLUMNS 
                       and col not in TRADING_TARGET_VARIABLES][:20]
        
        distribution_issues = []
        
        for col in feature_cols:
            if self.train_data[col].dtype in ['float32', 'float64']:
                train_mean = self.train_data[col].mean()
                val_mean = self.val_data[col].mean()
                test_mean = self.test_data[col].mean()
                
                train_std = self.train_data[col].std()
                val_std = self.val_data[col].std()
                test_std = self.test_data[col].std()
                
                # Проверка сдвига среднего
                mean_shift_val = abs(val_mean - train_mean) / (train_std + 1e-8)
                mean_shift_test = abs(test_mean - train_mean) / (train_std + 1e-8)
                
                if mean_shift_val > 2 or mean_shift_test > 2:
                    distribution_issues.append(col)
                    
        if distribution_issues:
            print_warning(f"Обнаружен сдвиг распределения в {len(distribution_issues)} признаках")
            for col in distribution_issues[:5]:
                print_warning(f"   - {col}")
            self.warnings.append("distribution_shift")
        else:
            print_success("Распределения согласованы между выборками")
            
        # 2. Проверка дисбаланса классов в целевых
        print_info("\n2. Проверка баланса целевых переменных...")
        
        imbalance_issues = []
        
        for target in BINARY_TARGETS:
            if target in self.train_data.columns:
                train_pos = self.train_data[target].mean()
                val_pos = self.val_data[target].mean() if target in self.val_data.columns else 0
                test_pos = self.test_data[target].mean() if target in self.test_data.columns else 0
                
                if train_pos < 0.05 or train_pos > 0.95:
                    imbalance_issues.append((target, train_pos))
                    
                # Проверка согласованности между выборками
                # ИСПРАВЛЕНО: Увеличен порог для криптовалют (20% вместо 10%)
                # Криптовалютные рынки нестационарны и могут иметь разные режимы в разные периоды
                if abs(train_pos - val_pos) > 0.2 or abs(train_pos - test_pos) > 0.2:
                    self.warnings.append(f"target_distribution_shift_{target}")
                    
        if imbalance_issues:
            print_error(f"Сильный дисбаланс в {len(imbalance_issues)} целевых переменных:")
            for target, pos_rate in imbalance_issues[:5]:
                print_error(f"   - {target}: {pos_rate*100:.1f}%")
            self.issues.append("severe_class_imbalance")
        else:
            print_success("Баланс целевых переменных в норме")
            
        # 3. Проверка уникальности данных
        print_info("\n3. Проверка уникальности и дубликатов...")
        
        # Проверка дубликатов по времени и символу
        duplicates = self.train_data.duplicated(subset=['datetime', 'symbol']).sum()
        if duplicates > 0:
            print_error(f"Найдено {duplicates} дубликатов по (datetime, symbol)")
            self.issues.append("duplicate_records")
        else:
            print_success("Дубликаты не обнаружены")
            
        # 4. Анализ временной согласованности
        print_info("\n4. Проверка временной согласованности...")
        
        # Проверяем нет ли пропусков во времени
        for symbol in self.train_data['symbol'].unique()[:5]:  # Проверяем первые 5 символов
            symbol_data = self.train_data[self.train_data['symbol'] == symbol].sort_values('datetime')
            time_diffs = pd.to_datetime(symbol_data['datetime']).diff()
            
            # Ожидаем 15-минутные интервалы
            expected_diff = pd.Timedelta(minutes=15)
            gaps = time_diffs[time_diffs > expected_diff * 2]
            
            if len(gaps) > 10:
                print_warning(f"Символ {symbol}: {len(gaps)} временных пропусков")
                
    def check_target_variables_v4(self):
        """Проверка корректности целевых переменных версии 4.0 БЕЗ УТЕЧЕК"""
        print_header("🎯 ПРОВЕРКА ЦЕЛЕВЫХ ПЕРЕМЕННЫХ v4.0 (БЕЗ УТЕЧЕК)")
        
        if self.train_data is None:
            print_error("Данные не загружены!")
            return
            
        print_info(f"Ожидается {len(TRADING_TARGET_VARIABLES)} целевых переменных")
        
        # 1. Проверка наличия всех целевых
        missing_targets = [t for t in TRADING_TARGET_VARIABLES if t not in self.train_data.columns]
        if missing_targets:
            print_error(f"Отсутствуют целевые переменные: {len(missing_targets)}")
            for target in missing_targets:
                print_error(f"   - {target}")
            self.issues.append("missing_target_variables")
        else:
            print_success("Все 20 целевых переменных присутствуют (без утечек)")
            
        # 2. Проверка диапазонов значений
        print_info("\n2. Проверка диапазонов целевых переменных...")
        
        for group_name, targets in TARGET_GROUPS.items():
            print_info(f"\n{group_name}:")
            
            for target in targets:
                if target in self.train_data.columns:
                    if target in REGRESSION_TARGETS:
                        # Для регрессионных целевых
                        stats = self.train_data[target].describe()
                        
                        if 'return' in target:
                            # Возвраты должны быть в разумных пределах
                            if abs(stats['mean']) > 0.1 or stats['max'] > 1.0:
                                print_warning(f"   {target}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")
                            else:
                                print_success(f"   {target}: mean={stats['mean']:.4f}, std={stats['std']:.4f} ✓")
                                
                    elif target in BINARY_TARGETS:
                        # Для бинарных целевых
                        pos_rate = self.train_data[target].mean() * 100
                        print_info(f"   {target}: {pos_rate:.1f}% положительных")
                        
                    elif target in MULTICLASS_TARGETS:
                        # Для многоклассовых
                        dist = self.train_data[target].value_counts(normalize=True) * 100
                        print_info(f"   {target}: {dict(dist.head(3))}")
                        
        # 3. Проверка симметричности LONG/SHORT
        print_info("\n3. Проверка симметричности LONG/SHORT целевых...")
        
        for level in ['1pct_4h', '2pct_4h', '3pct_12h', '5pct_12h']:
            long_target = f'long_will_reach_{level}'
            short_target = f'short_will_reach_{level}'
            
            if long_target in self.train_data.columns and short_target in self.train_data.columns:
                long_rate = self.train_data[long_target].mean() * 100
                short_rate = self.train_data[short_target].mean() * 100
                
                diff = abs(long_rate - short_rate)
                if diff > 20:
                    print_warning(f"   {level}: LONG={long_rate:.1f}%, SHORT={short_rate:.1f}% (разница {diff:.1f}%)")
                else:
                    print_success(f"   {level}: LONG={long_rate:.1f}%, SHORT={short_rate:.1f}% ✓")
                    
    def check_data_quality(self):
        """Проверка качества данных"""
        print_header("📊 ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
        
        if self.train_data is None:
            print_error("Данные не загружены!")
            return
            
        # 1. Пропущенные значения
        print_info("\n1. Анализ пропущенных значений...")
        
        nan_stats = self.train_data.isnull().sum()
        nan_cols = nan_stats[nan_stats > 0].sort_values(ascending=False)
        
        if len(nan_cols) > 0:
            total_nan = nan_cols.sum()
            total_cells = len(self.train_data) * len(self.train_data.columns)
            nan_pct = total_nan / total_cells * 100
            
            print_warning(f"Пропущенные значения в {len(nan_cols)} колонках ({nan_pct:.2f}% от всех данных)")
            
            for col, count in nan_cols.head(10).items():
                pct = count / len(self.train_data) * 100
                print_warning(f"   - {col}: {count:,} ({pct:.1f}%)")
                
            if nan_pct > 5:
                self.issues.append("high_missing_values")
        else:
            print_success("Пропущенные значения отсутствуют")
            
        # 2. Выбросы
        print_info("\n2. Анализ выбросов...")
        
        feature_cols = [col for col in self.train_data.columns 
                       if col not in SERVICE_COLUMNS 
                       and col not in TRADING_TARGET_VARIABLES][:30]
        
        outlier_cols = []
        for col in feature_cols:
            if self.train_data[col].dtype in ['float32', 'float64']:
                # Z-score метод
                z_scores = np.abs(stats.zscore(self.train_data[col].dropna()))
                outliers = (z_scores > 5).sum()
                
                if outliers > len(self.train_data) * 0.01:  # Более 1% выбросов
                    outlier_cols.append((col, outliers))
                    
        if outlier_cols:
            print_warning(f"Обнаружены выбросы в {len(outlier_cols)} признаках:")
            for col, count in outlier_cols[:5]:
                pct = count / len(self.train_data) * 100
                print_warning(f"   - {col}: {count} выбросов ({pct:.1f}%)")
        else:
            print_success("Выбросы в допустимых пределах")
            
        # 3. Проверка диапазонов технических индикаторов
        print_info("\n3. Проверка диапазонов технических индикаторов...")
        
        # RSI должен быть в [0, 100]
        if 'rsi' in self.train_data.columns:
            rsi_stats = self.train_data['rsi'].describe()
            if rsi_stats['min'] < -1 or rsi_stats['max'] > 101:
                print_error(f"RSI вне диапазона: [{rsi_stats['min']:.1f}, {rsi_stats['max']:.1f}]")
                self.issues.append("rsi_out_of_range")
            else:
                print_success(f"RSI в корректном диапазоне: [{rsi_stats['min']:.1f}, {rsi_stats['max']:.1f}]")
                
        # Stochastic должен быть в [0, 100]
        for indicator in ['stoch_k', 'stoch_d']:
            if indicator in self.train_data.columns:
                stats_ind = self.train_data[indicator].describe()
                if stats_ind['min'] < -1 or stats_ind['max'] > 101:
                    print_error(f"{indicator} вне диапазона: [{stats_ind['min']:.1f}, {stats_ind['max']:.1f}]")
                    self.issues.append(f"{indicator}_out_of_range")
                    
    def generate_report(self):
        """Генерация итогового отчета"""
        print_header("📋 ИТОГОВЫЙ ОТЧЕТ")
        
        # Подсчет проблем
        total_issues = len(self.issues)
        total_warnings = len(self.warnings)
        
        if total_issues == 0 and total_warnings == 0:
            print_success("✅ ДАННЫЕ ПОЛНОСТЬЮ КОРРЕКТНЫ!")
            print_success("🚀 Система готова к обучению")
            print_info("\nЗапустите обучение:")
            print(f"   {Colors.OKBLUE}python main.py --mode train{Colors.ENDC}")
            
        else:
            if total_issues > 0:
                print_error(f"❌ Обнаружено критических проблем: {total_issues}")
                print("\nКритические проблемы:")
                for issue in set(self.issues):
                    print_error(f"   - {issue}")
                    
            if total_warnings > 0:
                print_warning(f"⚠️  Обнаружено предупреждений: {total_warnings}")
                print("\nПредупреждения:")
                for warning in list(set(self.warnings))[:10]:
                    print_warning(f"   - {warning}")
                    
            # Рекомендации
            print(f"\n{Colors.WARNING}{Colors.BOLD}🔧 РЕКОМЕНДАЦИИ:{Colors.ENDC}")
            
            if 'future_data_in_features' in self.issues:
                print_info("1. Проверьте feature_engineering.py на использование будущих данных")
                
            if 'absolute_prices_in_targets' in self.issues:
                print_info("2. Убедитесь что целевые переменные - это проценты, а не цены")
                
            if 'insufficient_temporal_gap' in self.issues:
                print_info("3. Увеличьте gap между выборками в prepare_trading_data.py")
                
            if 'severe_class_imbalance' in self.issues:
                print_info("4. Рассмотрите балансировку классов или изменение порогов")
                
            print_info("\nДля пересоздания данных:")
            print(f"   {Colors.OKBLUE}rm -rf cache/features/*{Colors.ENDC}")
            print(f"   {Colors.OKBLUE}python prepare_trading_data.py --force-recreate{Colors.ENDC}")
            
        # Сохранение детального отчета
        self._save_detailed_report()
        
    def _save_detailed_report(self):
        """Сохранение детального отчета в файл"""
        report_path = Path('logs/data_validation_v4_report.txt')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ДЕТАЛЬНЫЙ ОТЧЕТ ВАЛИДАЦИИ ДАННЫХ v4.0 (БЕЗ УТЕЧЕК)\n")
            f.write("="*80 + "\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Статистика по датасетам
            for name in ['train', 'val', 'test']:
                data = getattr(self, f"{name}_data")
                if data is not None:
                    f.write(f"\n{name.upper()} DATASET:\n")
                    f.write(f"  Размер: {len(data):,} записей\n")
                    f.write(f"  Период: {data['datetime'].min()} - {data['datetime'].max()}\n")
                    f.write(f"  Символов: {data['symbol'].nunique()}\n")
                    f.write(f"  Признаков: {len(data.columns)}\n")
                    
            # Проблемы
            f.write(f"\nОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:\n")
            f.write(f"  Критических: {len(self.issues)}\n")
            f.write(f"  Предупреждений: {len(self.warnings)}\n")
            
            if self.issues:
                f.write("\nКРИТИЧЕСКИЕ ПРОБЛЕМЫ:\n")
                for issue in set(self.issues):
                    f.write(f"  - {issue}\n")
                    
            if self.warnings:
                f.write("\nПРЕДУПРЕЖДЕНИЯ:\n")
                for warning in set(self.warnings):
                    f.write(f"  - {warning}\n")
                    
        print_info(f"\n📄 Детальный отчет сохранен: {report_path}")


def main():
    """Основная функция"""
    verifier = DataCorrectnessVerifier()
    
    # Загрузка данных
    verifier.load_data()
    
    # Выполнение всех проверок
    verifier.check_data_leakage()
    verifier.check_overfitting_signs()
    verifier.check_target_variables_v4()
    verifier.check_data_quality()
    
    # Генерация отчета
    verifier.generate_report()


if __name__ == "__main__":
    main()