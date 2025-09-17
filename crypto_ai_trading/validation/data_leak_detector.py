"""
Детекция утечек данных из будущего в торговых моделях
Критически важно для временных рядов и торговых стратегий
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass
import warnings
import re
from pathlib import Path
import json
import logging
from scipy import stats
from scipy.stats import kstest, chi2_contingency
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from utils.logger import get_logger


@dataclass
class LeakDetectionConfig:
    """Конфигурация для детекции утечек"""
    # Временные параметры
    min_embargo_days: int = 1           # Минимальный gap между train/val/test
    max_future_correlation: float = 0.3  # Максимальная корреляция с будущими данными
    
    # Пороги подозрительности
    suspicious_performance_uplift: float = 1.5  # Подозрительное улучшение метрик
    suspicious_mutual_info: float = 0.5         # Подозрительная взаимная информация
    
    # Статистические тесты
    statistical_significance: float = 0.05      # Уровень значимости
    temporal_correlation_window: int = 30       # Окно для анализа временных корреляций
    
    # Подозрительные паттерны в названиях признаков
    suspicious_keywords: List[str] = None


@dataclass
class LeakageReport:
    """Отчет о найденных утечках"""
    feature_name: str
    leak_type: str                    # 'temporal', 'statistical', 'semantic', 'performance'
    severity: str                     # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    evidence: Dict[str, Any]          # Подтверждающие данные
    recommendations: List[str]        # Рекомендации по исправлению
    confidence: float                 # Уверенность в детекции (0-1)


class DataLeakDetector:
    """
    Детектор утечек данных из будущего
    
    Методы детекции:
    1. Анализ временных корреляций между признаками
    2. Статистические тесты на аномальные распределения  
    3. Семантический анализ названий признаков
    4. Анализ подозрительного улучшения производительности
    5. Проверка на forward-looking индикаторы
    """
    
    def __init__(self, config: LeakDetectionConfig = None):
        self.config = config or LeakDetectionConfig()
        self.logger = get_logger("DataLeakDetector")
        
        # Устанавливаем подозрительные ключевые слова по умолчанию
        if self.config.suspicious_keywords is None:
            self.config.suspicious_keywords = [
                'future', 'next', 'tomorrow', 'ahead', 'forward', 'predict',
                'target', 'label', 'y_', 'ground_truth', 'actual',
                'weekend', 'month_end', 'quarter_end', 'year_end',
                'day_of_week', 'day_of_month', 'day_of_year',
                'is_holiday', 'business_day', 'trading_session'
            ]
        
        self.leak_reports: List[LeakageReport] = []
        
        self.logger.info("🔍 Инициализация детектора утечек данных")
        self.logger.info(f"   🚨 Подозрительных паттернов: {len(self.config.suspicious_keywords)}")
        
    def detect_temporal_leaks(self, 
                            data: pd.DataFrame,
                            feature_columns: List[str],
                            target_columns: List[str],
                            time_column: str = 'datetime') -> List[LeakageReport]:
        """
        Детекция временных утечек через корреляционный анализ
        
        Args:
            data: DataFrame с данными
            feature_columns: Список признаков для проверки
            target_columns: Список целевых переменных
            time_column: Название колонки с временными метками
            
        Returns:
            Список отчетов об обнаруженных утечках
        """
        
        self.logger.info("⏰ Детекция временных утечек данных")
        
        if time_column not in data.columns:
            raise ValueError(f"Колонка {time_column} не найдена в данных")
        
        # Сортируем по времени
        data_sorted = data.sort_values(time_column).copy()
        
        temporal_leaks = []
        window = self.config.temporal_correlation_window
        
        for feature in feature_columns:
            if feature not in data_sorted.columns:
                continue
                
            self.logger.debug(f"   Анализ признака: {feature}")
            
            # Проверяем корреляции с будущими значениями целевых переменных
            for target in target_columns:
                if target not in data_sorted.columns:
                    continue
                
                # Рассчитываем корреляцию с будущими значениями target
                future_correlations = []
                
                for shift in range(1, min(window + 1, len(data_sorted) // 4)):
                    try:
                        # Сдвигаем target вперед (будущие значения)
                        future_target = data_sorted[target].shift(-shift)
                        current_feature = data_sorted[feature]
                        
                        # Удаляем NaN и рассчитываем корреляцию
                        valid_mask = ~(pd.isna(current_feature) | pd.isna(future_target))
                        if valid_mask.sum() < 10:  # Минимум точек для корреляции
                            continue
                            
                        corr = current_feature[valid_mask].corr(future_target[valid_mask])
                        if not np.isnan(corr):
                            future_correlations.append(abs(corr))
                            
                    except Exception as e:
                        self.logger.debug(f"Ошибка в корреляционном анализе: {str(e)}")
                        continue
                
                # Анализируем результаты
                if future_correlations:
                    max_future_corr = max(future_correlations)
                    mean_future_corr = np.mean(future_correlations)
                    
                    # Определяем подозрительность
                    if max_future_corr > self.config.max_future_correlation:
                        
                        # Определяем серьезность
                        if max_future_corr > 0.7:
                            severity = 'CRITICAL'
                        elif max_future_corr > 0.5:
                            severity = 'HIGH'
                        elif max_future_corr > 0.3:
                            severity = 'MEDIUM'
                        else:
                            severity = 'LOW'
                        
                        leak_report = LeakageReport(
                            feature_name=feature,
                            leak_type='temporal',
                            severity=severity,
                            evidence={
                                'max_future_correlation': max_future_corr,
                                'mean_future_correlation': mean_future_corr,
                                'target_variable': target,
                                'correlation_window': window,
                                'correlations_at_shifts': future_correlations[:10]  # Первые 10
                            },
                            recommendations=[
                                f"Проверить расчет признака {feature}",
                                f"Убедиться что {feature} не использует будущие данные",
                                "Пересмотреть логику создания признака",
                                "Добавить temporal lag если необходимо"
                            ],
                            confidence=min(max_future_corr * 2, 1.0)
                        )
                        
                        temporal_leaks.append(leak_report)
                        
                        self.logger.warning(f"🚨 ВРЕМЕННАЯ УТЕЧКА: {feature} -> {target} "
                                          f"(max_corr={max_future_corr:.3f})")
        
        return temporal_leaks
    
    def detect_statistical_leaks(self,
                                train_data: pd.DataFrame,
                                val_data: pd.DataFrame,
                                test_data: pd.DataFrame,
                                feature_columns: List[str]) -> List[LeakageReport]:
        """
        Детекция утечек через статистические аномалии
        
        Args:
            train_data: Обучающие данные
            val_data: Валидационные данные
            test_data: Тестовые данные
            feature_columns: Признаки для анализа
            
        Returns:
            Отчеты о статистических утечках
        """
        
        self.logger.info("📊 Детекция статистических утечек")
        
        statistical_leaks = []
        
        for feature in feature_columns:
            if not all(feature in df.columns for df in [train_data, val_data, test_data]):
                continue
            
            # Извлекаем значения признака
            train_vals = train_data[feature].dropna().values
            val_vals = val_data[feature].dropna().values
            test_vals = test_data[feature].dropna().values
            
            if any(len(vals) < 10 for vals in [train_vals, val_vals, test_vals]):
                continue
            
            # Тест Колмогорова-Смирнова на различия в распределениях
            ks_train_val = kstest(train_vals, val_vals)[1]
            ks_train_test = kstest(train_vals, test_vals)[1]
            ks_val_test = kstest(val_vals, test_vals)[1]
            
            min_p_value = min(ks_train_val, ks_train_test, ks_val_test)
            
            # Анализ моментов распределений
            moments_analysis = self._analyze_distribution_moments(
                train_vals, val_vals, test_vals, feature
            )
            
            # Тест на аномальные значения
            outliers_analysis = self._analyze_outliers(
                train_vals, val_vals, test_vals, feature
            )
            
            # Подозрительные различия в распределениях
            if min_p_value < self.config.statistical_significance:
                
                severity = 'LOW'
                if min_p_value < 0.001:
                    severity = 'HIGH'
                elif min_p_value < 0.01:
                    severity = 'MEDIUM'
                
                leak_report = LeakageReport(
                    feature_name=feature,
                    leak_type='statistical',
                    severity=severity,
                    evidence={
                        'ks_test_p_values': {
                            'train_vs_val': ks_train_val,
                            'train_vs_test': ks_train_test,
                            'val_vs_test': ks_val_test
                        },
                        'min_p_value': min_p_value,
                        'moments_analysis': moments_analysis,
                        'outliers_analysis': outliers_analysis
                    },
                    recommendations=[
                        "Проверить временное разделение данных",
                        "Убедиться в корректности признака на всех периодах",
                        "Проверить на data snooping"
                    ],
                    confidence=1 - min_p_value
                )
                
                statistical_leaks.append(leak_report)
                
                self.logger.warning(f"📊 СТАТИСТИЧЕСКАЯ УТЕЧКА: {feature} "
                                  f"(min_p={min_p_value:.4f})")
        
        return statistical_leaks
    
    def detect_semantic_leaks(self, feature_columns: List[str]) -> List[LeakageReport]:
        """
        Детекция утечек через анализ названий признаков
        
        Args:
            feature_columns: Список признаков для анализа
            
        Returns:
            Отчеты о семантических утечках
        """
        
        self.logger.info("🔤 Детекция семантических утечек")
        
        semantic_leaks = []
        
        for feature in feature_columns:
            feature_lower = feature.lower()
            
            # Проверяем на подозрительные ключевые слова
            suspicious_words = []
            for keyword in self.config.suspicious_keywords:
                if keyword in feature_lower:
                    suspicious_words.append(keyword)
            
            if suspicious_words:
                # Определяем серьезность на основе типа слов
                critical_words = ['future', 'next', 'tomorrow', 'ahead', 'target', 'label', 'y_']
                high_words = ['predict', 'ground_truth', 'actual']
                
                if any(word in suspicious_words for word in critical_words):
                    severity = 'CRITICAL'
                elif any(word in suspicious_words for word in high_words):
                    severity = 'HIGH'
                else:
                    severity = 'MEDIUM'
                
                leak_report = LeakageReport(
                    feature_name=feature,
                    leak_type='semantic',
                    severity=severity,
                    evidence={
                        'suspicious_keywords': suspicious_words,
                        'feature_name': feature,
                        'pattern_matches': self._extract_suspicious_patterns(feature)
                    },
                    recommendations=[
                        f"КРИТИЧНО: Переименовать признак {feature}",
                        "Проверить логику создания признака",
                        "Убедиться что не используются будущие данные"
                    ],
                    confidence=0.9 if severity == 'CRITICAL' else 0.7
                )
                
                semantic_leaks.append(leak_report)
                
                self.logger.warning(f"🔤 СЕМАНТИЧЕСКАЯ УТЕЧКА: {feature} "
                                  f"({', '.join(suspicious_words)})")
        
        return semantic_leaks
    
    def detect_performance_leaks(self,
                               baseline_metrics: Dict[str, float],
                               enhanced_metrics: Dict[str, float],
                               added_features: List[str]) -> List[LeakageReport]:
        """
        Детекция утечек через подозрительное улучшение производительности
        
        Args:
            baseline_metrics: Метрики базовой модели
            enhanced_metrics: Метрики улучшенной модели
            added_features: Список добавленных признаков
            
        Returns:
            Отчеты о производительностных утечках
        """
        
        self.logger.info("📈 Детекция утечек через анализ производительности")
        
        performance_leaks = []
        
        # Анализируем улучшение по каждой метрике
        suspicious_improvements = {}
        
        for metric, baseline_value in baseline_metrics.items():
            if metric in enhanced_metrics:
                enhanced_value = enhanced_metrics[metric]
                
                # Рассчитываем улучшение
                if baseline_value != 0:
                    improvement_ratio = enhanced_value / baseline_value
                else:
                    improvement_ratio = float('inf') if enhanced_value > 0 else 1.0
                
                # Проверяем на подозрительность
                if improvement_ratio > self.config.suspicious_performance_uplift:
                    suspicious_improvements[metric] = {
                        'baseline': baseline_value,
                        'enhanced': enhanced_value,
                        'improvement_ratio': improvement_ratio,
                        'absolute_improvement': enhanced_value - baseline_value
                    }
        
        # Если есть подозрительные улучшения
        if suspicious_improvements:
            
            # Определяем серьезность
            max_improvement = max(imp['improvement_ratio'] 
                                for imp in suspicious_improvements.values())
            
            if max_improvement > 3.0:
                severity = 'CRITICAL'
            elif max_improvement > 2.0:
                severity = 'HIGH'
            else:
                severity = 'MEDIUM'
            
            # Создаем отчет для группы добавленных признаков
            leak_report = LeakageReport(
                feature_name=', '.join(added_features[:5]),  # Первые 5 признаков
                leak_type='performance',
                severity=severity,
                evidence={
                    'suspicious_improvements': suspicious_improvements,
                    'added_features': added_features,
                    'max_improvement_ratio': max_improvement,
                    'baseline_metrics': baseline_metrics,
                    'enhanced_metrics': enhanced_metrics
                },
                recommendations=[
                    "КРИТИЧНО: Проверить все новые признаки на утечки",
                    "Провести детальный анализ каждого добавленного признака",
                    "Использовать строгий temporal split",
                    "Проверить на переобучение и data snooping"
                ],
                confidence=min((max_improvement - 1) / 2, 1.0)
            )
            
            performance_leaks.append(leak_report)
            
            self.logger.warning(f"📈 ПРОИЗВОДИТЕЛЬНОСТНАЯ УТЕЧКА: "
                              f"улучшение в {max_improvement:.1f}x раз")
        
        return performance_leaks
    
    def detect_mutual_information_leaks(self,
                                      data: pd.DataFrame,
                                      feature_columns: List[str],
                                      target_columns: List[str]) -> List[LeakageReport]:
        """
        Детекция утечек через анализ взаимной информации
        
        Args:
            data: Данные для анализа
            feature_columns: Список признаков
            target_columns: Список целевых переменных
            
        Returns:
            Отчеты об утечках взаимной информации
        """
        
        self.logger.info("🧠 Детекция утечек через взаимную информацию")
        
        mi_leaks = []
        
        for feature in feature_columns:
            if feature not in data.columns:
                continue
                
            feature_values = data[feature].dropna()
            if len(feature_values) < 50:  # Минимум данных
                continue
            
            for target in target_columns:
                if target not in data.columns:
                    continue
                
                # Выравниваем индексы
                common_idx = feature_values.index.intersection(data[target].dropna().index)
                if len(common_idx) < 50:
                    continue
                
                X = feature_values.loc[common_idx].values.reshape(-1, 1)
                y = data[target].loc[common_idx].values
                
                # Определяем тип целевой переменной
                if len(np.unique(y)) <= 10:  # Классификация
                    # Кодируем категории
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    mi_score = mutual_info_classif(X, y_encoded, random_state=42)[0]
                else:  # Регрессия
                    mi_score = mutual_info_regression(X, y, random_state=42)[0]
                
                # Проверяем на подозрительно высокую взаимную информацию
                if mi_score > self.config.suspicious_mutual_info:
                    
                    # Определяем серьезность
                    if mi_score > 0.8:
                        severity = 'CRITICAL'
                    elif mi_score > 0.6:
                        severity = 'HIGH'
                    else:
                        severity = 'MEDIUM'
                    
                    leak_report = LeakageReport(
                        feature_name=feature,
                        leak_type='mutual_information',
                        severity=severity,
                        evidence={
                            'mutual_information_score': mi_score,
                            'target_variable': target,
                            'feature_unique_values': len(np.unique(X)),
                            'target_unique_values': len(np.unique(y)),
                            'sample_size': len(X)
                        },
                        recommendations=[
                            f"Высокая взаимная информация между {feature} и {target}",
                            "Проверить на прямую связь с целевой переменной",
                            "Убедиться что признак не содержит целевую информацию"
                        ],
                        confidence=mi_score
                    )
                    
                    mi_leaks.append(leak_report)
                    
                    self.logger.warning(f"🧠 УТЕЧКА ВЗАИМНОЙ ИНФОРМАЦИИ: "
                                      f"{feature} -> {target} (MI={mi_score:.3f})")
        
        return mi_leaks
    
    def comprehensive_leak_detection(self,
                                   data: pd.DataFrame,
                                   feature_columns: List[str],
                                   target_columns: List[str],
                                   time_column: str = 'datetime',
                                   train_end_date: str = None,
                                   val_end_date: str = None) -> Dict:
        """
        Комплексная детекция всех типов утечек
        
        Args:
            data: Данные для анализа
            feature_columns: Список признаков
            target_columns: Список целевых переменных
            time_column: Колонка с временными метками
            train_end_date: Дата окончания обучающих данных
            val_end_date: Дата окончания валидационных данных
            
        Returns:
            Комплексный отчет о всех найденных утечках
        """
        
        self.logger.info("🔍 Запуск комплексной детекции утечек данных")
        self.logger.info(f"   📊 Признаков для анализа: {len(feature_columns)}")
        self.logger.info(f"   🎯 Целевых переменных: {len(target_columns)}")
        
        all_leaks = {
            'temporal_leaks': [],
            'statistical_leaks': [],
            'semantic_leaks': [],
            'mutual_information_leaks': [],
            'performance_leaks': []
        }
        
        # 1. Семантическая детекция (быстрая)
        try:
            semantic_leaks = self.detect_semantic_leaks(feature_columns)
            all_leaks['semantic_leaks'] = [leak.__dict__ for leak in semantic_leaks]
            self.logger.info(f"   🔤 Семантических утечек: {len(semantic_leaks)}")
        except Exception as e:
            self.logger.error(f"Ошибка в семантической детекции: {str(e)}")
        
        # 2. Временная детекция
        try:
            temporal_leaks = self.detect_temporal_leaks(
                data, feature_columns, target_columns, time_column
            )
            all_leaks['temporal_leaks'] = [leak.__dict__ for leak in temporal_leaks]
            self.logger.info(f"   ⏰ Временных утечек: {len(temporal_leaks)}")
        except Exception as e:
            self.logger.error(f"Ошибка в временной детекции: {str(e)}")
        
        # 3. Статистическая детекция (если есть временные границы)
        if train_end_date and val_end_date and time_column in data.columns:
            try:
                # Разделяем данные по времени
                train_data = data[data[time_column] <= train_end_date]
                val_data = data[(data[time_column] > train_end_date) & 
                               (data[time_column] <= val_end_date)]
                test_data = data[data[time_column] > val_end_date]
                
                if len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0:
                    statistical_leaks = self.detect_statistical_leaks(
                        train_data, val_data, test_data, feature_columns
                    )
                    all_leaks['statistical_leaks'] = [leak.__dict__ for leak in statistical_leaks]
                    self.logger.info(f"   📊 Статистических утечек: {len(statistical_leaks)}")
            except Exception as e:
                self.logger.error(f"Ошибка в статистической детекции: {str(e)}")
        
        # 4. Детекция взаимной информации
        try:
            mi_leaks = self.detect_mutual_information_leaks(
                data, feature_columns, target_columns
            )
            all_leaks['mutual_information_leaks'] = [leak.__dict__ for leak in mi_leaks]
            self.logger.info(f"   🧠 Утечек взаимной информации: {len(mi_leaks)}")
        except Exception as e:
            self.logger.error(f"Ошибка в детекции взаимной информации: {str(e)}")
        
        # Составляем итоговый отчет
        total_leaks = sum(len(leaks) for leaks in all_leaks.values() if isinstance(leaks, list))
        
        # Группируем по серьезности
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for leak_category in all_leaks.values():
            if isinstance(leak_category, list):
                for leak in leak_category:
                    if isinstance(leak, dict) and 'severity' in leak:
                        severity = leak['severity']
                        if severity in severity_counts:
                            severity_counts[severity] += 1
        
        # Генерируем рекомендации
        recommendations = self._generate_leak_recommendations(all_leaks, severity_counts)
        
        comprehensive_report = {
            'summary': {
                'total_leaks_detected': total_leaks,
                'severity_breakdown': severity_counts,
                'features_analyzed': len(feature_columns),
                'targets_analyzed': len(target_columns),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'detailed_results': all_leaks,
            'recommendations': recommendations,
            'risk_assessment': self._assess_leak_risk(severity_counts),
            'config': self.config.__dict__
        }
        
        # Сохраняем отчет
        self._save_leak_report(comprehensive_report)
        
        self.logger.info(f"🎯 Детекция завершена:")
        self.logger.info(f"   📊 Всего утечек: {total_leaks}")
        self.logger.info(f"   🚨 КРИТИЧНЫХ: {severity_counts['CRITICAL']}")
        self.logger.info(f"   ⚠️ ВЫСОКИХ: {severity_counts['HIGH']}")
        
        return comprehensive_report
    
    def _analyze_distribution_moments(self, train_vals, val_vals, test_vals, feature_name):
        """Анализ моментов распределения для детекции аномалий"""
        
        moments = {}
        
        for name, vals in [('train', train_vals), ('val', val_vals), ('test', test_vals)]:
            moments[name] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'skewness': stats.skew(vals),
                'kurtosis': stats.kurtosis(vals),
                'min': np.min(vals),
                'max': np.max(vals)
            }
        
        # Анализируем различия
        differences = {
            'mean_diff_train_val': abs(moments['train']['mean'] - moments['val']['mean']),
            'std_diff_train_val': abs(moments['train']['std'] - moments['val']['std']),
            'skew_diff_train_val': abs(moments['train']['skewness'] - moments['val']['skewness'])
        }
        
        return {'moments': moments, 'differences': differences}
    
    def _analyze_outliers(self, train_vals, val_vals, test_vals, feature_name):
        """Анализ выбросов в разных наборах данных"""
        
        # Определяем выбросы как значения за пределами 3 сигм от среднего
        def count_outliers(vals):
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            outliers = np.abs(vals - mean_val) > 3 * std_val
            return np.sum(outliers), np.sum(outliers) / len(vals)
        
        outliers_analysis = {}
        
        for name, vals in [('train', train_vals), ('val', val_vals), ('test', test_vals)]:
            count, pct = count_outliers(vals)
            outliers_analysis[name] = {
                'outliers_count': count,
                'outliers_percentage': pct
            }
        
        return outliers_analysis
    
    def _extract_suspicious_patterns(self, feature_name: str) -> List[str]:
        """Извлекает подозрительные паттерны из названия признака"""
        
        patterns = []
        feature_lower = feature_name.lower()
        
        # Регексы для подозрительных паттернов
        suspicious_regexes = [
            r'future_\w+',           # future_something
            r'next_\d+\w*',          # next_5min, next_day
            r'ahead_\w+',            # ahead_return
            r'target_\w*',           # target, target_var
            r'y_\w*',                # y_pred, y_true
            r'\w*_tomorrow\w*',      # price_tomorrow
            r'label_\w*',            # label_encoded
            r'ground_truth\w*',      # ground_truth
        ]
        
        for regex in suspicious_regexes:
            matches = re.findall(regex, feature_lower)
            patterns.extend(matches)
        
        return patterns
    
    def _generate_leak_recommendations(self, all_leaks: Dict, severity_counts: Dict) -> List[str]:
        """Генерирует рекомендации на основе найденных утечек"""
        
        recommendations = []
        
        # Общие рекомендации на основе серьезности
        if severity_counts['CRITICAL'] > 0:
            recommendations.extend([
                "🚨 КРИТИЧНО: Обнаружены критические утечки данных",
                "❌ НЕ ИСПОЛЬЗУЙТЕ модель в продакшене до устранения утечек",
                "🔄 Полностью пересмотрите процесс создания признаков",
                "📅 Используйте строгое временное разделение данных"
            ])
        
        if severity_counts['HIGH'] > 0:
            recommendations.extend([
                "⚠️ Обнаружены серьезные утечки данных",
                "🔍 Требуется детальная проверка всех подозрительных признаков",
                "📊 Проведите дополнительное тестирование на out-of-time данных"
            ])
        
        if severity_counts['MEDIUM'] > 0:
            recommendations.extend([
                "📋 Проверьте признаки со средним уровнем подозрительности",
                "🧪 Рассмотрите возможность исключения подозрительных признаков"
            ])
        
        # Специфичные рекомендации на основе типов утечек
        if len(all_leaks.get('semantic_leaks', [])) > 0:
            recommendations.append("🔤 Переименуйте признаки с подозрительными названиями")
        
        if len(all_leaks.get('temporal_leaks', [])) > 0:
            recommendations.extend([
                "⏰ Проверьте временную логику создания признаков",
                "🚫 Исключите признаки с высокой корреляцией с будущими данными"
            ])
        
        if len(all_leaks.get('statistical_leaks', [])) > 0:
            recommendations.extend([
                "📊 Проверьте корректность временного разделения данных",
                "🔄 Убедитесь в стабильности признаков во времени"
            ])
        
        if len(all_leaks.get('mutual_information_leaks', [])) > 0:
            recommendations.extend([
                "🧠 Проверьте признаки с высокой взаимной информацией",
                "🎯 Убедитесь что признаки не содержат целевую переменную"
            ])
        
        # Если утечек не найдено
        if sum(severity_counts.values()) == 0:
            recommendations.extend([
                "✅ Критических утечек данных не обнаружено",
                "🔍 Рекомендуется дополнительная проверка на независимых данных",
                "📊 Продолжите с walk-forward валидацией"
            ])
        
        return recommendations
    
    def _assess_leak_risk(self, severity_counts: Dict) -> Dict:
        """Оценивает общий риск утечек данных"""
        
        # Взвешенная оценка риска
        risk_weights = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        
        total_risk_score = sum(
            severity_counts.get(severity, 0) * weight
            for severity, weight in risk_weights.items()
        )
        
        max_possible_score = sum(severity_counts.values()) * 4 if sum(severity_counts.values()) > 0 else 1
        normalized_risk = total_risk_score / max_possible_score
        
        # Определяем уровень риска
        if normalized_risk >= 0.75 or severity_counts.get('CRITICAL', 0) > 0:
            risk_level = 'CRITICAL'
            risk_description = 'Критический риск утечек данных'
        elif normalized_risk >= 0.5 or severity_counts.get('HIGH', 0) > 0:
            risk_level = 'HIGH'
            risk_description = 'Высокий риск утечек данных'
        elif normalized_risk >= 0.25:
            risk_level = 'MEDIUM'  
            risk_description = 'Умеренный риск утечек данных'
        else:
            risk_level = 'LOW'
            risk_description = 'Низкий риск утечек данных'
        
        return {
            'risk_level': risk_level,
            'risk_score': normalized_risk,
            'risk_description': risk_description,
            'total_leaks': sum(severity_counts.values()),
            'critical_leaks': severity_counts.get('CRITICAL', 0),
            'actionable': risk_level in ['CRITICAL', 'HIGH']
        }
    
    def _save_leak_report(self, report: Dict):
        """Сохраняет отчет о детекции утечек"""
        
        # Создаем директорию
        reports_dir = Path("validation/leak_detection_results")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем JSON отчет
        json_file = reports_dir / f"leak_detection_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Создаем читаемый отчет
        readable_file = reports_dir / f"LEAK_DETECTION_REPORT_{timestamp}.md"
        readable_content = self._create_readable_leak_report(report)
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(readable_content)
        
        self.logger.info(f"💾 Отчет о детекции утечек сохранен:")
        self.logger.info(f"   📄 JSON: {json_file}")
        self.logger.info(f"   📖 Читаемый: {readable_file}")
    
    def _create_readable_leak_report(self, report: Dict) -> str:
        """Создает читаемый отчет о детекции утечек"""
        
        timestamp = report['summary']['analysis_timestamp']
        total_leaks = report['summary']['total_leaks_detected']
        severity_counts = report['summary']['severity_breakdown']
        risk_info = report['risk_assessment']
        
        readable_report = f"""# Data Leak Detection Report

**Дата анализа:** {timestamp}  
**Общий риск:** {risk_info['risk_level']} ({risk_info['risk_score']:.2%})  
**Всего утечек:** {total_leaks}

{risk_info['risk_description']}

## 📊 Сводка по серьезности

- 🚨 **КРИТИЧНЫЕ:** {severity_counts.get('CRITICAL', 0)}
- ⚠️ **ВЫСОКИЕ:** {severity_counts.get('HIGH', 0)}  
- 📋 **СРЕДНИЕ:** {severity_counts.get('MEDIUM', 0)}
- 📝 **НИЗКИЕ:** {severity_counts.get('LOW', 0)}

## 🔍 Детали по типам утечек

"""
        
        # Детали по каждому типу утечек
        leak_types = {
            'semantic_leaks': '🔤 Семантические утечки',
            'temporal_leaks': '⏰ Временные утечки', 
            'statistical_leaks': '📊 Статистические утечки',
            'mutual_information_leaks': '🧠 Утечки взаимной информации',
            'performance_leaks': '📈 Производительностные утечки'
        }
        
        detailed_results = report.get('detailed_results', {})
        
        for leak_type, title in leak_types.items():
            leaks = detailed_results.get(leak_type, [])
            if leaks:
                readable_report += f"### {title}\n\n"
                
                for i, leak in enumerate(leaks[:10], 1):  # Первые 10
                    if isinstance(leak, dict):
                        feature = leak.get('feature_name', 'Unknown')
                        severity = leak.get('severity', 'Unknown')
                        confidence = leak.get('confidence', 0)
                        
                        readable_report += f"{i}. **{feature}** - {severity} "
                        readable_report += f"(уверенность: {confidence:.1%})\n"
                        
                        # Добавляем ключевые доказательства
                        evidence = leak.get('evidence', {})
                        if evidence:
                            key_evidence = []
                            if 'suspicious_keywords' in evidence:
                                key_evidence.append(f"Ключевые слова: {', '.join(evidence['suspicious_keywords'])}")
                            if 'max_future_correlation' in evidence:
                                key_evidence.append(f"Корреляция с будущим: {evidence['max_future_correlation']:.3f}")
                            if 'mutual_information_score' in evidence:
                                key_evidence.append(f"Взаимная информация: {evidence['mutual_information_score']:.3f}")
                            
                            if key_evidence:
                                readable_report += f"   - {'; '.join(key_evidence)}\n"
                        
                        readable_report += "\n"
                
                if len(leaks) > 10:
                    readable_report += f"... и еще {len(leaks) - 10} утечек этого типа\n\n"
            else:
                readable_report += f"### {title}\n✅ Утечки не обнаружены\n\n"
        
        # Рекомендации
        recommendations = report.get('recommendations', [])
        readable_report += "## 💡 Рекомендации\n\n"
        
        for rec in recommendations:
            readable_report += f"- {rec}\n"
        
        readable_report += "\n---\n*Отчет сгенерирован системой детекции утечек данных*"
        
        return readable_report


def main():
    """Пример использования детектора утечек данных"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Детекция утечек данных в торговой модели")
    parser.add_argument('--data-file', required=True, help='CSV файл с данными')
    parser.add_argument('--feature-columns', nargs='+', help='Список признаков для анализа')
    parser.add_argument('--target-columns', nargs='+', help='Список целевых переменных')
    parser.add_argument('--time-column', default='datetime', help='Название колонки времени')
    parser.add_argument('--train-end', help='Дата окончания обучения (YYYY-MM-DD)')
    parser.add_argument('--val-end', help='Дата окончания валидации (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Загружаем данные
    print("📊 Загрузка данных...")
    data = pd.read_csv(args.data_file)
    
    if args.time_column in data.columns:
        data[args.time_column] = pd.to_datetime(data[args.time_column])
    
    # Автоматически определяем признаки и цели если не заданы
    if args.feature_columns is None:
        # Исключаем временные колонки и очевидные цели
        exclude_cols = {args.time_column, 'symbol', 'close', 'open', 'high', 'low', 'volume'}
        feature_columns = [col for col in data.columns 
                          if col not in exclude_cols and not col.startswith('future_')]
    else:
        feature_columns = args.feature_columns
    
    if args.target_columns is None:
        # Ищем колонки с future_ или target_ в названии
        target_columns = [col for col in data.columns 
                         if any(prefix in col.lower() 
                               for prefix in ['future_', 'target_', 'direction_', 'will_reach_'])]
    else:
        target_columns = args.target_columns
    
    print(f"🔍 Анализируем признаки: {len(feature_columns)}")
    print(f"🎯 Целевые переменные: {len(target_columns)}")
    
    # Создаем детектор
    config = LeakDetectionConfig()
    detector = DataLeakDetector(config)
    
    # Запускаем комплексную детекцию
    results = detector.comprehensive_leak_detection(
        data=data,
        feature_columns=feature_columns,
        target_columns=target_columns,
        time_column=args.time_column,
        train_end_date=args.train_end,
        val_end_date=args.val_end
    )
    
    # Выводим результаты
    print("✅ Детекция завершена!")
    print(f"📊 Найдено утечек: {results['summary']['total_leaks_detected']}")
    
    severity_counts = results['summary']['severity_breakdown']
    print(f"🚨 КРИТИЧНЫЕ: {severity_counts.get('CRITICAL', 0)}")
    print(f"⚠️ ВЫСОКИЕ: {severity_counts.get('HIGH', 0)}")
    print(f"📋 СРЕДНИЕ: {severity_counts.get('MEDIUM', 0)}")
    
    risk_info = results['risk_assessment']
    print(f"⚠️ Уровень риска: {risk_info['risk_level']}")
    
    if risk_info['actionable']:
        print("\n🚨 ТРЕБУЕТСЯ НЕМЕДЛЕННОЕ ДЕЙСТВИЕ!")
        print("❌ Модель НЕ ГОТОВА для продакшена")
    else:
        print("\n✅ Критических проблем не обнаружено")


if __name__ == "__main__":
    main()