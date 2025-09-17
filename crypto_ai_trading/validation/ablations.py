"""
Абляционное тестирование для новых компонентов системы
Систематическая проверка вклада каждого признака в производительность модели
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import yaml

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from data.feature_engineering import FeatureEngineer
from data.data_loader import CryptoDataLoader
from features.temporal import TemporalEmbeddings
from features.market_context import MarketContextFeatures
from features.normalization import AdaptiveNormalization


class AblationTester:
    """
    Система абляционного тестирования для оценки вклада различных компонентов
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Инициализация абляционного тестера
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.logger = get_logger("AblationTester")
        
        # Загрузка конфигурации
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Создание директорий для результатов
        self.results_dir = Path("validation/results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Временные результаты текущего запуска
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = self.results_dir / f"ablation_{timestamp}"
        self.current_run_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"🧪 Инициализирован абляционный тестер")
        self.logger.info(f"📁 Результаты будут сохранены в: {self.current_run_dir}")
        
    def test_temporal_features(self, 
                              symbols: List[str] = None,
                              sample_size: int = 10000) -> Dict:
        """
        Тестирование различных наборов временных признаков
        
        Args:
            symbols: Список символов для тестирования (None = все)
            sample_size: Размер выборки для каждого теста
            
        Returns:
            Словарь с результатами тестирования
        """
        self.logger.info("🕐 Начало абляционного тестирования временных признаков")
        
        # Загрузка базовых данных
        data_loader = CryptoDataLoader(self.config)
        
        # Определяем символы для тестирования
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']  # Тестовая выборка
        
        # Загружаем данные
        base_data = data_loader.get_data(
            symbols=symbols,
            start_date=self.config.get('data', {}).get('start_date'),
            limit=sample_size * len(symbols)
        )
        
        if base_data.empty:
            self.logger.error("❌ Не удалось загрузить данные для тестирования")
            return {}
        
        # Конфигурации для тестирования
        temporal_configs = {
            'baseline': {
                'allow_weekly_cycle': False,
                'allow_monthly_cycle': False,
                'allow_seasonal_cycle': False,
                'intraday_features': True,
                'session_features': True
            },
            'with_weekly': {
                'allow_weekly_cycle': True,
                'allow_monthly_cycle': False,
                'allow_seasonal_cycle': False,
                'intraday_features': True,
                'session_features': True
            },
            'with_monthly': {
                'allow_weekly_cycle': False,
                'allow_monthly_cycle': True,
                'allow_seasonal_cycle': False,
                'intraday_features': True,
                'session_features': True
            },
            'full_temporal': {
                'allow_weekly_cycle': True,
                'allow_monthly_cycle': True,
                'allow_seasonal_cycle': True,
                'intraday_features': True,
                'session_features': True
            }
        }
        
        results = {}
        
        for config_name, temp_config in temporal_configs.items():
            self.logger.info(f"   Тестируем конфигурацию: {config_name}")
            
            try:
                # Создаем TemporalEmbeddings с тестовой конфигурацией
                test_config = self.config.copy()
                test_config['features']['temporal'] = temp_config
                
                temporal_embedder = TemporalEmbeddings(test_config)
                
                # Применяем временные признаки
                enhanced_data = temporal_embedder.create_temporal_features(base_data.copy())
                
                # Анализ полученных признаков
                new_features = set(enhanced_data.columns) - set(base_data.columns)
                
                # Валидация на предмет утечек
                validation_results = self._validate_temporal_features(
                    enhanced_data, new_features, temp_config
                )
                
                # Статистика новых признаков
                feature_stats = self._calculate_feature_statistics(
                    enhanced_data, list(new_features)
                )
                
                results[config_name] = {
                    'config': temp_config,
                    'new_features_count': len(new_features),
                    'new_features': list(new_features),
                    'validation': validation_results,
                    'statistics': feature_stats,
                    'data_shape': enhanced_data.shape,
                    'risk_score': validation_results.get('risk_score', 0)
                }
                
                self.logger.info(f"     ✅ {config_name}: {len(new_features)} новых признаков")
                
            except Exception as e:
                self.logger.error(f"     ❌ Ошибка в {config_name}: {str(e)}")
                results[config_name] = {
                    'config': temp_config,
                    'error': str(e),
                    'risk_score': 100  # Максимальный риск при ошибке
                }
        
        # Сохранение результатов
        results_file = self.current_run_dir / "temporal_ablations.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 Результаты сохранены в: {results_file}")
        return results
    
    def test_market_context_features(self,
                                   symbols: List[str] = None,
                                   sample_size: int = 10000) -> Dict:
        """
        Тестирование рыночных контекстных признаков
        
        Args:
            symbols: Список символов для тестирования
            sample_size: Размер выборки
            
        Returns:
            Результаты тестирования
        """
        self.logger.info("📈 Начало тестирования рыночных контекстных признаков")
        
        # Загрузка данных
        data_loader = CryptoDataLoader(self.config)
        
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT']
        
        # Нужно больше символов для расчета рыночного контекста
        market_data = data_loader.get_data(
            symbols=symbols,
            start_date=self.config.get('data', {}).get('start_date'),
            limit=sample_size * len(symbols)
        )
        
        if market_data.empty or market_data['symbol'].nunique() < 3:
            self.logger.warning("⚠️ Недостаточно символов для расчета рыночного контекста")
            return {}
        
        # Конфигурации для тестирования
        market_configs = {
            'internal_only': {
                'internal_only': True,
                'fear_greed_components': 4,
                'market_breadth': True,
                'volatility_regime': True,
                'correlation_analysis': False
            },
            'with_correlations': {
                'internal_only': True,
                'fear_greed_components': 4,
                'market_breadth': True,
                'volatility_regime': True,
                'correlation_analysis': True
            },
            'extended_breadth': {
                'internal_only': True,
                'fear_greed_components': 6,
                'market_breadth': True,
                'volatility_regime': True,
                'correlation_analysis': True,
                'advanced_breadth': True
            }
        }
        
        results = {}
        
        for config_name, market_config in market_configs.items():
            self.logger.info(f"   Тестируем конфигурацию: {config_name}")
            
            try:
                # Создаем MarketContextFeatures с тестовой конфигурацией
                test_config = self.config.copy()
                test_config['features']['market_context'] = market_config
                
                market_context = MarketContextFeatures(test_config)
                
                # Применяем контекстные признаки
                enhanced_data = market_context.create_market_context_features(market_data.copy())
                
                # Анализ полученных признаков
                new_features = set(enhanced_data.columns) - set(market_data.columns)
                
                # Специальная валидация для рыночных признаков
                validation_results = self._validate_market_context_features(
                    enhanced_data, new_features, market_config
                )
                
                # Статистика новых признаков
                feature_stats = self._calculate_feature_statistics(
                    enhanced_data, list(new_features)
                )
                
                results[config_name] = {
                    'config': market_config,
                    'new_features_count': len(new_features),
                    'new_features': list(new_features),
                    'validation': validation_results,
                    'statistics': feature_stats,
                    'data_shape': enhanced_data.shape,
                    'risk_score': validation_results.get('risk_score', 0)
                }
                
                self.logger.info(f"     ✅ {config_name}: {len(new_features)} новых признаков")
                
            except Exception as e:
                self.logger.error(f"     ❌ Ошибка в {config_name}: {str(e)}")
                results[config_name] = {
                    'config': market_config,
                    'error': str(e),
                    'risk_score': 100
                }
        
        # Сохранение результатов
        results_file = self.current_run_dir / "market_context_ablations.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 Результаты сохранены в: {results_file}")
        return results
    
    def test_normalization_methods(self,
                                 symbols: List[str] = None,
                                 sample_size: int = 5000) -> Dict:
        """
        Тестирование различных методов нормализации
        
        Args:
            symbols: Список символов
            sample_size: Размер выборки
            
        Returns:
            Результаты тестирования
        """
        self.logger.info("🔄 Начало тестирования методов нормализации")
        
        # Загрузка данных с базовыми техническими индикаторами
        data_loader = CryptoDataLoader(self.config)
        feature_engineer = FeatureEngineer(self.config)
        
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT']  # Меньше символов для быстрого теста
        
        raw_data = data_loader.get_data(
            symbols=symbols,
            start_date=self.config.get('data', {}).get('start_date'),
            limit=sample_size * len(symbols)
        )
        
        # Создаем базовые технические индикаторы для тестирования нормализации
        base_data = feature_engineer.create_features(raw_data.copy())
        
        # Методы нормализации для тестирования
        normalization_methods = {
            'revin': {
                'method': 'revin',
                'window': 1000,
                'min_periods': 100,
                'causal_only': True
            },
            'robust': {
                'method': 'robust',
                'window': 1000,
                'min_periods': 100,
                'causal_only': True
            },
            'adaptive_zscore': {
                'method': 'adaptive_zscore',
                'window': 1000,
                'min_periods': 100,
                'causal_only': True
            },
            'quantile': {
                'method': 'quantile',
                'window': 1000,
                'min_periods': 100,
                'causal_only': True
            }
        }
        
        results = {}
        
        # Выбираем признаки для нормализации (избегаем bounded features)
        feature_columns = [col for col in base_data.columns 
                          if col not in ['datetime', 'symbol', 'timestamp'] and
                          not any(indicator in col.lower() for indicator in 
                                ['rsi', 'stoch_k', 'stoch_d', 'adx']) and
                          base_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        self.logger.info(f"   Тестируем нормализацию для {len(feature_columns)} признаков")
        
        for method_name, norm_config in normalization_methods.items():
            self.logger.info(f"   Тестируем метод: {method_name}")
            
            try:
                # Создаем нормализатор с тестовой конфигурацией
                normalizer = AdaptiveNormalization(**norm_config)
                
                # Применяем нормализацию
                test_data = base_data.copy()
                normalized_data = normalizer.fit_transform(
                    test_data, feature_columns, 'symbol'
                )
                
                # Валидация качества нормализации
                validation_results = normalizer.validate_normalization(
                    base_data, normalized_data, feature_columns
                )
                
                # Дополнительные проверки
                additional_checks = self._additional_normalization_checks(
                    base_data, normalized_data, feature_columns
                )
                
                results[method_name] = {
                    'config': norm_config,
                    'validation_results': validation_results,
                    'additional_checks': additional_checks,
                    'overall_quality': validation_results.get('overall_quality', 0),
                    'features_processed': len(feature_columns),
                    'risk_score': self._calculate_normalization_risk(validation_results)
                }
                
                self.logger.info(f"     ✅ {method_name}: качество {validation_results.get('overall_quality', 0):.2%}")
                
            except Exception as e:
                self.logger.error(f"     ❌ Ошибка в {method_name}: {str(e)}")
                results[method_name] = {
                    'config': norm_config,
                    'error': str(e),
                    'risk_score': 100
                }
        
        # Сохранение результатов
        results_file = self.current_run_dir / "normalization_ablations.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 Результаты сохранены в: {results_file}")
        return results
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Генерация комплексного отчета по всем абляционным тестам
        
        Returns:
            Сводный отчет
        """
        self.logger.info("📋 Генерация комплексного отчета")
        
        # Запуск всех тестов
        temporal_results = self.test_temporal_features()
        market_results = self.test_market_context_features()
        norm_results = self.test_normalization_methods()
        
        # Сводная оценка рисков
        risk_assessment = self._assess_overall_risk(
            temporal_results, market_results, norm_results
        )
        
        # Рекомендации по конфигурации
        recommendations = self._generate_recommendations(
            temporal_results, market_results, norm_results, risk_assessment
        )
        
        # Итоговый отчет
        comprehensive_report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'temporal_features': {
                    'configurations_tested': len(temporal_results),
                    'best_config': self._find_best_config(temporal_results),
                    'average_risk': np.mean([r.get('risk_score', 0) for r in temporal_results.values()])
                },
                'market_context': {
                    'configurations_tested': len(market_results),
                    'best_config': self._find_best_config(market_results),
                    'average_risk': np.mean([r.get('risk_score', 0) for r in market_results.values()])
                },
                'normalization': {
                    'methods_tested': len(norm_results),
                    'best_method': self._find_best_config(norm_results),
                    'average_quality': np.mean([r.get('overall_quality', 0) for r in norm_results.values()])
                }
            },
            'risk_assessment': risk_assessment,
            'recommendations': recommendations,
            'detailed_results': {
                'temporal_features': temporal_results,
                'market_context': market_results,
                'normalization': norm_results
            }
        }
        
        # Сохранение комплексного отчета
        report_file = self.current_run_dir / "comprehensive_ablation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=str)
        
        # Создание читаемого отчета
        readable_report = self._create_readable_report(comprehensive_report)
        readable_file = self.current_run_dir / "ABLATION_REPORT.md"
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(readable_report)
        
        self.logger.info(f"📊 Комплексный отчет сохранен:")
        self.logger.info(f"   JSON: {report_file}")
        self.logger.info(f"   Markdown: {readable_file}")
        
        return comprehensive_report
    
    def _validate_temporal_features(self, df: pd.DataFrame, 
                                  features: set, 
                                  config: Dict) -> Dict:
        """Валидация временных признаков на предмет утечек"""
        
        risk_factors = []
        risk_score = 0
        
        for feature in features:
            feature_lower = feature.lower()
            
            # Проверка на опасные паттерны
            if any(risky in feature_lower for risky in [
                'day_sin', 'day_cos', 'month_', 'quarter_', 'seasonal_'
            ]):
                risk_factors.append(f"ВЫСОКИЙ РИСК: {feature} - может использовать будущую информацию")
                risk_score += 30
            
            # Проверка на NaN и бесконечности
            if feature in df.columns:
                nan_pct = df[feature].isna().sum() / len(df)
                if nan_pct > 0.1:
                    risk_factors.append(f"МНОГО NaN: {feature} ({nan_pct:.1%})")
                    risk_score += 10
                
                if np.isinf(df[feature]).any():
                    risk_factors.append(f"БЕСКОНЕЧНОСТИ: {feature}")
                    risk_score += 20
        
        # Проверка конфигурации
        if config.get('allow_weekly_cycle', False):
            risk_factors.append("КОНФИГ: разрешены недельные циклы")
            risk_score += 15
            
        if config.get('allow_monthly_cycle', False):
            risk_factors.append("КОНФИГ: разрешены месячные циклы")
            risk_score += 25
            
        if config.get('allow_seasonal_cycle', False):
            risk_factors.append("КОНФИГ: разрешены сезонные циклы")
            risk_score += 35
        
        return {
            'risk_factors': risk_factors,
            'risk_score': min(risk_score, 100),
            'safe_features': len(features) - len([f for f in risk_factors if 'РИСК:' in f])
        }
    
    def _validate_market_context_features(self, df: pd.DataFrame, 
                                        features: set, 
                                        config: Dict) -> Dict:
        """Валидация рыночных контекстных признаков"""
        
        risk_factors = []
        risk_score = 0
        
        # Проверка на использование внешних данных
        if not config.get('internal_only', True):
            risk_factors.append("ВЫСОКИЙ РИСК: используются внешние источники данных")
            risk_score += 40
        
        # Проверка качества признаков
        for feature in features:
            if feature in df.columns:
                # Проверка на константные значения
                if df[feature].nunique() <= 1:
                    risk_factors.append(f"КОНСТАНТНЫЙ: {feature}")
                    risk_score += 5
                
                # Проверка на корреляцию с будущими данными (упрощенная)
                if 'fear_greed' in feature.lower():
                    # Простая проверка: значения не должны резко изменяться
                    volatility = df[feature].std()
                    if volatility > 50:  # Слишком волатильный Fear & Greed
                        risk_factors.append(f"ВОЛАТИЛЬНЫЙ: {feature} (std={volatility:.1f})")
                        risk_score += 10
        
        return {
            'risk_factors': risk_factors,
            'risk_score': min(risk_score, 100),
            'internal_only': config.get('internal_only', True)
        }
    
    def _additional_normalization_checks(self, original: pd.DataFrame,
                                       normalized: pd.DataFrame,
                                       feature_columns: List[str]) -> Dict:
        """Дополнительные проверки качества нормализации"""
        
        checks = {
            'extreme_values_count': 0,
            'mean_shift_issues': [],
            'variance_issues': [],
            'distribution_changes': []
        }
        
        for col in feature_columns:
            if col in normalized.columns and col in original.columns:
                norm_values = normalized[col].dropna()
                orig_values = original[col].dropna()
                
                # Подсчет экстремальных значений после нормализации
                extreme_count = ((norm_values < -5) | (norm_values > 5)).sum()
                checks['extreme_values_count'] += extreme_count
                
                # Проверка смещения среднего
                if abs(norm_values.mean()) > 0.2:
                    checks['mean_shift_issues'].append(col)
                
                # Проверка дисперсии
                if norm_values.std() < 0.5 or norm_values.std() > 2.0:
                    checks['variance_issues'].append(col)
        
        return checks
    
    def _calculate_feature_statistics(self, df: pd.DataFrame, 
                                    features: List[str]) -> Dict:
        """Расчет статистик по новым признакам"""
        
        stats = {
            'count': len(features),
            'numeric_features': 0,
            'categorical_features': 0,
            'binary_features': 0,
            'nan_percentage': {},
            'variance': {},
            'correlation_with_price': {}
        }
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            values = df[feature].dropna()
            if len(values) == 0:
                continue
            
            # Определение типа признака
            if values.dtype in ['int64', 'float64', 'int32', 'float32']:
                stats['numeric_features'] += 1
                
                # Проверка на бинарность
                unique_vals = values.unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                    stats['binary_features'] += 1
                
                # Статистики
                stats['nan_percentage'][feature] = df[feature].isna().sum() / len(df)
                stats['variance'][feature] = float(values.var()) if len(values) > 1 else 0
                
                # Корреляция с ценой (если есть)
                if 'close' in df.columns:
                    corr = df[feature].corr(df['close'])
                    if not np.isnan(corr):
                        stats['correlation_with_price'][feature] = float(corr)
            else:
                stats['categorical_features'] += 1
        
        return stats
    
    def _calculate_normalization_risk(self, validation_results: Dict) -> int:
        """Расчет риска для метода нормализации"""
        
        risk_score = 0
        quality = validation_results.get('overall_quality', 0)
        
        # Снижаем риск при высоком качестве
        if quality > 0.8:
            risk_score = 10
        elif quality > 0.6:
            risk_score = 25
        elif quality > 0.4:
            risk_score = 50
        else:
            risk_score = 75
        
        # Увеличиваем риск при плохих показателях
        failed_features = validation_results.get('total_features', 1) - validation_results.get('features_passed', 0)
        risk_score += failed_features * 5
        
        return min(risk_score, 100)
    
    def _assess_overall_risk(self, temporal_results: Dict,
                           market_results: Dict,
                           norm_results: Dict) -> Dict:
        """Оценка общего риска всех компонентов"""
        
        # Средние риски по категориям
        temporal_risk = np.mean([r.get('risk_score', 0) for r in temporal_results.values()]) if temporal_results else 0
        market_risk = np.mean([r.get('risk_score', 0) for r in market_results.values()]) if market_results else 0
        norm_risk = np.mean([r.get('risk_score', 0) for r in norm_results.values()]) if norm_results else 0
        
        overall_risk = (temporal_risk * 0.4 + market_risk * 0.3 + norm_risk * 0.3)
        
        risk_level = "НИЗКИЙ"
        if overall_risk > 60:
            risk_level = "ВЫСОКИЙ"
        elif overall_risk > 30:
            risk_level = "СРЕДНИЙ"
        
        return {
            'overall_risk_score': overall_risk,
            'risk_level': risk_level,
            'component_risks': {
                'temporal_features': temporal_risk,
                'market_context': market_risk,
                'normalization': norm_risk
            },
            'recommendations': self._get_risk_recommendations(overall_risk)
        }
    
    def _get_risk_recommendations(self, risk_score: float) -> List[str]:
        """Генерация рекомендаций на основе уровня риска"""
        
        recommendations = []
        
        if risk_score > 60:
            recommendations.extend([
                "🚨 ВЫСОКИЙ РИСК: рекомендуется использовать только базовую конфигурацию",
                "🔒 Отключить все потенциально опасные временные признаки",
                "📊 Использовать только internal_only режим для рыночного контекста",
                "🧪 Провести дополнительное тестирование на исторических данных"
            ])
        elif risk_score > 30:
            recommendations.extend([
                "⚠️ СРЕДНИЙ РИСК: требуется осторожность при внедрении",
                "✅ Можно использовать безопасные временные признаки",
                "📈 Рыночный контекст только с internal данными",
                "🔍 Мониторинг качества предсказаний после внедрения"
            ])
        else:
            recommendations.extend([
                "✅ НИЗКИЙ РИСК: компоненты можно безопасно использовать",
                "🚀 Рекомендуется поэтапное внедрение с мониторингом",
                "📊 Можно экспериментировать с расширенными функциями"
            ])
        
        return recommendations
    
    def _find_best_config(self, results: Dict) -> str:
        """Поиск лучшей конфигурации по минимальному риску"""
        
        if not results:
            return "Нет данных"
        
        best_config = min(results.keys(), 
                         key=lambda k: results[k].get('risk_score', 100))
        return best_config
    
    def _generate_recommendations(self, temporal_results: Dict,
                                market_results: Dict,
                                norm_results: Dict,
                                risk_assessment: Dict) -> Dict:
        """Генерация итоговых рекомендаций"""
        
        recommendations = {
            'production_config': {
                'features': {
                    'temporal': {
                        'allow_weekly_cycle': False,
                        'allow_monthly_cycle': False,
                        'allow_seasonal_cycle': False,
                        'intraday_features': True,
                        'session_features': True
                    },
                    'market_context': {
                        'internal_only': True,
                        'fear_greed_components': 4,
                        'market_breadth': True,
                        'volatility_regime': True,
                        'correlation_analysis': False
                    },
                    'normalization': {
                        'method': self._find_best_config(norm_results),
                        'window': 1000,
                        'min_periods': 100,
                        'causal_only': True
                    }
                }
            },
            'implementation_stages': [
                {
                    'stage': 1,
                    'name': "Базовые компоненты",
                    'components': ['normalization', 'basic_temporal'],
                    'risk': "Низкий",
                    'timeline': "1-2 недели"
                },
                {
                    'stage': 2,
                    'name': "Рыночный контекст",
                    'components': ['market_context_internal'],
                    'risk': "Низкий-Средний",
                    'timeline': "2-3 недели"
                },
                {
                    'stage': 3,
                    'name': "Расширенные функции",
                    'components': ['advanced_temporal', 'correlations'],
                    'risk': "Средний",
                    'timeline': "4-6 недель"
                }
            ],
            'monitoring_requirements': [
                "Отслеживание качества предсказаний после каждого этапа",
                "Мониторинг распределения признаков на новых данных",
                "Проверка корреляции с историческими паттернами",
                "Контроль утечек данных через back-testing"
            ]
        }
        
        return recommendations
    
    def _create_readable_report(self, report: Dict) -> str:
        """Создание читаемого отчета в формате Markdown"""
        
        timestamp = report['timestamp']
        risk_assessment = report['risk_assessment']
        recommendations = report['recommendations']
        
        md_content = f"""# Отчет по абляционному тестированию
        
**Дата:** {timestamp}  
**Общий уровень риска:** {risk_assessment['risk_level']} ({risk_assessment['overall_risk_score']:.1f}/100)

## 📊 Сводка тестирования

### Временные признаки
- **Конфигураций протестировано:** {report['test_summary']['temporal_features']['configurations_tested']}
- **Лучшая конфигурация:** `{report['test_summary']['temporal_features']['best_config']}`
- **Средний риск:** {report['test_summary']['temporal_features']['average_risk']:.1f}/100

### Рыночный контекст
- **Конфигураций протестировано:** {report['test_summary']['market_context']['configurations_tested']}
- **Лучшая конфигурация:** `{report['test_summary']['market_context']['best_config']}`
- **Средний риск:** {report['test_summary']['market_context']['average_risk']:.1f}/100

### Нормализация
- **Методов протестировано:** {report['test_summary']['normalization']['methods_tested']}
- **Лучший метод:** `{report['test_summary']['normalization']['best_method']}`
- **Среднее качество:** {report['test_summary']['normalization']['average_quality']:.1%}

## ⚠️ Оценка рисков

"""
        
        # Добавляем рекомендации по рискам
        for rec in risk_assessment['recommendations']:
            md_content += f"- {rec}\n"
        
        md_content += f"""
## 🚀 Рекомендуемая конфигурация

```yaml
{yaml.dump(recommendations['production_config'], default_flow_style=False)}
```

## 📋 Поэтапный план внедрения

"""
        
        for stage in recommendations['implementation_stages']:
            md_content += f"""### Этап {stage['stage']}: {stage['name']}
- **Компоненты:** {', '.join(stage['components'])}
- **Риск:** {stage['risk']}
- **Временные рамки:** {stage['timeline']}

"""
        
        md_content += """## 🔍 Требования к мониторингу

"""
        for req in recommendations['monitoring_requirements']:
            md_content += f"- {req}\n"
        
        md_content += f"""
## 📈 Результаты компонентов

### Риски по компонентам:
- **Временные признаки:** {risk_assessment['component_risks']['temporal_features']:.1f}/100
- **Рыночный контекст:** {risk_assessment['component_risks']['market_context']:.1f}/100
- **Нормализация:** {risk_assessment['component_risks']['normalization']:.1f}/100

---
*Отчет сгенерирован автоматически системой абляционного тестирования*
"""
        
        return md_content


def main():
    """Главная функция для запуска абляционного тестирования"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Абляционное тестирование компонентов системы")
    parser.add_argument('--test', choices=['temporal', 'market', 'normalization', 'all'], 
                       default='all', help='Тип теста для запуска')
    parser.add_argument('--symbols', nargs='+', 
                       help='Список символов для тестирования')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Размер выборки для тестирования')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Путь к конфигурационному файлу')
    
    args = parser.parse_args()
    
    # Инициализация тестера
    tester = AblationTester(args.config)
    
    print("🧪 Запуск абляционного тестирования...")
    print(f"📊 Тест: {args.test}")
    print(f"🎯 Символы: {args.symbols or 'по умолчанию'}")
    print(f"📏 Размер выборки: {args.sample_size}")
    
    # Запуск тестов
    if args.test == 'temporal':
        results = tester.test_temporal_features(args.symbols, args.sample_size)
    elif args.test == 'market':
        results = tester.test_market_context_features(args.symbols, args.sample_size)
    elif args.test == 'normalization':
        results = tester.test_normalization_methods(args.symbols, args.sample_size)
    elif args.test == 'all':
        results = tester.generate_comprehensive_report()
    
    print(f"✅ Тестирование завершено!")
    print(f"📁 Результаты сохранены в: {tester.current_run_dir}")


if __name__ == "__main__":
    main()