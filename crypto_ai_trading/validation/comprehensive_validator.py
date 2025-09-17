#!/usr/bin/env python3
"""
Комплексная система валидации торговых моделей
Объединяет все методы валидации в единый пайплайн
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger

# Импортируем наши валидаторы
from validation.walk_forward_validator import WalkForwardValidator, WalkForwardConfig
from validation.monte_carlo_tester import MonteCarloTester, MonteCarloConfig
from validation.data_leak_detector import DataLeakDetector, LeakDetectionConfig
from validation.prediction_quality_analyzer import PredictionQualityAnalyzer
from validation.regime_stress_tester import RegimeStressTester, MarketRegimeConfig
try:
    from validation.ablations import AblationTester
except ImportError:
    AblationTester = None

# Импортируем торговые компоненты
from trading.unified_backtester import UnifiedBacktester
from data.data_loader import CryptoDataLoader


class ComprehensiveValidator:
    """
    Комплексная система валидации торговых моделей
    
    Включает:
    1. Walk-Forward валидацию
    2. Monte Carlo тестирование
    3. Детекцию утечек данных
    4. Анализ качества предсказаний
    5. Стресс-тестирование рыночных режимов
    6. Абляционное тестирование (опционально)
    7. Реалистичный бэктестинг
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Инициализация комплексного валидатора
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.logger = get_logger("ComprehensiveValidator")
        
        # Загружаем конфигурацию
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # Создаем директорию для результатов
        self.results_dir = Path("validation/comprehensive_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Временные файлы для текущего запуска
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.results_dir / f"validation_session_{self.current_session}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Результаты валидации
        self.validation_results = {
            'session_id': self.current_session,
            'timestamp': datetime.now().isoformat(),
            'config_used': self.config,
            'results': {}
        }
        
        self.logger.info("🚀 Инициализация комплексной системы валидации")
        self.logger.info(f"📁 Результаты будут сохранены в: {self.session_dir}")
        
    def run_data_leak_detection(self, 
                              data: pd.DataFrame,
                              feature_columns: List[str],
                              target_columns: List[str]) -> Dict:
        """
        Запускает детекцию утечек данных
        
        Args:
            data: Данные для анализа
            feature_columns: Список признаков
            target_columns: Список целевых переменных
            
        Returns:
            Результаты детекции утечек
        """
        
        self.logger.info("🔍 === ДЕТЕКЦИЯ УТЕЧЕК ДАННЫХ ===")
        
        try:
            # Создаем детектор утечек
            leak_config = LeakDetectionConfig()
            detector = DataLeakDetector(leak_config)
            
            # Определяем временные границы из конфигурации
            time_column = 'datetime' if 'datetime' in data.columns else 'timestamp'
            
            # Запускаем комплексную детекцию
            leak_results = detector.comprehensive_leak_detection(
                data=data,
                feature_columns=feature_columns,
                target_columns=target_columns,
                time_column=time_column
            )
            
            self.validation_results['results']['data_leak_detection'] = leak_results
            
            # Сохраняем отдельный отчет
            leak_report_path = self.session_dir / "data_leak_detection_report.json"
            with open(leak_report_path, 'w', encoding='utf-8') as f:
                json.dump(leak_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Выводим критические предупреждения
            risk_info = leak_results.get('risk_assessment', {})
            if risk_info.get('risk_level') in ['CRITICAL', 'HIGH']:
                self.logger.error(f"🚨 КРИТИЧЕСКИЕ УТЕЧКИ ОБНАРУЖЕНЫ!")
                self.logger.error(f"   Уровень риска: {risk_info['risk_level']}")
                self.logger.error(f"   Критичных утечек: {risk_info.get('critical_leaks', 0)}")
                
                # Останавливаем валидацию если есть критические утечки
                if risk_info.get('critical_leaks', 0) > 0:
                    self.logger.error("❌ Валидация прервана из-за критических утечек данных")
                    return leak_results
            else:
                self.logger.info("✅ Критических утечек данных не обнаружено")
            
            return leak_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в детекции утечек: {str(e)}")
            return {'error': str(e)}
    
    def run_prediction_quality_analysis(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Запускает анализ качества предсказаний
        
        Args:
            y_true: Истинные значения (N, 20)
            y_pred: Предсказанные значения (N, 20)
            y_pred_proba: Вероятности (опционально)
            
        Returns:
            Результаты анализа качества
        """
        
        self.logger.info("📊 === АНАЛИЗ КАЧЕСТВА ПРЕДСКАЗАНИЙ ===")
        
        try:
            # Создаем анализатор
            analyzer = PredictionQualityAnalyzer()
            
            # Запускаем анализ
            quality_results = analyzer.analyze_all_variables(y_true, y_pred, y_pred_proba)
            
            self.validation_results['results']['prediction_quality'] = quality_results
            
            # Сохраняем результаты
            analyzer.save_analysis_results(quality_results, str(self.session_dir))
            
            # Выводим ключевые метрики
            summary = quality_results.get('summary', {})
            
            if 'regression' in summary:
                reg_info = summary['regression']
                self.logger.info(f"📈 Регрессия - средний R²: {reg_info.get('mean_r2', 0):.3f}")
                
            if 'classification' in summary:
                class_info = summary['classification'] 
                self.logger.info(f"🎯 Классификация - средний F1: {class_info.get('mean_f1', 0):.3f}")
                
            if 'binary_classification' in summary:
                binary_info = summary['binary_classification']
                self.logger.info(f"🎲 Бинарная - средний AUC: {binary_info.get('mean_auc', 0):.3f}")
            
            return quality_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в анализе качества предсказаний: {str(e)}")
            return {'error': str(e)}
    
    def run_monte_carlo_testing(self, returns: np.ndarray) -> Dict:
        """
        Запускает Monte Carlo тестирование
        
        Args:
            returns: Массив доходностей торговой стратегии
            
        Returns:
            Результаты Monte Carlo тестов
        """
        
        self.logger.info("🎲 === MONTE CARLO ТЕСТИРОВАНИЕ ===")
        
        try:
            # Создаем конфигурацию для Monte Carlo
            mc_config = MonteCarloConfig(
                n_permutations=1000,
                n_bootstrap=500,
                confidence_level=0.95
            )
            
            # Создаем тестер
            mc_tester = MonteCarloTester(mc_config)
            
            # Запускаем комплексную валидацию
            mc_results = mc_tester.comprehensive_validation(
                strategy_returns=returns,
                save_results=False  # Мы сами сохраним результаты
            )
            
            self.validation_results['results']['monte_carlo'] = mc_results
            
            # Сохраняем результаты
            mc_report_path = self.session_dir / "monte_carlo_validation_report.json"
            with open(mc_report_path, 'w', encoding='utf-8') as f:
                json.dump(mc_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Выводим ключевые результаты
            assessment = mc_results.get('overall_assessment', {})
            
            sig_info = assessment.get('statistical_significance', {})
            self.logger.info(f"📊 Статистическая значимость: {sig_info.get('significance_score', 0):.1%}")
            
            robust_info = assessment.get('robustness', {})
            self.logger.info(f"🛡️ Устойчивость: {robust_info.get('robustness_rating', 'НЕИЗВЕСТНО')}")
            
            # Предупреждения о рисках
            if not sig_info.get('overall_significant', False):
                self.logger.warning("⚠️ Стратегия не демонстрирует статистическую значимость!")
                
            if assessment.get('overfitting', {}).get('overfitting_detected', False):
                self.logger.warning("⚠️ Обнаружены признаки переобучения!")
            
            return mc_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в Monte Carlo тестировании: {str(e)}")
            return {'error': str(e)}
    
    def run_walk_forward_validation(self, 
                                   model_trainer_func,
                                   data: pd.DataFrame) -> Dict:
        """
        Запускает Walk-Forward валидацию
        
        Args:
            model_trainer_func: Функция обучения модели
            data: Данные для валидации
            
        Returns:
            Результаты Walk-Forward валидации
        """
        
        self.logger.info("🔄 === WALK-FORWARD ВАЛИДАЦИЯ ===")
        
        try:
            # Создаем конфигурацию
            wf_config = WalkForwardConfig(
                training_window_days=180,
                validation_window_days=30,
                test_window_days=30,
                step_size_days=7,
                embargo_days=1
            )
            
            # Создаем валидатор
            wf_validator = WalkForwardValidator(self.config, wf_config)
            
            # Запускаем анализ
            wf_results = wf_validator.run_walk_forward_analysis(
                model_trainer=model_trainer_func,
                data=data,
                save_results=False
            )
            
            self.validation_results['results']['walk_forward'] = wf_results
            
            # Сохраняем результаты
            wf_report_path = self.session_dir / "walk_forward_validation_report.json"
            with open(wf_report_path, 'w', encoding='utf-8') as f:
                json.dump(wf_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Выводим результаты
            summary = wf_results.get('summary', {})
            self.logger.info(f"📈 Средняя доходность: {summary.get('mean_return', 0):.2%}")
            self.logger.info(f"📊 Средний Sharpe: {summary.get('mean_sharpe', 0):.2f}")
            
            stability = wf_results.get('stability', {})
            self.logger.info(f"🔄 Стабильность: {stability.get('stability_score', 0):.2f}")
            
            # Предупреждения
            if stability.get('performance_degradation', False):
                self.logger.warning("⚠️ Обнаружена деградация производительности во времени!")
                
            overfitting = wf_results.get('overfitting', {})
            if overfitting.get('overfitting_detected', False):
                self.logger.warning("⚠️ Обнаружены признаки переобучения!")
            
            return wf_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в Walk-Forward валидации: {str(e)}")
            return {'error': str(e)}
    
    def run_regime_stress_testing(self, 
                                price_data: pd.DataFrame,
                                predictions: pd.DataFrame,
                                actual_returns: pd.DataFrame,
                                trading_results: pd.DataFrame = None) -> Dict:
        """
        Запускает стресс-тестирование рыночных режимов
        
        Args:
            price_data: Данные цен (timestamp, symbol, close, volume)
            predictions: Предсказания модели
            actual_returns: Фактические доходности
            trading_results: Результаты торговли (опционально)
            
        Returns:
            Результаты стресс-тестирования режимов
        """
        
        self.logger.info("🌪️ === СТРЕСС-ТЕСТИРОВАНИЕ РЫНОЧНЫХ РЕЖИМОВ ===")
        
        try:
            # Создаем конфигурацию для режимов
            regime_config = MarketRegimeConfig(
                trend_window=30,
                volatility_window=20,
                volume_window=30,
                bull_threshold=0.05,
                bear_threshold=-0.05,
                high_vol_percentile=80,
                low_liquidity_percentile=20,
                crisis_vol_threshold=0.8,
                min_regime_samples=100
            )
            
            # Создаем стресс-тестер
            stress_tester = RegimeStressTester(regime_config)
            
            # Определяем рыночные режимы
            self.logger.info("🔍 Определение рыночных режимов...")
            regime_data = stress_tester.identify_market_regimes(price_data)
            
            # Тестируем производительность по режимам
            self.logger.info("🧪 Анализ производительности по режимам...")
            regime_results = stress_tester.test_model_performance_by_regime(
                regime_data, predictions, actual_returns, trading_results
            )
            
            # Генерируем отчет
            self.logger.info("📊 Генерация отчета по режимам...")
            stress_report = stress_tester.generate_regime_stress_report()
            
            # Комбинируем все результаты
            combined_results = {
                'regime_classification': {
                    'total_samples': len(regime_data),
                    'regimes_identified': regime_data['regime'].value_counts().to_dict(),
                    'regime_periods': {
                        regime: {
                            'start_date': regime_data[regime_data['regime'] == regime]['timestamp'].min(),
                            'end_date': regime_data[regime_data['regime'] == regime]['timestamp'].max(),
                            'duration_days': (regime_data[regime_data['regime'] == regime]['timestamp'].max() - 
                                            regime_data[regime_data['regime'] == regime]['timestamp'].min()).days
                        }
                        for regime in regime_data['regime'].unique() if pd.notna(regime)
                    }
                },
                'performance_by_regime': regime_results,
                'stress_test_report': stress_report,
                'regime_raw_data': regime_data  # Для дальнейшего анализа
            }
            
            self.validation_results['results']['regime_stress_testing'] = combined_results
            
            # Сохраняем результаты
            regime_report_path = self.session_dir / "regime_stress_testing_report.json"
            # Не сохраняем raw_data в JSON (слишком большой)
            save_data = combined_results.copy()
            save_data.pop('regime_raw_data', None)
            
            with open(regime_report_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Экспортируем детальные результаты в CSV
            regime_csv_path = self.session_dir / "regime_analysis_detailed.csv"
            stress_tester.export_regime_analysis(str(regime_csv_path))
            
            # Выводим ключевые результаты
            summary = stress_report.get('summary', {})
            self.logger.info(f"🎯 Протестировано режимов: {summary.get('total_regimes_tested', 0)}")
            self.logger.info(f"📈 Средний Sharpe по режимам: {summary.get('avg_sharpe_ratio', 0):.3f}")
            self.logger.info(f"🎲 Консистентность: {summary.get('regime_consistency', 0):.3f}")
            
            # Предупреждения
            risk_assessment = stress_report.get('risk_assessment', {})
            high_risk_regimes = risk_assessment.get('high_risk_regimes', [])
            
            if len(high_risk_regimes) > 0:
                self.logger.warning(f"⚠️ Высокорискованные режимы: {', '.join(high_risk_regimes)}")
            
            robustness_score = risk_assessment.get('regime_robustness_score', 0)
            if robustness_score < 0.6:
                self.logger.warning(f"⚠️ Низкая робастность модели: {robustness_score:.1%}")
            
            worst_case = risk_assessment.get('worst_case_scenario', {})
            if worst_case.get('max_expected_loss', 0) < -0.20:
                self.logger.error(f"🚨 Критическая просадка в худшем сценарии: {worst_case.get('max_expected_loss', 0):.1%}")
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в стресс-тестировании режимов: {str(e)}")
            return {'error': str(e)}
    
    def run_backtesting_validation(self, model, test_loader) -> Dict:
        """
        Запускает реалистичный бэктестинг
        
        Args:
            model: Обученная модель
            test_loader: DataLoader с тестовыми данными
            
        Returns:
            Результаты бэктестинга
        """
        
        self.logger.info("💰 === РЕАЛИСТИЧНЫЙ БЭКТЕСТИНГ ===")
        
        try:
            # Создаем бэктестер
            backtester = UnifiedBacktester(self.config)
            
            # Запускаем бэктест
            backtest_results = backtester.run_backtest(model, test_loader)
            
            self.validation_results['results']['backtesting'] = backtest_results
            
            # Сохраняем результаты
            bt_report_path = self.session_dir / "backtesting_report.json"
            with open(bt_report_path, 'w', encoding='utf-8') as f:
                json.dump(backtest_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Выводим ключевые метрики
            self.logger.info(f"📊 Всего сделок: {backtest_results.get('total_trades', 0)}")
            self.logger.info(f"🎯 Win Rate: {backtest_results.get('win_rate', 0):.1%}")
            self.logger.info(f"📈 Общая доходность: {backtest_results.get('total_return', 0):.1%}")
            self.logger.info(f"📊 Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"📉 Max Drawdown: {backtest_results.get('max_drawdown', 0):.1%}")
            
            # Проверка на реалистичность результатов
            if backtest_results.get('win_rate', 0) > 0.65:
                self.logger.warning("⚠️ Слишком высокий Win Rate - возможна переоптимизация")
                
            if backtest_results.get('sharpe_ratio', 0) > 3.0:
                self.logger.warning("⚠️ Нереалистично высокий Sharpe Ratio")
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в бэктестинге: {str(e)}")
            return {'error': str(e)}
    
    def run_comprehensive_validation(self,
                                   model_trainer_func=None,
                                   model=None,
                                   test_loader=None,
                                   data: pd.DataFrame = None,
                                   y_true: np.ndarray = None,
                                   y_pred: np.ndarray = None,
                                   returns: np.ndarray = None,
                                   feature_columns: List[str] = None,
                                   target_columns: List[str] = None,
                                   price_data: pd.DataFrame = None,
                                   predictions: pd.DataFrame = None,
                                   actual_returns: pd.DataFrame = None,
                                   trading_results: pd.DataFrame = None) -> Dict:
        """
        Запускает комплексную валидацию со всеми компонентами
        
        Args:
            model_trainer_func: Функция обучения модели (для walk-forward)
            model: Обученная модель (для бэктестинга)
            test_loader: DataLoader (для бэктестинга)
            data: Исходные данные (для детекции утечек)
            y_true: Истинные значения (для анализа качества)
            y_pred: Предсказания (для анализа качества)
            returns: Доходности стратегии (для Monte Carlo)
            feature_columns: Список признаков
            target_columns: Список целевых переменных
            price_data: Данные цен для стресс-тестирования режимов
            predictions: Предсказания модели для режимов
            actual_returns: Фактические доходности для режимов
            trading_results: Результаты торговли для режимов
            
        Returns:
            Комплексные результаты валидации
        """
        
        self.logger.info("🎯 === ЗАПУСК КОМПЛЕКСНОЙ ВАЛИДАЦИИ ===")
        self.logger.info(f"📅 Сессия: {self.current_session}")
        
        validation_pipeline = []
        
        # 1. ДЕТЕКЦИЯ УТЕЧЕК ДАННЫХ (приоритет #1)
        if data is not None and feature_columns is not None and target_columns is not None:
            validation_pipeline.append('data_leak_detection')
            
        # 2. АНАЛИЗ КАЧЕСТВА ПРЕДСКАЗАНИЙ  
        if y_true is not None and y_pred is not None:
            validation_pipeline.append('prediction_quality')
            
        # 3. MONTE CARLO ТЕСТИРОВАНИЕ
        if returns is not None:
            validation_pipeline.append('monte_carlo')
            
        # 4. WALK-FORWARD ВАЛИДАЦИЯ (если есть trainer)
        if model_trainer_func is not None and data is not None:
            validation_pipeline.append('walk_forward')
            
        # 5. СТРЕСС-ТЕСТИРОВАНИЕ РЕЖИМОВ (если есть данные цен)
        if (price_data is not None and predictions is not None and 
            actual_returns is not None):
            validation_pipeline.append('regime_stress_testing')
            
        # 6. БЭКТЕСТИНГ (если есть модель)
        if model is not None and test_loader is not None:
            validation_pipeline.append('backtesting')
        
        self.logger.info(f"📋 Пайплайн валидации: {' -> '.join(validation_pipeline)}")
        
        # Выполняем пайплайн
        validation_successful = True
        critical_issues = []
        
        for step in validation_pipeline:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"🔄 Выполнение этапа: {step}")
            
            try:
                if step == 'data_leak_detection':
                    results = self.run_data_leak_detection(data, feature_columns, target_columns)
                    
                    # Критическая проверка - останавливаем если есть критические утечки
                    risk_info = results.get('risk_assessment', {})
                    if risk_info.get('critical_leaks', 0) > 0:
                        critical_issues.append("Обнаружены критические утечки данных")
                        self.logger.error("🚨 Критические утечки! Дальнейшая валидация может быть недостоверной")
                        validation_successful = False
                        break
                        
                elif step == 'prediction_quality':
                    results = self.run_prediction_quality_analysis(y_true, y_pred)
                    
                elif step == 'monte_carlo':
                    results = self.run_monte_carlo_testing(returns)
                    
                elif step == 'walk_forward':
                    results = self.run_walk_forward_validation(model_trainer_func, data)
                    
                elif step == 'regime_stress_testing':
                    results = self.run_regime_stress_testing(price_data, predictions, actual_returns, trading_results)
                    
                elif step == 'backtesting':
                    results = self.run_backtesting_validation(model, test_loader)
                
                if 'error' in results:
                    critical_issues.append(f"Ошибка в {step}: {results['error']}")
                    validation_successful = False
                    
            except Exception as e:
                self.logger.error(f"❌ Критическая ошибка в {step}: {str(e)}")
                critical_issues.append(f"Критическая ошибка в {step}")
                validation_successful = False
        
        # Генерируем итоговый отчет
        final_report = self._generate_final_report(validation_successful, critical_issues)
        
        self.validation_results['final_report'] = final_report
        self.validation_results['validation_successful'] = validation_successful
        self.validation_results['critical_issues'] = critical_issues
        
        # Сохраняем комплексные результаты
        self._save_comprehensive_results()
        
        # Выводим итоги
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🎯 === ИТОГИ КОМПЛЕКСНОЙ ВАЛИДАЦИИ ===")
        
        if validation_successful:
            self.logger.info("✅ Валидация успешно завершена")
            self.logger.info(f"📊 Общая оценка модели: {final_report.get('overall_grade', 'N/A')}")
        else:
            self.logger.error("❌ Валидация завершена с критическими проблемами")
            
        if critical_issues:
            self.logger.error("🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
            for issue in critical_issues:
                self.logger.error(f"   • {issue}")
                
        self.logger.info(f"📁 Все результаты сохранены в: {self.session_dir}")
        
        return self.validation_results
    
    def _generate_final_report(self, validation_successful: bool, critical_issues: List[str]) -> Dict:
        """Генерирует итоговый отчет валидации"""
        
        results = self.validation_results['results']
        
        # Собираем ключевые метрики
        key_metrics = {}
        
        # Из анализа качества предсказаний
        if 'prediction_quality' in results:
            pq = results['prediction_quality'].get('summary', {})
            if 'classification' in pq:
                key_metrics['avg_f1_score'] = pq['classification'].get('mean_f1')
        
        # Из Monte Carlo
        if 'monte_carlo' in results:
            mc = results['monte_carlo'].get('overall_assessment', {})
            key_metrics['statistical_significance'] = mc.get('statistical_significance', {}).get('overall_significant')
            key_metrics['robustness_rating'] = mc.get('robustness', {}).get('robustness_rating')
        
        # Из Walk-Forward
        if 'walk_forward' in results:
            wf = results['walk_forward'].get('summary', {})
            key_metrics['mean_sharpe'] = wf.get('mean_sharpe')
            key_metrics['stability_score'] = results['walk_forward'].get('stability', {}).get('stability_score')
        
        # Из бэктестинга
        if 'backtesting' in results:
            bt = results['backtesting']
            key_metrics['win_rate'] = bt.get('win_rate')
            key_metrics['total_return'] = bt.get('total_return')
            key_metrics['max_drawdown'] = bt.get('max_drawdown')
        
        # Из детекции утечек
        if 'data_leak_detection' in results:
            leak = results['data_leak_detection'].get('risk_assessment', {})
            key_metrics['leak_risk_level'] = leak.get('risk_level')
            key_metrics['critical_leaks'] = leak.get('critical_leaks', 0)
        
        # Из стресс-тестирования режимов
        if 'regime_stress_testing' in results:
            regime_report = results['regime_stress_testing'].get('stress_test_report', {})
            summary = regime_report.get('summary', {})
            risk_assessment = regime_report.get('risk_assessment', {})
            
            key_metrics['regime_avg_sharpe'] = summary.get('avg_sharpe_ratio')
            key_metrics['regime_consistency'] = summary.get('regime_consistency')
            key_metrics['regime_robustness'] = risk_assessment.get('regime_robustness_score')
            key_metrics['high_risk_regimes_count'] = len(risk_assessment.get('high_risk_regimes', []))
            key_metrics['worst_case_loss'] = risk_assessment.get('worst_case_scenario', {}).get('max_expected_loss')
        
        # Общая оценка модели
        grade_points = 0
        total_checks = 0
        
        # Проверка F1 score
        if key_metrics.get('avg_f1_score') is not None:
            if key_metrics['avg_f1_score'] > 0.4:
                grade_points += 2
            elif key_metrics['avg_f1_score'] > 0.3:
                grade_points += 1
            total_checks += 2
        
        # Проверка статистической значимости
        if key_metrics.get('statistical_significance') is not None:
            if key_metrics['statistical_significance']:
                grade_points += 2
            total_checks += 2
        
        # Проверка Win Rate
        if key_metrics.get('win_rate') is not None:
            if key_metrics['win_rate'] > 0.45:
                grade_points += 2
            elif key_metrics['win_rate'] > 0.35:
                grade_points += 1
            total_checks += 2
        
        # Проверка утечек
        if key_metrics.get('critical_leaks', 0) == 0:
            grade_points += 2
        total_checks += 2
        
        # Проверка робастности по режимам
        if key_metrics.get('regime_robustness') is not None:
            if key_metrics['regime_robustness'] > 0.7:
                grade_points += 2
            elif key_metrics['regime_robustness'] > 0.5:
                grade_points += 1
            total_checks += 2
        
        # Проверка консистентности по режимам
        if key_metrics.get('regime_consistency') is not None:
            if key_metrics['regime_consistency'] < 0.5:  # Меньше = лучше
                grade_points += 2
            elif key_metrics['regime_consistency'] < 1.0:
                grade_points += 1
            total_checks += 2
        
        # Финальная оценка
        if total_checks > 0:
            grade_percentage = grade_points / total_checks
            if grade_percentage >= 0.8:
                overall_grade = "ОТЛИЧНО"
            elif grade_percentage >= 0.6:
                overall_grade = "ХОРОШО"  
            elif grade_percentage >= 0.4:
                overall_grade = "УДОВЛЕТВОРИТЕЛЬНО"
            else:
                overall_grade = "НЕУДОВЛЕТВОРИТЕЛЬНО"
        else:
            overall_grade = "НЕ ОЦЕНЕНО"
        
        # Рекомендации
        recommendations = []
        
        if not validation_successful:
            recommendations.append("🚨 КРИТИЧНО: Устранить все обнаруженные проблемы перед продакшеном")
        
        if key_metrics.get('critical_leaks', 0) > 0:
            recommendations.append("❌ НЕ ИСПОЛЬЗОВАТЬ модель до устранения утечек данных")
        
        if key_metrics.get('avg_f1_score', 0) < 0.3:
            recommendations.append("📊 Улучшить качество предсказаний модели")
        
        if key_metrics.get('win_rate', 0) < 0.4:
            recommendations.append("🎯 Оптимизировать торговую стратегию")
        
        if not key_metrics.get('statistical_significance', False):
            recommendations.append("📈 Проверить статистическую значимость результатов")
        
        if key_metrics.get('regime_robustness', 1) < 0.6:
            recommendations.append("🌪️ Улучшить робастность модели к различным рыночным режимам")
        
        if key_metrics.get('regime_consistency', 0) > 1.0:
            recommendations.append("📊 Высокая нестабильность между режимами - рассмотреть адаптивные стратегии")
        
        if key_metrics.get('high_risk_regimes_count', 0) > 0:
            recommendations.append(f"⚠️ Обнаружено {key_metrics.get('high_risk_regimes_count')} высокорискованных режимов - усилить risk management")
        
        if key_metrics.get('worst_case_loss', 0) < -0.20:
            recommendations.append(f"🚨 Критическая просадка в худшем сценарии: {key_metrics.get('worst_case_loss', 0):.1%} - пересмотреть стратегию")
        
        if not recommendations:
            recommendations.append("✅ Модель готова к дополнительному тестированию")
        
        return {
            'overall_grade': overall_grade,
            'grade_percentage': grade_percentage if total_checks > 0 else 0,
            'key_metrics': key_metrics,
            'validation_successful': validation_successful,
            'critical_issues_count': len(critical_issues),
            'recommendations': recommendations,
            'ready_for_production': validation_successful and key_metrics.get('critical_leaks', 0) == 0
        }
    
    def _save_comprehensive_results(self):
        """Сохраняет комплексные результаты валидации"""
        
        # JSON отчет
        json_path = self.session_dir / "comprehensive_validation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Человеко-читаемый отчет
        report_path = self.session_dir / "COMPREHENSIVE_VALIDATION_REPORT.md"
        readable_report = self._create_readable_comprehensive_report()
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(readable_report)
        
        self.logger.info(f"💾 Комплексные результаты сохранены:")
        self.logger.info(f"   📄 JSON: {json_path}")
        self.logger.info(f"   📖 Отчет: {report_path}")
    
    def _create_readable_comprehensive_report(self) -> str:
        """Создает человеко-читаемый отчет"""
        
        final_report = self.validation_results.get('final_report', {})
        timestamp = self.validation_results['timestamp']
        session_id = self.validation_results['session_id']
        
        report = f"""# Комплексный отчет валидации модели

**Дата:** {timestamp}  
**Сессия:** {session_id}  
**Общая оценка:** {final_report.get('overall_grade', 'N/A')}  
**Готовность к продакшену:** {'✅ ДА' if final_report.get('ready_for_production', False) else '❌ НЕТ'}

## 🎯 Ключевые метрики

"""
        
        key_metrics = final_report.get('key_metrics', {})
        
        if key_metrics.get('avg_f1_score') is not None:
            report += f"- **Средний F1 Score:** {key_metrics['avg_f1_score']:.3f}\n"
        
        if key_metrics.get('win_rate') is not None:
            report += f"- **Win Rate:** {key_metrics['win_rate']:.1%}\n"
            
        if key_metrics.get('total_return') is not None:
            report += f"- **Общая доходность:** {key_metrics['total_return']:.1%}\n"
            
        if key_metrics.get('mean_sharpe') is not None:
            report += f"- **Средний Sharpe:** {key_metrics['mean_sharpe']:.2f}\n"
        
        if key_metrics.get('statistical_significance') is not None:
            sig_status = "✅ ДА" if key_metrics['statistical_significance'] else "❌ НЕТ"
            report += f"- **Статистическая значимость:** {sig_status}\n"
        
        if key_metrics.get('leak_risk_level'):
            report += f"- **Риск утечек данных:** {key_metrics['leak_risk_level']}\n"
        
        report += "\n## 📋 Результаты по этапам валидации\n\n"
        
        # Описание каждого этапа
        results = self.validation_results.get('results', {})
        
        if 'data_leak_detection' in results:
            report += "### 🔍 Детекция утечек данных\n"
            leak_info = results['data_leak_detection'].get('summary', {})
            report += f"- Утечек обнаружено: {leak_info.get('total_leaks_detected', 0)}\n"
            report += f"- Критичных: {leak_info.get('severity_breakdown', {}).get('CRITICAL', 0)}\n\n"
        
        if 'prediction_quality' in results:
            report += "### 📊 Качество предсказаний\n"
            pq_summary = results['prediction_quality'].get('summary', {})
            if 'classification' in pq_summary:
                report += f"- Средний F1 (направления): {pq_summary['classification'].get('mean_f1', 0):.3f}\n"
            if 'binary_classification' in pq_summary:
                report += f"- Средний AUC (уровни): {pq_summary['binary_classification'].get('mean_auc', 0):.3f}\n"
            report += "\n"
        
        if 'monte_carlo' in results:
            report += "### 🎲 Monte Carlo тестирование\n"
            mc_info = results['monte_carlo'].get('overall_assessment', {})
            sig_info = mc_info.get('statistical_significance', {})
            robust_info = mc_info.get('robustness', {})
            report += f"- Статистическая значимость: {'✅' if sig_info.get('overall_significant', False) else '❌'}\n"
            report += f"- Устойчивость: {robust_info.get('robustness_rating', 'N/A')}\n\n"
        
        if 'walk_forward' in results:
            report += "### 🔄 Walk-Forward валидация\n"
            wf_summary = results['walk_forward'].get('summary', {})
            report += f"- Средняя доходность: {wf_summary.get('mean_return', 0):.2%}\n"
            report += f"- Средний Sharpe: {wf_summary.get('mean_sharpe', 0):.2f}\n"
            stability = results['walk_forward'].get('stability', {})
            report += f"- Стабильность: {stability.get('stability_score', 0):.2f}\n\n"
        
        if 'backtesting' in results:
            report += "### 💰 Бэктестинг\n"
            bt = results['backtesting']
            report += f"- Всего сделок: {bt.get('total_trades', 0)}\n"
            report += f"- Win Rate: {bt.get('win_rate', 0):.1%}\n"
            report += f"- Доходность: {bt.get('total_return', 0):.1%}\n"
            report += f"- Max Drawdown: {bt.get('max_drawdown', 0):.1%}\n\n"
        
        # Рекомендации
        recommendations = final_report.get('recommendations', [])
        if recommendations:
            report += "## 💡 Рекомендации\n\n"
            for rec in recommendations:
                report += f"- {rec}\n"
            report += "\n"
        
        # Критические проблемы
        critical_issues = self.validation_results.get('critical_issues', [])
        if critical_issues:
            report += "## 🚨 Критические проблемы\n\n"
            for issue in critical_issues:
                report += f"- {issue}\n"
            report += "\n"
        
        report += "---\n*Отчет сгенерирован системой комплексной валидации*"
        
        return report


def main():
    """Главная функция для запуска комплексной валидации"""
    
    parser = argparse.ArgumentParser(description="Комплексная валидация торговой модели")
    parser.add_argument('--config', default='config/config.yaml', help='Конфигурационный файл')
    parser.add_argument('--data-file', help='CSV файл с данными (для детекции утечек)')
    parser.add_argument('--predictions-true', help='NPY файл с истинными значениями')
    parser.add_argument('--predictions-pred', help='NPY файл с предсказаниями')
    parser.add_argument('--returns-file', help='NPY файл с доходностями стратегии')
    parser.add_argument('--skip-leak-detection', action='store_true', help='Пропустить детекцию утечек')
    parser.add_argument('--skip-monte-carlo', action='store_true', help='Пропустить Monte Carlo')
    parser.add_argument('--skip-walk-forward', action='store_true', help='Пропустить Walk-Forward')
    
    args = parser.parse_args()
    
    print("🚀 Запуск комплексной системы валидации...")
    
    # Создаем валидатор
    validator = ComprehensiveValidator(args.config)
    
    # Загружаем данные
    data = None
    if args.data_file and not args.skip_leak_detection:
        print(f"📊 Загрузка данных из {args.data_file}")
        data = pd.read_csv(args.data_file)
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
    
    y_true = None
    y_pred = None
    if args.predictions_true and args.predictions_pred:
        print(f"🎯 Загрузка предсказаний...")
        y_true = np.load(args.predictions_true)
        y_pred = np.load(args.predictions_pred)
        print(f"   Истинные: {y_true.shape}")
        print(f"   Предсказания: {y_pred.shape}")
    
    returns = None
    if args.returns_file and not args.skip_monte_carlo:
        print(f"📈 Загрузка доходностей из {args.returns_file}")
        returns = np.load(args.returns_file)
        print(f"   Доходности: {returns.shape}")
    
    # Автоматическое определение признаков и целей
    feature_columns = None
    target_columns = None
    
    if data is not None:
        # Исключаем технические колонки
        exclude_cols = {'datetime', 'timestamp', 'symbol', 'close', 'open', 'high', 'low', 'volume'}
        
        # Целевые переменные (с известными префиксами)
        target_columns = [col for col in data.columns 
                         if any(prefix in col.lower() for prefix in 
                               ['future_', 'direction_', 'will_reach_', 'max_drawdown', 'max_rally'])]
        
        # Признаки (все остальные числовые колонки)
        feature_columns = [col for col in data.columns 
                          if col not in exclude_cols and col not in target_columns
                          and data[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        print(f"🔍 Найдено признаков: {len(feature_columns)}")
        print(f"🎯 Найдено целевых переменных: {len(target_columns)}")
    
    # Заглушка для функции обучения модели
    def dummy_trainer(train_data, val_data):
        """Заглушка для обучения модели"""
        return f"trained_model_on_{len(train_data)}_samples"
    
    model_trainer = None if args.skip_walk_forward else dummy_trainer
    
    # Запускаем комплексную валидацию
    results = validator.run_comprehensive_validation(
        model_trainer_func=model_trainer,
        model=None,  # Заглушка
        test_loader=None,  # Заглушка
        data=data,
        y_true=y_true,
        y_pred=y_pred,
        returns=returns,
        feature_columns=feature_columns,
        target_columns=target_columns
    )
    
    # Выводим итоги
    print("\n" + "="*60)
    print("🎯 ИТОГИ ВАЛИДАЦИИ:")
    
    final_report = results.get('final_report', {})
    print(f"📊 Общая оценка: {final_report.get('overall_grade', 'N/A')}")
    print(f"🚀 Готовность к продакшену: {'✅ ДА' if final_report.get('ready_for_production', False) else '❌ НЕТ'}")
    
    if results.get('critical_issues'):
        print(f"🚨 Критических проблем: {len(results['critical_issues'])}")
        for issue in results['critical_issues'][:3]:  # Первые 3
            print(f"   • {issue}")
    
    print(f"📁 Все результаты сохранены в: {validator.session_dir}")


if __name__ == "__main__":
    main()