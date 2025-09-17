"""
Система стресс-тестирования модели под различными рыночными режимами.

Этот модуль анализирует производительность модели при различных рыночных условиях:
- Бычий рынок (восходящий тренд)
- Медвежий рынок (нисходящий тренд)  
- Боковой рынок (флэт)
- Кризисные периоды (высокая волатильность)
- Различные уровни ликвидности
- Периоды высокой корреляции между активами
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketRegimeConfig:
    """Конфигурация для определения рыночных режимов"""
    # Параметры для классификации режимов
    trend_window: int = 30  # Окно для определения тренда (дни)
    volatility_window: int = 20  # Окно для волатильности
    volume_window: int = 30  # Окно для объемного анализа
    
    # Пороги для классификации
    bull_threshold: float = 0.05  # +5% за период = бычий рынок
    bear_threshold: float = -0.05  # -5% за период = медвежий рынок
    high_vol_percentile: float = 80  # 80-й перцентиль = высокая волатильность
    low_liquidity_percentile: float = 20  # 20-й перцентиль = низкая ликвидность
    crisis_vol_threshold: float = 0.8  # VIX эквивалент для крипто
    
    # Минимальные размеры выборок
    min_regime_samples: int = 100

@dataclass
class RegimeAnalysisResult:
    """Результат анализа конкретного режима"""
    regime_name: str
    sample_count: int
    date_range: Tuple[datetime, datetime]
    
    # Метрики производительности
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_return: float
    profit_factor: float
    
    # Статистика предсказаний
    prediction_accuracy: float
    prediction_precision: float
    prediction_recall: float
    prediction_f1: float
    
    # Рыночные характеристики периода
    market_return: float
    market_volatility: float
    avg_volume: float
    correlation_to_btc: float

class RegimeStressTester:
    """
    Основной класс для стресс-тестирования модели под различными рыночными режимами.
    """
    
    def __init__(self, config: MarketRegimeConfig = None):
        self.config = config or MarketRegimeConfig()
        self.regime_results = {}
        
    def identify_market_regimes(self, 
                               price_data: pd.DataFrame, 
                               volume_data: pd.DataFrame = None,
                               btc_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Определяет рыночные режимы на основе цены, объема и корреляции с BTC.
        
        Args:
            price_data: DataFrame с ценами (columns: timestamp, symbol, close)
            volume_data: DataFrame с объемами (опционально)
            btc_data: DataFrame с ценами BTC для корреляции (опционально)
            
        Returns:
            DataFrame с классифицированными режимами
        """
        print("🔍 Определение рыночных режимов...")
        
        regimes = []
        
        for symbol in price_data['symbol'].unique():
            symbol_data = price_data[price_data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(symbol_data) < self.config.trend_window:
                continue
                
            # Расчет индикаторов режимов
            symbol_data['returns'] = symbol_data['close'].pct_change()
            symbol_data['rolling_return'] = symbol_data['close'].pct_change(self.config.trend_window)
            symbol_data['volatility'] = symbol_data['returns'].rolling(self.config.volatility_window).std()
            symbol_data['volume_ma'] = symbol_data.get('volume', pd.Series(0, index=symbol_data.index)).rolling(self.config.volume_window).mean()
            
            # Определение режимов
            conditions = []
            regime_labels = []
            
            # Бычий рынок
            bull_condition = symbol_data['rolling_return'] >= self.config.bull_threshold
            conditions.append(bull_condition)
            regime_labels.append('bull_market')
            
            # Медвежий рынок  
            bear_condition = symbol_data['rolling_return'] <= self.config.bear_threshold
            conditions.append(bear_condition)
            regime_labels.append('bear_market')
            
            # Высокая волатильность (кризис)
            vol_threshold = symbol_data['volatility'].quantile(self.config.high_vol_percentile / 100)
            crisis_condition = symbol_data['volatility'] >= vol_threshold
            conditions.append(crisis_condition)
            regime_labels.append('crisis')
            
            # Низкая ликвидность (если есть данные по объему)
            if 'volume' in symbol_data.columns and symbol_data['volume'].sum() > 0:
                volume_threshold = symbol_data['volume'].quantile(self.config.low_liquidity_percentile / 100)
                low_liquidity_condition = symbol_data['volume'] <= volume_threshold
                conditions.append(low_liquidity_condition)
                regime_labels.append('low_liquidity')
            
            # Боковой рынок (по умолчанию)
            symbol_data['regime'] = 'sideways'
            
            # Применение условий (приоритет по порядку)
            for condition, label in zip(conditions, regime_labels):
                symbol_data.loc[condition, 'regime'] = label
            
            # Корреляция с BTC (если доступна)
            if btc_data is not None:
                btc_returns = btc_data.set_index('timestamp')['close'].pct_change()
                symbol_returns_indexed = symbol_data.set_index('timestamp')['returns']
                correlation = symbol_returns_indexed.rolling(30).corr(btc_returns)
                symbol_data['btc_correlation'] = correlation.values
            else:
                symbol_data['btc_correlation'] = np.nan
                
            symbol_data['symbol'] = symbol
            regimes.append(symbol_data)
        
        regime_df = pd.concat(regimes, ignore_index=True)
        
        # Статистика режимов
        regime_stats = regime_df.groupby('regime').agg({
            'timestamp': ['count', 'min', 'max'],
            'rolling_return': 'mean',
            'volatility': 'mean',
            'btc_correlation': 'mean'
        }).round(4)
        
        print("\n📊 Статистика рыночных режимов:")
        print(regime_stats)
        
        return regime_df
    
    def test_model_performance_by_regime(self, 
                                       regime_data: pd.DataFrame,
                                       predictions: pd.DataFrame,
                                       actual_returns: pd.DataFrame,
                                       trading_results: pd.DataFrame = None) -> Dict[str, RegimeAnalysisResult]:
        """
        Тестирует производительность модели для каждого рыночного режима.
        
        Args:
            regime_data: DataFrame с классифицированными режимами
            predictions: Предсказания модели
            actual_returns: Фактические доходности
            trading_results: Результаты торговли (опционально)
            
        Returns:
            Словарь с результатами для каждого режима
        """
        print("🧪 Тестирование производительности по режимам...")
        
        results = {}
        
        # Объединяем данные по timestamp и symbol
        merged_data = regime_data.merge(
            predictions, on=['timestamp', 'symbol'], how='inner'
        ).merge(
            actual_returns, on=['timestamp', 'symbol'], how='inner'
        )
        
        if trading_results is not None:
            merged_data = merged_data.merge(
                trading_results, on=['timestamp', 'symbol'], how='left'
            )
        
        for regime in merged_data['regime'].unique():
            if pd.isna(regime):
                continue
                
            regime_subset = merged_data[merged_data['regime'] == regime].copy()
            
            if len(regime_subset) < self.config.min_regime_samples:
                print(f"⚠️  Недостаточно данных для режима {regime} ({len(regime_subset)} < {self.config.min_regime_samples})")
                continue
            
            # Анализ производительности
            result = self._analyze_regime_performance(regime, regime_subset)
            results[regime] = result
            
        self.regime_results = results
        return results
    
    def _analyze_regime_performance(self, 
                                  regime_name: str, 
                                  data: pd.DataFrame) -> RegimeAnalysisResult:
        """Анализирует производительность для конкретного режима"""
        
        # Временные границы
        date_range = (data['timestamp'].min(), data['timestamp'].max())
        
        # Торговые метрики
        if 'trade_return' in data.columns:
            trades = data.dropna(subset=['trade_return'])
            total_return = trades['trade_return'].sum()
            avg_trade_return = trades['trade_return'].mean()
            win_rate = (trades['trade_return'] > 0).mean()
            
            # Sharpe ratio (дневная частота)
            if trades['trade_return'].std() > 0:
                sharpe_ratio = trades['trade_return'].mean() / trades['trade_return'].std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            cumulative = (1 + trades['trade_return']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Profit factor
            winning_trades = trades[trades['trade_return'] > 0]['trade_return'].sum()
            losing_trades = abs(trades[trades['trade_return'] < 0]['trade_return'].sum())
            profit_factor = winning_trades / losing_trades if losing_trades > 0 else np.inf
            
        else:
            # Если нет торговых данных, используем предсказанные доходности
            total_return = data['predicted_return'].sum() if 'predicted_return' in data.columns else 0
            avg_trade_return = data['predicted_return'].mean() if 'predicted_return' in data.columns else 0
            win_rate = (data['predicted_return'] > 0).mean() if 'predicted_return' in data.columns else 0
            sharpe_ratio = 0
            max_drawdown = 0
            profit_factor = 1
        
        # Метрики предсказаний
        if 'direction_prediction' in data.columns and 'actual_direction' in data.columns:
            # Классификационные метрики для направления
            correct_predictions = (data['direction_prediction'] == data['actual_direction']).sum()
            total_predictions = len(data)
            prediction_accuracy = correct_predictions / total_predictions
            
            # Precision, Recall, F1 для основных классов
            from sklearn.metrics import precision_recall_fscore_support
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    data['actual_direction'], 
                    data['direction_prediction'], 
                    average='weighted',
                    zero_division=0
                )
                prediction_precision = precision
                prediction_recall = recall
                prediction_f1 = f1
            except:
                prediction_precision = 0
                prediction_recall = 0
                prediction_f1 = 0
        else:
            prediction_accuracy = 0
            prediction_precision = 0
            prediction_recall = 0
            prediction_f1 = 0
        
        # Рыночные характеристики
        market_return = data['rolling_return'].mean() if 'rolling_return' in data.columns else 0
        market_volatility = data['volatility'].mean() if 'volatility' in data.columns else 0
        avg_volume = data['volume'].mean() if 'volume' in data.columns else 0
        correlation_to_btc = data['btc_correlation'].mean() if 'btc_correlation' in data.columns else 0
        
        return RegimeAnalysisResult(
            regime_name=regime_name,
            sample_count=len(data),
            date_range=date_range,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            profit_factor=profit_factor,
            prediction_accuracy=prediction_accuracy,
            prediction_precision=prediction_precision,
            prediction_recall=prediction_recall,
            prediction_f1=prediction_f1,
            market_return=market_return,
            market_volatility=market_volatility,
            avg_volume=avg_volume,
            correlation_to_btc=correlation_to_btc
        )
    
    def generate_regime_stress_report(self) -> Dict[str, Any]:
        """Генерирует комплексный отчет по стресс-тестированию режимов"""
        
        if not self.regime_results:
            return {"error": "Нет результатов для генерации отчета. Запустите test_model_performance_by_regime сначала."}
        
        print("\n" + "="*80)
        print("📈 ОТЧЕТ ПО СТРЕСС-ТЕСТИРОВАНИЮ РЫНОЧНЫХ РЕЖИМОВ")
        print("="*80)
        
        report = {
            "summary": {},
            "regime_analysis": {},
            "risk_assessment": {},
            "recommendations": []
        }
        
        # Сводная статистика
        all_sharpe = [r.sharpe_ratio for r in self.regime_results.values()]
        all_win_rates = [r.win_rate for r in self.regime_results.values()]
        all_drawdowns = [r.max_drawdown for r in self.regime_results.values()]
        all_f1_scores = [r.prediction_f1 for r in self.regime_results.values()]
        
        report["summary"] = {
            "total_regimes_tested": len(self.regime_results),
            "avg_sharpe_ratio": np.mean(all_sharpe),
            "min_sharpe_ratio": np.min(all_sharpe),
            "max_sharpe_ratio": np.max(all_sharpe),
            "avg_win_rate": np.mean(all_win_rates),
            "avg_max_drawdown": np.mean(all_drawdowns),
            "worst_drawdown": np.min(all_drawdowns),
            "avg_prediction_f1": np.mean(all_f1_scores),
            "regime_consistency": np.std(all_sharpe) / np.mean(all_sharpe) if np.mean(all_sharpe) != 0 else np.inf
        }
        
        print(f"\n🎯 СВОДНАЯ СТАТИСТИКА:")
        print(f"Протестировано режимов: {report['summary']['total_regimes_tested']}")
        print(f"Средний Sharpe Ratio: {report['summary']['avg_sharpe_ratio']:.3f}")
        print(f"Диапазон Sharpe: [{report['summary']['min_sharpe_ratio']:.3f}, {report['summary']['max_sharpe_ratio']:.3f}]")
        print(f"Средний Win Rate: {report['summary']['avg_win_rate']:.1%}")
        print(f"Средняя просадка: {report['summary']['avg_max_drawdown']:.1%}")
        print(f"Худшая просадка: {report['summary']['worst_drawdown']:.1%}")
        print(f"Средний F1 Score: {report['summary']['avg_prediction_f1']:.3f}")
        print(f"Консистентность по режимам: {report['summary']['regime_consistency']:.3f}")
        
        # Детальный анализ по режимам
        print(f"\n📊 ДЕТАЛЬНЫЙ АНАЛИЗ ПО РЕЖИМАМ:")
        print("-" * 120)
        print(f"{'Режим':<15} {'Выборка':<8} {'Sharpe':<8} {'Win Rate':<10} {'Просадка':<10} {'F1 Score':<10} {'Рын.доходн.':<12}")
        print("-" * 120)
        
        for regime_name, result in self.regime_results.items():
            report["regime_analysis"][regime_name] = {
                "sample_count": result.sample_count,
                "performance_metrics": {
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor
                },
                "prediction_metrics": {
                    "accuracy": result.prediction_accuracy,
                    "precision": result.prediction_precision,
                    "recall": result.prediction_recall,
                    "f1_score": result.prediction_f1
                },
                "market_characteristics": {
                    "market_return": result.market_return,
                    "market_volatility": result.market_volatility,
                    "avg_volume": result.avg_volume,
                    "btc_correlation": result.correlation_to_btc
                }
            }
            
            print(f"{regime_name:<15} {result.sample_count:<8} {result.sharpe_ratio:<8.3f} "
                  f"{result.win_rate:<10.1%} {result.max_drawdown:<10.1%} {result.prediction_f1:<10.3f} "
                  f"{result.market_return:<12.1%}")
        
        # Оценка рисков
        risk_regimes = []
        high_risk_threshold = -0.15  # Просадка > 15%
        
        for regime_name, result in self.regime_results.items():
            if result.max_drawdown < high_risk_threshold:
                risk_regimes.append(regime_name)
        
        report["risk_assessment"] = {
            "high_risk_regimes": risk_regimes,
            "stable_regimes": [name for name in self.regime_results.keys() if name not in risk_regimes],
            "regime_robustness_score": 1 - len(risk_regimes) / len(self.regime_results),
            "worst_case_scenario": {
                "regime": min(self.regime_results.items(), key=lambda x: x[1].sharpe_ratio)[0],
                "max_expected_loss": min(all_drawdowns)
            }
        }
        
        print(f"\n⚠️  ОЦЕНКА РИСКОВ:")
        print(f"Высокорискованные режимы: {risk_regimes}")
        print(f"Стабильные режимы: {report['risk_assessment']['stable_regimes']}")
        print(f"Оценка робастности: {report['risk_assessment']['regime_robustness_score']:.1%}")
        print(f"Худший сценарий: {report['risk_assessment']['worst_case_scenario']}")
        
        # Рекомендации
        recommendations = []
        
        if report["summary"]["regime_consistency"] > 1.0:
            recommendations.append("Высокая нестабильность между режимами - рассмотрите адаптивные стратегии")
        
        if report["summary"]["worst_drawdown"] < -0.20:
            recommendations.append("Критическая просадка обнаружена - усильте risk management")
        
        if report["summary"]["avg_prediction_f1"] < 0.4:
            recommendations.append("Низкое качество предсказаний - требуется улучшение модели")
        
        if len(risk_regimes) > len(self.regime_results) * 0.4:
            recommendations.append("Более 40% режимов высокорискованные - пересмотрите стратегию")
        
        # Специфичные рекомендации по режимам
        for regime_name, result in self.regime_results.items():
            if result.sharpe_ratio < 0.5:
                recommendations.append(f"Режим {regime_name}: низкий Sharpe ratio - избегайте торговли в данных условиях")
            
            if result.win_rate < 0.4:
                recommendations.append(f"Режим {regime_name}: низкий win rate - улучшите селекцию сигналов")
        
        report["recommendations"] = recommendations
        
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*80)
        
        return report
    
    def export_regime_analysis(self, filepath: str):
        """Экспортирует результаты анализа в файл"""
        if not self.regime_results:
            print("❌ Нет данных для экспорта")
            return
        
        # Создаем DataFrame с результатами
        export_data = []
        for regime_name, result in self.regime_results.items():
            export_data.append({
                'regime': regime_name,
                'sample_count': result.sample_count,
                'date_start': result.date_range[0],
                'date_end': result.date_range[1],
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'avg_trade_return': result.avg_trade_return,
                'profit_factor': result.profit_factor,
                'prediction_accuracy': result.prediction_accuracy,
                'prediction_precision': result.prediction_precision,
                'prediction_recall': result.prediction_recall,
                'prediction_f1': result.prediction_f1,
                'market_return': result.market_return,
                'market_volatility': result.market_volatility,
                'avg_volume': result.avg_volume,
                'correlation_to_btc': result.correlation_to_btc
            })
        
        df = pd.DataFrame(export_data)
        df.to_csv(filepath, index=False)
        print(f"✅ Результаты экспортированы в {filepath}")


def run_regime_stress_test(price_data: pd.DataFrame, 
                          predictions: pd.DataFrame,
                          actual_returns: pd.DataFrame,
                          trading_results: pd.DataFrame = None,
                          config: MarketRegimeConfig = None,
                          export_path: str = None) -> Dict[str, Any]:
    """
    Convenience функция для запуска полного стресс-тестирования режимов.
    
    Args:
        price_data: Данные цен
        predictions: Предсказания модели  
        actual_returns: Фактические доходности
        trading_results: Результаты торговли (опционально)
        config: Конфигурация (опционально)
        export_path: Путь для экспорта (опционально)
        
    Returns:
        Полный отчет по стресс-тестированию
    """
    tester = RegimeStressTester(config)
    
    # Определяем режимы
    regime_data = tester.identify_market_regimes(price_data)
    
    # Тестируем производительность
    results = tester.test_model_performance_by_regime(
        regime_data, predictions, actual_returns, trading_results
    )
    
    # Генерируем отчет
    report = tester.generate_regime_stress_report()
    
    # Экспортируем если нужно
    if export_path:
        tester.export_regime_analysis(export_path)
    
    return report