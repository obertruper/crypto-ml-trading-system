#!/usr/bin/env python3
"""
Скрипт для запуска комплексной валидации торговой модели
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from validation.comprehensive_validator import ComprehensiveValidator
from utils.logger import get_logger

def create_sample_data():
    """Создает образцы данных для демонстрации валидации"""
    
    logger = get_logger("DataGenerator")
    logger.info("📊 Создание демонстрационных данных...")
    
    # Временные параметры
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    dates = pd.date_range(start_date, end_date, freq='15T')
    
    # Список символов
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    
    # Создаем данные цен
    price_data = []
    predictions = []
    actual_returns = []
    
    for symbol in symbols:
        # Генерируем цены (случайное блуждание с трендом)
        n_points = len(dates)
        returns = np.random.normal(0.0001, 0.02, n_points)  # Маленькая положительная доходность
        returns[0] = 0
        
        prices = 100 * np.exp(np.cumsum(returns))  # Цены от случайного блуждания
        volumes = np.random.lognormal(10, 1, n_points)  # Логнормальные объемы
        
        symbol_prices = pd.DataFrame({
            'timestamp': dates,
            'symbol': symbol,
            'close': prices,
            'volume': volumes
        })
        price_data.append(symbol_prices)
        
        # Генерируем предсказания модели
        # Берем только каждый 4-й час для предсказаний (для реалистичности)
        pred_dates = dates[::16]  # Каждые 4 часа
        n_preds = len(pred_dates)
        
        symbol_predictions = pd.DataFrame({
            'timestamp': pred_dates,
            'symbol': symbol,
            'direction_prediction': np.random.choice(['LONG', 'SHORT', 'FLAT'], n_preds, p=[0.4, 0.3, 0.3]),
            'predicted_return': np.random.normal(0.002, 0.01, n_preds),  # Предсказанная доходность
            'confidence': np.random.uniform(0.1, 0.9, n_preds)
        })
        predictions.append(symbol_predictions)
        
        # Генерируем фактические доходности
        actual_symbol_returns = pd.DataFrame({
            'timestamp': pred_dates,
            'symbol': symbol,
            'actual_return_15m': np.random.normal(0.001, 0.015, n_preds),
            'actual_return_1h': np.random.normal(0.004, 0.025, n_preds),
            'actual_return_4h': np.random.normal(0.016, 0.04, n_preds),
            'actual_return_12h': np.random.normal(0.048, 0.08, n_preds),
            'actual_direction': np.random.choice(['LONG', 'SHORT', 'FLAT'], n_preds, p=[0.45, 0.35, 0.2])
        })
        actual_returns.append(actual_symbol_returns)
    
    # Объединяем данные
    price_data_df = pd.concat(price_data, ignore_index=True)
    predictions_df = pd.concat(predictions, ignore_index=True)  
    actual_returns_df = pd.concat(actual_returns, ignore_index=True)
    
    logger.info(f"✅ Создано данных: цены={len(price_data_df)}, предсказания={len(predictions_df)}, доходности={len(actual_returns_df)}")
    
    return price_data_df, predictions_df, actual_returns_df

def create_sample_features_and_targets():
    """Создает образцы признаков и целевых переменных для тестирования утечек"""
    
    n_samples = 10000
    n_features = 171  # Как в реальной модели
    
    # Создаем признаки (технические индикаторы)
    feature_data = np.random.randn(n_samples, n_features)
    
    # Создаем временные метки
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='15T')
    
    # Добавляем систематическую утечку в некоторые признаки (для демонстрации)
    # Признак который "знает" будущую доходность
    future_return = np.random.normal(0.001, 0.02, n_samples)
    feature_data[:, 50] = future_return * 0.8 + np.random.normal(0, 0.01, n_samples)  # Коррелирует с будущим
    
    feature_columns = [f'feature_{i:03d}' for i in range(n_features)]
    feature_columns[50] = 'suspicious_future_correlated_indicator'  # Подозрительное название
    
    # Создаем DataFrame
    data = pd.DataFrame(feature_data, columns=feature_columns)
    data['timestamp'] = timestamps
    data['symbol'] = np.random.choice(['BTCUSDT', 'ETHUSDT', 'ADAUSDT'], n_samples)
    
    # Целевые переменные (20 как в реальной модели)
    target_columns = [
        'future_return_15m', 'future_return_1h', 'future_return_4h', 'future_return_12h',
        'direction_15m', 'direction_1h', 'direction_4h', 'direction_12h',
        'long_tp1_4h', 'long_tp2_4h', 'long_tp3_4h', 'long_tp1_12h',
        'short_tp1_4h', 'short_tp2_4h', 'short_tp3_4h', 'short_tp1_12h',
        'max_drawdown_1h', 'max_rally_1h', 'max_drawdown_4h', 'risk_reward_ratio'
    ]
    
    # Создаем целевые переменные с некоторой зависимостью от признаков
    for i, target in enumerate(target_columns):
        if 'return' in target:
            data[target] = (feature_data[:, i % n_features].sum(axis=0 if feature_data.ndim == 1 else 1) * 0.01 + 
                           future_return + np.random.normal(0, 0.015, n_samples))
        elif 'direction' in target:
            data[target] = np.random.choice(['LONG', 'SHORT', 'FLAT'], n_samples, p=[0.4, 0.3, 0.3])
        elif 'tp' in target or 'drawdown' in target or 'rally' in target:
            data[target] = np.random.uniform(0, 1, n_samples)
        else:
            data[target] = np.random.normal(0, 1, n_samples)
    
    return data, feature_columns, target_columns

def create_sample_predictions():
    """Создает образцы предсказаний модели для анализа качества"""
    
    n_samples = 5000
    
    # Истинные значения
    y_true_regression = np.random.normal(0.002, 0.02, (n_samples, 4))  # 4 регрессионные цели
    y_true_classification = np.random.choice([0, 1, 2], (n_samples, 4))  # 4 классификационные цели
    y_true_binary = np.random.choice([0, 1], (n_samples, 12))  # 12 бинарных целей
    
    y_true = np.concatenate([y_true_regression, y_true_classification, y_true_binary], axis=1)
    
    # Предсказания (с некоторым шумом)
    y_pred_regression = y_true_regression + np.random.normal(0, 0.01, (n_samples, 4))
    y_pred_classification = np.random.rand(n_samples, 4, 3)  # Вероятности для 3 классов
    y_pred_binary = np.random.rand(n_samples, 12)  # Вероятности для бинарных целей
    
    # Нормализуем вероятности классификации
    y_pred_classification = y_pred_classification / y_pred_classification.sum(axis=2, keepdims=True)
    
    y_pred = np.concatenate([y_pred_regression, y_pred_classification.reshape(n_samples, -1), y_pred_binary], axis=1)
    
    return y_true, y_pred

def main():
    """Основная функция демонстрации"""
    
    parser = argparse.ArgumentParser(description='Комплексная валидация торговой модели')
    parser.add_argument('--config', default='config/config.yaml', help='Путь к файлу конфигурации')
    parser.add_argument('--demo', action='store_true', help='Запустить демонстрацию с синтетическими данными')
    parser.add_argument('--full', action='store_true', help='Запустить полную валидацию')
    parser.add_argument('--regime-only', action='store_true', help='Запустить только стресс-тестирование режимов')
    
    args = parser.parse_args()
    
    logger = get_logger("ComprehensiveValidation")
    logger.info("🚀 Запуск комплексной валидации модели")
    
    try:
        # Создаем валидатор
        validator = ComprehensiveValidator(args.config)
        
        if args.demo or args.regime_only:
            logger.info("🎭 Режим демонстрации с синтетическими данными")
            
            # Создаем демонстрационные данные
            price_data, predictions, actual_returns = create_sample_data()
            
            if args.regime_only:
                # Запускаем только стресс-тестирование режимов
                logger.info("🌪️ Запуск стресс-тестирования рыночных режимов...")
                
                results = validator.run_regime_stress_testing(
                    price_data=price_data,
                    predictions=predictions,
                    actual_returns=actual_returns
                )
                
                if 'error' not in results:
                    logger.info("✅ Стресс-тестирование завершено успешно!")
                    logger.info(f"📊 Результаты сохранены в: {validator.session_dir}")
                else:
                    logger.error(f"❌ Ошибка: {results['error']}")
                    
            else:
                # Полная демонстрация
                data, feature_columns, target_columns = create_sample_features_and_targets()
                y_true, y_pred = create_sample_predictions()
                returns = np.random.normal(0.001, 0.02, 1000)  # Синтетические доходности стратегии
                
                # Запускаем комплексную валидацию
                results = validator.run_comprehensive_validation(
                    data=data,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    y_true=y_true,
                    y_pred=y_pred,
                    returns=returns,
                    price_data=price_data,
                    predictions=predictions,
                    actual_returns=actual_returns
                )
                
                # Выводим итоги
                if results['validation_successful']:
                    logger.info("✅ Комплексная валидация завершена успешно!")
                    logger.info(f"🎯 Общая оценка: {results['final_report']['overall_grade']}")
                else:
                    logger.error("❌ Валидация завершена с проблемами")
                    logger.error("🚨 Критические проблемы:")
                    for issue in results['critical_issues']:
                        logger.error(f"   • {issue}")
                
                logger.info(f"📁 Все результаты в: {validator.session_dir}")
                
        else:
            logger.info("📋 Для запуска валидации нужны реальные данные модели")
            logger.info("Используйте --demo для демонстрации или --regime-only для тестирования режимов")
            logger.info("\nПример использования:")
            logger.info("python validation/run_comprehensive_validation.py --demo")
            logger.info("python validation/run_comprehensive_validation.py --regime-only")
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    main()