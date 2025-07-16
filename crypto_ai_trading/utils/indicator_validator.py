"""
Валидатор технических индикаторов - проверка корректности диапазонов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.logger import get_logger

class IndicatorValidator:
    """Класс для проверки корректности технических индикаторов"""
    
    def __init__(self):
        self.logger = get_logger("IndicatorValidator")
        
        # Определяем ожидаемые диапазоны для критических индикаторов
        self.critical_ranges = {
            # Осцилляторы [0, 100]
            'rsi': (0, 100, "RSI должен быть в диапазоне [0, 100]"),
            'stoch_k': (0, 100, "Stochastic %K должен быть в диапазоне [0, 100]"),
            'stoch_d': (0, 100, "Stochastic %D должен быть в диапазоне [0, 100]"),
            'adx': (0, 100, "ADX должен быть в диапазоне [0, 100]"),
            'adx_pos': (0, 100, "ADX +DI должен быть в диапазоне [0, 100]"),
            'adx_neg': (0, 100, "ADX -DI должен быть в диапазоне [0, 100]"),
            
            # Вероятности и позиции [0, 1]
            'toxicity': (0, 1, "Toxicity должен быть в диапазоне [0, 1]"),
            'bb_position': (0, 1, "BB Position должен быть в диапазоне [0, 1]"),
            'close_position': (0, 1, "Close Position должен быть в диапазоне [0, 1]"),
            'psar_trend': (0, 1, "PSAR Trend должен быть в диапазоне [0, 1]"),
            
            # Бинарные индикаторы [0, 1]
            'rsi_oversold': (0, 1, "RSI Oversold должен быть в диапазоне [0, 1]"),
            'rsi_overbought': (0, 1, "RSI Overbought должен быть в диапазоне [0, 1]"),
        }
        
        # Диапазоны для соотношений (обычно близко к 1.0)
        # Расширенные границы для реальных рыночных условий
        self.ratio_ranges = {
            'close_vwap_ratio': (0.5, 2.0, "Close/VWAP ratio должен быть в диапазоне [0.5, 2.0]"),
            'close_open_ratio': (0.7, 1.3, "Close/Open ratio должен быть в диапазоне [0.7, 1.3]"),  # ±30% движение возможно
            'high_low_ratio': (1.0, 2.0, "High/Low ratio должен быть в диапазоне [1.0, 2.0]"),  # Волатильные дни могут иметь большой диапазон
        }
        
        # Разумные диапазоны для доходностей (%)
        self.return_ranges = {
            'returns': (-50, 50, "Returns не должны превышать ±50% для 15-минутных свечей"),
        }
    
    def validate_dataframe(self, df: pd.DataFrame, strict: bool = True) -> Dict[str, List[str]]:
        """
        Полная валидация DataFrame с индикаторами
        
        Args:
            df: DataFrame для проверки
            strict: Строгий режим (ошибки при нарушениях)
            
        Returns:
            Словарь с результатами проверки
        """
        results = {
            'errors': [],
            'warnings': [],
            'info': [],
            'statistics': {}
        }
        
        self.logger.info("🔍 Начало валидации технических индикаторов...")
        
        # Проверка критических индикаторов
        for indicator, (min_val, max_val, message) in self.critical_ranges.items():
            if indicator in df.columns:
                result = self._check_range(df, indicator, min_val, max_val, message)
                if result['status'] == 'error':
                    results['errors'].extend(result['messages'])
                elif result['status'] == 'warning':
                    results['warnings'].extend(result['messages'])
                else:
                    results['info'].extend(result['messages'])
                
                # Сохраняем статистики
                results['statistics'][indicator] = result['stats']
        
        # Проверка соотношений
        for ratio, (min_val, max_val, message) in self.ratio_ranges.items():
            if ratio in df.columns:
                result = self._check_range(df, ratio, min_val, max_val, message, tolerance=0.1)
                if result['status'] == 'warning':
                    results['warnings'].extend(result['messages'])
                else:
                    results['info'].extend(result['messages'])
                
                results['statistics'][ratio] = result['stats']
        
        # Проверка доходностей
        for return_col, (min_val, max_val, message) in self.return_ranges.items():
            if return_col in df.columns:
                result = self._check_range(df, return_col, min_val, max_val, message, tolerance=0.05)
                if result['status'] == 'warning':
                    results['warnings'].extend(result['messages'])
                else:
                    results['info'].extend(result['messages'])
                
                results['statistics'][return_col] = result['stats']
        
        # Проверка на признаки нормализации
        self._check_normalization_signs(df, results)
        
        # Проверка экстремальных значений
        self._check_extreme_values(df, results)
        
        # Выводим результаты
        self._log_results(results)
        
        # В строгом режиме поднимаем исключение при ошибках
        if strict and results['errors']:
            raise ValueError(f"Найдены критические ошибки в индикаторах: {results['errors']}")
        
        return results
    
    def _check_range(self, df: pd.DataFrame, column: str, min_val: float, max_val: float, 
                    message: str, tolerance: float = 0.01) -> Dict:
        """Проверка диапазона значений индикатора"""
        
        stats = df[column].describe()
        actual_min = stats['min']
        actual_max = stats['max']
        actual_mean = stats['mean']
        actual_std = stats['std']
        
        # Определяем статус
        status = 'ok'
        messages = []
        
        # Проверка на выход за границы
        if actual_min < min_val - tolerance or actual_max > max_val + tolerance:
            status = 'error'
            messages.append(f"❌ {column}: Значения выходят за допустимый диапазон!")
            messages.append(f"   Ожидается: [{min_val}, {max_val}], Фактически: [{actual_min:.4f}, {actual_max:.4f}]")
            messages.append(f"   {message}")
        
        # Проверка на признаки нормализации (для критических индикаторов)
        elif column in self.critical_ranges:
            # Специальная логика для разных типов индикаторов
            if column == 'toxicity':
                # ИСПРАВЛЕНО: Toxicity = exp(-price_impact*20), распределение от 0.3 до 1.0
                if actual_mean < 0.3 or actual_mean > 1.0:
                    status = 'error'
                    messages.append(f"❌ {column}: Значение вне допустимого диапазона")
                elif actual_mean < 0.6 or actual_mean > 0.9:
                    status = 'warning'
                    messages.append(f"⚠️ {column}: Необычное среднее значение")
                    messages.append(f"   Mean={actual_mean:.4f} (ожидается ≈0.7-0.85)")
            elif column in ['rsi_oversold', 'rsi_overbought']:
                # Бинарные индикаторы - редко активны, mean обычно низкий
                if actual_mean > 0.3:  # Если более 30% времени активен - подозрительно
                    status = 'warning'
                    messages.append(f"⚠️ {column}: Слишком часто активен")
                    messages.append(f"   Mean={actual_mean:.4f} (ожидается <0.3)")
            elif column == 'psar_trend':
                # Бинарный тренд, должен быть сбалансирован
                if actual_mean < 0.3 or actual_mean > 0.7:
                    status = 'warning'
                    messages.append(f"⚠️ {column}: Несбалансированный тренд")
                    messages.append(f"   Mean={actual_mean:.4f} (ожидается 0.3-0.7)")
            elif column in ['bb_position', 'close_position']:
                # Позиционные индикаторы, обычно около 0.5
                if actual_mean < 0.3 or actual_mean > 0.7:
                    status = 'warning'
                    messages.append(f"⚠️ {column}: Смещенная позиция")
                    messages.append(f"   Mean={actual_mean:.4f} (ожидается 0.3-0.7)")
            else:
                # Для остальных индикаторов (RSI, Stochastic, ADX)
                expected_mean = (min_val + max_val) / 2
                # Проверка на z-score нормализацию (mean≈0, std≈1)
                if abs(actual_mean) < 1.0 and 0.8 < actual_std < 1.2:
                    status = 'error'
                    messages.append(f"❌ {column}: ПОДОЗРЕНИЕ НА Z-SCORE НОРМАЛИЗАЦИЮ!")
                    messages.append(f"   Mean={actual_mean:.4f}, Std={actual_std:.4f}")
                    messages.append(f"   Ожидался mean≈{expected_mean:.1f} для {column}")
                # Проверка на min-max нормализацию (все значения в [0,1] когда должны быть в [0,100])
                elif column in ['rsi', 'stoch_k', 'stoch_d', 'adx', 'adx_pos', 'adx_neg']:
                    if actual_max < 2.0 and actual_min > -0.1:
                        status = 'error'
                        messages.append(f"❌ {column}: ПОДОЗРЕНИЕ НА MIN-MAX НОРМАЛИЗАЦИЮ!")
                        messages.append(f"   Range=[{actual_min:.4f}, {actual_max:.4f}] вместо [0, 100]")
        
        # Предупреждения
        elif actual_min < min_val or actual_max > max_val:
            status = 'warning'
            messages.append(f"⚠️ {column}: Незначительный выход за границы")
            messages.append(f"   Диапазон: [{actual_min:.4f}, {actual_max:.4f}], Ожидается: [{min_val}, {max_val}]")
        
        else:
            messages.append(f"✅ {column}: Корректный диапазон [{actual_min:.4f}, {actual_max:.4f}]")
        
        return {
            'status': status,
            'messages': messages,
            'stats': {
                'min': actual_min,
                'max': actual_max, 
                'mean': actual_mean,
                'std': actual_std,
                'count': len(df[column])
            }
        }
    
    def _check_normalization_signs(self, df: pd.DataFrame, results: Dict):
        """Проверка признаков нормализации в данных"""

        # Ищем колонки с подозрительными статистиками
        suspicious_cols = []

        ignore_cols = {
            'price_direction', 'volume_zscore', 'momentum_1h',
            'trend_1h_strength', 'future_return_3', 'future_return_4',
            'target_return_1h'
        }

        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ['id', 'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']:
                continue
            if col in ignore_cols:
                continue

            stats = df[col].describe()
            
            # Признаки стандартизации: mean≈0, std≈1
            if abs(stats['mean']) < 0.1 and 0.8 < stats['std'] < 1.2:
                suspicious_cols.append(f"{col} (mean={stats['mean']:.3f}, std={stats['std']:.3f})")
        
        if suspicious_cols:
            results['warnings'].append("⚠️ Обнаружены признаки нормализации в следующих колонках:")
            for col in suspicious_cols[:10]:  # Показываем первые 10
                results['warnings'].append(f"   - {col}")
            
            if len(suspicious_cols) > 10:
                results['warnings'].append(f"   ... и еще {len(suspicious_cols) - 10} колонок")
    
    def _check_extreme_values(self, df: pd.DataFrame, results: Dict):
        """Проверка экстремальных значений с учетом контекста"""
        
        # Колонки, где большие значения допустимы
        price_cols = ['open', 'high', 'low', 'close', 'vwap', 'psar', 
                      'sma_', 'ema_', 'bb_high', 'bb_low', 'bb_middle',
                      'local_high_', 'local_low_', 'daily_high', 'daily_low',
                      'long_liquidation_price', 'short_liquidation_price', 
                      'long_optimal_entry_price', 'short_optimal_entry_price',
                      'btc_close', 'future_high_', 'future_low_']
        
        volume_cols = ['volume', 'turnover', 'directed_volume', 'obv', 'obv_ema',
                       'volume_cumsum_', 'liquidity_score']
        
        # Колонки с ID и временными метками
        system_cols = ['id', 'timestamp']
        
        # Индикаторы с ограниченными диапазонами
        limited_indicators = {
            'rsi': 100, 'stoch_k': 100, 'stoch_d': 100, 'adx': 100,
            'toxicity': 1, 'bb_position': 1, 'close_position': 1
        }
        
        extreme_cols = []
        context_errors = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Пропускаем системные колонки
            if col in system_cols:
                continue
            
            # Проверяем, является ли колонка ценовой
            is_price_col = any(price_prefix in col for price_prefix in price_cols)
            is_volume_col = any(volume_prefix in col for volume_prefix in volume_cols)
            
            max_abs = df[col].abs().max()
            
            # Для индикаторов с известными ограничениями
            if col in limited_indicators:
                if max_abs > limited_indicators[col] * 1.1:  # 10% толерантность
                    context_errors.append(f"{col}: max={max_abs:.2f}, но должен быть <= {limited_indicators[col]}")
            
            # Для ценовых колонок - проверяем реалистичность
            elif is_price_col:
                # BTC может быть 100k+, другие монеты обычно меньше
                if 'btc' in col.lower() or df['symbol'].str.contains('BTC').any():
                    if max_abs > 200000:  # Предел для BTC
                        extreme_cols.append(f"{col} (max_abs={max_abs:.2e}) - слишком высоко даже для BTC")
                else:
                    if max_abs > 100000:  # Предел для других монет
                        extreme_cols.append(f"{col} (max_abs={max_abs:.2e}) - нереально высокая цена")
            
            # Для объемных колонок - очень большие значения допустимы
            elif is_volume_col:
                if max_abs > 1e12:  # Триллион - уже подозрительно
                    extreme_cols.append(f"{col} (max_abs={max_abs:.2e}) - слишком большой объем")
            
            # Для остальных колонок - старая логика
            else:
                if max_abs > 1000:
                    # Проверяем, не является ли это trend или другим допустимым индикатором
                    if 'trend' in col or 'range' in col or 'width' in col or 'atr' in col or 'macd' in col:
                        # Для трендов и волатильности большие значения могут быть нормальны
                        if max_abs > 10000:
                            extreme_cols.append(f"{col} (max_abs={max_abs:.2e})")
                    else:
                        extreme_cols.append(f"{col} (max_abs={max_abs:.2e})")
        
        # Сообщения об ошибках
        if context_errors:
            results['errors'].append("❌ Индикаторы выходят за допустимые пределы:")
            for error in context_errors:
                results['errors'].append(f"   - {error}")
        
        if extreme_cols:
            results['warnings'].append("⚠️ Обнаружены большие значения (требуют проверки):")
            for col in extreme_cols[:10]:  # Показываем первые 10
                results['warnings'].append(f"   - {col}")
            if len(extreme_cols) > 10:
                results['warnings'].append(f"   ... и еще {len(extreme_cols) - 10} колонок")
            results['warnings'].append("   Рекомендуется проверить корректность расчетов")
    
    def _log_results(self, results: Dict):
        """Логирование результатов валидации"""
        
        total_errors = len(results['errors'])
        total_warnings = len(results['warnings'])
        total_checked = len(results['statistics'])
        
        if total_errors == 0 and total_warnings == 0:
            self.logger.info(f"✅ Валидация пройдена успешно! Проверено {total_checked} индикаторов")
        else:
            self.logger.warning(f"⚠️ Валидация завершена: {total_errors} ошибок, {total_warnings} предупреждений")
        
        # Выводим ошибки
        for error in results['errors']:
            self.logger.error(error)
        
        # Выводим предупреждения  
        for warning in results['warnings']:
            self.logger.warning(warning)
        
        # Выводим информационные сообщения
        for info in results['info'][:5]:  # Показываем первые 5
            self.logger.info(info)
        
        if len(results['info']) > 5:
            self.logger.info(f"... и еще {len(results['info']) - 5} корректных индикаторов")
    
    def validate_batch(self, batch_data: np.ndarray, indicator_names: List[str]) -> bool:
        """
        Быстрая валидация батча во время обучения
        
        Args:
            batch_data: Numpy array с данными батча
            indicator_names: Названия признаков в порядке колонок
            
        Returns:
            True если батч корректен, False если есть проблемы
        """
        
        # Проверка на NaN/Inf
        if np.isnan(batch_data).any() or np.isinf(batch_data).any():
            self.logger.warning("⚠️ Батч содержит NaN или Inf значения!")
            return False
        
        # Проверка экстремальных значений
        max_abs = np.abs(batch_data).max()
        if max_abs > 1000:
            self.logger.warning(f"⚠️ Батч содержит экстремальные значения: max_abs={max_abs:.2e}")
            return False
        
        # Проверка критических индикаторов (если есть в батче)
        for i, name in enumerate(indicator_names):
            if name in self.critical_ranges:
                min_val, max_val, _ = self.critical_ranges[name]
                col_data = batch_data[:, i] if batch_data.ndim > 1 else batch_data
                
                if col_data.min() < min_val - 0.1 or col_data.max() > max_val + 0.1:
                    self.logger.warning(f"⚠️ {name} выходит за границы в батче: [{col_data.min():.4f}, {col_data.max():.4f}]")
                    return False
        
        return True
    
    def create_validation_report(self, df: pd.DataFrame) -> str:
        """Создание детального отчета о валидации"""
        
        results = self.validate_dataframe(df, strict=False)
        
        report = []
        report.append("=" * 80)
        report.append("📊 ОТЧЕТ ВАЛИДАЦИИ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ")
        report.append("=" * 80)
        report.append(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Размер данных: {len(df):,} записей, {len(df.columns)} колонок")
        report.append("")
        
        # Статистика валидации
        total_errors = len(results['errors'])
        total_warnings = len(results['warnings'])
        total_checked = len(results['statistics'])
        
        report.append("📈 ОБЩАЯ СТАТИСТИКА:")
        report.append(f"   - Проверено индикаторов: {total_checked}")
        report.append(f"   - Критические ошибки: {total_errors}")
        report.append(f"   - Предупреждения: {total_warnings}")
        report.append("")
        
        # Детали по индикаторам
        if results['statistics']:
            report.append("📊 ДЕТАЛЬНАЯ СТАТИСТИКА ПО ИНДИКАТОРАМ:")
            for indicator, stats in results['statistics'].items():
                report.append(f"   {indicator}:")
                report.append(f"      Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
                report.append(f"      Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                report.append(f"      Записей: {stats['count']:,}")
            report.append("")
        
        # Ошибки
        if results['errors']:
            report.append("❌ КРИТИЧЕСКИЕ ОШИБКИ:")
            for error in results['errors']:
                report.append(f"   {error}")
            report.append("")
        
        # Предупреждения
        if results['warnings']:
            report.append("⚠️ ПРЕДУПРЕЖДЕНИЯ:")
            for warning in results['warnings']:
                report.append(f"   {warning}")
            report.append("")
        
        # Рекомендации
        report.append("💡 РЕКОМЕНДАЦИИ:")
        if total_errors > 0:
            report.append("   1. Пересоздать данные с исправленной нормализацией")
            report.append("   2. Проверить функцию _normalize_features() в feature_engineering.py")
            report.append("   3. Убедиться что технические индикаторы исключены из нормализации")
        elif total_warnings > 0:
            report.append("   1. Проверить качество исходных данных")
            report.append("   2. Рассмотреть фильтрацию аномальных значений")
        else:
            report.append("   ✅ Данные готовы к обучению модели!")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def validate_indicators_quick(df: pd.DataFrame) -> bool:
    """Быстрая проверка основных индикаторов"""
    validator = IndicatorValidator()
    
    # Проверяем только критические индикаторы
    critical_indicators = ['rsi', 'stoch_k', 'stoch_d', 'adx', 'toxicity']
    
    for indicator in critical_indicators:
        if indicator in df.columns:
            stats = df[indicator].describe()
            
            # RSI, Stochastic, ADX должны быть в [0, 100]
            if indicator in ['rsi', 'stoch_k', 'stoch_d', 'adx']:
                if stats['min'] < -1 or stats['max'] > 101:
                    print(f"❌ {indicator}: выход за границы [0, 100]")
                    return False
                if abs(stats['mean']) < 1 and stats['std'] < 2:
                    print(f"❌ {indicator}: подозрение на нормализацию (mean≈0, std≈1)")
                    return False
            
            # Toxicity должен быть в [0, 1]
            elif indicator == 'toxicity':
                if stats['min'] < -0.1 or stats['max'] > 1.1:
                    print(f"❌ {indicator}: выход за границы [0, 1]")
                    return False
    
    print("✅ Быстрая проверка индикаторов пройдена")
    return True