"""
Валидатор важности признаков для XGBoost v3.0
Проверяет что модель не переобучается на временных паттернах
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any

try:
    from config.feature_mapping import get_feature_category, get_temporal_blacklist
    USE_FEATURE_MAPPING = True
except ImportError:
    USE_FEATURE_MAPPING = False

logger = logging.getLogger(__name__)


class FeatureImportanceValidator:
    """Класс для валидации важности признаков после обучения"""
    
    def __init__(self, max_temporal_importance: float = 5.0):
        """
        Args:
            max_temporal_importance: Максимальная допустимая важность временных признаков (%)
        """
        self.max_temporal_importance = max_temporal_importance
        self.validation_results = {}
        
    def validate_model_feature_importance(self, model: Any, feature_names: List[str], 
                                        model_name: str = "unknown") -> Dict:
        """
        Валидация важности признаков одной модели
        
        Args:
            model: Обученная модель XGBoost
            feature_names: Список имен признаков
            model_name: Имя модели для логирования
            
        Returns:
            Dict с результатами валидации
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"⚠️ Модель {model_name} не имеет feature_importances_")
            return {'valid': False, 'reason': 'no_feature_importances'}
        
        # Получаем важности
        importances = model.feature_importances_
        
        # Создаем DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Категоризируем признаки
        if USE_FEATURE_MAPPING:
            importance_df['category'] = importance_df['feature'].apply(get_feature_category)
        else:
            importance_df['category'] = importance_df['feature'].apply(self._categorize_feature_fallback)
        
        # Анализируем по категориям
        category_analysis = self._analyze_by_categories(importance_df)
        
        # Проверяем на переобучение
        validation_result = self._check_overfitting(category_analysis, model_name)
        
        # Сохраняем результаты
        self.validation_results[model_name] = {
            'category_analysis': category_analysis,
            'validation': validation_result,
            'top_features': importance_df.head(10)['feature'].tolist()
        }
        
        return validation_result
    
    def validate_ensemble_importance(self, models: Dict[str, Any], feature_names: List[str]) -> Dict:
        """
        Валидация важности признаков для ансамбля моделей
        
        Args:
            models: Словарь моделей {direction: {'ensemble': ensemble_model}}
            feature_names: Список имен признаков
            
        Returns:
            Dict с результатами валидации ансамбля
        """
        logger.info("🔍 ВАЛИДАЦИЯ ВАЖНОСТИ ПРИЗНАКОВ АНСАМБЛЯ")
        
        all_importances = {}
        model_results = {}
        
        # Собираем важности со всех моделей
        for direction in ['buy', 'sell']:
            if direction not in models:
                continue
                
            ensemble = models[direction]['ensemble']
            
            for i, model in enumerate(ensemble.models):
                model_name = f"{direction}_model_{i}"
                result = self.validate_model_feature_importance(model, feature_names, model_name)
                model_results[model_name] = result
                
                # Собираем важности для усреднения
                if hasattr(model, 'feature_importances_'):
                    for feat, imp in zip(feature_names, model.feature_importances_):
                        if feat not in all_importances:
                            all_importances[feat] = []
                        all_importances[feat].append(imp)
        
        # Усредняем важности
        avg_importances = {feat: np.mean(imps) for feat, imps in all_importances.items()}
        
        # Создаем общий анализ
        avg_importance_df = pd.DataFrame({
            'feature': list(avg_importances.keys()),
            'importance': list(avg_importances.values())
        }).sort_values('importance', ascending=False)
        
        if USE_FEATURE_MAPPING:
            avg_importance_df['category'] = avg_importance_df['feature'].apply(get_feature_category)
        else:
            avg_importance_df['category'] = avg_importance_df['feature'].apply(self._categorize_feature_fallback)
        
        # Анализ по категориям
        ensemble_category_analysis = self._analyze_by_categories(avg_importance_df)
        ensemble_validation = self._check_overfitting(ensemble_category_analysis, "ensemble")
        
        # Логируем результаты
        self._log_ensemble_results(ensemble_category_analysis, ensemble_validation, model_results)
        
        return {
            'ensemble_validation': ensemble_validation,
            'category_analysis': ensemble_category_analysis,
            'model_results': model_results,
            'top_features': avg_importance_df.head(15)['feature'].tolist()
        }
    
    def _analyze_by_categories(self, importance_df: pd.DataFrame) -> Dict:
        """Анализ важности по категориям"""
        total_importance = importance_df['importance'].sum()
        
        category_stats = {}
        for category in ['technical', 'temporal', 'btc_related', 'symbol', 'other']:
            cat_features = importance_df[importance_df['category'] == category]
            
            if len(cat_features) > 0:
                cat_importance = cat_features['importance'].sum()
                percentage = (cat_importance / total_importance) * 100 if total_importance > 0 else 0
                
                category_stats[category] = {
                    'count': len(cat_features),
                    'total_importance': cat_importance,
                    'percentage': percentage,
                    'top_features': cat_features.head(3)['feature'].tolist()
                }
            else:
                category_stats[category] = {
                    'count': 0,
                    'total_importance': 0,
                    'percentage': 0,
                    'top_features': []
                }
        
        return category_stats
    
    def _check_overfitting(self, category_analysis: Dict, model_name: str) -> Dict:
        """Проверка на переобучение по временным паттернам"""
        temporal_percentage = category_analysis['temporal']['percentage']
        technical_percentage = category_analysis['technical']['percentage']
        
        issues = []
        severity = "ok"
        
        # Проверка 1: Слишком высокая важность временных признаков
        if temporal_percentage > self.max_temporal_importance:
            issues.append(f"Temporal признаки: {temporal_percentage:.1f}% > {self.max_temporal_importance}%")
            severity = "critical"
        
        # Проверка 2: Технические признаки должны доминировать
        if technical_percentage < 70:
            issues.append(f"Технические признаки слишком слабые: {technical_percentage:.1f}% < 70%")
            if severity != "critical":
                severity = "warning"
        
        # Проверка 3: Временные важнее технических (критично!)
        if temporal_percentage > technical_percentage:
            issues.append("КРИТИЧНО: Temporal важнее Technical!")
            severity = "critical"
        
        # Проверка 4: Проблемные признаки в топе
        if USE_FEATURE_MAPPING:
            blacklist = get_temporal_blacklist()
            top_temporal = category_analysis['temporal']['top_features']
            problematic = [f for f in top_temporal if f in blacklist]
            if problematic:
                issues.append(f"Проблемные temporal в топе: {problematic}")
                if severity != "critical":
                    severity = "warning"
        
        result = {
            'valid': len(issues) == 0,
            'severity': severity,
            'issues': issues,
            'temporal_percentage': temporal_percentage,
            'technical_percentage': technical_percentage
        }
        
        # Логируем для этой модели
        if issues:
            logger.warning(f"⚠️ Проблемы в модели {model_name}:")
            for issue in issues:
                logger.warning(f"   - {issue}")
        else:
            logger.info(f"✅ Модель {model_name}: признаки валидны")
        
        return result
    
    def _log_ensemble_results(self, category_analysis: Dict, ensemble_validation: Dict, 
                            model_results: Dict):
        """Логирование результатов ансамбля"""
        logger.info("\n" + "="*60)
        logger.info("📊 ИТОГОВЫЙ АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        logger.info("="*60)
        
        # Статистика по категориям
        logger.info("📈 Распределение важности по категориям:")
        for category, stats in category_analysis.items():
            if stats['count'] > 0:
                logger.info(f"   {category}: {stats['count']} признаков, "
                          f"{stats['percentage']:.1f}% важности")
                if stats['top_features']:
                    logger.info(f"      Топ: {', '.join(stats['top_features'])}")
        
        # Общая валидация
        if ensemble_validation['valid']:
            logger.info("✅ ВАЛИДАЦИЯ ПРОЙДЕНА: Модель не переобучается на временных паттернах")
        else:
            logger.error("❌ ВАЛИДАЦИЯ НЕ ПРОЙДЕНА!")
            logger.error(f"   Серьезность: {ensemble_validation['severity'].upper()}")
            for issue in ensemble_validation['issues']:
                logger.error(f"   - {issue}")
        
        # Статистика по отдельным моделям
        critical_models = [name for name, result in model_results.items() 
                          if result.get('severity') == 'critical']
        if critical_models:
            logger.warning(f"⚠️ Модели с критическими проблемами: {', '.join(critical_models)}")
    
    def _categorize_feature_fallback(self, feature_name: str) -> str:
        """Fallback категоризация если feature_mapping недоступен"""
        feature_lower = feature_name.lower()
        
        # Временные
        if any(t in feature_lower for t in ['hour', 'dow', 'weekend']):
            return 'temporal'
        
        # BTC related
        if 'btc_' in feature_lower:
            return 'btc_related'
        
        # Symbol
        if feature_lower.startswith('is_'):
            return 'symbol'
        
        # По умолчанию считаем техническими
        return 'technical'
    
    def get_recommendations(self) -> List[str]:
        """Получить рекомендации по улучшению модели"""
        recommendations = []
        
        if not self.validation_results:
            return ["Нет данных для анализа"]
        
        # Анализируем все результаты
        critical_count = sum(1 for result in self.validation_results.values() 
                           if result['validation'].get('severity') == 'critical')
        
        if critical_count > 0:
            recommendations.append("КРИТИЧНО: Модель переобучается на временных паттернах")
            recommendations.append("1. Уменьшить квоту temporal признаков до 1%")
            recommendations.append("2. Исключить dow_sin, dow_cos, is_weekend из обучения")
            recommendations.append("3. Увеличить регуляризацию (alpha, lambda)")
            recommendations.append("4. Использовать временное разделение данных")
        
        return recommendations