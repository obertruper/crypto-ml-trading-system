"""
Feature Selection для XGBoost v3.0
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Класс для отбора признаков"""
    
    # Иерархия признаков для криптотрейдинга
    FEATURE_HIERARCHY = {
        'primary': [  # Основные технические индикаторы
            'rsi_val', 'rsi_ma', 'rsi_val_ma_10', 'rsi_val_ma_60',
            'macd_val', 'macd_signal', 'macd_hist', 'macd_signal_ratio',
            'bb_position', 'bb_width', 'bb_upper', 'bb_lower',
            'adx_val', 'adx_plus_di', 'adx_minus_di', 'adx_diff',
            'atr', 'atr_percent', 'volume_ratio', 'volume_ratio_ma',
            'stoch_k', 'stoch_d', 'stoch_signal',
            'williams_r', 'mfi', 'cci', 'cmf', 'obv', 'obv_slope'
        ],
        'secondary': [  # Корреляция с BTC и трендовые
            'btc_correlation_5', 'btc_correlation_20', 'btc_correlation_60',
            'btc_volatility', 'btc_volume_ratio', 'btc_price_ratio',
            'ema_15', 'sma_20', 'vwap', 'price_to_vwap',
            'sar', 'sar_distance', 'sar_trend',
            'ich_tenkan', 'ich_kijun', 'ich_senkou_a', 'ich_senkou_b'
        ],
        'auxiliary': [  # Временные и категориальные
            'market_regime_low_vol', 'market_regime_med_vol', 'market_regime_high_vol',
            'dow_sin', 'dow_cos', 'hour_sin', 'hour_cos',
            'is_btc', 'is_eth', 'is_bnb', 'is_doge', 'is_other'
        ]
    }
    
    def __init__(self, method: str = "importance", top_k: int = 50,
                 primary_ratio: float = 0.7, auxiliary_ratio: float = 0.2):
        """
        Args:
            method: Метод отбора признаков (importance, rfe, mutual_info, chi2, hierarchical)
            top_k: Количество лучших признаков для отбора
            primary_ratio: Минимальная доля основных признаков (0.7 = 70%)
            auxiliary_ratio: Максимальная доля вспомогательных признаков (0.2 = 20%)
        """
        self.method = method
        self.top_k = top_k
        self.primary_ratio = primary_ratio
        self.auxiliary_ratio = auxiliary_ratio
        self.selected_features = []
        self.feature_scores = {}
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       feature_names: List[str] = None, group_name: str = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Отбор признаков
        
        Args:
            X: Признаки
            y: Целевая переменная
            feature_names: Имена признаков
            
        Returns:
            X_selected: DataFrame с отобранными признаками
            selected_features: Список отобранных признаков
        """
        if feature_names is None:
            feature_names = list(X.columns)
            
        logger.info(f"🔍 Отбор признаков методом: {self.method}")
        logger.info(f"   Исходное количество признаков: {X.shape[1]}")
        
        if self.method == "importance":
            selected_features = self._select_by_importance(X, y, feature_names)
        elif self.method == "rfe":
            selected_features = self._select_by_rfe(X, y, feature_names)
        elif self.method == "mutual_info":
            selected_features = self._select_by_mutual_info(X, y, feature_names)
        elif self.method == "chi2":
            selected_features = self._select_by_chi2(X, y, feature_names)
        elif self.method == "combined":
            selected_features = self._select_combined(X, y, feature_names)
        elif self.method == "hierarchical":
            selected_features = self._select_hierarchical(X, y, feature_names, group_name)
        else:
            # По умолчанию используем все признаки
            selected_features = feature_names
            
        self.selected_features = selected_features
        
        # Отбираем только выбранные признаки
        X_selected = X[selected_features]
        
        logger.info(f"✅ Отобрано признаков: {len(selected_features)}")
        logger.info(f"📊 Топ-10 признаков: {selected_features[:10]}")
        
        return X_selected, selected_features
        
    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series, 
                            feature_names: List[str]) -> List[str]:
        """Отбор по важности признаков из XGBoost"""
        # Обучаем простую модель XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        # Получаем важность признаков
        importance = model.feature_importances_
        
        # Создаем DataFrame с важностью
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Сохраняем scores
        self.feature_scores = dict(zip(importance_df['feature'], 
                                      importance_df['importance']))
        
        # Отбираем топ-K признаков
        selected = importance_df.head(self.top_k)['feature'].tolist()
        
        # Применяем иерархическую коррекцию
        selected = self._apply_hierarchy_correction(selected, feature_names)
                
        return selected[:self.top_k]
        
    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series, 
                      feature_names: List[str]) -> List[str]:
        """Рекурсивное удаление признаков"""
        # Используем RandomForest для RFE
        estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        selector = RFE(estimator, n_features_to_select=self.top_k, step=0.1)
        selector.fit(X, y)
        
        # Отбираем признаки
        selected = [feature_names[i] for i in range(len(feature_names)) 
                   if selector.support_[i]]
        
        return selected
        
    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series, 
                              feature_names: List[str]) -> List[str]:
        """Отбор по взаимной информации"""
        selector = SelectKBest(mutual_info_classif, k=self.top_k)
        selector.fit(X, y)
        
        # Получаем маску отобранных признаков
        mask = selector.get_support()
        selected = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        
        # Сохраняем scores
        self.feature_scores = dict(zip(feature_names, selector.scores_))
        
        return selected
        
    def _select_by_chi2(self, X: pd.DataFrame, y: pd.Series, 
                       feature_names: List[str]) -> List[str]:
        """Отбор по chi2 (только для неотрицательных признаков)"""
        # Преобразуем в неотрицательные значения
        X_positive = X - X.min() + 1e-6
        
        selector = SelectKBest(chi2, k=self.top_k)
        selector.fit(X_positive, y)
        
        # Получаем маску отобранных признаков
        mask = selector.get_support()
        selected = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        
        return selected
        
    def _select_combined(self, X: pd.DataFrame, y: pd.Series, 
                        feature_names: List[str]) -> List[str]:
        """Комбинированный отбор признаков"""
        # Получаем признаки разными методами
        importance_features = set(self._select_by_importance(X, y, feature_names))
        mutual_info_features = set(self._select_by_mutual_info(X, y, feature_names))
        
        # Объединяем результаты (пересечение)
        combined = list(importance_features.intersection(mutual_info_features))
        
        # Если пересечение слишком маленькое, берем объединение
        if len(combined) < self.top_k // 2:
            combined = list(importance_features.union(mutual_info_features))
            
        # Сортируем по важности
        if self.feature_scores:
            combined.sort(key=lambda x: self.feature_scores.get(x, 0), reverse=True)
            
        return combined[:self.top_k]
        
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Получить отчет о важности признаков"""
        if not self.feature_scores:
            return pd.DataFrame()
            
        df = pd.DataFrame(list(self.feature_scores.items()), 
                         columns=['feature', 'score'])
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Добавляем нормализованную важность
        if df['score'].sum() > 0:
            df['normalized_score'] = df['score'] / df['score'].sum()
            df['cumulative_score'] = df['normalized_score'].cumsum()
        
        return df
    
    def _select_hierarchical(self, X: pd.DataFrame, y: pd.Series,
                           feature_names: List[str], group_name: str = None) -> List[str]:
        """Иерархический отбор признаков для криптотрейдинга с жесткими квотами"""
        logger.info("🏗️ Использую иерархический отбор признаков с распределением 60/20/10/10")
        
        # Сначала получаем важность всех признаков
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        # Создаем DataFrame с важностью
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        
        # УЛУЧШЕННАЯ категоризация признаков
        def get_category(feature):
            feature_lower = feature.lower()
            
            # 1. Технические индикаторы (80%)
            technical_patterns = [
                'rsi', 'macd', 'bb_', 'bollinger', 'adx', 'atr', 'stoch', 'williams', 
                'mfi', 'cci', 'cmf', 'obv', 'ema', 'sma', 'vwap', 'sar', 'ich_',
                'aroon', 'kc_', 'dc_', 'volume_ratio', 'price_momentum', 'volatility',
                'hl_spread', 'close_ratio', 'upper_shadow', 'lower_shadow',
                # Добавляем паттерны свечей и market regime в технические
                'hammer', 'doji', 'engulfing', 'consecutive', 'pattern', 'candle',
                'market_regime', 'regime', 'trend', 'divergence',
                # Добавляем производные признаки
                'gk_volatility', 'cumulative', 'position', 'log_return', 'log_volume',
                'price_to_', 'ratio', 'interaction', 'efficiency', 'signal'
            ]
            for pattern in technical_patterns:
                if pattern in feature_lower:
                    return 'technical'
            
            # 2. BTC корреляция (10%)
            btc_patterns = ['btc_', 'bitcoin']
            for pattern in btc_patterns:
                if pattern in feature_lower:
                    return 'btc_related'
            
            # 3. Временные признаки (5% - уменьшаем для борьбы с переобучением)
            time_patterns = ['hour', 'dow', 'day', 'week', 'month', 'time', 'weekend']
            # Исключаем паттерны свечей и режимы рынка из временных
            exclude_patterns = ['hammer', 'consecutive', 'market_regime', 'pattern', 'candle']
            
            # Проверяем что это действительно временной признак
            is_temporal = False
            for pattern in time_patterns:
                if pattern in feature_lower:
                    is_temporal = True
                    break
            
            # Проверяем исключения
            if is_temporal:
                for exclude in exclude_patterns:
                    if exclude in feature_lower:
                        is_temporal = False
                        break
                        
            if is_temporal:
                return 'temporal'
            
            # 4. Все остальное (10%)
            return 'other'
        
        importance_df['category'] = importance_df['feature'].apply(get_category)
        
        # Логируем текущее распределение
        logger.info("📊 Текущее распределение признаков:")
        for cat in ['technical', 'temporal', 'btc_related', 'other']:
            count = len(importance_df[importance_df['category'] == cat])
            logger.info(f"   {cat}: {count} признаков")
        
        # Сортируем по важности внутри каждой категории
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Определяем квоты в зависимости от группы монет
        if group_name:
            # Получаем адаптивные веса для группы
            from ..data.preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor(None)
            weights = preprocessor.get_group_weights(group_name)
            
            n_technical = int(self.top_k * weights.get('technical', 0.6))
            n_temporal = int(self.top_k * weights.get('temporal', 0.2))
            n_btc = int(self.top_k * weights.get('btc_related', 0.1))
            n_other = self.top_k - n_technical - n_temporal - n_btc
            
            logger.info(f"🎯 Адаптивные квоты для группы '{group_name}':")
            logger.info(f"   Technical: {n_technical} ({weights.get('technical', 0.6)*100:.0f}%)")
            logger.info(f"   Temporal: {n_temporal} ({weights.get('temporal', 0.2)*100:.0f}%)")
            logger.info(f"   BTC: {n_btc} ({weights.get('btc_related', 0.1)*100:.0f}%)")
            logger.info(f"   Other: {n_other}")
        else:
            # ИЗМЕНЕНО: Уменьшаем временные признаки для борьбы с переобучением
            # Было: 60/20/10/10, стало: 80/5/10/5
            n_technical = int(self.top_k * 0.8)    # 80% - увеличиваем технические
            n_temporal = int(self.top_k * 0.05)    # 5% - резко уменьшаем временные
            n_btc = int(self.top_k * 0.1)          # 10% - оставляем BTC
            n_other = self.top_k - n_technical - n_temporal - n_btc  # 5%
        
        selected = []
        
        # 1. Отбираем технические индикаторы (80%)
        technical_features = importance_df[importance_df['category'] == 'technical']
        if len(technical_features) < n_technical:
            logger.warning(f"⚠️ Недостаточно технических признаков: {len(technical_features)} < {n_technical}")
            selected.extend(technical_features['feature'].tolist())
            # НЕ добавляем лучшие из других категорий - это нарушает квоты!
            # Просто используем то что есть
        else:
            selected.extend(technical_features.head(n_technical)['feature'].tolist())
        
        # 2. Отбираем временные признаки (5%)
        temporal_features = importance_df[importance_df['category'] == 'temporal']
        temporal_features = temporal_features[~temporal_features['feature'].isin(selected)]
        selected.extend(temporal_features.head(n_temporal)['feature'].tolist())
        
        # 3. Отбираем BTC-related признаки (10%)
        btc_features = importance_df[importance_df['category'] == 'btc_related']
        btc_features = btc_features[~btc_features['feature'].isin(selected)]
        selected.extend(btc_features.head(n_btc)['feature'].tolist())
        
        # 4. Отбираем остальные признаки (10%)
        other_features = importance_df[importance_df['category'] == 'other']
        other_features = other_features[~other_features['feature'].isin(selected)]
        selected.extend(other_features.head(n_other)['feature'].tolist())
        
        # Если не хватает признаков, добираем лучшие по важности
        if len(selected) < self.top_k:
            remaining = self.top_k - len(selected)
            all_remaining = importance_df[~importance_df['feature'].isin(selected)]
            selected.extend(all_remaining.head(remaining)['feature'].tolist())
        
        # Логируем финальное распределение
        logger.info("✅ ФИНАЛЬНОЕ распределение отобранных признаков:")
        final_counts = {}
        for feature in selected:
            cat = get_category(feature)
            final_counts[cat] = final_counts.get(cat, 0) + 1
        
        for cat in ['technical', 'temporal', 'btc_related', 'other']:
            count = final_counts.get(cat, 0)
            percentage = count/len(selected)*100 if selected else 0
            target = {'technical': 80, 'temporal': 5, 'btc_related': 10, 'other': 5}.get(cat, 0)
            status = "✅" if abs(percentage - target) < 5 else "⚠️"
            logger.info(f"   {cat}: {count} ({percentage:.1f}%) {status} [цель: {target}%]")
        
        # Логируем топ признаки по категориям
        logger.info("\n📋 Топ признаки по категориям:")
        for cat in ['technical', 'temporal', 'btc_related', 'other']:
            cat_features = [f for f in selected if get_category(f) == cat][:5]
            if cat_features:
                logger.info(f"   {cat}: {', '.join(cat_features)}")
        
        # Сохраняем scores
        self.feature_scores = dict(zip(importance_df['feature'], 
                                      importance_df['importance']))
        
        return selected
    
    def _apply_hierarchy_correction(self, selected: List[str], 
                                  feature_names: List[str]) -> List[str]:
        """Применяет иерархическую коррекцию к отобранным признакам"""
        # Считаем текущее распределение
        def get_category(feature):
            for category, features in self.FEATURE_HIERARCHY.items():
                if any(f in feature for f in features):
                    return category
            return 'auxiliary'
        
        category_counts = {'primary': 0, 'secondary': 0, 'auxiliary': 0}
        for feature in selected:
            category = get_category(feature)
            category_counts[category] += 1
        
        # Проверяем квоты
        n_primary_needed = int(self.top_k * self.primary_ratio)
        n_auxiliary_max = int(self.top_k * self.auxiliary_ratio)
        
        # Если недостаточно primary признаков
        if category_counts['primary'] < n_primary_needed:
            # Добавляем важные технические индикаторы
            for feature in self.FEATURE_HIERARCHY['primary']:
                if feature in feature_names and feature not in selected:
                    selected.append(feature)
                    category_counts['primary'] += 1
                    if category_counts['primary'] >= n_primary_needed:
                        break
        
        # Если слишком много auxiliary признаков
        if category_counts['auxiliary'] > n_auxiliary_max:
            # Удаляем лишние auxiliary признаки
            auxiliary_to_remove = category_counts['auxiliary'] - n_auxiliary_max
            removed = 0
            selected_copy = selected.copy()
            for feature in reversed(selected_copy):
                if get_category(feature) == 'auxiliary':
                    selected.remove(feature)
                    removed += 1
                    if removed >= auxiliary_to_remove:
                        break
        
        return selected