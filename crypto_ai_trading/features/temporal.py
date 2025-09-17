"""
Временные эмбеддинги с защитой от утечек данных
Этап 1: Безопасные признаки без weekly/monthly циклов
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from utils.logger import get_logger


class TemporalEmbeddings:
    """
    Создание безопасных временных признаков
    Управляется через config.yaml для контроля рискованных признаков
    """
    
    # РАЗРЕШЕННЫЕ признаки - безопасные для использования
    ALLOWED_TEMPORAL = {
        'intraday': [
            'hour_sin', 'hour_cos',
            'minutes_from_day_open', 'normalized_time_of_day',
            'london_fix', 'power_hour'
        ],
        'sessions': [
            'asian_session', 'european_session', 'us_session',
            'session_overlap', 'asian_session_strength',
            'european_session_strength', 'us_session_strength'
        ]
    }
    
    # ОГРАНИЧЕННЫЕ признаки - потенциально опасные
    RESTRICTED_TEMPORAL = {
        'weekly': [
            'day_sin', 'day_cos', 'is_weekend', 'monday_effect', 
            'friday_effect', 'weekend_decay'
        ],
        'monthly': [
            'month_sin', 'month_cos', 'monthday_sin', 'monthday_cos',
            'month_start', 'month_end'
        ],
        'seasonal': [
            'quarter_end', 'tax_season_us', 'summer_period', 
            'year_end', 'options_expiry_week', 'futures_rollover_week'
        ]
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("TemporalEmbeddings")
        
        # Конфигурация разрешенных признаков
        temporal_config = config.get('features', {}).get('temporal', {})
        self.allow_weekly_cycle = temporal_config.get('allow_weekly_cycle', False)
        self.allow_monthly_cycle = temporal_config.get('allow_monthly_cycle', False)  
        self.allow_seasonal = temporal_config.get('allow_seasonal', False)
        
        self.logger.info(f"🕐 Временные эмбеддинги инициализированы:")
        self.logger.info(f"   - Weekly cycle: {self.allow_weekly_cycle}")
        self.logger.info(f"   - Monthly cycle: {self.allow_monthly_cycle}")
        self.logger.info(f"   - Seasonal: {self.allow_seasonal}")
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание временных признаков с учетом конфигурации безопасности
        
        Args:
            df: DataFrame с колонкой datetime
            
        Returns:
            DataFrame с добавленными временными признаками
        """
        df = df.copy()
        
        # Извлекаем компоненты времени
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['dayofweek'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        
        # Создаем безопасные признаки
        df = self._create_intraday_features(df)
        df = self._create_session_features(df)
        
        # Условно создаем ограниченные признаки
        if self.allow_weekly_cycle:
            df = self._create_weekly_features(df)
            self.logger.info("✅ Weekly cycle признаки добавлены")
            
        if self.allow_monthly_cycle:
            df = self._create_monthly_features(df)
            self.logger.info("✅ Monthly cycle признаки добавлены")
            
        if self.allow_seasonal:
            df = self._create_seasonal_features(df)
            self.logger.info("✅ Seasonal признаки добавлены")
        
        # Логируем созданные признаки
        created_features = [col for col in df.columns 
                          if any(pattern in col for pattern in 
                                ['tmp_', 'session_', 'hour_', 'day_', 'month_'])]
        self.logger.info(f"🎯 Создано временных признаков: {len(created_features)}")
        
        return df
    
    def _create_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Внутридневные признаки - всегда безопасные"""
        
        # 1. ЦИКЛИЧЕСКОЕ КОДИРОВАНИЕ ЧАСА (безопасно)
        df['tmp_hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['tmp_hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 2. ПОЗИЦИЯ В ДНЕ
        df['tmp_minutes_from_day_open'] = df['hour'] * 60 + df['minute']
        df['tmp_minutes_to_day_close'] = 1440 - df['tmp_minutes_from_day_open']
        df['tmp_normalized_time_of_day'] = df['tmp_minutes_from_day_open'] / 1440
        
        # 3. КЛЮЧЕВЫЕ ВРЕМЕННЫЕ МЕТКИ
        # London Fix (16:00 UTC)
        df['tmp_london_fix'] = ((df['hour'] == 16) & (df['minute'] < 30)).astype(float)
        
        # Power Hour (последние часы торговли традиционных рынков)
        df['tmp_power_hour'] = ((df['hour'] == 20) | (df['hour'] == 21)).astype(float)
        
        # Intensity по времени дня (пиковая активность)
        df['tmp_time_intensity'] = np.where(
            (df['hour'] >= 8) & (df['hour'] <= 22),  # Активные часы
            1.0 - abs(df['hour'] - 15) / 15,  # Пик в 15:00 UTC
            0.3  # Низкая активность ночью
        )
        
        return df
    
    def _create_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Торговые сессии - безопасные признаки"""
        
        # 1. ОСНОВНЫЕ СЕССИИ (UTC время)
        # Азиатская сессия: 00:00 - 08:00 UTC
        df['tmp_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(float)
        df['tmp_asian_session_strength'] = np.where(
            df['tmp_asian_session'] == 1,
            1.0 - abs(df['hour'] - 4) / 4,  # Пик в 04:00 UTC
            0.0
        )
        
        # Европейская сессия: 08:00 - 16:00 UTC
        df['tmp_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(float)
        df['tmp_european_session_strength'] = np.where(
            df['tmp_european_session'] == 1,
            1.0 - abs(df['hour'] - 12) / 4,  # Пик в 12:00 UTC
            0.0
        )
        
        # Американская сессия: 14:00 - 22:00 UTC
        df['tmp_us_session'] = ((df['hour'] >= 14) & (df['hour'] < 22)).astype(float)
        df['tmp_us_session_strength'] = np.where(
            df['tmp_us_session'] == 1,
            1.0 - abs(df['hour'] - 18) / 4,  # Пик в 18:00 UTC
            0.0
        )
        
        # 2. ПЕРЕКРЫТИЯ СЕССИЙ
        # Европа + США (высокая ликвидность)
        df['tmp_session_overlap'] = ((df['tmp_european_session'] + df['tmp_us_session']) > 1).astype(float)
        
        # Общая активность сессий
        df['tmp_session_activity'] = (
            df['tmp_asian_session_strength'] + 
            df['tmp_european_session_strength'] + 
            df['tmp_us_session_strength']
        ).clip(0, 1)
        
        # 3. МЕЖСЕССИОННЫЕ ПЕРИОДЫ (низкая ликвидность)
        df['tmp_inter_session'] = ((df['tmp_asian_session'] + df['tmp_european_session'] + 
                                   df['tmp_us_session']) == 0).astype(float)
        
        return df
    
    def _create_weekly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Недельные признаки - ОГРАНИЧЕННЫЕ (включаются по флагу)"""
        
        self.logger.warning("⚠️ Создаем weekly cycle признаки - возможны утечки!")
        
        # 1. ЦИКЛИЧЕСКОЕ КОДИРОВАНИЕ ДНЯ НЕДЕЛИ
        df['tmp_day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['tmp_day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # 2. СПЕЦИАЛЬНЫЕ ДНИ
        df['tmp_is_weekend'] = (df['dayofweek'] >= 5).astype(float)
        df['tmp_weekend_decay'] = df['tmp_is_weekend'] * (1 - df['hour'] / 48)
        
        # 3. ЭФФЕКТЫ ДНЕЙ НЕДЕЛИ
        # Понедельничный эффект
        df['tmp_monday_effect'] = (df['dayofweek'] == 0).astype(float) * (1 - df['hour'] / 24)
        
        # Пятничный эффект  
        df['tmp_friday_effect'] = (df['dayofweek'] == 4).astype(float) * (df['hour'] / 24)
        
        return df
    
    def _create_monthly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Месячные признаки - ОГРАНИЧЕННЫЕ (включаются по флагу)"""
        
        self.logger.warning("⚠️ Создаем monthly cycle признаки - возможны утечки!")
        
        # 1. ЦИКЛИЧЕСКОЕ КОДИРОВАНИЕ
        df['tmp_monthday_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['tmp_monthday_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        df['tmp_month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['tmp_month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 2. НАЧАЛО/КОНЕЦ МЕСЯЦА
        df['tmp_month_start'] = (df['day'] <= 5).astype(float)
        df['tmp_month_end'] = (df['day'] >= 26).astype(float)
        
        return df
    
    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Сезонные признаки - ОГРАНИЧЕННЫЕ (включаются по флагу)"""
        
        self.logger.warning("⚠️ Создаем seasonal признаки - высокий риск утечек!")
        
        # 1. КВАРТАЛЫ
        df['tmp_quarter_end'] = ((df['month'] % 3 == 0) & (df['day'] >= 25)).astype(float)
        
        # 2. НАЛОГОВЫЕ ПЕРИОДЫ
        df['tmp_tax_season_us'] = ((df['month'] == 4) | (df['month'] == 3)).astype(float)
        
        # 3. СЕЗОННЫЕ ПЕРИОДЫ
        df['tmp_summer_period'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(float)
        df['tmp_year_end'] = ((df['month'] == 12) & (df['day'] >= 15)).astype(float)
        
        # 4. ПРОИЗВОДНЫЕ СОБЫТИЯ
        # Options expiry (последняя пятница месяца)
        df['tmp_options_expiry_week'] = ((df['day'] >= 22) & (df['dayofweek'] == 4)).astype(float)
        
        # Futures rollover (3-я пятница)
        df['tmp_futures_rollover_week'] = ((df['day'] >= 15) & (df['day'] <= 21) & 
                                          (df['dayofweek'] == 4)).astype(float)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Получить список всех возможных временных признаков"""
        features = []
        
        # Всегда доступные
        features.extend(['tmp_' + f for f in self.ALLOWED_TEMPORAL['intraday']])
        features.extend(['tmp_' + f for f in self.ALLOWED_TEMPORAL['sessions']])
        features.extend(['tmp_session_activity', 'tmp_inter_session', 'tmp_time_intensity'])
        
        # Условно доступные
        if self.allow_weekly_cycle:
            features.extend(['tmp_' + f for f in self.RESTRICTED_TEMPORAL['weekly']])
            
        if self.allow_monthly_cycle:
            features.extend(['tmp_' + f for f in self.RESTRICTED_TEMPORAL['monthly']])
            
        if self.allow_seasonal:
            features.extend(['tmp_' + f for f in self.RESTRICTED_TEMPORAL['seasonal']])
        
        return features
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Валидация созданных признаков на предмет утечек"""
        validation_results = {}
        
        # Проверяем на аномальные паттерны
        for feature in self.get_feature_names():
            if feature in df.columns:
                # Проверка на константность (подозрительно)
                is_constant = df[feature].nunique() <= 2
                
                # Проверка на периодичность (может указывать на утечки)
                variance = df[feature].var()
                is_too_periodic = variance < 0.01
                
                validation_results[feature] = {
                    'exists': True,
                    'constant': is_constant,
                    'too_periodic': is_too_periodic,
                    'suspicious': is_constant or is_too_periodic
                }
        
        # Логируем подозрительные признаки
        suspicious = [f for f, v in validation_results.items() if v.get('suspicious', False)]
        if suspicious:
            self.logger.warning(f"⚠️ Подозрительные временные признаки: {suspicious}")
        
        return validation_results