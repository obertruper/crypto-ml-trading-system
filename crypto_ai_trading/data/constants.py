"""
Константы для определения структуры данных
"""

# Служебные колонки (не признаки и не целевые)
SERVICE_COLUMNS = ['id', 'symbol', 'datetime', 'timestamp', 'sector']

# Основные целевые переменные для торговой модели v4.0
# 20 переменных без утечек данных (убраны переменные с утечками)
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
    'max_rally_4h',        
    
    # УДАЛЕНО: best_action, signal_strength, risk_reward_ratio, optimal_hold_time
    # Эти переменные содержали утечки данных и будут генерироваться
    # в trading/signal_generator.py на основе предсказаний модели
]

# Дополнительные целевые переменные v4.0 - legacy совместимость
ADDITIONAL_TARGET_VARIABLES = [
    # Legacy переменные для совместимости (создаются в feature_engineering)
    'best_direction',  # Упрощенная версия best_action
    'long_tp1_reached', 'long_tp2_reached', 'long_tp3_reached',
    'short_tp1_reached', 'short_tp2_reached', 'short_tp3_reached',
    'long_expected_value', 'short_expected_value'
]

# Все целевые переменные (20 основных + дополнительные legacy)
# Убираем дубликаты через set()
ALL_TARGET_VARIABLES = list(set(TRADING_TARGET_VARIABLES + ADDITIONAL_TARGET_VARIABLES))

# Префиксы для автоматического определения целевых переменных v4.0
TARGET_PREFIXES = (
    'future_return_',      # Базовые возвраты
    'direction_',          # Направления движения
    'long_will_reach_',    # LONG прибыль
    'short_will_reach_',   # SHORT прибыль
    'max_drawdown_',       # Риск-метрики
    'max_rally_',          # Риск-метрики
)

# Расширенные префиксы (включая legacy)
EXTENDED_TARGET_PREFIXES = TARGET_PREFIXES + (
    'best_direction',      # Legacy
    'long_tp', 'short_tp', # Legacy
    'long_expected', 'short_expected'  # Legacy
)

def get_feature_columns(df_columns):
    """Получить список признаков из колонок DataFrame"""
    return [col for col in df_columns 
            if col not in SERVICE_COLUMNS 
            and col not in ALL_TARGET_VARIABLES]

def get_target_columns(df_columns, use_extended=False):
    """Получить список целевых переменных из колонок DataFrame"""
    if use_extended:
        # Возвращает все 49 переменных
        return [col for col in df_columns if col in ALL_TARGET_VARIABLES]
    else:
        # Возвращает только основные 20 для обучения
        return [col for col in df_columns if col in TRADING_TARGET_VARIABLES]

def validate_data_structure(df):
    """Проверить структуру данных"""
    missing_targets = [col for col in TRADING_TARGET_VARIABLES if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Отсутствуют обязательные целевые переменные: {missing_targets}")
    
    feature_cols = get_feature_columns(df.columns)
    target_cols = get_target_columns(df.columns)
    
    return {
        'n_features': len(feature_cols),
        'n_targets': len(target_cols),
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'missing_targets': missing_targets
    }