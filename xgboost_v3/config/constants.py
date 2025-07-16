"""
Константы и параметры для XGBoost v3.0
Централизованное хранение всех магических чисел и пороговых значений
"""

# Epsilon значения для защиты от деления на ноль
EPSILON = 1e-8
EPSILON_PRICE = 1e-8  # Для цен
EPSILON_VOLUME = 1e-8  # Для объемов
EPSILON_STD = 1e-8    # Для стандартных отклонений

# Параметры для рыночных признаков
MARKET_FEATURES = {
    # Коэффициенты для синтетических данных (ВРЕМЕННО - нужно заменить на реальные данные)
    'btc_synthetic_min': 0.8,
    'btc_synthetic_max': 1.2,
    
    # Окна для корреляций
    'correlation_windows': [20, 60],
    
    # Периоды для возвратов (в 15-минутных свечах)
    'return_periods': {
        '1h': 4,   # 4 * 15 мин = 1 час
        '4h': 16   # 16 * 15 мин = 4 часа
    },
    
    # Квантили для режимов волатильности
    'volatility_quantiles': [0.33, 0.67]
}

# Параметры для OHLC признаков
OHLC_FEATURES = {
    # Множитель для приближения EMA50 через EMA15
    'ema50_multiplier': 3.33,
    
    # Пороги для паттернов свечей
    'candle_patterns': {
        'hammer_body_size': 0.002,
        'hammer_shadow_ratio': 2.0,
        'doji_body_size': 0.001
    }
}

# Параметры для дивергенций
DIVERGENCE_PARAMS = {
    # Пороги изменений для определения дивергенций
    'price_change_threshold': 0.01,      # 1% изменение цены
    'rsi_change_threshold': 0.01,        # 1% изменение RSI
    'volume_price_threshold': 0.02,      # 2% изменение цены для volume дивергенции
    'volume_change_threshold': -0.1,     # -10% изменение объема
    
    # Окно для расчета дивергенций (из конфига)
    'default_window': 14
}

# Параметры для ансамбля
ENSEMBLE_PARAMS = {
    # Вариации параметров для разнообразия моделей
    'model_variations': [
        {'max_depth': 6, 'learning_rate': 0.01, 'subsample': 0.8},
        {'max_depth': 8, 'learning_rate': 0.005, 'subsample': 0.7},
        {'max_depth': 5, 'learning_rate': 0.02, 'subsample': 0.9},
        {'max_depth': 7, 'learning_rate': 0.01, 'subsample': 0.75},
        {'max_depth': 6, 'learning_rate': 0.015, 'subsample': 0.85},
        # Дополнительные вариации для больших ансамблей
        {'max_depth': 9, 'learning_rate': 0.008, 'subsample': 0.65},
        {'max_depth': 4, 'learning_rate': 0.025, 'subsample': 0.95},
        {'max_depth': 7, 'learning_rate': 0.012, 'subsample': 0.82}
    ],
    
    # Параметры подвыборки
    'subsample_ratio': 0.8,              # Доля данных для каждой модели
    'bootstrap': True,                   # Использовать bootstrap sampling
    
    # Параметры взвешивания
    'score_normalization': {
        'clip_min': -2,                  # Минимальное значение после нормализации
        'clip_max': 2,                   # Максимальное значение после нормализации
        'similarity_threshold': 0.01,     # Порог для определения похожих моделей
    },
    
    # Параметры сглаживания весов
    'weight_smoothing': {
        'extreme_weight_threshold': 0.9,  # Порог для экстремальных весов
        'model_weight': 0.8,             # Вес модели при сглаживании
        'uniform_weight': 0.2            # Вес равномерного распределения
    },
    
    # Пороги для голосования
    'voting_threshold': 0.5              # Порог для бинарной классификации
}

# Параметры для валидации данных
VALIDATION_PARAMS = {
    # Максимальная допустимая корреляция признака с целевой переменной
    'max_feature_target_correlation': 0.95,
    
    # Минимальное количество уникальных значений для признака
    'min_unique_values': 2,
    
    # Параметры для обработки выбросов
    'outlier_std_threshold': 5.0         # Количество стандартных отклонений для выброса
}

# Стратегии заполнения пропущенных значений
FILLNA_STRATEGIES = {
    # Осцилляторы с диапазоном 0-100
    'rsi_val': 50, 'rsi_ma': 50,
    'stoch_k': 50, 'stoch_d': 50, 'stoch_signal': 50,
    'aroon_up': 50, 'aroon_down': 50, 'aroon_oscillator': 0,
    'mfi': 50,
    
    # Трендовые индикаторы
    'adx_val': 25, 'adx_plus_di': 25, 'adx_minus_di': 25, 'adx_diff': 0,
    'cci': 0, 'williams_r': -50,
    
    # Волатильность и объем
    'atr': 0, 'bb_width': 0, 'volume_ratio': 1,
    'obv': 0, 'obv_slope': 0, 'cmf': 0,
    
    # Позиционные индикаторы
    'bb_position': 0.5,
    'sar_trend': 0, 'sar_distance': 0,
    
    # Ichimoku
    'ich_tenkan': 0, 'ich_kijun': 0, 'ich_senkou_a': 0, 
    'ich_senkou_b': 0, 'ich_chikou': 0,
    'ich_tenkan_kijun_signal': 0, 'ich_price_kumo': 0,
    
    # Bollinger Bands и другие bands
    'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0,
    'kc_upper': 0, 'kc_lower': 0,
    'dc_upper': 0, 'dc_lower': 0,
    
    # MACD компоненты  
    'macd_val': 0, 'macd_signal': 0, 'macd_hist': 0, 'macd_signal_ratio': 0,
    
    # Базовые цены
    'ema_15': 0, 'sar': 0
}

# Параметры для загрузки данных BTC
BTC_DATA_PARAMS = {
    'symbol': 'BTCUSDT',
    'source': 'database',  # 'database' или 'api'
    'fallback_to_synthetic': False,  # Использовать синтетические данные если нет реальных
    'cache_duration': 3600,  # Кэширование в секундах
}

# Топовые символы для one-hot encoding
TOP_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'SOLUSDT', 'DOTUSDT', 'MATICUSDT', 'SHIBUSDT',
    'AVAXUSDT', 'LTCUSDT', 'UNIUSDT', 'LINKUSDT'
]

# Параметры для проверки утечки данных
DATA_LEAKAGE_PARAMS = {
    'check_n_features': 10,  # Количество признаков для проверки
    'check_all_features': False,  # Проверять все признаки (медленно)
    'random_sample': True  # Использовать случайную выборку
}

# Параметры для оптимизации гиперпараметров
OPTUNA_PARAMS = {
    # Диапазоны для гиперпараметров (СИЛЬНАЯ РЕГУЛЯРИЗАЦИЯ против переобучения)
    'max_depth': {'min': 2, 'max': 4},  # Только неглубокие деревья
    'learning_rate': {'min': 0.005, 'max': 0.02, 'log': True},  # Медленное обучение
    'subsample': {'min': 0.3, 'max': 0.6},  # Меньше данных на дерево
    'colsample_bytree': {'min': 0.3, 'max': 0.6},  # Меньше признаков
    'colsample_bylevel': {'min': 0.3, 'max': 0.6},  # Меньше признаков на уровень
    'min_child_weight': {'min': 50, 'max': 200},  # Большие листья
    'gamma': {'min': 5.0, 'max': 20.0},  # Сильная обрезка
    'reg_alpha': {'min': 1.0, 'max': 10.0},  # Сильная L1
    'reg_lambda': {'min': 2.0, 'max': 10.0},  # Сильная L2
    
    # Параметры для scale_pos_weight
    'scale_pos_weight_factor': {'min': 0.3, 'max': 1.0},
    
    # Параметры обучения для CV
    'cv_n_estimators': 500,  # Увеличиваем для лучшей оценки
    'cv_early_stopping_rounds': 50,
    
    # Семя для воспроизводимости
    'random_state': 42,
    'pruner_warmup_steps': 20
}

# Параметры для предсказаний
PREDICTION_PARAMS = {
    'classification_threshold': 1.5,      # Порог для создания меток классификации в % изменения цены
    'probability_threshold': 0.5,         # Порог вероятности для бинарной классификации (0-1)
    'optimize_threshold': True,           # Оптимизировать порог на валидации
    'threshold_search_range': (0.2, 0.8), # Расширенный диапазон поиска оптимального порога вероятности
    'threshold_search_steps': 20          # Количество шагов при поиске
}