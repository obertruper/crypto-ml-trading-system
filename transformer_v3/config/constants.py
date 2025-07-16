"""
Константы для Transformer v3.0
Взято из train_universal_transformer.py и адаптировано
"""

# Полный список технических индикаторов (49 штук)
TECHNICAL_INDICATORS = [
    # Трендовые индикаторы
    'ema_15', 'adx_val', 'adx_plus_di', 'adx_minus_di',
    'macd_val', 'macd_signal', 'macd_hist', 'sar',
    'ichimoku_conv', 'ichimoku_base', 'aroon_up', 'aroon_down',
    'dpo',
    
    # Осцилляторы
    'rsi_val', 'stoch_k', 'stoch_d', 'cci_val', 'williams_r',
    'roc', 'ult_osc', 'mfi',
    
    # Волатильность
    'atr_val', 'bb_upper', 'bb_lower', 'bb_basis',
    'donchian_upper', 'donchian_lower',
    
    # Объемные индикаторы
    'obv', 'cmf', 'volume_sma', 'volume_ratio',
    
    # Vortex индикаторы
    'vortex_vip', 'vortex_vin',
    
    # Производные индикаторы
    'macd_signal_ratio', 'adx_diff', 'bb_position',
    'rsi_dist_from_mid', 'stoch_diff', 'vortex_ratio',
    'ichimoku_diff', 'atr_norm',
    
    # Временные признаки
    'hour', 'day_of_week', 'is_weekend',
    
    # Ценовые паттерны
    'price_change_1', 'price_change_4', 'price_change_16',
    'volatility_4', 'volatility_16'
]

# Инженерные признаки (добавляются к техническим)
ENGINEERED_FEATURES = [
    "rsi_oversold",      # RSI < 30
    "rsi_overbought",    # RSI > 70
    "macd_bullish",      # MACD > MACD Signal
    "bb_near_lower",     # Цена близко к нижней границе Bollinger
    "bb_near_upper",     # Цена близко к верхней границе Bollinger
    "strong_trend",      # ADX > 25
    "high_volume"        # Volume ratio > 2.0
]

# Параметры последовательностей
SEQUENCE_PARAMS = {
    'sequence_length': 50,  # 50 свечей = 12.5 часов на 15м таймфрейме (упрощено)
    'stride': 10,  # Увеличенный шаг для снижения корреляции
    'min_sequences_per_symbol': 100,  # Минимум последовательностей на символ
}

# Параметры риска (из конфигурации проекта)
RISK_PARAMS = {
    'stop_loss': 1.1,  # % от входа
    'take_profit': 5.8,  # % максимальная цель
    'partial_targets': [1.2, 2.4, 3.5],  # % уровни частичного закрытия
    'partial_sizes': [0.2, 0.3, 0.3],  # Размеры частичных закрытий
    'breakeven_level': 1.2,  # % для перевода в безубыток
}

# Параметры целевых значений
TARGET_PARAMS = {
    'lookforward_candles': 100,  # Анализ 100 свечей вперед
    'expected_return_levels': [-1.1, 0.48, 1.56, 2.49, 3.17, 5.8],  # Основные уровни expected returns
    'classification_threshold': 0.3,  # % порог для бинарной классификации
}

# Параметры нормализации
NORMALIZATION_PARAMS = {
    'scaler_type': 'robust',  # robust, standard, minmax
    'clip_outliers': True,
    'outlier_threshold': 5.0,  # Стандартные отклонения для клиппинга
}

# Параметры валидации
VALIDATION_PARAMS = {
    'time_based_split': True,  # Временное разделение данных
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'gap_periods': 100,  # Зазор между train/val/test для предотвращения утечки
}

# Символы для исключения
EXCLUDE_SYMBOLS = ['TESTUSDT', 'TESTBTC', 'TESTBNB']

# GPU параметры
GPU_PARAMS = {
    'memory_growth': True,
    'mixed_precision': True,
    'xla_jit': False,  # Экспериментальная JIT компиляция
}

# Параметры визуализации
VISUALIZATION_PARAMS = {
    'figure_size': (15, 10),
    'dpi': 150,
    'save_dpi': 150,
    'style': 'seaborn-v0_8',  # Обновленное название стиля
    'update_frequency': 5,  # Обновление графиков каждые N эпох
}

# Метрики для мониторинга
METRICS_CONFIG = {
    'regression': {
        'primary': 'mae',
        'secondary': ['rmse', 'r2', 'direction_accuracy'],
        'monitor': 'val_mae'
    },
    'classification_binary': {
        'primary': 'accuracy', 
        'secondary': ['precision', 'recall', 'f1', 'auc'],
        'monitor': 'val_accuracy'
    }
}

# Параметры логирования
LOGGING_CONFIG = {
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'encoding': 'utf-8',
    'level': 'INFO'
}

# Параметры сохранения
SAVE_CONFIG = {
    'save_frequency': 10,  # Сохранение checkpoint каждые N эпох
    'keep_n_checkpoints': 3,  # Количество checkpoint для хранения
    'save_optimizer_state': True,
    'save_training_history': True
}