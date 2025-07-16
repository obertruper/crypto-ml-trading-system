"""
Конфигурация архитектуры Temporal Fusion Transformer
"""

# Архитектура TFT компонентов
TFT_ARCHITECTURE = {
    # Variable Selection Network
    'vsn': {
        'hidden_units': 128,
        'dropout': 0.1,
        'activation': 'gelu'
    },
    
    # LSTM Encoder
    'lstm': {
        'units': 128,
        'return_sequences': True,
        'dropout': 0.1,
        'recurrent_dropout': 0.1
    },
    
    # Gated Residual Network
    'grn': {
        'units': 128,
        'dropout': 0.1,
        'use_time_distributed': False
    },
    
    # Multi-Head Attention
    'attention': {
        'num_heads': 8,
        'key_dim': 128,
        'dropout': 0.1,
        'use_bias': True
    },
    
    # Transformer Blocks
    'transformer': {
        'num_blocks': 4,
        'ff_dim': 256,
        'ff_activation': 'gelu',
        'norm_epsilon': 1e-6
    },
    
    # Output Network
    'output': {
        'hidden_layers': [256, 128],
        'dropout': 0.2,
        'activation': 'gelu'
    }
}

# Параметры оптимизации
OPTIMIZATION_PARAMS = {
    'optimizer': {
        'adam': {
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'amsgrad': False
        },
        'adamw': {
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'weight_decay': 0.01
        },
        'sgd': {
            'momentum': 0.9,
            'nesterov': True
        }
    },
    
    'learning_rate_schedule': {
        'type': 'reduce_on_plateau',  # reduce_on_plateau, exponential, cosine
        'reduce_factor': 0.5,
        'patience': 7,
        'min_lr': 1e-7,
        'cooldown': 3
    },
    
    'gradient_clipping': {
        'enabled': True,
        'max_norm': 1.0,
        'type': 'norm'  # norm, value
    }
}

# Параметры callbacks
CALLBACK_PARAMS = {
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 20,
        'mode': 'min',
        'restore_best_weights': True,
        'verbose': 1
    },
    
    'model_checkpoint': {
        'monitor': 'val_loss',
        'save_best_only': True,
        'save_weights_only': False,
        'mode': 'min',
        'verbose': 1
    },
    
    'tensorboard': {
        'histogram_freq': 1,
        'write_graph': True,
        'write_images': False,
        'update_freq': 'epoch',
        'profile_batch': 0
    },
    
    'csv_logger': {
        'separator': ',',
        'append': False
    }
}

# Loss функции для разных задач
LOSS_FUNCTIONS = {
    'regression': {
        'huber': {'delta': 1.0},
        'mse': {},
        'mae': {},
        'mape': {},
        'logcosh': {}
    },
    'classification_binary': {
        'binary_crossentropy': {
            'from_logits': False,
            'label_smoothing': 0.05
        },
        'focal_loss': {
            'gamma': 2.0,
            'alpha': 0.25
        }
    }
}

# Метрики для мониторинга
METRICS = {
    'regression': [
        'mae',
        'mse',
        'mape',
        'cosine_similarity'
    ],
    'classification_binary': [
        'accuracy',
        'precision',
        'recall',
        'auc'
    ]
}

# Регуляризация
REGULARIZATION = {
    'l1_l2': {
        'l1': 0.0,
        'l2': 0.01
    },
    'dropout': {
        'rate': 0.2,
        'noise_shape': None,
        'seed': 42
    },
    'batch_norm': {
        'momentum': 0.99,
        'epsilon': 0.001,
        'center': True,
        'scale': True
    },
    'layer_norm': {
        'epsilon': 1e-6,
        'center': True,
        'scale': True
    }
}

# Инициализация весов
WEIGHT_INIT = {
    'method': 'glorot_uniform',  # glorot_uniform, glorot_normal, he_uniform, he_normal
    'seed': 42
}

# Параметры для интерпретации модели
INTERPRETATION_PARAMS = {
    'attention_weights': {
        'save': True,
        'visualize': True,
        'top_k': 10  # Топ K важных временных шагов
    },
    'feature_importance': {
        'method': 'permutation',  # permutation, gradient
        'n_repeats': 10,
        'save': True
    }
}

# Экспериментальные функции
EXPERIMENTAL = {
    'use_mixed_precision': True,
    'xla_compile': False,
    'distribute_strategy': None,  # None, 'mirrored', 'multi_worker_mirrored'
    'cache_dataset': True,
    'prefetch_buffer': 'auto'
}