"""
Модуль для работы с конфигурационными файлами
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import json


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML файла
    
    Args:
        config_path: путь к файлу конфигурации
        
    Returns:
        словарь с конфигурацией
    """
    if config_path is None:
        config_path = "config/config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Загружаем переменные окружения если есть
    config = _substitute_env_vars(config)
    
    # Валидация конфигурации
    config = _validate_config(config)
    
    return config


def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Подстановка переменных окружения в конфигурацию
    
    Формат: ${ENV_VAR_NAME:default_value}
    """
    def _process_value(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            # Извлекаем имя переменной и значение по умолчанию
            var_content = value[2:-1]
            if ':' in var_content:
                var_name, default_value = var_content.split(':', 1)
            else:
                var_name = var_content
                default_value = None
            
            # Получаем значение из окружения
            env_value = os.environ.get(var_name, default_value)
            if env_value is None:
                raise ValueError(f"Переменная окружения {var_name} не найдена и не задано значение по умолчанию")
            
            # Пытаемся преобразовать в правильный тип
            try:
                # Числа
                if '.' in env_value:
                    return float(env_value)
                else:
                    return int(env_value)
            except ValueError:
                # Булевы значения
                if env_value.lower() in ('true', 'false'):
                    return env_value.lower() == 'true'
                # Иначе возвращаем как строку
                return env_value
        
        elif isinstance(value, dict):
            return {k: _process_value(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [_process_value(item) for item in value]
        
        else:
            return value
    
    return _process_value(config)


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Валидация и дополнение конфигурации значениями по умолчанию
    """
    # Проверяем обязательные секции
    required_sections = ['model', 'data', 'optimizer']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Обязательная секция '{section}' отсутствует в конфигурации")
    
    # Дополняем значениями по умолчанию
    defaults = {
        'model': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout': 0.1,
            'context_window': 168,
            'early_stopping_patience': 10
        },
        'data': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        },
        'optimizer': {
            'name': 'AdamW',
            'params': {
                'weight_decay': 0.01
            }
        },
        'performance': {
            'device': 'cuda',
            'num_workers': 4,
            'pin_memory': True,
            'mixed_precision': False
        },
        'logging': {
            'level': 'INFO',
            'tensorboard': True
        }
    }
    
    # Рекурсивное слияние с defaults
    def merge_dicts(default, custom):
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    config = merge_dicts(defaults, config)
    
    return config


def save_config(config: Dict[str, Any], path: str):
    """
    Сохранение конфигурации в файл
    
    Args:
        config: словарь с конфигурацией
        path: путь для сохранения
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.yaml' or path.suffix == '.yml':
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    elif path.suffix == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Слияние нескольких конфигураций
    
    Более поздние конфигурации перезаписывают более ранние
    
    Args:
        *configs: конфигурации для слияния
        
    Returns:
        объединенная конфигурация
    """
    result = {}
    
    def merge_dicts(dict1, dict2):
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    for config in configs:
        result = merge_dicts(result, config)
    
    return result


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлечение конфигурации модели
    
    Args:
        config: полная конфигурация
        
    Returns:
        конфигурация модели
    """
    model_config = config.get('model', {})
    
    # Добавляем некоторые параметры из других секций
    if 'data' in config:
        if 'n_features' not in model_config:
            # Количество признаков может быть в data секции
            model_config['n_features'] = config['data'].get('n_features', 171)
    
    return model_config


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлечение конфигурации для обучения
    
    Args:
        config: полная конфигурация
        
    Returns:
        конфигурация обучения
    """
    training_config = {
        'epochs': config['model'].get('epochs', 100),
        'batch_size': config['model'].get('batch_size', 32),
        'learning_rate': config['model'].get('learning_rate', 0.001),
        'optimizer': config.get('optimizer', {}),
        'scheduler': config.get('scheduler', {}),
        'loss': config.get('loss', {}),
        'early_stopping': config['model'].get('early_stopping_patience', 10),
        'performance': config.get('performance', {})
    }
    
    return training_config