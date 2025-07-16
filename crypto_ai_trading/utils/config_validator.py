"""
Валидация конфигурации с использованием Pydantic
"""

from pydantic import BaseModel, validator, Field, ValidationError
from typing import List, Optional, Dict, Any, Union
import yaml
from pathlib import Path
import logging


class DatabaseConfig(BaseModel):
    """Конфигурация базы данных"""
    host: str
    port: int = Field(ge=1, le=65535)
    database: str
    user: str
    password: str
    
    @validator('host')
    def host_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Host не может быть пустым')
        return v


class ModelConfig(BaseModel):
    """Конфигурация модели"""
    context_window: int = Field(gt=0, description="Размер контекстного окна")
    pred_len: int = Field(gt=0, description="Длина предсказания")
    batch_size: int = Field(gt=0, description="Размер батча")
    epochs: int = Field(gt=0, description="Количество эпох")
    learning_rate: float = Field(gt=0, lt=1, description="Скорость обучения")
    d_model: int = Field(gt=0, description="Размерность модели")
    n_heads: int = Field(gt=0, description="Количество голов внимания")
    n_layers: Optional[int] = Field(gt=0, description="Количество слоев", default=3)
    e_layers: Optional[int] = Field(gt=0, description="Количество слоев энкодера", default=3)
    d_ff: Optional[int] = Field(gt=0, description="Размерность FFN", default=512)
    patch_len: int = Field(gt=0, description="Длина патча")
    stride: int = Field(gt=0, description="Шаг патча")
    dropout: float = Field(ge=0, le=1, description="Dropout rate")
    input_size: Optional[int] = Field(gt=0, description="Размер входа", default=100)
    output_size: Optional[int] = Field(gt=0, description="Размер выхода", default=1)
    
    @validator('n_heads')
    def heads_divisible_by_d_model(cls, v, values):
        if 'd_model' in values and values['d_model'] % v != 0:
            raise ValueError(f'd_model ({values["d_model"]}) должен быть кратен n_heads ({v})')
        return v
    
    @validator('stride')
    def stride_less_than_patch_len(cls, v, values):
        if 'patch_len' in values and v > values['patch_len']:
            raise ValueError(f'stride ({v}) не может быть больше patch_len ({values["patch_len"]})')
        return v


class TradingConfig(BaseModel):
    """Конфигурация торговли"""
    min_confidence_threshold: float = Field(ge=0, le=1, description="Минимальная уверенность", default=0.65)
    max_positions: int = Field(gt=0, description="Максимум позиций", default=5)
    max_daily_trades: int = Field(gt=0, description="Максимум сделок в день", default=15)
    multiframe_confirmation: bool = Field(default=True)
    rebalance_interval: str = Field(default="1h")


class RiskManagementConfig(BaseModel):
    """Конфигурация риск-менеджмента"""
    max_concurrent_positions: int = Field(gt=0, default=10)
    max_positions_per_symbol: int = Field(gt=0, default=1)
    stop_loss_pct: float = Field(gt=0, default=2.0)
    risk_reward_ratio: float = Field(gt=0, default=3.0)
    take_profit_targets: List[float] = Field(min_items=1, default=[1.5, 2.5, 4.0])
    partial_close_sizes: List[int] = Field(min_items=1, default=[40, 40, 20])
    
    @validator('partial_close_sizes')
    def sizes_sum_to_100(cls, v):
        total = sum(v)
        if total != 100:
            raise ValueError(f'Сумма partial_close_sizes должна быть 100%, получено {total}%')
        return v


class PerformanceConfig(BaseModel):
    """Конфигурация производительности"""
    cache_features: bool = Field(default=True)
    cache_dir: str = Field(default="cache")
    max_cache_size_mb: int = Field(gt=0, default=500)
    max_cache_files: int = Field(gt=0, default=10)
    num_workers: int = Field(ge=0, default=4)
    device: str = Field(default="cuda")
    mixed_precision: bool = Field(default=True)


class DataConfig(BaseModel):
    """Конфигурация данных"""
    symbols: List[str] = Field(min_items=1, description="Список символов")
    start_date: str = Field(description="Дата начала")
    end_date: str = Field(description="Дата окончания")
    interval_minutes: int = Field(gt=0, default=15)
    train_ratio: float = Field(gt=0, le=1, default=0.7)
    val_ratio: float = Field(gt=0, le=1, default=0.15)
    test_ratio: float = Field(gt=0, le=1, default=0.15)
    
    @validator('symbols')
    def no_test_symbols(cls, v):
        test_symbols = [s for s in v if 'TEST' in s.upper()]
        if test_symbols:
            raise ValueError(f'Обнаружены тестовые символы: {test_symbols}')
        return v
    
    @validator('test_ratio')
    def ratios_sum_to_one(cls, v, values):
        if 'train_ratio' in values and 'val_ratio' in values:
            total = values['train_ratio'] + values['val_ratio'] + v
            if abs(total - 1.0) > 0.001:
                raise ValueError(f'Сумма train_ratio + val_ratio + test_ratio должна быть 1.0, получено {total}')
        return v


class Config(BaseModel):
    """Основная конфигурация"""
    database: DatabaseConfig
    model: ModelConfig
    trading: TradingConfig
    risk_management: RiskManagementConfig
    performance: PerformanceConfig
    data: DataConfig
    
    class Config:
        extra = "allow"  # Разрешаем дополнительные поля для обратной совместимости


def load_config(config_path: str = None) -> Config:
    """Загрузка и валидация конфигурации"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Валидация через Pydantic
    try:
        config = Config(**config_dict)
        return config
    except Exception as e:
        raise ValueError(f"Ошибка валидации конфигурации: {e}")


def validate_tensor_shapes(outputs: Any, targets: Any, context: str = ""):
    """Валидация размерностей тензоров"""
    import torch
    
    if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError(f"Ожидались torch тензоры в {context}")
    
    if outputs.shape != targets.shape:
        raise ValueError(
            f"Несоответствие размеров в {context}: "
            f"outputs {outputs.shape} != targets {targets.shape}"
        )
    
    if torch.isnan(outputs).any():
        raise ValueError(f"Обнаружены NaN значения в outputs ({context})")
    
    if torch.isnan(targets).any():
        raise ValueError(f"Обнаружены NaN значения в targets ({context})")


def validate_dataframe(df, required_columns: List[str], context: str = ""):
    """Валидация DataFrame"""
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Ожидался pandas DataFrame в {context}")
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки в {context}: {missing_cols}")
    
    if df.empty:
        raise ValueError(f"Пустой DataFrame в {context}")
    
    # Проверка на NaN в критических колонках
    critical_cols = ['datetime', 'symbol', 'close']
    for col in critical_cols:
        if col in df.columns and df[col].isna().any():
            nan_count = df[col].isna().sum()
            raise ValueError(f"Обнаружено {nan_count} NaN значений в колонке {col} ({context})")


def validate_config(config: Union[dict, Config]) -> bool:
    """Валидация конфигурации
    
    Args:
        config: Словарь конфигурации или объект Config
        
    Returns:
        bool: True если конфигурация валидна
    """
    try:
        if isinstance(config, dict):
            # Пытаемся создать объект Config из словаря
            Config(**config)
        elif isinstance(config, Config):
            # Уже валидированный объект
            pass
        else:
            raise ValueError("config должен быть dict или Config")
        
        logging.info("✅ Конфигурация валидна")
        return True
        
    except ValidationError as e:
        logging.error(f"❌ Ошибки валидации конфигурации:")
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            logging.error(f"  {field_path}: {error['msg']}")
        return False
    except Exception as e:
        logging.error(f"❌ Ошибка при валидации: {str(e)}")
        return False