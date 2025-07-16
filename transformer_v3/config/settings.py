"""
Централизованная конфигурация для Transformer v3.0
Адаптировано из XGBoost v3 для Temporal Fusion Transformer
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path


# Константы из основного проекта
EXCLUDE_SYMBOLS = [
    'TESTUSDT',  # Тестовый токен Bybit
    'USTUSDT',   # UST (Terra USD) - деактивирован
    'LUNAUSDT',  # LUNA - старый токен Terra
    'FTTUSDT',   # FTX Token - банкрот
    'SRMUSER',   # Serum - связан с FTX
    'GFTUSDT',   # Gifto - делистинг
    'ANCUSDT',   # Anchor Protocol - Terra экосистема
    'MIRRUSDT',  # Mirror Protocol - Terra экосистема
    'RAYUSDT'    # Возможные проблемы с ликвидностью
]


@dataclass
class DatabaseConfig:
    """Настройки подключения к БД"""
    host: str = "localhost"
    port: int = 5555
    database: str = "crypto_trading"
    user: str = "ruslan"
    password: str = ""
    
    def __post_init__(self):
        """Автоматическая настройка для сервера"""
        import os
        # Если запускается на сервере vast.ai
        if os.path.exists('/workspace') and not os.path.exists('/Users'):
            # На сервере используем туннель на порт 5555
            self.host = "127.0.0.1"
            self.port = 5555
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Параметры для psycopg2"""
        params = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user
        }
        if self.password:
            params["password"] = self.password
        return params


@dataclass
class ModelConfig:
    """Настройки модели Temporal Fusion Transformer"""
    # Архитектура
    sequence_length: int = 50   # Уменьшенная длина последовательности
    d_model: int = 128          # Уменьшенная размерность модели
    num_heads: int = 8          # Количество голов внимания
    num_transformer_blocks: int = 3  # Меньше блоков трансформера
    ff_dim: int = 256           # Уменьшенная размерность feedforward
    dropout_rate: float = 0.3   # Увеличенный dropout для регуляризации
    
    # LSTM параметры
    lstm_units: int = 128       # Уменьшенная размерность LSTM
    
    # GRN параметры
    grn_units: int = 128        # Уменьшенная размерность GRN
    
    # Выходной слой
    mlp_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # GPU настройки
    use_mixed_precision: bool = False  # Временно отключено для стабильности (NaN fix)
    memory_growth: bool = True  # Динамическое выделение памяти GPU
    use_multi_gpu: bool = True  # Использовать multi-GPU обучение если доступно
    num_gpus: int = -1  # -1 = использовать все доступные GPU
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для создания модели"""
        return {
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_transformer_blocks': self.num_transformer_blocks,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'lstm_units': self.lstm_units,
            'grn_units': self.grn_units,
            'mlp_units': self.mlp_units
        }


@dataclass
class TrainingConfig:
    """Настройки процесса обучения"""
    # Режимы работы
    task_type: str = "regression"  # regression, classification_binary
    
    # Параметры обучения
    batch_size: int = 128             # Еще меньше для лучшей генерализации
    epochs: int = 200                 # Больше эпох для полного обучения
    learning_rate: float = 0.0003     # Оптимальный для cosine annealing
    optimizer: str = "adamw"  # adamw для weight decay
    
    # Gradient accumulation для эмуляции еще больших батчей
    gradient_accumulation_steps: int = 1  # Отключаем, так как используем большой batch_size
    
    # Параметры оптимизатора
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    
    # Early stopping
    early_stopping_patience: int = 25    # Больше терпения
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    restore_best_weights: bool = True
    
    # Learning rate schedule
    reduce_lr_patience: int = 8          # Больше терпения для LR
    reduce_lr_factor: float = 0.7        # Мягче снижение LR
    reduce_lr_min: float = 5e-7          # Минимальный LR
    
    # Регуляризация
    loss_function: str = "huber"  # huber, mse, mae для регрессии
    huber_delta: float = 1.0
    label_smoothing: float = 0.05  # Для классификации
    
    # Данные
    validation_split: float = 0.15
    test_split: float = 0.15
    sequence_stride: int = 5  # Шаг при создании последовательностей
    
    # Аугментация данных
    use_augmentation: bool = True
    noise_level: float = 0.005  # Меньше шума для стабильности
    
    # Балансировка классов (для классификации)
    balance_method: str = "class_weight"  # none, class_weight, smote
    classification_threshold: float = 0.3  # Порог для бинарной классификации (%)
    
    # Логирование и визуализация
    log_dir: str = "logs"
    tensorboard: bool = True
    save_plots: bool = True
    plot_update_freq: int = 5  # Обновление графиков каждые N эпох
    verbose: int = 1
    
    # Сохранение моделей
    save_models: bool = True
    model_dir: str = "trained_model"
    checkpoint_monitor: str = "val_loss"
    save_best_only: bool = True


@dataclass
class Config:
    """Главный класс конфигурации"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Загрузка конфигурации из YAML файла"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Обновляем значения из файла
        if 'database' in data:
            for key, value in data['database'].items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)
                    
        if 'model' in data:
            for key, value in data['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
                    
        if 'training' in data:
            for key, value in data['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
                    
        return config
    
    def validate(self) -> bool:
        """Валидация конфигурации"""
        # Проверка режима задачи
        valid_tasks = ["regression", "classification_binary"]
        if self.training.task_type not in valid_tasks:
            raise ValueError(f"Неверный task_type: {self.training.task_type}")
        
        # Проверка параметров модели
        if self.model.sequence_length < 10:
            raise ValueError("sequence_length должен быть >= 10")
            
        if self.model.d_model % self.model.num_heads != 0:
            raise ValueError("d_model должен быть кратен num_heads")
        
        # Проверка параметров обучения
        if not 0 < self.training.validation_split < 1:
            raise ValueError("validation_split должен быть между 0 и 1")
            
        if not 0 < self.training.test_split < 1:
            raise ValueError("test_split должен быть между 0 и 1")
            
        if self.training.validation_split + self.training.test_split >= 1:
            raise ValueError("Сумма validation_split и test_split должна быть < 1")
        
        return True
    
    def get_log_dir(self) -> Path:
        """Получить директорию для логов"""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(self.training.log_dir) / f"transformer_v3_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def save(self, path: str):
        """Сохранение конфигурации в файл"""
        data = {
            'database': self.database.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__
        }
        
        # Преобразуем списки и другие сложные типы
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif hasattr(obj, '__dict__'):
                return convert_types(obj.__dict__)
            else:
                return obj
                
        data = convert_types(data)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def __str__(self) -> str:
        """Красивый вывод конфигурации"""
        return f"""
Transformer v3.0 Configuration
==============================
Task: {self.training.task_type}
Model: Temporal Fusion Transformer
Sequence Length: {self.model.sequence_length}
Model Dimension: {self.model.d_model}
Transformer Blocks: {self.model.num_transformer_blocks}

Training Parameters:
- Batch Size: {self.training.batch_size}
- Epochs: {self.training.epochs}
- Learning Rate: {self.training.learning_rate}
- Optimizer: {self.training.optimizer}
- Early Stopping Patience: {self.training.early_stopping_patience}

Database: {self.database.database} @ {self.database.host}:{self.database.port}
Log Directory: {self.training.log_dir}
"""