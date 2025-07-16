"""
Централизованная конфигурация для XGBoost v3.0
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Настройки подключения к БД"""
    host: str = "localhost"
    port: int = 5555
    database: str = "crypto_trading"
    user: str = "ruslan"
    password: str = ""
    
    @property
    def connection_string(self) -> str:
        return f"host={self.host} port={self.port} dbname={self.database} user={self.user}"


@dataclass
class ModelConfig:
    """Настройки модели XGBoost"""
    # Основные параметры
    objective: str = "binary:logistic"
    eval_metric: str = "auc"  # AUC лучше для оценки качества разделения классов
    n_estimators: int = 500  # Меньше деревьев для простоты
    early_stopping_rounds: int = 30  # Быстрее останавливаемся при переобучении
    
    # Гиперпараметры по умолчанию (УСИЛЕНА РЕГУЛЯРИЗАЦИЯ против переобучения)
    max_depth: int = 3  # Неглубокие деревья (было 5)
    learning_rate: float = 0.01  # Медленное обучение (было 0.03)
    subsample: float = 0.5  # Меньше данных на дерево (было 0.7)
    colsample_bytree: float = 0.5  # Меньше признаков (было 0.7)
    min_child_weight: int = 100  # Большие листья (было 20)
    gamma: float = 10.0  # Сильная обрезка (было 1.0)
    reg_alpha: float = 2.0  # Сильная L1 регуляризация (было 0.5)
    reg_lambda: float = 5.0  # Сильная L2 регуляризация (было 2.0)
    
    # GPU настройки
    tree_method: str = "auto"  # auto, hist, gpu_hist
    predictor: str = "auto"    # auto, cpu_predictor, gpu_predictor
    gpu_id: Optional[int] = None  # ID GPU для использования
    
    # Балансировка классов
    scale_pos_weight: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для XGBoost"""
        params = {
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'tree_method': self.tree_method,
            'predictor': self.predictor,
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0
        }
        
        if self.scale_pos_weight is not None:
            params['scale_pos_weight'] = self.scale_pos_weight
            
        if self.gpu_id is not None:
            params['gpu_id'] = self.gpu_id
            
        return params


@dataclass
class TrainingConfig:
    """Настройки процесса обучения"""
    # Режимы работы
    task_type: str = "classification_binary"  # classification_binary, classification_multi, regression
    test_mode: bool = False
    use_cache: bool = True
    
    # Данные
    test_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    exclude_symbols: List[str] = field(default_factory=lambda: ["TESTUSDT", "TESTBTC"])
    validation_split: float = 0.2
    
    # Балансировка
    balance_method: str = "none"  # Отключаем SMOTE, полагаемся на scale_pos_weight
    smote_k_neighbors: int = 5
    
    # Оптимизация
    optuna_trials: int = 50  # Уменьшаем для тест режима, но достаточно для поиска
    optuna_cv_folds: int = 3
    
    # Ансамбль
    ensemble_size: int = 3  # Меньше моделей, но качественнее
    ensemble_method: str = "weighted"  # weighted, voting, stacking
    
    # Отбор признаков
    feature_selection_method: str = "importance"  # importance, hierarchical, rfe, mutual_info, chi2, combined
    
    # Пороги
    classification_threshold: float = 0.7  # Снижаем порог для более реалистичной задачи
    multiclass_thresholds: List[float] = field(default_factory=lambda: [-0.7, 0.0, 0.7, 1.4])
    
    # Оптимизация порогов
    optimize_threshold: bool = True  # Автоматически находить оптимальный порог
    threshold_metric: str = "f1"  # F1-score лучше для несбалансированных классов
    
    # Логирование
    log_dir: str = "logs"
    save_plots: bool = True
    save_models: bool = True
    verbose: int = 1


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
                setattr(config.database, key, value)
                
        if 'model' in data:
            for key, value in data['model'].items():
                setattr(config.model, key, value)
                
        if 'training' in data:
            for key, value in data['training'].items():
                setattr(config.training, key, value)
                
        return config
    
    def validate(self) -> bool:
        """Валидация конфигурации"""
        # Проверка режима задачи
        valid_tasks = ["classification_binary", "classification_multi", "regression"]
        if self.training.task_type not in valid_tasks:
            raise ValueError(f"Неверный task_type: {self.training.task_type}")
        
        # Проверка методов балансировки
        valid_balance = ["none", "smote", "adasyn", "class_weight"]
        if self.training.balance_method not in valid_balance:
            raise ValueError(f"Неверный balance_method: {self.training.balance_method}")
        
        # Проверка порогов
        if self.training.classification_threshold < 0 or self.training.classification_threshold > 100:
            raise ValueError("classification_threshold должен быть от 0 до 100")
        
        return True
    
    def get_log_dir(self) -> Path:
        """Получить директорию для логов"""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(self.training.log_dir) / f"xgboost_v3_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def save(self, path: str):
        """Сохранение конфигурации в файл"""
        data = {
            'database': self.database.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def __str__(self) -> str:
        """Красивый вывод конфигурации"""
        return f"""
XGBoost v3.0 Configuration
==========================
Task: {self.training.task_type}
Test Mode: {self.training.test_mode}
Balance Method: {self.training.balance_method}
Ensemble Size: {self.training.ensemble_size}
Classification Threshold: {self.training.classification_threshold}%

Model Parameters:
- Objective: {self.model.objective}
- Max Depth: {self.model.max_depth}
- Learning Rate: {self.model.learning_rate}
- Tree Method: {self.model.tree_method}

Database: {self.database.database} @ {self.database.host}:{self.database.port}
"""