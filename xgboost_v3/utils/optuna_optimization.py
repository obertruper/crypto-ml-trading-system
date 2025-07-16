"""
Оптимизации для ускорения Optuna
"""

import os
import logging
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from typing import Dict, Any, Optional, Callable
import joblib
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Оптимизированный класс для работы с Optuna"""
    
    def __init__(self, 
                 study_name: str,
                 storage: Optional[str] = None,
                 direction: str = "maximize",
                 load_if_exists: bool = True):
        """
        Args:
            study_name: Имя исследования
            storage: URL базы данных для хранения (None = в памяти)
            direction: Направление оптимизации
            load_if_exists: Загрузить существующее исследование
        """
        self.study_name = study_name
        self.direction = direction
        
        # Определяем оптимальные параметры на основе ресурсов
        cpu_count = psutil.cpu_count(logical=True)
        self.is_powerful = cpu_count >= 64
        
        # Создаем или загружаем исследование
        self.study = self._create_study(storage, load_if_exists)
        
    def _create_study(self, storage: Optional[str], load_if_exists: bool) -> optuna.Study:
        """Создание оптимизированного исследования"""
        
        # Выбираем оптимальный sampler
        if self.is_powerful:
            # Для мощных серверов - более агрессивный sampler
            sampler = TPESampler(
                n_startup_trials=10,  # Меньше случайных попыток
                n_ei_candidates=48,   # Больше кандидатов для EI
                multivariate=True,    # Учитывать зависимости между параметрами
                seed=42
            )
            # Более агрессивный pruner
            pruner = HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3
            )
        else:
            # Для обычных серверов - стандартные настройки
            sampler = TPESampler(
                n_startup_trials=20,
                n_ei_candidates=24,
                multivariate=True,
                seed=42
            )
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        
        # Создаем исследование
        try:
            if storage:
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=storage,
                    sampler=sampler,
                    pruner=pruner,
                    direction=self.direction,
                    load_if_exists=load_if_exists
                )
                logger.info(f"✅ Создано исследование {self.study_name} с БД: {storage}")
            else:
                study = optuna.create_study(
                    study_name=self.study_name,
                    sampler=sampler,
                    pruner=pruner,
                    direction=self.direction
                )
                logger.info(f"✅ Создано исследование {self.study_name} в памяти")
                
            return study
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания исследования: {e}")
            raise
    
    def optimize_parallel(self,
                         objective: Callable,
                         n_trials: int,
                         n_jobs: Optional[int] = None,
                         timeout: Optional[float] = None,
                         gc_after_trial: bool = True,
                         show_progress_bar: bool = True) -> optuna.Study:
        """
        Параллельная оптимизация с улучшениями
        
        Args:
            objective: Целевая функция
            n_trials: Количество попыток
            n_jobs: Количество параллельных процессов
            timeout: Таймаут в секундах
            gc_after_trial: Запускать сборку мусора после каждой попытки
            show_progress_bar: Показывать прогресс
            
        Returns:
            Оптимизированное исследование
        """
        # Определяем оптимальное количество воркеров
        if n_jobs is None:
            cpu_count = psutil.cpu_count(logical=True)
            if self.is_powerful:
                # Для мощных серверов - больше воркеров
                n_jobs = min(32, cpu_count // 4)
            else:
                # Для обычных - консервативно
                n_jobs = min(4, cpu_count // 2)
        
        logger.info(f"🚀 Запуск параллельной оптимизации: {n_trials} попыток, {n_jobs} воркеров")
        
        try:
            # Оптимизация с дополнительными параметрами
            self.study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                timeout=timeout,
                gc_after_trial=gc_after_trial,
                show_progress_bar=show_progress_bar,
                catch=(Exception,)  # Ловим все исключения чтобы не прерывать оптимизацию
            )
            
            # Логируем результаты
            best_trial = self.study.best_trial
            logger.info(f"✅ Оптимизация завершена!")
            logger.info(f"   Лучшее значение: {best_trial.value:.4f}")
            logger.info(f"   Лучшие параметры: {best_trial.params}")
            
            return self.study
            
        except Exception as e:
            logger.error(f"❌ Ошибка оптимизации: {e}")
            raise
    
    def get_best_params(self) -> Dict[str, Any]:
        """Получить лучшие параметры"""
        return self.study.best_params
    
    def save_study(self, filepath: str) -> None:
        """Сохранить исследование в файл"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.study, filepath)
        logger.info(f"✅ Исследование сохранено: {filepath}")
    
    @staticmethod
    def load_study(filepath: str) -> optuna.Study:
        """Загрузить исследование из файла"""
        study = joblib.load(filepath)
        logger.info(f"✅ Исследование загружено: {filepath}")
        return study
    
    def create_visualization_cache(self, cache_dir: str = "optuna_cache") -> None:
        """
        Создать кэш визуализаций для быстрого доступа
        
        Args:
            cache_dir: Директория для кэша
        """
        os.makedirs(cache_dir, exist_ok=True)
        
        # Сохраняем важные визуализации
        visualizations = {
            'optimization_history': optuna.visualization.plot_optimization_history,
            'param_importances': optuna.visualization.plot_param_importances,
            'parallel_coordinate': optuna.visualization.plot_parallel_coordinate,
            'slice': optuna.visualization.plot_slice
        }
        
        for name, plot_func in visualizations.items():
            try:
                fig = plot_func(self.study)
                fig.write_html(os.path.join(cache_dir, f"{name}.html"))
                logger.info(f"✅ Сохранена визуализация: {name}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось создать визуализацию {name}: {e}")


def create_fast_objective(model_class, X_train, y_train, X_val, y_val,
                         use_gpu: bool = True,
                         early_stopping_rounds: int = 50) -> Callable:
    """
    Создать оптимизированную целевую функцию для Optuna
    
    Args:
        model_class: Класс модели (например, XGBClassifier)
        X_train, y_train: Обучающие данные
        X_val, y_val: Валидационные данные
        use_gpu: Использовать GPU
        early_stopping_rounds: Раунды для ранней остановки
        
    Returns:
        Целевая функция для Optuna
    """
    def objective(trial):
        # Предлагаем гиперпараметры
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        }
        
        # Добавляем GPU параметры если доступно
        if use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'
        else:
            params['tree_method'] = 'hist'
        
        # Создаем модель
        model = model_class(**params, random_state=42, n_jobs=-1)
        
        # Обучаем с ранней остановкой
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        # Предсказываем и оцениваем
        y_pred = model.predict_proba(X_val)[:, 1]
        
        # Можно использовать разные метрики
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(y_val, y_pred)
        
        return score
    
    return objective