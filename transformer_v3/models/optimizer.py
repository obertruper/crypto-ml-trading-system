"""
Optuna оптимизация гиперпараметров для TFT
Адаптировано из XGBoost v3.0
"""

import optuna
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from typing import Dict, Tuple, Callable, Optional
import tempfile
import shutil
from pathlib import Path

from config import Config
from .tft_model import create_tft_model

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Оптимизация гиперпараметров TFT с помощью Optuna
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.study = None
        self.best_params = None
        self.trial_results = []
        
    def optimize(self,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray, 
                y_val: np.ndarray,
                n_trials: int = 50,
                model_type: str = "buy") -> Dict:
        """
        Оптимизация гиперпараметров
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            X_val: Валидационные данные
            y_val: Валидационные метки
            n_trials: Количество попыток оптимизации
            model_type: Тип модели (buy/sell)
            
        Returns:
            Лучшие найденные параметры
        """
        logger.info(f"🔍 Запуск Optuna оптимизации для {model_type} модели...")
        logger.info(f"   Количество попыток: {n_trials}")
        
        # Сохраняем данные в атрибуты для objective функции
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Создаем study
        study_name = f"tft_optimization_{model_type}"
        self.study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Запускаем оптимизацию
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=None,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        
        logger.info("✅ Оптимизация завершена")
        logger.info(f"   Лучший результат: {self.study.best_value:.4f}")
        logger.info(f"   Лучшие параметры: {self.best_params}")
        
        return self.best_params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective функция для Optuna
        
        Args:
            trial: Optuna trial объект
            
        Returns:
            Значение метрики для минимизации
        """
        # Предлагаем гиперпараметры
        params = self._suggest_parameters(trial)
        
        # Обновляем конфигурацию
        config = self._update_config_with_params(params)
        
        try:
            # Создаем модель с новыми параметрами
            model = create_tft_model(config, self.input_shape)
            
            # Callbacks для раннего завершения
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=max(5, params.get('early_stopping_patience', 10) // 2),
                    restore_best_weights=True,
                    verbose=0
                )
            ]
            
            # Обучаем модель
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=min(params.get('epochs', 50), 30),  # Ограничиваем эпохи для оптимизации
                batch_size=params.get('batch_size', 32),
                callbacks=callbacks,
                verbose=0
            )
            
            # Возвращаем лучшую val_loss
            best_val_loss = min(history.history['val_loss'])
            
            # Сохраняем результат
            self.trial_results.append({
                'trial_number': trial.number,
                'params': params,
                'val_loss': best_val_loss,
                'epochs_trained': len(history.history['val_loss'])
            })
            
            # Освобождаем память
            del model
            tf.keras.backend.clear_session()
            
            return best_val_loss
            
        except Exception as e:
            logger.warning(f"⚠️ Trial {trial.number} failed: {e}")
            # Возвращаем большое значение для неудачных попыток
            return float('inf')
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict:
        """
        Предложение гиперпараметров для оптимизации
        """
        params = {}
        
        # Архитектура модели
        params['hidden_size'] = trial.suggest_categorical('hidden_size', [64, 96, 128, 160, 192, 224, 256])
        params['lstm_layers'] = trial.suggest_int('lstm_layers', 1, 3)
        params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.05, 0.3, step=0.05)
        params['state_size'] = trial.suggest_categorical('state_size', [32, 48, 64, 80, 96])
        
        # Обучение
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', [16, 24, 32, 48, 64])
        params['epochs'] = trial.suggest_int('epochs', 30, 100)
        
        # Регуляризация
        params['l2_regularization'] = trial.suggest_float('l2_regularization', 0.001, 0.1, log=True)
        params['gradient_clip_val'] = trial.suggest_float('gradient_clip_val', 0.5, 2.0, step=0.1)
        
        # Early stopping
        params['early_stopping_patience'] = trial.suggest_int('early_stopping_patience', 8, 20)
        params['reduce_lr_patience'] = trial.suggest_int('reduce_lr_patience', 3, 8)
        params['reduce_lr_factor'] = trial.suggest_float('reduce_lr_factor', 0.3, 0.7, step=0.1)
        
        # Проверяем совместимость параметров
        if params['hidden_size'] % params['num_heads'] != 0:
            # Корректируем hidden_size чтобы был кратен num_heads
            params['hidden_size'] = (params['hidden_size'] // params['num_heads']) * params['num_heads']
        
        return params
    
    def _update_config_with_params(self, params: Dict) -> Config:
        """
        Обновление конфигурации с новыми параметрами
        """
        config = Config()
        
        # Копируем основные настройки
        config.training = self.config.training
        config.database = self.config.database
        
        # Обновляем параметры модели
        for key, value in params.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
        
        return config
    
    def get_optimization_history(self) -> Dict:
        """
        Получение истории оптимизации
        """
        if self.study is None:
            return {}
        
        trials_df = self.study.trials_dataframe()
        
        return {
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'trials_df': trials_df,
            'trial_results': self.trial_results
        }
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Создание графиков истории оптимизации
        """
        if self.study is None:
            logger.warning("⚠️ Оптимизация не запущена, нечего отображать")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Optuna Optimization History', fontsize=16)
            
            # 1. Optimization history
            trials = self.study.trials
            values = [t.value for t in trials if t.value is not None]
            trial_numbers = [t.number for t in trials if t.value is not None]
            
            axes[0, 0].plot(trial_numbers, values, 'b-', alpha=0.7)
            axes[0, 0].axhline(y=self.study.best_value, color='r', linestyle='--', 
                              label=f'Best: {self.study.best_value:.4f}')
            axes[0, 0].set_title('Optimization History')
            axes[0, 0].set_xlabel('Trial')
            axes[0, 0].set_ylabel('Objective Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Parameter importance (если доступно)
            try:
                importance = optuna.importance.get_param_importances(self.study)
                if importance:
                    params = list(importance.keys())[:10]  # Топ-10
                    importances = [importance[p] for p in params]
                    
                    axes[0, 1].barh(params, importances)
                    axes[0, 1].set_title('Parameter Importance')
                    axes[0, 1].set_xlabel('Importance')
            except:
                axes[0, 1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # 3. Best trial parameters
            best_params = self.study.best_params
            if best_params:
                param_names = list(best_params.keys())[:8]  # Топ-8 для отображения
                param_values = [best_params[p] for p in param_names]
                
                axes[1, 0].barh(param_names, param_values)
                axes[1, 0].set_title('Best Parameters')
                axes[1, 0].set_xlabel('Value')
            
            # 4. Convergence
            if len(values) > 1:
                best_so_far = []
                current_best = float('inf')
                for val in values:
                    if val < current_best:
                        current_best = val
                    best_so_far.append(current_best)
                
                axes[1, 1].plot(trial_numbers, best_so_far, 'g-', linewidth=2)
                axes[1, 1].set_title('Convergence')
                axes[1, 1].set_xlabel('Trial')
                axes[1, 1].set_ylabel('Best Value So Far')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"   ✅ График оптимизации сохранен: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Не удалось создать график оптимизации: {e}")
    
    def suggest_best_config(self) -> Config:
        """
        Создание конфигурации с лучшими найденными параметрами
        """
        if self.best_params is None:
            logger.warning("⚠️ Оптимизация не проводилась, возвращается исходная конфигурация")
            return self.config
        
        return self._update_config_with_params(self.best_params)