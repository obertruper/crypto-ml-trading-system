"""
Optuna оптимизатор для подбора гиперпараметров XGBoost
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
import logging
from typing import Dict, Tuple, Optional, Callable

from config import Config
from config.constants import OPTUNA_PARAMS

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Класс для оптимизации гиперпараметров с помощью Optuna"""
    
    def __init__(self, config: Config):
        self.config = config
        self.best_params = None
        self.best_score = None
        self.study = None
        
    def optimize(self,
                X_train: pd.DataFrame,
                y_train: pd.Series,
                n_trials: Optional[int] = None,
                direction: Optional[str] = None,
                model_type: str = "buy") -> Dict:
        """
        Оптимизация гиперпараметров
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие метки
            n_trials: Количество попыток оптимизации
            direction: Направление оптимизации ("minimize" или "maximize")
            model_type: Тип модели (buy/sell)
            
        Returns:
            Лучшие найденные параметры
        """
        if n_trials is None:
            n_trials = self.config.training.optuna_trials
            
        if direction is None:
            direction = "minimize" if self.config.training.task_type == "regression" else "maximize"
            
        logger.info(f"\n🔧 Запуск Optuna оптимизации гиперпараметров...")
        logger.info(f"🔍 Начало Optuna оптимизации для {model_type}...")
        
        # Определяем оптимальные настройки для сервера
        import multiprocessing
        import time
        cpu_count = multiprocessing.cpu_count()
        
        # Создаем study с поддержкой параллелизма для мощных серверов
        if cpu_count > 64:
            logger.info(f"🚀 Мощный сервер ({cpu_count} CPU) - оптимизация без межпроцессной синхронизации")
            # Не используем storage для избежания блокировок
            # XGBoost сам эффективно использует все ядра
            
            self.study = optuna.create_study(
                direction=direction,
                sampler=TPESampler(seed=OPTUNA_PARAMS['random_state']),
                pruner=MedianPruner(n_warmup_steps=OPTUNA_PARAMS['pruner_warmup_steps'])
            )
            
            # Отключаем параллелизм Optuna, так как XGBoost сам параллелится
            logger.info(f"   XGBoost будет использовать все {cpu_count} ядер")
            
            # Оптимизация без параллелизма Optuna
            self.study.optimize(
                lambda trial: self._objective(trial, X_train, y_train),
                n_trials=n_trials,
                n_jobs=1,  # Однопоточная оптимизация, XGBoost сам параллелится
                show_progress_bar=True
            )
        else:
            # Обычная оптимизация для небольших систем
            self.study = optuna.create_study(
                direction=direction,
                sampler=TPESampler(seed=OPTUNA_PARAMS['random_state']),
                pruner=MedianPruner(n_warmup_steps=OPTUNA_PARAMS['pruner_warmup_steps'])
            )
            
            # Оптимизация
            self.study.optimize(
                lambda trial: self._objective(trial, X_train, y_train),
                n_trials=n_trials,
                show_progress_bar=True
            )
        
        # Сохраняем лучшие параметры
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"\n✅ Оптимизация завершена!")
        logger.info(f"🏆 Лучший score: {self.best_score:.5f}")
        logger.info(f"🎯 Лучшие параметры:")
        for param, value in self.best_params.items():
            logger.info(f"   {param}: {value}")
            
        return self.best_params
        
    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Целевая функция для оптимизации"""
        
        # Предлагаем параметры
        params = {
            'objective': self.config.model.objective,
            'eval_metric': self.config.model.eval_metric,
            'max_depth': trial.suggest_int('max_depth', 
                                         OPTUNA_PARAMS['max_depth']['min'], 
                                         OPTUNA_PARAMS['max_depth']['max']),
            'learning_rate': trial.suggest_float('learning_rate', 
                                               OPTUNA_PARAMS['learning_rate']['min'], 
                                               OPTUNA_PARAMS['learning_rate']['max'], 
                                               log=OPTUNA_PARAMS['learning_rate']['log']),
            'subsample': trial.suggest_float('subsample', 
                                           OPTUNA_PARAMS['subsample']['min'], 
                                           OPTUNA_PARAMS['subsample']['max']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 
                                                   OPTUNA_PARAMS['colsample_bytree']['min'], 
                                                   OPTUNA_PARAMS['colsample_bytree']['max']),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 
                                                    OPTUNA_PARAMS['colsample_bylevel']['min'], 
                                                    OPTUNA_PARAMS['colsample_bylevel']['max']),
            'min_child_weight': trial.suggest_int('min_child_weight', 
                                                OPTUNA_PARAMS['min_child_weight']['min'], 
                                                OPTUNA_PARAMS['min_child_weight']['max']),
            'gamma': trial.suggest_float('gamma', 
                                       OPTUNA_PARAMS['gamma']['min'], 
                                       OPTUNA_PARAMS['gamma']['max']),
            'reg_alpha': trial.suggest_float('reg_alpha', 
                                           OPTUNA_PARAMS['reg_alpha']['min'], 
                                           OPTUNA_PARAMS['reg_alpha']['max']),
            'reg_lambda': trial.suggest_float('reg_lambda', 
                                            OPTUNA_PARAMS['reg_lambda']['min'], 
                                            OPTUNA_PARAMS['reg_lambda']['max']),
            'n_jobs': -1,
            'random_state': OPTUNA_PARAMS['random_state'],
            'verbosity': 0
        }
        
        # Добавляем scale_pos_weight для классификации
        if self.config.training.task_type == "classification_binary":
            n_positive = (y == 1).sum()
            n_negative = (y == 0).sum()
            if n_positive > 0:
                scale_pos_weight = n_negative / n_positive
                # Позволяем Optuna немного варьировать этот параметр
                params['scale_pos_weight'] = trial.suggest_float(
                    'scale_pos_weight', 
                    scale_pos_weight * OPTUNA_PARAMS['scale_pos_weight_factor']['min'], 
                    scale_pos_weight * OPTUNA_PARAMS['scale_pos_weight_factor']['max']
                )
                
        # GPU параметры
        if self._check_gpu_available():
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'
        else:
            params['tree_method'] = 'hist'
            
        # Кросс-валидация
        scores = self._cross_validate(X, y, params)
        
        # Возвращаем среднее значение
        return np.mean(scores)
        
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, params: Dict) -> np.ndarray:
        """Кросс-валидация для оценки параметров"""
        n_folds = self.config.training.optuna_cv_folds
        
        # Выбираем тип разбиения
        if self.config.training.task_type == "regression":
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=OPTUNA_PARAMS['random_state'])
            splits = kf.split(X)
        else:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=OPTUNA_PARAMS['random_state'])
            splits = kf.split(X, y)
            
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            
            # Создаем DMatrix
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
            
            # Обучаем модель
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=OPTUNA_PARAMS['cv_n_estimators'],
                evals=[(dval, 'val')],
                early_stopping_rounds=OPTUNA_PARAMS['cv_early_stopping_rounds'],
                verbose_eval=False
            )
            
            # Оцениваем
            predictions = model.predict(dval)
            
            if self.config.training.task_type == "regression":
                score = mean_absolute_error(y_fold_val, predictions)
            else:
                # Для классификации используем ROC-AUC
                try:
                    score = roc_auc_score(y_fold_val, predictions)
                except:
                    score = 0.5
                    
            scores.append(score)
            
        return np.array(scores)
        
    def _check_gpu_available(self) -> bool:
        """Проверка доступности GPU"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except:
            return False
            
    def get_optimization_history(self) -> pd.DataFrame:
        """Получить историю оптимизации как DataFrame"""
        if self.study is None:
            return pd.DataFrame()
            
        history = []
        
        for trial in self.study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                **trial.params
            }
            history.append(trial_data)
            
        df = pd.DataFrame(history)
        return df.sort_values('value', ascending=(self.study.direction.name == 'MINIMIZE'))
        
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Визуализация истории оптимизации"""
        if self.study is None:
            return
            
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Optuna Optimization History', fontsize=16)
        
        # 1. История значений
        ax = axes[0, 0]
        trials_df = self.get_optimization_history()
        completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
        
        ax.plot(completed_trials['number'], completed_trials['value'], 'b-', alpha=0.5)
        ax.scatter(completed_trials['number'], completed_trials['value'], c='blue', alpha=0.5)
        
        # Лучшее значение
        best_idx = completed_trials['value'].idxmin() if self.study.direction.name == 'MINIMIZE' else completed_trials['value'].idxmax()
        best_trial = completed_trials.loc[best_idx]
        ax.scatter(best_trial['number'], best_trial['value'], c='red', s=100, label=f'Best: {best_trial["value"]:.5f}')
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Параллельные координаты для важных параметров
        ax = axes[0, 1]
        important_params = ['max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
        
        if len(completed_trials) > 0:
            # Нормализуем параметры для визуализации
            param_data = completed_trials[important_params].values
            param_data_norm = (param_data - param_data.min(axis=0)) / (param_data.max(axis=0) - param_data.min(axis=0) + 1e-8)
            
            for i in range(len(param_data_norm)):
                color = plt.cm.viridis(completed_trials.iloc[i]['value'] / completed_trials['value'].max())
                ax.plot(range(len(important_params)), param_data_norm[i], alpha=0.3, color=color)
                
            ax.set_xticks(range(len(important_params)))
            ax.set_xticklabels(important_params, rotation=45)
            ax.set_ylabel('Normalized Value')
            ax.set_title('Parameter Relationships')
            
        # 3. Важность параметров
        ax = axes[1, 0]
        importances = optuna.importance.get_param_importances(self.study)
        
        if importances:
            params = list(importances.keys())
            values = list(importances.values())
            
            y_pos = np.arange(len(params))
            ax.barh(y_pos, values)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.set_xlabel('Importance')
            ax.set_title('Parameter Importance')
            
        # 4. Распределение лучших параметров
        ax = axes[1, 1]
        if len(completed_trials) > 10:
            # Берем топ 10% лучших
            n_best = max(1, len(completed_trials) // 10)
            best_trials = completed_trials.nsmallest(n_best, 'value') if self.study.direction.name == 'MINIMIZE' else completed_trials.nlargest(n_best, 'value')
            
            text = "Best Parameters Distribution:\n\n"
            for param in ['max_depth', 'learning_rate', 'subsample']:
                if param in best_trials.columns:
                    mean_val = best_trials[param].mean()
                    std_val = best_trials[param].std()
                    text += f"{param}: {mean_val:.3f} ± {std_val:.3f}\n"
                    
            ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=12, verticalalignment='center')
            ax.axis('off')
            ax.set_title('Best Trials Statistics')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 График оптимизации сохранен: {save_path}")
            
        plt.close()
        
    def suggest_next_params(self) -> Dict:
        """Предложить следующие параметры для ручной настройки"""
        if self.best_params is None:
            return {}
            
        # Создаем вариации лучших параметров
        variations = []
        
        for param, value in self.best_params.items():
            if isinstance(value, int):
                variations.append({
                    param: value + np.random.choice([-1, 0, 1])
                })
            elif isinstance(value, float):
                variations.append({
                    param: value * np.random.uniform(0.8, 1.2)
                })
                
        return variations