"""
Кэширование для Optuna оптимизации
"""

import os
import hashlib
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OptunaCache:
    """Кэш для результатов Optuna оптимизации"""
    
    def __init__(self, cache_dir: str = "cache/optuna"):
        """
        Args:
            cache_dir: Директория для кэша
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, 
                      dataset_hash: str,
                      model_type: str,
                      task_type: str,
                      n_features: int,
                      n_samples: int) -> str:
        """Генерация уникального ключа для кэша"""
        key_data = {
            'dataset': dataset_hash,
            'model': model_type,
            'task': task_type,
            'features': n_features,
            'samples': n_samples
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_dataset_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Получить хэш датасета для проверки изменений"""
        # Берем сэмпл данных для хэширования (первые и последние строки)
        sample_size = min(1000, len(X))
        X_sample = pd.concat([X.head(sample_size//2), X.tail(sample_size//2)])
        y_sample = pd.concat([y.head(sample_size//2), y.tail(sample_size//2)])
        
        # Создаем строку для хэширования
        data_str = f"{X_sample.columns.tolist()}{X_sample.values.tobytes()}{y_sample.values.tobytes()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def check_cache(self, 
                   X: pd.DataFrame,
                   y: pd.Series,
                   model_type: str,
                   task_type: str) -> Optional[Dict[str, Any]]:
        """
        Проверить наличие кэшированных результатов
        
        Args:
            X: Признаки
            y: Целевая переменная
            model_type: Тип модели (buy/sell)
            task_type: Тип задачи
            
        Returns:
            Кэшированные параметры или None
        """
        try:
            # Получаем хэш данных
            dataset_hash = self._get_dataset_hash(X, y)
            
            # Генерируем ключ кэша
            cache_key = self._get_cache_key(
                dataset_hash=dataset_hash,
                model_type=model_type,
                task_type=task_type,
                n_features=X.shape[1],
                n_samples=X.shape[0]
            )
            
            # Путь к файлу кэша
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            meta_file = self.cache_dir / f"{cache_key}_meta.json"
            
            if cache_file.exists() and meta_file.exists():
                # Загружаем метаданные
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                
                # Проверяем актуальность
                if (meta.get('n_features') == X.shape[1] and 
                    abs(meta.get('n_samples', 0) - X.shape[0]) < X.shape[0] * 0.1):  # 10% допуск
                    
                    # Загружаем параметры
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    logger.info(f"✅ Найден кэш оптимизации для {model_type}")
                    logger.info(f"   Лучший score: {cached_data['best_score']:.4f}")
                    logger.info(f"   Количество trials: {cached_data['n_trials']}")
                    
                    return cached_data
                else:
                    logger.info(f"⚠️ Кэш устарел для {model_type} (изменился размер данных)")
                    
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при проверке кэша: {e}")
        
        return None
    
    def save_cache(self,
                  X: pd.DataFrame,
                  y: pd.Series,
                  model_type: str,
                  task_type: str,
                  best_params: Dict[str, Any],
                  best_score: float,
                  n_trials: int,
                  study: Optional[Any] = None) -> None:
        """
        Сохранить результаты в кэш
        
        Args:
            X: Признаки
            y: Целевая переменная
            model_type: Тип модели
            task_type: Тип задачи
            best_params: Лучшие параметры
            best_score: Лучший score
            n_trials: Количество попыток
            study: Объект Optuna Study (опционально)
        """
        try:
            # Получаем хэш данных
            dataset_hash = self._get_dataset_hash(X, y)
            
            # Генерируем ключ кэша
            cache_key = self._get_cache_key(
                dataset_hash=dataset_hash,
                model_type=model_type,
                task_type=task_type,
                n_features=X.shape[1],
                n_samples=X.shape[0]
            )
            
            # Данные для сохранения
            cache_data = {
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': n_trials,
                'model_type': model_type,
                'task_type': task_type
            }
            
            # Метаданные
            meta_data = {
                'dataset_hash': dataset_hash,
                'n_features': X.shape[1],
                'n_samples': X.shape[0],
                'feature_names': X.columns.tolist(),
                'model_type': model_type,
                'task_type': task_type,
                'best_score': best_score,
                'n_trials': n_trials,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Сохраняем
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            meta_file = self.cache_dir / f"{cache_key}_meta.json"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            # Опционально сохраняем study
            if study is not None:
                study_file = self.cache_dir / f"{cache_key}_study.pkl"
                with open(study_file, 'wb') as f:
                    pickle.dump(study, f)
            
            logger.info(f"✅ Результаты оптимизации сохранены в кэш для {model_type}")
            
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить кэш: {e}")
    
    def get_cache_info(self) -> pd.DataFrame:
        """Получить информацию о всех кэшированных результатах"""
        cache_info = []
        
        for meta_file in self.cache_dir.glob("*_meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                
                cache_info.append({
                    'model_type': meta.get('model_type'),
                    'task_type': meta.get('task_type'),
                    'n_features': meta.get('n_features'),
                    'n_samples': meta.get('n_samples'),
                    'best_score': meta.get('best_score'),
                    'n_trials': meta.get('n_trials'),
                    'timestamp': meta.get('timestamp')
                })
            except:
                continue
        
        if cache_info:
            return pd.DataFrame(cache_info).sort_values('timestamp', ascending=False)
        else:
            return pd.DataFrame()
    
    def clear_old_cache(self, days: int = 7) -> int:
        """
        Удалить старый кэш
        
        Args:
            days: Удалить кэш старше указанного количества дней
            
        Returns:
            Количество удаленных файлов
        """
        count = 0
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        
        for meta_file in self.cache_dir.glob("*_meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                
                timestamp = pd.Timestamp(meta.get('timestamp', '2000-01-01'))
                
                if timestamp < cutoff_date:
                    # Удаляем все связанные файлы
                    base_name = meta_file.stem.replace('_meta', '')
                    for pattern in [f"{base_name}.pkl", f"{base_name}_meta.json", f"{base_name}_study.pkl"]:
                        file_path = self.cache_dir / pattern
                        if file_path.exists():
                            file_path.unlink()
                            count += 1
                            
            except:
                continue
        
        if count > 0:
            logger.info(f"🗑️ Удалено {count} старых файлов кэша")
        
        return count