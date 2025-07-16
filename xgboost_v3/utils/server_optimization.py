"""
Автоматическая оптимизация для мощных серверов
"""

import os
import multiprocessing
import psutil
import logging

logger = logging.getLogger(__name__)


class ServerOptimizer:
    """Класс для автоматической оптимизации под характеристики сервера"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        self.server_type = os.environ.get('SERVER_TYPE', 'auto')
        
        # Автоматическое определение типа сервера
        if self.server_type == 'auto':
            if self.cpu_count >= 64 and self.ram_gb >= 100:
                self.server_type = 'powerful'
            else:
                self.server_type = 'normal'
                
        logger.info(f"🖥️ Сервер: {self.cpu_count} CPU, {self.ram_gb:.0f}GB RAM")
        logger.info(f"📋 Тип: {self.server_type}")
    
    def get_data_loader_workers(self, default=10):
        """Оптимальное количество воркеров для загрузки данных"""
        if self.server_type == 'powerful':
            workers = min(50, self.cpu_count // 2)
            logger.info(f"🚀 DataLoader: {workers} воркеров (мощный сервер)")
            return workers
        return min(default, self.cpu_count)
    
    def get_batch_size(self, default=10):
        """Оптимальный размер батча"""
        if self.server_type == 'powerful' and self.ram_gb > 100:
            batch_size = 50
            logger.info(f"🚀 Batch size: {batch_size} (много RAM)")
            return batch_size
        return default
    
    def get_optuna_jobs(self, default=1):
        """Количество параллельных Optuna воркеров"""
        if self.server_type == 'powerful':
            n_jobs = min(32, self.cpu_count // 4)
            logger.info(f"🚀 Optuna: {n_jobs} параллельных воркеров")
            return n_jobs
        return default
    
    def get_adasyn_params(self, n_samples):
        """Параметры ADASYN для балансировки"""
        if self.server_type == 'powerful':
            n_neighbors = min(15, n_samples // 1000)
            n_jobs = min(32, self.cpu_count // 4)
            logger.info(f"🚀 ADASYN: {n_neighbors} соседей, {n_jobs} воркеров")
            return {'n_neighbors': n_neighbors, 'n_jobs': n_jobs}
        
        return {'n_neighbors': min(5, n_samples // 1000), 'n_jobs': 1}
    
    def get_xgboost_params(self):
        """Дополнительные параметры для XGBoost"""
        params = {}
        
        if self.server_type == 'powerful':
            # Больше потоков для построения деревьев
            params['nthread'] = -1
            
            # Больше памяти для гистограмм
            if self.ram_gb > 200:
                params['max_bin'] = 512  # Больше бинов для точности
            else:
                params['max_bin'] = 256
                
            logger.info(f"🚀 XGBoost: max_bin={params.get('max_bin', 256)}")
            
        return params
    
    def should_use_parallel_optuna(self):
        """Использовать ли параллельную Optuna оптимизацию"""
        return self.server_type == 'powerful'
    
    def get_feature_engineering_workers(self):
        """Воркеры для параллельной инженерии признаков"""
        if self.server_type == 'powerful':
            return self.cpu_count // 2
        return self.cpu_count
    
    def log_optimization_summary(self):
        """Вывести сводку оптимизаций"""
        logger.info("\n🔧 Применённые оптимизации:")
        logger.info(f"   Тип сервера: {self.server_type}")
        logger.info(f"   CPU: {self.cpu_count}")
        logger.info(f"   RAM: {self.ram_gb:.0f} GB")
        
        if self.server_type == 'powerful':
            logger.info("   ✅ Включена параллельная Optuna")
            logger.info("   ✅ Увеличены батчи данных")
            logger.info("   ✅ Больше воркеров для загрузки")
            logger.info("   ✅ Оптимизированы параметры ADASYN")
        else:
            logger.info("   📋 Стандартные настройки")


# Глобальный экземпляр оптимизатора
_optimizer = None

def get_optimizer():
    """Получить глобальный экземпляр оптимизатора"""
    global _optimizer
    if _optimizer is None:
        _optimizer = ServerOptimizer()
    return _optimizer