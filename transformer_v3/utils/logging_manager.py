"""
Менеджер логирования для Transformer v3
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from config import LOGGING_CONFIG


class LoggingManager:
    """Централизованное управление логированием"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Настройка логирования"""
        # Основной логгер
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
        
        # Удаляем существующие хендлеры
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Форматтер
        formatter = logging.Formatter(
            LOGGING_CONFIG['format'],
            datefmt=LOGGING_CONFIG['date_format']
        )
        
        # Консольный хендлер
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Файловый хендлер
        log_file = self.log_dir / 'training.log'
        file_handler = logging.FileHandler(
            log_file, 
            encoding=LOGGING_CONFIG['encoding']
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Отдельный файл для ошибок
        error_file = self.log_dir / 'errors.log'
        error_handler = logging.FileHandler(
            error_file,
            encoding=LOGGING_CONFIG['encoding']
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # Логгер для TensorFlow
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.setLevel(logging.WARNING)
        
        logger.info(f"📝 Логирование настроено. Лог-файл: {log_file}")
        
    def get_logger(self, name: str) -> logging.Logger:
        """Получить логгер с заданным именем"""
        return logging.getLogger(name)