"""
Управление логированием
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import colorlog


class LoggingManager:
    """Класс для настройки системы логирования"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self, level=logging.INFO):
        """Настройка логирования с цветным выводом"""
        
        # Создаем форматтер для файла
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Создаем цветной форматтер для консоли
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(levelname)s - %(message)s%(reset)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        # Файловый обработчик
        file_handler = logging.FileHandler(
            self.log_dir / 'training.log',
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        
        # Консольный обработчик
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        
        # Настраиваем корневой логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Удаляем существующие обработчики
        root_logger.handlers = []
        
        # Добавляем наши обработчики
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Отключаем логи от некоторых библиотек
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        # Логируем начало
        logger = logging.getLogger(__name__)
        logger.info(f"📝 Логирование настроено. Файл: {self.log_dir / 'training.log'}")