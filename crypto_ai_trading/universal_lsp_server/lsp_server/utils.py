"""
Утилиты для Universal LSP Server
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Set
import colorlog

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Настройка логирования"""
    
    # Преобразуем уровень в константу
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Формат для консоли (с цветами)
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Формат для файла (без цветов)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Удаляем существующие обработчики
    root_logger.handlers.clear()
    
    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Файловый обработчик (если указан)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    # Уменьшаем уровень логирования для некоторых библиотек
    logging.getLogger('pygls').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Найти корневую директорию проекта"""
    
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()
    
    # Маркеры корневой директории проекта
    root_markers = [
        '.git',
        '.hg',
        '.svn',
        'pyproject.toml',
        'setup.py',
        'setup.cfg',
        'requirements.txt',
        'Pipfile',
        'poetry.lock',
        'package.json',
        '.project',
        '.vscode',
        '.idea'
    ]
    
    current = start_path
    while current != current.parent:
        for marker in root_markers:
            if (current / marker).exists():
                return current
        current = current.parent
    
    # Если не нашли маркеры, возвращаем текущую директорию
    return start_path

def is_python_file(path: Path) -> bool:
    """Проверить, является ли файл Python файлом"""
    
    # Проверяем расширение
    if path.suffix in ['.py', '.pyi', '.pyx']:
        return True
    
    # Проверяем shebang для файлов без расширения
    if not path.suffix:
        try:
            with open(path, 'rb') as f:
                first_line = f.readline()
                if first_line.startswith(b'#!/') and b'python' in first_line:
                    return True
        except:
            pass
    
    return False

def get_file_hash(content: str) -> str:
    """Получить хеш содержимого файла"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

def get_relative_path(path: Path, base: Path) -> str:
    """Получить относительный путь"""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)

def format_file_size(size_bytes: int) -> str:
    """Форматировать размер файла"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def find_python_files(directory: Path, 
                     exclude_dirs: Set[str] = None,
                     exclude_patterns: List[str] = None,
                     follow_symlinks: bool = False) -> List[Path]:
    """Найти все Python файлы в директории"""
    
    if exclude_dirs is None:
        exclude_dirs = {
            '__pycache__', '.git', '.venv', 'venv', 
            'node_modules', '.tox', 'build', 'dist'
        }
    
    if exclude_patterns is None:
        exclude_patterns = ['*.pyc', '*.pyo', '*~', '.DS_Store']
    
    python_files = []
    
    for path in directory.rglob('*'):
        # Пропускаем директории
        if path.is_dir():
            continue
        
        # Проверяем исключения
        should_exclude = False
        
        # Проверяем директории
        for part in path.parts:
            if part in exclude_dirs:
                should_exclude = True
                break
        
        if should_exclude:
            continue
        
        # Проверяем паттерны
        for pattern in exclude_patterns:
            if path.match(pattern):
                should_exclude = True
                break
        
        if should_exclude:
            continue
        
        # Проверяем символические ссылки
        if not follow_symlinks and path.is_symlink():
            continue
        
        # Проверяем, что это Python файл
        if is_python_file(path):
            python_files.append(path)
    
    return python_files

def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Обрезать строку до максимальной длины"""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix

def sanitize_path(path: str) -> str:
    """Очистить путь от небезопасных символов"""
    # Удаляем потенциально опасные символы
    dangerous_chars = ['..', '~', '$', '`', '|', '&', ';', '>', '<', '\n', '\r']
    
    sanitized = path
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    return sanitized

def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Рекурсивное объединение словарей"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

class CircuitBreaker:
    """Простой circuit breaker для защиты от повторяющихся ошибок"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
    
    def call(self, func, *args, **kwargs):
        """Вызвать функцию через circuit breaker"""
        import time
        
        # Проверяем, нужно ли сбросить состояние
        if self.is_open and self.last_failure_time:
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.reset()
        
        # Если открыт, не выполняем
        if self.is_open:
            raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e
    
    def record_failure(self):
        """Записать неудачу"""
        import time
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
    
    def reset(self):
        """Сбросить состояние"""
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False