"""
Конфигурация Universal LSP Server
Поддерживает загрузку из файла, переменных окружения и аргументов командной строки
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла, если он есть
load_dotenv()

@dataclass
class ServerConfig:
    """Конфигурация сервера"""
    name: str = "universal-lsp-server"
    version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 2087
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_stdio: bool = False  # для работы через stdio вместо TCP

@dataclass
class IndexingConfig:
    """Конфигурация индексации"""
    extensions: List[str] = field(default_factory=lambda: [".py", ".pyi", ".pyx"])
    exclude_dirs: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", ".venv", "venv", "node_modules", 
        ".tox", "build", "dist", "*.egg-info", ".pytest_cache",
        ".mypy_cache", ".coverage", "htmlcov"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: ["*.pyc", "*.pyo", "*~", ".DS_Store"])
    max_file_size: int = 1048576  # 1MB
    parallel_indexing: bool = True
    max_workers: int = 4
    follow_symlinks: bool = False
    index_hidden_files: bool = False

@dataclass
class ContextConfig:
    """Конфигурация контекста"""
    max_tokens: int = 100000
    max_related_files: int = 10
    max_recent_edits: int = 20
    include_documentation: bool = True
    import_depth: int = 3
    context_format: str = "markdown"  # markdown, json, xml
    include_file_tree: bool = True
    include_git_info: bool = True

@dataclass
class FeaturesConfig:
    """Конфигурация возможностей LSP"""
    completion: bool = True
    hover: bool = True
    definition: bool = True
    references: bool = True
    symbols: bool = True
    diagnostics: bool = True
    formatting: bool = True
    code_actions: bool = True
    semantic_tokens: bool = False
    folding_range: bool = True
    selection_range: bool = True

@dataclass
class CacheConfig:
    """Конфигурация кеширования"""
    enabled: bool = True
    directory: str = ".lsp_cache"
    ttl: int = 3600  # 1 час
    max_size: int = 100  # MB
    compression: bool = True

@dataclass
class Config:
    """Полная конфигурация сервера"""
    server: ServerConfig = field(default_factory=ServerConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Дополнительные настройки
    project_root: Optional[str] = None
    workspace_folders: List[str] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Загрузка конфигурации из файла"""
        if not config_path.exists():
            logger.warning(f"Файл конфигурации не найден: {config_path}")
            return cls()
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            return cls._from_dict(data)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return cls()
    
    @classmethod
    def from_env(cls) -> "Config":
        """Загрузка конфигурации из переменных окружения"""
        config = cls()
        
        # Переменные окружения имеют приоритет
        env_mappings = {
            'LSP_HOST': ('server', 'host'),
            'LSP_PORT': ('server', 'port', int),
            'LSP_LOG_LEVEL': ('server', 'log_level'),
            'LSP_LOG_FILE': ('server', 'log_file'),
            'LSP_ENABLE_STDIO': ('server', 'enable_stdio', lambda x: x.lower() == 'true'),
            'LSP_PROJECT_ROOT': ('project_root',),
            'LSP_MAX_TOKENS': ('context', 'max_tokens', int),
            'LSP_CACHE_ENABLED': ('cache', 'enabled', lambda x: x.lower() == 'true'),
            'LSP_CACHE_DIR': ('cache', 'directory'),
        }
        
        for env_key, mapping in env_mappings.items():
            value = os.getenv(env_key)
            if value is not None:
                try:
                    # Преобразуем значение если нужно
                    if len(mapping) > 2:
                        value = mapping[2](value)
                    
                    # Устанавливаем значение
                    if len(mapping) == 1:
                        setattr(config, mapping[0], value)
                    else:
                        section = getattr(config, mapping[0])
                        setattr(section, mapping[1], value)
                except Exception as e:
                    logger.warning(f"Ошибка обработки {env_key}: {e}")
        
        return config
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Загрузка конфигурации с приоритетами:
        1. Переменные окружения
        2. Файл конфигурации
        3. Значения по умолчанию
        """
        # Начинаем с дефолтной конфигурации
        config = cls()
        
        # Загружаем из файла если указан
        if config_path and config_path.exists():
            config = cls.from_file(config_path)
        
        # Переопределяем из переменных окружения
        env_config = cls.from_env()
        config._merge(env_config)
        
        # Автоопределение project_root если не задан
        if not config.project_root:
            config.project_root = os.getcwd()
        
        return config
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Создание конфигурации из словаря"""
        config = cls()
        
        # Обрабатываем каждую секцию
        if 'server' in data:
            config.server = ServerConfig(**data['server'])
        if 'indexing' in data:
            config.indexing = IndexingConfig(**data['indexing'])
        if 'context' in data:
            config.context = ContextConfig(**data['context'])
        if 'features' in data:
            config.features = FeaturesConfig(**data['features'])
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        
        # Дополнительные настройки
        config.project_root = data.get('project_root')
        config.workspace_folders = data.get('workspace_folders', [])
        config.custom_settings = data.get('custom_settings', {})
        
        return config
    
    def _merge(self, other: "Config"):
        """Объединение с другой конфигурацией (other имеет приоритет)"""
        # Обновляем только непустые значения
        for section_name in ['server', 'indexing', 'context', 'features', 'cache']:
            self_section = getattr(self, section_name)
            other_section = getattr(other, section_name)
            
            for field_name in self_section.__dataclass_fields__:
                other_value = getattr(other_section, field_name)
                default_value = self_section.__dataclass_fields__[field_name].default
                
                # Обновляем только если значение отличается от дефолтного
                if other_value != default_value:
                    setattr(self_section, field_name, other_value)
        
        # Обновляем остальные поля
        if other.project_root:
            self.project_root = other.project_root
        if other.workspace_folders:
            self.workspace_folders = other.workspace_folders
        if other.custom_settings:
            self.custom_settings.update(other.custom_settings)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'server': asdict(self.server),
            'indexing': asdict(self.indexing),
            'context': asdict(self.context),
            'features': asdict(self.features),
            'cache': asdict(self.cache),
            'project_root': self.project_root,
            'workspace_folders': self.workspace_folders,
            'custom_settings': self.custom_settings,
        }
    
    def save(self, config_path: Path):
        """Сохранение конфигурации в файл"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Конфигурация сохранена: {config_path}")