"""
Universal LSP Server - универсальный Language Server Protocol сервер
для улучшения работы с кодом в AI-ассистентах
"""

__version__ = "1.0.0"
__author__ = "Universal LSP Server Contributors"

from .server import UniversalLSPServer
from .config import Config
from .indexer import ProjectIndexer
from .context import ContextManager

__all__ = [
    "UniversalLSPServer",
    "Config",
    "ProjectIndexer",
    "ContextManager",
]