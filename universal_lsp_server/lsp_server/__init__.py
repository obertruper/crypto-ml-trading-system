"""
Universal LSP Server - Language Server Protocol для Python
Оптимизирован для работы с AI ассистентами
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .server import LSPServer
from .indexer import ProjectIndexer
from .context import AIContextExporter

__all__ = ["LSPServer", "ProjectIndexer", "AIContextExporter"]