"""
Базовые тесты для Universal LSP Server
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from lsp_server import UniversalLSPServer, Config
from lsp_server.indexer import ProjectIndexer
from lsp_server.context import ContextManager


@pytest.fixture
def temp_project():
    """Создать временный проект для тестов"""
    temp_dir = tempfile.mkdtemp()
    
    # Создаем структуру проекта
    project_dir = Path(temp_dir) / "test_project"
    project_dir.mkdir()
    
    # Создаем тестовые файлы
    (project_dir / "main.py").write_text("""
def hello_world():
    '''Приветствие миру'''
    return "Hello, World!"

class Calculator:
    '''Простой калькулятор'''
    
    def add(self, a: int, b: int) -> int:
        '''Сложение двух чисел'''
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        '''Умножение двух чисел'''
        return a * b
""")
    
    (project_dir / "utils.py").write_text("""
import os
from typing import List, Dict

def get_files(directory: str) -> List[str]:
    '''Получить список файлов в директории'''
    return os.listdir(directory)

def parse_config(config_path: str) -> Dict[str, any]:
    '''Парсинг конфигурации'''
    # TODO: Implement
    pass
""")
    
    # Создаем поддиректорию
    (project_dir / "tests").mkdir()
    (project_dir / "tests" / "test_main.py").write_text("""
from main import hello_world, Calculator

def test_hello():
    assert hello_world() == "Hello, World!"

def test_calculator():
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.multiply(3, 4) == 12
""")
    
    yield project_dir
    
    # Очистка
    shutil.rmtree(temp_dir)


@pytest.fixture
def config():
    """Конфигурация для тестов"""
    cfg = Config()
    cfg.server.log_level = "DEBUG"
    cfg.indexing.parallel_indexing = False  # Для предсказуемости тестов
    cfg.cache.enabled = False  # Отключаем кеш в тестах
    return cfg


class TestConfig:
    """Тесты конфигурации"""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию"""
        config = Config()
        assert config.server.port == 2087
        assert config.server.host == "127.0.0.1"
        assert config.indexing.extensions == [".py", ".pyi", ".pyx"]
    
    def test_config_from_env(self, monkeypatch):
        """Тест загрузки из переменных окружения"""
        monkeypatch.setenv("LSP_HOST", "0.0.0.0")
        monkeypatch.setenv("LSP_PORT", "3000")
        monkeypatch.setenv("LSP_LOG_LEVEL", "DEBUG")
        
        config = Config.from_env()
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 3000
        assert config.server.log_level == "DEBUG"
    
    def test_config_merge(self):
        """Тест объединения конфигураций"""
        config1 = Config()
        config2 = Config()
        config2.server.port = 3000
        config2.context.max_tokens = 50000
        
        config1._merge(config2)
        assert config1.server.port == 3000
        assert config1.context.max_tokens == 50000


class TestIndexer:
    """Тесты индексатора"""
    
    @pytest.mark.asyncio
    async def test_index_directory(self, temp_project, config):
        """Тест индексации директории"""
        indexer = ProjectIndexer(config.indexing)
        
        result = await indexer.index_directory(temp_project)
        
        assert result['total_files'] == 3
        assert result['indexed_files'] == 3
        assert result['total_symbols'] > 0
        assert len(result['errors']) == 0
    
    @pytest.mark.asyncio
    async def test_find_symbols(self, temp_project, config):
        """Тест поиска символов"""
        indexer = ProjectIndexer(config.indexing)
        await indexer.index_directory(temp_project)
        
        # Поиск функции
        symbols = await indexer.search_symbols("hello")
        assert len(symbols) > 0
        assert any(s.name == "hello_world" for s in symbols)
        
        # Поиск класса
        symbols = await indexer.search_symbols("Calc")
        assert len(symbols) > 0
        assert any(s.name == "Calculator" for s in symbols)
    
    @pytest.mark.asyncio
    async def test_get_symbol_at_position(self, temp_project, config):
        """Тест получения символа в позиции"""
        indexer = ProjectIndexer(config.indexing)
        await indexer.index_directory(temp_project)
        
        main_py = str(temp_project / "main.py")
        
        # Позиция внутри функции hello_world
        symbol = await indexer.get_symbol_at_position(main_py, 1, 10)
        assert symbol is not None
        assert symbol.name == "hello_world"
        assert symbol.kind == "function"


class TestContextManager:
    """Тесты менеджера контекста"""
    
    def test_export_markdown(self, temp_project, config):
        """Тест экспорта контекста в Markdown"""
        context_mgr = ContextManager(config.context)
        
        # Добавляем тестовые данные
        from lsp_server.indexer import FileIndex, SymbolInfo
        
        file_index = FileIndex(
            path=str(temp_project / "main.py"),
            last_modified=0,
            content_hash="test_hash"
        )
        
        file_index.symbols["hello_world"] = SymbolInfo(
            name="hello_world",
            kind="function",
            file_path=str(temp_project / "main.py"),
            line=1,
            column=0,
            end_line=3,
            end_column=0,
            docstring="Приветствие миру"
        )
        
        context_mgr.add_file_context(file_index)
        
        # Экспортируем
        markdown = context_mgr.export_context(str(temp_project / "main.py"), "markdown")
        
        assert "hello_world" in markdown
        assert "function" in markdown
        assert "Приветствие миру" in markdown
    
    def test_get_related_files(self, config):
        """Тест получения связанных файлов"""
        context_mgr = ContextManager(config.context)
        
        # TODO: Добавить более сложный тест с импортами
        related = context_mgr.get_related_files("/test/file.py")
        assert isinstance(related, list)


class TestServer:
    """Тесты сервера"""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, temp_project, config):
        """Тест инициализации сервера"""
        config.project_root = str(temp_project)
        server = UniversalLSPServer(config)
        
        assert server.config == config
        assert server.project_root == temp_project
        assert server.indexer is not None
        assert server.context_manager is not None
    
    @pytest.mark.asyncio
    async def test_server_capabilities(self, config):
        """Тест возможностей сервера"""
        server = UniversalLSPServer(config)
        
        # Проверяем, что обработчики зарегистрированы
        assert hasattr(server, 'lsp')
        assert len(server.lsp._feature_handlers) > 0


class TestUtils:
    """Тесты утилит"""
    
    def test_find_project_root(self, temp_project):
        """Тест поиска корня проекта"""
        from lsp_server.utils import find_project_root
        
        # Создаем .git директорию
        (temp_project / ".git").mkdir()
        
        # Из поддиректории
        subdir = temp_project / "subdir"
        subdir.mkdir()
        
        root = find_project_root(subdir)
        assert root == temp_project
    
    def test_is_python_file(self):
        """Тест определения Python файлов"""
        from lsp_server.utils import is_python_file
        
        assert is_python_file(Path("test.py"))
        assert is_python_file(Path("test.pyi"))
        assert is_python_file(Path("test.pyx"))
        assert not is_python_file(Path("test.txt"))
        assert not is_python_file(Path("test.js"))
    
    def test_file_hash(self):
        """Тест хеширования файлов"""
        from lsp_server.utils import get_file_hash
        
        content1 = "Hello, World!"
        content2 = "Hello, World!"
        content3 = "Goodbye, World!"
        
        hash1 = get_file_hash(content1)
        hash2 = get_file_hash(content2)
        hash3 = get_file_hash(content3)
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # Укороченный хеш


if __name__ == "__main__":
    pytest.main([__file__, "-v"])