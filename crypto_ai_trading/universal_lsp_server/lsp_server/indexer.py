"""
Индексатор проекта для Universal LSP Server
"""

import ast
import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import aiofiles

from .config import IndexingConfig
from .utils import is_python_file, get_file_hash

logger = logging.getLogger(__name__)

@dataclass
class SymbolInfo:
    """Информация о символе в коде"""
    name: str
    kind: str  # class, function, method, variable, etc.
    file_path: str
    line: int
    column: int
    end_line: int
    end_column: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    references: List[Tuple[str, int]] = field(default_factory=list)
    type_annotation: Optional[str] = None
    module: Optional[str] = None

@dataclass
class FileIndex:
    """Индекс файла"""
    path: str
    last_modified: float
    content_hash: str
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)

class PythonASTAnalyzer(ast.NodeVisitor):
    """Анализатор Python AST для извлечения символов"""
    
    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.splitlines()
        self.symbols: Dict[str, SymbolInfo] = {}
        self.imports: List[str] = []
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.symbol_stack: List[str] = []
        
    def get_line_content(self, lineno: int) -> str:
        """Получить содержимое строки"""
        if 0 <= lineno - 1 < len(self.lines):
            return self.lines[lineno - 1]
        return ""
    
    def get_docstring(self, node: ast.AST) -> Optional[str]:
        """Извлечь docstring из узла"""
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return None
        
        if node.body and isinstance(node.body[0], ast.Expr):
            expr = node.body[0]
            if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                return expr.value.value
        return None
    
    def get_signature(self, node: ast.FunctionDef) -> str:
        """Получить сигнатуру функции"""
        args = []
        
        # Позиционные аргументы
        for i, arg in enumerate(node.args.args):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            
            # Значения по умолчанию
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                arg_str += f" = {ast.unparse(node.args.defaults[default_idx])}"
            
            args.append(arg_str)
        
        # *args
        if node.args.vararg:
            arg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(arg_str)
        
        # **kwargs
        if node.args.kwarg:
            arg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(arg_str)
        
        # Возвращаемый тип
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"
        
        return f"{node.name}({', '.join(args)}){returns}"
    
    def add_symbol(self, name: str, kind: str, node: ast.AST, 
                   signature: Optional[str] = None,
                   docstring: Optional[str] = None,
                   type_annotation: Optional[str] = None):
        """Добавить символ в индекс"""
        
        # Определяем родителя
        parent = None
        if self.current_class:
            parent = self.current_class
        elif self.current_function and kind == "variable":
            parent = self.current_function
        
        # Полное имя символа
        if parent:
            full_name = f"{parent}.{name}"
        else:
            full_name = name
        
        symbol = SymbolInfo(
            name=name,
            kind=kind,
            file_path=self.file_path,
            line=node.lineno - 1,  # 0-based
            column=node.col_offset,
            end_line=getattr(node, 'end_lineno', node.lineno) - 1,
            end_column=getattr(node, 'end_col_offset', node.col_offset),
            signature=signature,
            docstring=docstring,
            parent=parent,
            type_annotation=type_annotation,
            module=Path(self.file_path).stem
        )
        
        self.symbols[full_name] = symbol
    
    def visit_Import(self, node: ast.Import):
        """Обработка import"""
        for alias in node.names:
            self.imports.append(alias.name)
            # Добавляем импорт как символ
            self.add_symbol(
                alias.asname or alias.name,
                "import",
                node,
                signature=f"import {alias.name}"
            )
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Обработка from ... import ..."""
        if node.module:
            self.imports.append(node.module)
            for alias in node.names:
                name = alias.asname or alias.name
                self.add_symbol(
                    name,
                    "import",
                    node,
                    signature=f"from {node.module} import {alias.name}"
                )
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Обработка определения класса"""
        docstring = self.get_docstring(node)
        
        # Базовые классы
        bases = [ast.unparse(base) for base in node.bases]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
        
        self.add_symbol(node.name, "class", node, signature, docstring)
        
        # Входим в контекст класса
        old_class = self.current_class
        self.current_class = node.name if not old_class else f"{old_class}.{node.name}"
        
        self.generic_visit(node)
        
        # Выходим из контекста
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Обработка определения функции"""
        self._visit_function(node, is_async=False)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Обработка определения async функции"""
        self._visit_function(node, is_async=True)
    
    def _visit_function(self, node, is_async: bool):
        """Общая логика для функций"""
        docstring = self.get_docstring(node)
        signature = self.get_signature(node)
        if is_async:
            signature = f"async {signature}"
        
        kind = "method" if self.current_class else "function"
        self.add_symbol(node.name, kind, node, signature, docstring)
        
        # Входим в контекст функции
        old_function = self.current_function
        if self.current_class:
            self.current_function = f"{self.current_class}.{node.name}"
        else:
            self.current_function = node.name
        
        self.generic_visit(node)
        
        # Выходим из контекста
        self.current_function = old_function
    
    def visit_Assign(self, node: ast.Assign):
        """Обработка присваивания"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Определяем тип если возможно
                type_annotation = None
                if isinstance(node.value, ast.Constant):
                    type_annotation = type(node.value.value).__name__
                
                self.add_symbol(
                    target.id,
                    "variable",
                    target,
                    type_annotation=type_annotation
                )
        
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Обработка аннотированного присваивания"""
        if isinstance(node.target, ast.Name):
            type_annotation = ast.unparse(node.annotation)
            self.add_symbol(
                node.target.id,
                "variable",
                node.target,
                type_annotation=type_annotation
            )
        
        self.generic_visit(node)

class ProjectIndexer:
    """Индексатор проекта"""
    
    def __init__(self, config: IndexingConfig):
        self.config = config
        self.file_index: Dict[str, FileIndex] = {}
        self.symbol_index: Dict[str, List[SymbolInfo]] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
    
    async def index_directory(self, directory: Path) -> Dict[str, Any]:
        """Индексировать директорию"""
        start_time = time.time()
        
        logger.info(f"Начинаю индексацию: {directory}")
        
        # Собираем файлы для индексации
        files_to_index = self._collect_files(directory)
        total_files = len(files_to_index)
        
        logger.info(f"Найдено {total_files} файлов для индексации")
        
        # Индексируем файлы
        if self.config.parallel_indexing:
            results = await self._index_files_parallel(files_to_index)
        else:
            results = await self._index_files_sequential(files_to_index)
        
        # Обновляем индексы
        total_symbols = 0
        errors = []
        
        for file_path, result in results:
            if isinstance(result, FileIndex):
                self.file_index[file_path] = result
                
                # Обновляем символьный индекс
                for symbol in result.symbols.values():
                    if symbol.name not in self.symbol_index:
                        self.symbol_index[symbol.name] = []
                    self.symbol_index[symbol.name].append(symbol)
                    total_symbols += 1
            else:
                errors.append(f"{file_path}: {result}")
        
        # Статистика
        end_time = time.time()
        total_size = sum(
            Path(f).stat().st_size 
            for f in files_to_index 
            if Path(f).exists()
        )
        
        return {
            'total_files': total_files,
            'indexed_files': len(self.file_index),
            'total_symbols': total_symbols,
            'total_size': total_size,
            'indexing_time': end_time - start_time,
            'errors': errors
        }
    
    def _collect_files(self, directory: Path) -> List[str]:
        """Собрать файлы для индексации"""
        files = []
        
        for pattern in self.config.extensions:
            for file_path in directory.rglob(f"*{pattern}"):
                # Проверяем исключения
                if self._should_exclude(file_path):
                    continue
                
                # Проверяем размер
                if file_path.stat().st_size > self.config.max_file_size:
                    logger.warning(f"Пропускаю большой файл: {file_path}")
                    continue
                
                files.append(str(file_path))
        
        return files
    
    def _should_exclude(self, path: Path) -> bool:
        """Проверить, нужно ли исключить путь"""
        # Проверяем директории
        for part in path.parts:
            if part in self.config.exclude_dirs:
                return True
            
            # Проверяем паттерны
            for pattern in self.config.exclude_patterns:
                if Path(part).match(pattern):
                    return True
        
        # Проверяем скрытые файлы
        if not self.config.index_hidden_files and path.name.startswith('.'):
            return True
        
        # Проверяем символические ссылки
        if not self.config.follow_symlinks and path.is_symlink():
            return True
        
        return False
    
    async def _index_files_parallel(self, files: List[str]) -> List[Tuple[str, Any]]:
        """Параллельная индексация файлов"""
        loop = asyncio.get_event_loop()
        
        tasks = []
        for file_path in files:
            task = loop.run_in_executor(
                self.executor,
                self._index_file_sync,
                file_path
            )
            tasks.append((file_path, task))
        
        results = []
        for file_path, task in tasks:
            try:
                result = await task
                results.append((file_path, result))
            except Exception as e:
                logger.error(f"Ошибка индексации {file_path}: {e}")
                results.append((file_path, str(e)))
        
        return results
    
    async def _index_files_sequential(self, files: List[str]) -> List[Tuple[str, Any]]:
        """Последовательная индексация файлов"""
        results = []
        
        for file_path in files:
            try:
                result = await self._index_file(file_path)
                results.append((file_path, result))
            except Exception as e:
                logger.error(f"Ошибка индексации {file_path}: {e}")
                results.append((file_path, str(e)))
        
        return results
    
    def _index_file_sync(self, file_path: str) -> FileIndex:
        """Синхронная индексация файла"""
        return asyncio.run(self._index_file(file_path))
    
    async def _index_file(self, file_path: str) -> FileIndex:
        """Индексировать один файл"""
        path = Path(file_path)
        
        # Читаем содержимое
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Создаем индекс файла
        file_index = FileIndex(
            path=file_path,
            last_modified=path.stat().st_mtime,
            content_hash=get_file_hash(content)
        )
        
        try:
            # Парсим AST
            tree = ast.parse(content, filename=file_path)
            
            # Анализируем
            analyzer = PythonASTAnalyzer(file_path, content)
            analyzer.visit(tree)
            
            # Сохраняем результаты
            file_index.imports = analyzer.imports
            file_index.symbols = analyzer.symbols
            
            # Определяем экспорты (публичные символы верхнего уровня)
            for name, symbol in analyzer.symbols.items():
                if not name.startswith('_') and not symbol.parent:
                    file_index.exports.append(name)
            
        except SyntaxError as e:
            file_index.errors.append(f"Синтаксическая ошибка: {e}")
        except Exception as e:
            file_index.errors.append(f"Ошибка анализа: {e}")
        
        return file_index
    
    async def get_symbol_at_position(self, file_path: str, line: int, column: int) -> Optional[SymbolInfo]:
        """Получить символ в указанной позиции"""
        if file_path not in self.file_index:
            return None
        
        file_index = self.file_index[file_path]
        
        # Ищем символ в позиции
        for symbol in file_index.symbols.values():
            if (symbol.line <= line <= symbol.end_line and
                symbol.column <= column <= symbol.end_column):
                return symbol
        
        return None
    
    async def find_definition(self, file_path: str, line: int, column: int) -> Optional[SymbolInfo]:
        """Найти определение символа"""
        # Получаем символ в позиции
        symbol = await self.get_symbol_at_position(file_path, line, column)
        if not symbol:
            return None
        
        # Ищем определение
        if symbol.name in self.symbol_index:
            definitions = self.symbol_index[symbol.name]
            
            # Приоритет: текущий файл > импортированные
            for defn in definitions:
                if defn.file_path == file_path:
                    return defn
            
            # Возвращаем первое найденное
            if definitions:
                return definitions[0]
        
        return None
    
    async def find_references(self, file_path: str, line: int, column: int,
                            include_declaration: bool = True) -> List[SymbolInfo]:
        """Найти все ссылки на символ"""
        # Получаем символ
        symbol = await self.get_symbol_at_position(file_path, line, column)
        if not symbol:
            return []
        
        references = []
        
        # Ищем во всех файлах
        for file_idx in self.file_index.values():
            # TODO: Здесь нужен более сложный анализ для поиска использований
            # Пока возвращаем только определения
            for sym in file_idx.symbols.values():
                if sym.name == symbol.name:
                    if include_declaration or sym != symbol:
                        references.append(sym)
        
        return references
    
    async def get_document_symbols(self, file_path: str) -> List[SymbolInfo]:
        """Получить все символы документа"""
        if file_path not in self.file_index:
            return []
        
        return list(self.file_index[file_path].symbols.values())
    
    async def search_symbols(self, query: str) -> List[SymbolInfo]:
        """Поиск символов по запросу"""
        query_lower = query.lower()
        results = []
        
        for name, symbols in self.symbol_index.items():
            if query_lower in name.lower():
                results.extend(symbols)
        
        # Сортируем по релевантности
        results.sort(key=lambda s: (
            not s.name.lower().startswith(query_lower),  # Приоритет тем, что начинаются с запроса
            len(s.name),  # Короче = лучше
            s.name.lower()
        ))
        
        return results[:100]  # Ограничиваем количество результатов
    
    def load_index(self, index_data: Dict[str, Any]):
        """Загрузить индекс из данных"""
        # TODO: Реализовать загрузку из сохраненных данных
        pass