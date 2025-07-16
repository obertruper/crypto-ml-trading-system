#!/usr/bin/env python3
"""
Enhanced Python LSP Server для улучшения работы с Claude Code
Основные возможности:
- Продвинутая индексация проекта
- Управление контекстом и памятью
- Интеграция с документацией
- Оптимизация для работы с LLM
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import ast

from lsprotocol import types
from pygls.server import LanguageServer
from pygls.workspace import TextDocument

# Настройка логирования
logging.basicConfig(
    filename='enhanced-lsp.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Информация о символе в коде"""
    name: str
    kind: types.SymbolKind
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


@dataclass
class FileIndex:
    """Индекс файла с метаинформацией"""
    path: str
    last_modified: float
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    content_hash: Optional[str] = None


@dataclass
class ContextWindow:
    """Окно контекста для LLM"""
    current_file: str
    related_files: List[str] = field(default_factory=list)
    symbols_in_scope: List[SymbolInfo] = field(default_factory=list)
    recent_edits: List[Dict[str, Any]] = field(default_factory=list)
    documentation_refs: List[str] = field(default_factory=list)
    max_tokens: int = 8000


class PythonASTAnalyzer(ast.NodeVisitor):
    """Анализатор AST для извлечения символов и метаданных"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.symbols: Dict[str, SymbolInfo] = {}
        self.imports: List[str] = []
        self.current_class: Optional[str] = None
        self.symbol_stack: List[str] = []
        
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node: ast.ClassDef):
        symbol = SymbolInfo(
            name=node.name,
            kind=types.SymbolKind.Class,
            file_path=self.file_path,
            line=node.lineno - 1,
            column=node.col_offset,
            end_line=node.end_lineno - 1 if node.end_lineno else node.lineno - 1,
            end_column=node.end_col_offset if node.end_col_offset else 0,
            docstring=ast.get_docstring(node),
            parent=self.current_class
        )
        
        # Извлекаем базовые классы
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
        if bases:
            symbol.signature = f"class {node.name}({', '.join(bases)})"
        else:
            symbol.signature = f"class {node.name}"
            
        self.symbols[node.name] = symbol
        
        # Обрабатываем вложенные элементы
        old_class = self.current_class
        self.current_class = node.name
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Создаем сигнатуру функции
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
            
        signature = f"def {node.name}({', '.join(args)})"
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
            
        symbol = SymbolInfo(
            name=node.name,
            kind=types.SymbolKind.Function if not self.current_class else types.SymbolKind.Method,
            file_path=self.file_path,
            line=node.lineno - 1,
            column=node.col_offset,
            end_line=node.end_lineno - 1 if node.end_lineno else node.lineno - 1,
            end_column=node.end_col_offset if node.end_col_offset else 0,
            signature=signature,
            docstring=ast.get_docstring(node),
            parent=self.current_class
        )
        
        # Извлекаем аннотации типов
        if node.returns:
            symbol.type_annotation = ast.unparse(node.returns)
            
        full_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
        self.symbols[full_name] = symbol
        
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # Обрабатываем как обычную функцию, но с пометкой async
        self.visit_FunctionDef(node)
        full_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
        if full_name in self.symbols:
            self.symbols[full_name].signature = "async " + self.symbols[full_name].signature


class EnhancedPythonLSPServer(LanguageServer):
    """Расширенный LSP сервер для Python с оптимизацией под LLM"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Индекс проекта
        self.project_index: Dict[str, FileIndex] = {}
        self.symbol_index: Dict[str, List[SymbolInfo]] = defaultdict(list)
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Контекст для LLM
        self.context_windows: Dict[str, ContextWindow] = {}
        self.documentation_cache: Dict[str, str] = {}
        
        # Настройки
        self.index_extensions = {'.py', '.pyi', '.pyx'}
        self.max_file_size = 1024 * 1024  # 1MB
        self.context_token_limit = 8000
        
        # Статистика
        self.stats = {
            'indexed_files': 0,
            'total_symbols': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    async def index_workspace(self, workspace_path: Path):
        """Индексация всего рабочего пространства"""
        logger.info(f"Starting workspace indexing: {workspace_path}")
        
        try:
            for file_path in workspace_path.rglob('*'):
                if file_path.suffix in self.index_extensions:
                    if file_path.stat().st_size <= self.max_file_size:
                        await self.index_file(file_path)
                        
            logger.info(f"Indexing complete: {self.stats['indexed_files']} files, "
                       f"{self.stats['total_symbols']} symbols")
        except Exception as e:
            logger.error(f"Error during workspace indexing: {e}")
            
    async def index_file(self, file_path: Path):
        """Индексация одного файла"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Парсим AST
            tree = ast.parse(content, filename=str(file_path))
            analyzer = PythonASTAnalyzer(str(file_path))
            analyzer.visit(tree)
            
            # Создаем индекс файла
            file_index = FileIndex(
                path=str(file_path),
                last_modified=file_path.stat().st_mtime,
                imports=analyzer.imports,
                symbols=analyzer.symbols
            )
            
            # Обновляем глобальные индексы
            self.project_index[str(file_path)] = file_index
            
            # Обновляем индекс символов
            for symbol_name, symbol_info in analyzer.symbols.items():
                self.symbol_index[symbol_name].append(symbol_info)
                
            # Обновляем граф импортов
            for imp in analyzer.imports:
                self.import_graph[str(file_path)].add(imp)
                
            self.stats['indexed_files'] += 1
            self.stats['total_symbols'] += len(analyzer.symbols)
            
            logger.debug(f"Indexed {file_path}: {len(analyzer.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            
    def get_context_window(self, uri: str) -> ContextWindow:
        """Получение окна контекста для файла"""
        if uri not in self.context_windows:
            self.context_windows[uri] = ContextWindow(
                current_file=uri,
                max_tokens=self.context_token_limit
            )
        return self.context_windows[uri]
        
    def update_context_window(self, uri: str, document: TextDocument):
        """Обновление окна контекста при изменении документа"""
        context = self.get_context_window(uri)
        
        # Добавляем информацию о последних изменениях
        context.recent_edits.append({
            'timestamp': time.time(),
            'uri': uri,
            'version': document.version
        })
        
        # Ограничиваем количество записей
        if len(context.recent_edits) > 10:
            context.recent_edits.pop(0)
            
        # Обновляем связанные файлы
        if uri in self.project_index:
            file_index = self.project_index[uri]
            for imp in file_index.imports:
                # Пытаемся найти импортированные модули
                for path, index in self.project_index.items():
                    if path.endswith(f"{imp}.py"):
                        if path not in context.related_files:
                            context.related_files.append(path)
                            
        # Ограничиваем количество связанных файлов
        context.related_files = context.related_files[:5]
        
    def get_symbol_context(self, symbol_name: str) -> List[SymbolInfo]:
        """Получение контекста для символа"""
        symbols = self.symbol_index.get(symbol_name, [])
        
        # Добавляем информацию о родительских классах
        enriched_symbols = []
        for symbol in symbols:
            if symbol.parent:
                parent_symbols = self.symbol_index.get(symbol.parent, [])
                symbol.references.extend([(s.file_path, s.line) for s in parent_symbols])
            enriched_symbols.append(symbol)
            
        return enriched_symbols
        
    def generate_llm_context(self, uri: str, position: types.Position) -> str:
        """Генерация контекста для LLM"""
        context_parts = []
        context_window = self.get_context_window(uri)
        
        # Добавляем информацию о текущем файле
        if uri in self.project_index:
            file_index = self.project_index[uri]
            context_parts.append(f"Current file: {file_index.path}")
            context_parts.append(f"Imports: {', '.join(file_index.imports)}")
            
            # Добавляем символы в области видимости
            relevant_symbols = []
            for symbol in file_index.symbols.values():
                if symbol.line <= position.line <= symbol.end_line:
                    relevant_symbols.append(symbol)
                    
            if relevant_symbols:
                context_parts.append("\nSymbols in scope:")
                for symbol in relevant_symbols:
                    context_parts.append(f"- {symbol.kind.name}: {symbol.name}")
                    if symbol.signature:
                        context_parts.append(f"  Signature: {symbol.signature}")
                    if symbol.docstring:
                        context_parts.append(f"  Doc: {symbol.docstring[:100]}...")
                        
        # Добавляем информацию о связанных файлах
        if context_window.related_files:
            context_parts.append("\nRelated files:")
            for related_file in context_window.related_files[:3]:
                if related_file in self.project_index:
                    rel_index = self.project_index[related_file]
                    context_parts.append(f"- {related_file}")
                    # Добавляем ключевые символы
                    key_symbols = list(rel_index.symbols.values())[:5]
                    for sym in key_symbols:
                        context_parts.append(f"  - {sym.kind.name}: {sym.name}")
                        
        return "\n".join(context_parts)


# Создаем сервер
server = EnhancedPythonLSPServer("enhanced-python-lsp", "v1.0.0")


@server.feature(types.INITIALIZE)
async def initialize(params: types.InitializeParams):
    """Инициализация сервера"""
    logger.info("Server initialization started")
    
    # Индексируем рабочее пространство
    if params.workspace_folders:
        for folder in params.workspace_folders:
            workspace_path = Path(folder.uri.replace("file://", ""))
            asyncio.create_task(server.index_workspace(workspace_path))
            
    return types.InitializeResult(
        capabilities=types.ServerCapabilities(
            text_document_sync=types.TextDocumentSyncOptions(
                open_close=True,
                change=types.TextDocumentSyncKind.Incremental,
                save=True
            ),
            completion_provider=types.CompletionOptions(
                trigger_characters=[".", "(", "[", '"', "'"],
                resolve_provider=True
            ),
            hover_provider=True,
            definition_provider=True,
            references_provider=True,
            document_symbol_provider=True,
            workspace_symbol_provider=True,
            code_action_provider=True,
            document_formatting_provider=True,
            rename_provider=True,
            signature_help_provider=types.SignatureHelpOptions(
                trigger_characters=["(", ","]
            )
        )
    )


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
async def did_open(params: types.DidOpenTextDocumentParams):
    """Обработка открытия документа"""
    uri = params.text_document.uri
    document = server.workspace.get_text_document(uri)
    
    # Индексируем файл
    file_path = Path(uri.replace("file://", ""))
    await server.index_file(file_path)
    
    # Обновляем контекст
    server.update_context_window(uri, document)
    
    logger.info(f"Document opened: {uri}")


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(params: types.DidChangeTextDocumentParams):
    """Обработка изменения документа"""
    uri = params.text_document.uri
    document = server.workspace.get_text_document(uri)
    
    # Обновляем контекст
    server.update_context_window(uri, document)
    
    # Планируем переиндексацию
    asyncio.create_task(_reindex_file_delayed(uri))


async def _reindex_file_delayed(uri: str, delay: float = 1.0):
    """Отложенная переиндексация файла"""
    await asyncio.sleep(delay)
    file_path = Path(uri.replace("file://", ""))
    if file_path.exists():
        await server.index_file(file_path)


@server.feature(types.TEXT_DOCUMENT_COMPLETION)
def completion(params: types.CompletionParams) -> types.CompletionList:
    """Автодополнение с учетом контекста"""
    uri = params.text_document.uri
    document = server.workspace.get_text_document(uri)
    
    # Генерируем контекст для LLM
    llm_context = server.generate_llm_context(uri, params.position)
    
    # Получаем строку под курсором
    current_line = document.lines[params.position.line]
    prefix = current_line[:params.position.character]
    
    items = []
    
    # Ищем подходящие символы
    for symbol_name, symbols in server.symbol_index.items():
        if symbol_name.lower().startswith(prefix.lower()):
            for symbol in symbols[:5]:  # Ограничиваем количество
                item = types.CompletionItem(
                    label=symbol_name,
                    kind=_symbol_kind_to_completion_kind(symbol.kind),
                    detail=symbol.signature or symbol_name,
                    documentation=types.MarkupContent(
                        kind=types.MarkupKind.Markdown,
                        value=f"{symbol.docstring or 'No documentation'}\n\n"
                              f"File: {symbol.file_path}\n"
                              f"Line: {symbol.line + 1}"
                    )
                )
                
                # Добавляем контекст LLM в дополнительные данные
                item.data = {
                    'llm_context': llm_context,
                    'symbol_info': {
                        'name': symbol.name,
                        'kind': symbol.kind.name,
                        'file': symbol.file_path
                    }
                }
                
                items.append(item)
                
    return types.CompletionList(is_incomplete=False, items=items)


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(params: types.HoverParams) -> Optional[types.Hover]:
    """Информация при наведении с расширенным контекстом"""
    uri = params.text_document.uri
    document = server.workspace.get_text_document(uri)
    
    # Получаем слово под курсором
    word = document.word_at_position(params.position)
    if not word:
        return None
        
    # Ищем символ
    symbols = server.symbol_index.get(word, [])
    if not symbols:
        return None
        
    symbol = symbols[0]  # Берем первый найденный
    
    # Формируем содержимое
    content_parts = [f"**{symbol.kind.name}**: `{symbol.name}`"]
    
    if symbol.signature:
        content_parts.append(f"\n```python\n{symbol.signature}\n```")
        
    if symbol.docstring:
        content_parts.append(f"\n{symbol.docstring}")
        
    # Добавляем информацию о контексте
    content_parts.append(f"\n\n---\n**Location**: {symbol.file_path}:{symbol.line + 1}")
    
    if symbol.references:
        content_parts.append(f"\n**References**: {len(symbol.references)} locations")
        
    # Генерируем LLM контекст
    llm_context = server.generate_llm_context(uri, params.position)
    
    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value="\n".join(content_parts)
        ),
        range=types.Range(
            start=types.Position(line=params.position.line, character=0),
            end=types.Position(line=params.position.line, character=len(document.lines[params.position.line]))
        )
    )


@server.feature(types.TEXT_DOCUMENT_DEFINITION)
def definition(params: types.DefinitionParams) -> Optional[List[types.Location]]:
    """Переход к определению"""
    uri = params.text_document.uri
    document = server.workspace.get_text_document(uri)
    
    word = document.word_at_position(params.position)
    if not word:
        return None
        
    symbols = server.symbol_index.get(word, [])
    locations = []
    
    for symbol in symbols:
        locations.append(
            types.Location(
                uri=f"file://{symbol.file_path}",
                range=types.Range(
                    start=types.Position(line=symbol.line, character=symbol.column),
                    end=types.Position(line=symbol.end_line, character=symbol.end_column)
                )
            )
        )
        
    return locations if locations else None


@server.feature(types.TEXT_DOCUMENT_REFERENCES)
def references(params: types.ReferenceParams) -> Optional[List[types.Location]]:
    """Поиск ссылок на символ"""
    uri = params.text_document.uri
    document = server.workspace.get_text_document(uri)
    
    word = document.word_at_position(params.position)
    if not word:
        return None
        
    locations = []
    
    # Ищем все вхождения символа
    for file_path, file_index in server.project_index.items():
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines):
                # Простой поиск по тексту (можно улучшить с помощью AST)
                if word in line:
                    # Находим позицию слова в строке
                    col = line.find(word)
                    if col >= 0:
                        locations.append(
                            types.Location(
                                uri=f"file://{file_path}",
                                range=types.Range(
                                    start=types.Position(line=line_num, character=col),
                                    end=types.Position(line=line_num, character=col + len(word))
                                )
                            )
                        )
        except Exception as e:
            logger.error(f"Error searching references in {file_path}: {e}")
            
    return locations if locations else None


@server.feature(types.WORKSPACE_SYMBOL)
def workspace_symbol(params: types.WorkspaceSymbolParams) -> List[types.SymbolInformation]:
    """Поиск символов в рабочем пространстве"""
    query = params.query.lower()
    symbols = []
    
    for symbol_name, symbol_list in server.symbol_index.items():
        if query in symbol_name.lower():
            for symbol in symbol_list[:10]:  # Ограничиваем количество
                symbols.append(
                    types.SymbolInformation(
                        name=symbol.name,
                        kind=symbol.kind,
                        location=types.Location(
                            uri=f"file://{symbol.file_path}",
                            range=types.Range(
                                start=types.Position(line=symbol.line, character=symbol.column),
                                end=types.Position(line=symbol.end_line, character=symbol.end_column)
                            )
                        ),
                        container_name=symbol.parent
                    )
                )
                
    return symbols


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(params: types.DocumentSymbolParams) -> List[types.DocumentSymbol]:
    """Символы в документе"""
    uri = params.text_document.uri
    file_path = uri.replace("file://", "")
    
    if file_path not in server.project_index:
        return []
        
    file_index = server.project_index[file_path]
    symbols = []
    
    # Группируем символы по родителям
    root_symbols = []
    child_symbols = defaultdict(list)
    
    for symbol in file_index.symbols.values():
        if symbol.parent:
            child_symbols[symbol.parent].append(symbol)
        else:
            root_symbols.append(symbol)
            
    # Создаем иерархию символов
    def create_document_symbol(symbol: SymbolInfo) -> types.DocumentSymbol:
        children = []
        for child in child_symbols.get(symbol.name, []):
            children.append(create_document_symbol(child))
            
        return types.DocumentSymbol(
            name=symbol.name,
            detail=symbol.signature or "",
            kind=symbol.kind,
            range=types.Range(
                start=types.Position(line=symbol.line, character=symbol.column),
                end=types.Position(line=symbol.end_line, character=symbol.end_column)
            ),
            selection_range=types.Range(
                start=types.Position(line=symbol.line, character=symbol.column),
                end=types.Position(line=symbol.line, character=symbol.column + len(symbol.name))
            ),
            children=children
        )
        
    for symbol in root_symbols:
        symbols.append(create_document_symbol(symbol))
        
    return symbols


@server.command("enhanced_lsp.get_context")
def get_context_command(args):
    """Команда для получения контекста LLM"""
    uri = args[0] if args else None
    if not uri:
        return {"error": "No URI provided"}
        
    context_window = server.get_context_window(uri)
    
    return {
        "current_file": context_window.current_file,
        "related_files": context_window.related_files,
        "recent_edits": len(context_window.recent_edits),
        "symbols_in_scope": len(context_window.symbols_in_scope),
        "stats": server.stats
    }


@server.command("enhanced_lsp.reindex_workspace")
async def reindex_workspace_command(args):
    """Команда для переиндексации рабочего пространства"""
    workspace_folders = server.workspace.folders
    if not workspace_folders:
        return {"error": "No workspace folders"}
        
    # Сбрасываем индексы
    server.project_index.clear()
    server.symbol_index.clear()
    server.import_graph.clear()
    server.stats['indexed_files'] = 0
    server.stats['total_symbols'] = 0
    
    # Переиндексируем
    for folder_uri, folder in workspace_folders.items():
        workspace_path = Path(folder_uri.replace("file://", ""))
        await server.index_workspace(workspace_path)
        
    return {
        "status": "success",
        "indexed_files": server.stats['indexed_files'],
        "total_symbols": server.stats['total_symbols']
    }


def _symbol_kind_to_completion_kind(symbol_kind: types.SymbolKind) -> types.CompletionItemKind:
    """Преобразование типа символа в тип автодополнения"""
    mapping = {
        types.SymbolKind.Class: types.CompletionItemKind.Class,
        types.SymbolKind.Function: types.CompletionItemKind.Function,
        types.SymbolKind.Method: types.CompletionItemKind.Method,
        types.SymbolKind.Variable: types.CompletionItemKind.Variable,
        types.SymbolKind.Constant: types.CompletionItemKind.Constant,
        types.SymbolKind.Module: types.CompletionItemKind.Module,
        types.SymbolKind.Property: types.CompletionItemKind.Property,
    }
    return mapping.get(symbol_kind, types.CompletionItemKind.Text)


if __name__ == "__main__":
    logger.info("Starting Enhanced Python LSP Server")
    server.start_io()
