"""
Основной класс Universal LSP Server
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from lsprotocol import types
from pygls.server import LanguageServer
from pygls.workspace import TextDocument

from .config import Config
from .indexer import ProjectIndexer
from .context import ContextManager
from .handlers import setup_handlers
from .cache import CacheManager

logger = logging.getLogger(__name__)

class UniversalLSPServer(LanguageServer):
    """Универсальный LSP сервер с расширенными возможностями"""
    
    def __init__(self, config: Config):
        super().__init__(
            name=config.server.name,
            version=config.server.version,
            max_workers=config.indexing.max_workers
        )
        
        self.config = config
        self.project_root = Path(config.project_root or ".").resolve()
        
        # Инициализируем компоненты
        self.indexer = ProjectIndexer(config.indexing)
        self.context_manager = ContextManager(config.context)
        self.cache_manager = CacheManager(config.cache) if config.cache.enabled else None
        
        # Состояние сервера
        self.initialized = False
        self.indexing_task: Optional[asyncio.Task] = None
        self.workspace_folders: List[Path] = []
        
        # Настраиваем обработчики
        setup_handlers(self)
        
        logger.info(f"Universal LSP Server инициализирован для {self.project_root}")
    
    async def initialize_server(self, params: types.InitializeParams):
        """Инициализация сервера при подключении клиента"""
        logger.info("Инициализация сервера...")
        
        # Определяем рабочие папки
        if params.workspace_folders:
            self.workspace_folders = [
                Path(folder.uri.replace("file://", ""))
                for folder in params.workspace_folders
            ]
        elif params.root_uri:
            root_path = Path(params.root_uri.replace("file://", ""))
            self.workspace_folders = [root_path]
        elif params.root_path:
            self.workspace_folders = [Path(params.root_path)]
        else:
            self.workspace_folders = [self.project_root]
        
        # Обновляем конфигурацию если есть настройки от клиента
        if params.initialization_options:
            self._update_config_from_client(params.initialization_options)
        
        # Запускаем индексацию в фоне
        self.indexing_task = asyncio.create_task(self._index_workspace())
        
        self.initialized = True
        logger.info(f"Сервер инициализирован. Рабочие папки: {self.workspace_folders}")
    
    async def _index_workspace(self):
        """Индексация всех рабочих папок"""
        try:
            for folder in self.workspace_folders:
                if folder.exists():
                    logger.info(f"Индексация папки: {folder}")
                    
                    # Проверяем кеш
                    if self.cache_manager:
                        cached_index = await self.cache_manager.get_index(folder)
                        if cached_index:
                            logger.info(f"Использую кешированный индекс для {folder}")
                            self.indexer.load_index(cached_index)
                            continue
                    
                    # Индексируем
                    index_data = await self.indexer.index_directory(folder)
                    
                    # Сохраняем в кеш
                    if self.cache_manager:
                        await self.cache_manager.save_index(folder, index_data)
                    
                    # Обновляем контекст
                    self.context_manager.update_from_index(index_data)
                    
                    logger.info(f"Индексация завершена: {index_data['total_files']} файлов")
                    
        except Exception as e:
            logger.error(f"Ошибка индексации: {e}", exc_info=True)
    
    def _update_config_from_client(self, options: Dict[str, Any]):
        """Обновление конфигурации из настроек клиента"""
        # Здесь можно обработать специфичные настройки от клиента
        if "maxTokens" in options:
            self.config.context.max_tokens = options["maxTokens"]
        
        if "excludeDirs" in options:
            self.config.indexing.exclude_dirs.extend(options["excludeDirs"])
        
        # Добавляем кастомные настройки
        self.config.custom_settings.update(options)
    
    async def get_completion_items(self, params: types.CompletionParams) -> List[types.CompletionItem]:
        """Получение элементов автодополнения"""
        document = self.workspace.get_text_document(params.text_document.uri)
        position = params.position
        
        # Получаем контекст
        context = await self.context_manager.get_completion_context(
            document.path,
            position.line,
            position.character
        )
        
        # Генерируем предложения
        items = []
        
        # Добавляем символы из текущего файла
        for symbol in context.local_symbols:
            item = types.CompletionItem(
                label=symbol.name,
                kind=self._symbol_kind_to_completion_kind(symbol.kind),
                detail=symbol.signature,
                documentation=symbol.docstring,
                insert_text=symbol.name,
            )
            items.append(item)
        
        # Добавляем импортированные символы
        for symbol in context.imported_symbols:
            item = types.CompletionItem(
                label=symbol.name,
                kind=self._symbol_kind_to_completion_kind(symbol.kind),
                detail=f"from {symbol.module}",
                documentation=symbol.docstring,
                insert_text=symbol.name,
            )
            items.append(item)
        
        return items
    
    async def get_hover_info(self, params: types.HoverParams) -> Optional[types.Hover]:
        """Получение информации при наведении"""
        document = self.workspace.get_text_document(params.text_document.uri)
        position = params.position
        
        # Получаем символ под курсором
        symbol = await self.indexer.get_symbol_at_position(
            document.path,
            position.line,
            position.character
        )
        
        if not symbol:
            return None
        
        # Формируем содержимое
        contents = []
        
        # Сигнатура
        if symbol.signature:
            contents.append(types.MarkedString(
                language="python",
                value=symbol.signature
            ))
        
        # Документация
        if symbol.docstring:
            contents.append(types.MarkedString(
                language="markdown",
                value=symbol.docstring
            ))
        
        # Дополнительная информация
        info_parts = []
        if symbol.type_annotation:
            info_parts.append(f"Type: `{symbol.type_annotation}`")
        if symbol.file_path != document.path:
            info_parts.append(f"Defined in: `{Path(symbol.file_path).name}`")
        
        if info_parts:
            contents.append(types.MarkedString(
                language="markdown",
                value="\n\n".join(info_parts)
            ))
        
        return types.Hover(contents=contents)
    
    async def get_definition(self, params: types.DefinitionParams) -> Optional[types.Location]:
        """Получение определения символа"""
        document = self.workspace.get_text_document(params.text_document.uri)
        position = params.position
        
        # Находим определение
        definition = await self.indexer.find_definition(
            document.path,
            position.line,
            position.character
        )
        
        if not definition:
            return None
        
        return types.Location(
            uri=Path(definition.file_path).as_uri(),
            range=types.Range(
                start=types.Position(line=definition.line, character=definition.column),
                end=types.Position(line=definition.end_line, character=definition.end_column)
            )
        )
    
    async def get_references(self, params: types.ReferenceParams) -> List[types.Location]:
        """Получение всех ссылок на символ"""
        document = self.workspace.get_text_document(params.text_document.uri)
        position = params.position
        
        # Находим все ссылки
        references = await self.indexer.find_references(
            document.path,
            position.line,
            position.character,
            include_declaration=params.context.include_declaration
        )
        
        locations = []
        for ref in references:
            locations.append(types.Location(
                uri=Path(ref.file_path).as_uri(),
                range=types.Range(
                    start=types.Position(line=ref.line, character=ref.column),
                    end=types.Position(line=ref.line, character=ref.column + len(ref.name))
                )
            ))
        
        return locations
    
    async def get_symbols(self, params: types.DocumentSymbolParams) -> List[types.DocumentSymbol]:
        """Получение символов документа"""
        document = self.workspace.get_text_document(params.text_document.uri)
        
        # Получаем символы из индекса
        symbols = await self.indexer.get_document_symbols(document.path)
        
        # Преобразуем в LSP формат
        return self._convert_to_document_symbols(symbols)
    
    async def get_workspace_symbols(self, params: types.WorkspaceSymbolParams) -> List[types.SymbolInformation]:
        """Поиск символов во всем проекте"""
        query = params.query.lower()
        
        # Ищем символы
        symbols = await self.indexer.search_symbols(query)
        
        # Преобразуем в LSP формат
        result = []
        for symbol in symbols:
            result.append(types.SymbolInformation(
                name=symbol.name,
                kind=symbol.kind,
                location=types.Location(
                    uri=Path(symbol.file_path).as_uri(),
                    range=types.Range(
                        start=types.Position(line=symbol.line, character=symbol.column),
                        end=types.Position(line=symbol.end_line, character=symbol.end_column)
                    )
                ),
                container_name=symbol.parent
            ))
        
        return result
    
    def _symbol_kind_to_completion_kind(self, symbol_kind: types.SymbolKind) -> types.CompletionItemKind:
        """Конвертация типа символа в тип автодополнения"""
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
    
    def _convert_to_document_symbols(self, symbols: List[Any]) -> List[types.DocumentSymbol]:
        """Конвертация символов в формат LSP"""
        # Здесь должна быть логика преобразования
        # Это заглушка для примера
        result = []
        for symbol in symbols:
            doc_symbol = types.DocumentSymbol(
                name=symbol.name,
                kind=symbol.kind,
                range=types.Range(
                    start=types.Position(line=symbol.line, character=0),
                    end=types.Position(line=symbol.end_line, character=0)
                ),
                selection_range=types.Range(
                    start=types.Position(line=symbol.line, character=symbol.column),
                    end=types.Position(line=symbol.line, character=symbol.column + len(symbol.name))
                ),
                detail=symbol.signature,
                children=[]
            )
            result.append(doc_symbol)
        
        return result
    
    async def shutdown(self):
        """Корректное завершение работы сервера"""
        logger.info("Завершение работы сервера...")
        
        # Отменяем задачу индексации
        if self.indexing_task and not self.indexing_task.done():
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except asyncio.CancelledError:
                pass
        
        # Сохраняем кеш
        if self.cache_manager:
            await self.cache_manager.flush()
        
        logger.info("Сервер остановлен")