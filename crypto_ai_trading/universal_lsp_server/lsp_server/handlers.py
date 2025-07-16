"""
Обработчики LSP запросов для Universal LSP Server
"""

import logging
from typing import Optional, List

from lsprotocol import types
from pygls.server import LanguageServer

logger = logging.getLogger(__name__)

def setup_handlers(server: LanguageServer):
    """Настройка всех обработчиков LSP"""
    
    @server.feature(types.TEXT_DOCUMENT_COMPLETION)
    async def completions(params: types.CompletionParams) -> Optional[types.CompletionList]:
        """Обработчик автодополнения"""
        try:
            items = await server.get_completion_items(params)
            return types.CompletionList(is_incomplete=False, items=items)
        except Exception as e:
            logger.error(f"Ошибка автодополнения: {e}", exc_info=True)
            return None
    
    @server.feature(types.TEXT_DOCUMENT_HOVER)
    async def hover(params: types.HoverParams) -> Optional[types.Hover]:
        """Обработчик информации при наведении"""
        try:
            return await server.get_hover_info(params)
        except Exception as e:
            logger.error(f"Ошибка hover: {e}", exc_info=True)
            return None
    
    @server.feature(types.TEXT_DOCUMENT_DEFINITION)
    async def definition(params: types.DefinitionParams) -> Optional[types.Location]:
        """Обработчик перехода к определению"""
        try:
            return await server.get_definition(params)
        except Exception as e:
            logger.error(f"Ошибка definition: {e}", exc_info=True)
            return None
    
    @server.feature(types.TEXT_DOCUMENT_REFERENCES)
    async def references(params: types.ReferenceParams) -> Optional[List[types.Location]]:
        """Обработчик поиска ссылок"""
        try:
            return await server.get_references(params)
        except Exception as e:
            logger.error(f"Ошибка references: {e}", exc_info=True)
            return None
    
    @server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
    async def document_symbols(params: types.DocumentSymbolParams) -> Optional[List[types.DocumentSymbol]]:
        """Обработчик символов документа"""
        try:
            return await server.get_symbols(params)
        except Exception as e:
            logger.error(f"Ошибка document symbols: {e}", exc_info=True)
            return None
    
    @server.feature(types.WORKSPACE_SYMBOL)
    async def workspace_symbols(params: types.WorkspaceSymbolParams) -> Optional[List[types.SymbolInformation]]:
        """Обработчик поиска символов в проекте"""
        try:
            return await server.get_workspace_symbols(params)
        except Exception as e:
            logger.error(f"Ошибка workspace symbols: {e}", exc_info=True)
            return None
    
    @server.feature(types.INITIALIZE)
    async def initialize(params: types.InitializeParams) -> types.InitializeResult:
        """Обработчик инициализации"""
        await server.initialize_server(params)
        
        # Возвращаем возможности сервера
        return types.InitializeResult(
            capabilities=types.ServerCapabilities(
                # Синхронизация текста
                text_document_sync=types.TextDocumentSyncOptions(
                    open_close=True,
                    change=types.TextDocumentSyncKind.Incremental if server.config.performance.incremental_sync 
                           else types.TextDocumentSyncKind.Full,
                    save=types.SaveOptions(include_text=True)
                ),
                
                # Возможности
                completion_provider=types.CompletionOptions(
                    trigger_characters=[".", "(", "[", '"', "'", " "],
                    resolve_provider=True
                ) if server.config.features.completion else None,
                
                hover_provider=server.config.features.hover,
                definition_provider=server.config.features.definition,
                references_provider=server.config.features.references,
                document_symbol_provider=server.config.features.symbols,
                workspace_symbol_provider=server.config.features.symbols,
                
                # Дополнительные возможности
                code_action_provider=server.config.features.code_actions,
                document_formatting_provider=server.config.features.formatting,
                document_range_formatting_provider=server.config.features.formatting,
                rename_provider=types.RenameOptions(prepare_provider=True),
                
                # Экспериментальные
                semantic_tokens_provider=types.SemanticTokensOptions(
                    legend=types.SemanticTokensLegend(
                        token_types=["class", "function", "variable", "parameter"],
                        token_modifiers=["declaration", "definition", "readonly"]
                    ),
                    full=True
                ) if server.config.experimental.semantic_highlighting else None,
            ),
            
            server_info=types.InitializeResultServerInfo(
                name=server.config.server.name,
                version=server.config.server.version
            )
        )
    
    @server.feature(types.INITIALIZED)
    def initialized(params: types.InitializedParams):
        """Обработчик завершения инициализации"""
        logger.info("Клиент завершил инициализацию")
    
    @server.feature(types.TEXT_DOCUMENT_DID_OPEN)
    def did_open(params: types.DidOpenTextDocumentParams):
        """Обработчик открытия документа"""
        logger.debug(f"Открыт документ: {params.text_document.uri}")
    
    @server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(params: types.DidChangeTextDocumentParams):
        """Обработчик изменения документа"""
        # Записываем изменения в контекст
        for change in params.content_changes:
            if isinstance(change, types.TextDocumentContentChangeEvent_Type1):
                server.context_manager.record_edit(
                    params.text_document.uri,
                    change.range.start.line,
                    change.text,
                    "change"
                )
    
    @server.feature(types.TEXT_DOCUMENT_DID_SAVE)
    async def did_save(params: types.DidSaveTextDocumentParams):
        """Обработчик сохранения документа"""
        logger.debug(f"Сохранен документ: {params.text_document.uri}")
        
        # Переиндексируем файл
        file_path = params.text_document.uri.replace("file://", "")
        # TODO: Вызвать переиндексацию файла
    
    @server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
    def did_close(params: types.DidCloseTextDocumentParams):
        """Обработчик закрытия документа"""
        logger.debug(f"Закрыт документ: {params.text_document.uri}")
    
    @server.feature(types.SHUTDOWN)
    async def shutdown(params=None):
        """Обработчик завершения работы"""
        await server.shutdown()
        return None
    
    @server.feature(types.EXIT)
    def exit_server(params=None):
        """Обработчик выхода"""
        logger.info("Выход из сервера")
    
    # Дополнительные обработчики
    
    @server.feature(types.TEXT_DOCUMENT_CODE_ACTION)
    async def code_action(params: types.CodeActionParams) -> Optional[List[types.CodeAction]]:
        """Обработчик code actions"""
        if not server.config.features.code_actions:
            return None
        
        actions = []
        
        # Пример: добавить импорт
        actions.append(types.CodeAction(
            title="Добавить импорт",
            kind=types.CodeActionKind.QuickFix,
            command=types.Command(
                title="Добавить импорт",
                command="universal-lsp.addImport",
                arguments=[params.text_document.uri]
            )
        ))
        
        return actions
    
    @server.feature(types.TEXT_DOCUMENT_FORMATTING)
    async def formatting(params: types.DocumentFormattingParams) -> Optional[List[types.TextEdit]]:
        """Обработчик форматирования документа"""
        if not server.config.features.formatting:
            return None
        
        # TODO: Реализовать форматирование через black/autopep8
        return None
    
    @server.feature(types.TEXT_DOCUMENT_RENAME)
    async def rename(params: types.RenameParams) -> Optional[types.WorkspaceEdit]:
        """Обработчик переименования"""
        # TODO: Реализовать переименование
        return None
    
    @server.feature(types.TEXT_DOCUMENT_FOLDING_RANGE)
    async def folding_range(params: types.FoldingRangeParams) -> Optional[List[types.FoldingRange]]:
        """Обработчик сворачивания кода"""
        # TODO: Реализовать определение блоков для сворачивания
        return None
    
    # Кастомные команды
    
    @server.command("universal-lsp.getContext")
    async def get_context(file_path: str, format: str = "markdown") -> str:
        """Получить контекст файла для AI"""
        return server.context_manager.export_context(file_path, format)
    
    @server.command("universal-lsp.reindex")
    async def reindex_workspace():
        """Переиндексировать проект"""
        if server.indexing_task and not server.indexing_task.done():
            server.indexing_task.cancel()
        
        server.indexing_task = server.loop.create_task(server._index_workspace())
        return "Переиндексация запущена"
    
    @server.command("universal-lsp.getStats")
    async def get_stats() -> dict:
        """Получить статистику сервера"""
        stats = {
            "indexed_files": len(server.indexer.file_index),
            "total_symbols": len(server.indexer.symbol_index),
            "cache_enabled": server.cache_manager is not None,
            "workspace_folders": [str(f) for f in server.workspace_folders]
        }
        
        if server.workspace_context:
            stats["workspace"] = {
                "root": server.workspace_context.root_path,
                "files": server.workspace_context.total_files,
                "symbols": server.workspace_context.total_symbols
            }
        
        return stats