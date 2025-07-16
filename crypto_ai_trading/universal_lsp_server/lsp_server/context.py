"""
Менеджер контекста для Universal LSP Server
Управляет контекстным окном для оптимальной работы с AI
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import deque
import json

from .config import ContextConfig
from .indexer import SymbolInfo, FileIndex

logger = logging.getLogger(__name__)

@dataclass
class CompletionContext:
    """Контекст для автодополнения"""
    current_file: str
    line: int
    column: int
    prefix: str = ""
    local_symbols: List[SymbolInfo] = field(default_factory=list)
    imported_symbols: List[SymbolInfo] = field(default_factory=list)
    available_imports: List[str] = field(default_factory=list)
    recent_symbols: List[SymbolInfo] = field(default_factory=list)

@dataclass
class FileContext:
    """Контекст файла"""
    path: str
    symbols: List[SymbolInfo]
    imports: List[str]
    exports: List[str]
    dependencies: Set[str]
    related_files: List[str] = field(default_factory=list)

@dataclass
class WorkspaceContext:
    """Контекст всего проекта"""
    root_path: str
    total_files: int
    total_symbols: int
    file_tree: Dict[str, Any] = field(default_factory=dict)
    symbol_graph: Dict[str, List[str]] = field(default_factory=dict)
    import_graph: Dict[str, List[str]] = field(default_factory=dict)

class ContextManager:
    """Менеджер контекста для LSP"""
    
    def __init__(self, config: ContextConfig):
        self.config = config
        self.file_contexts: Dict[str, FileContext] = {}
        self.workspace_context: Optional[WorkspaceContext] = None
        self.recent_edits: deque = deque(maxlen=config.max_recent_edits)
        self.symbol_usage_stats: Dict[str, int] = {}
        
    def update_from_index(self, index_data: Dict[str, Any]):
        """Обновить контекст из данных индекса"""
        # Обновляем контекст рабочего пространства
        if not self.workspace_context:
            self.workspace_context = WorkspaceContext(
                root_path=index_data.get('root_path', ''),
                total_files=index_data.get('total_files', 0),
                total_symbols=index_data.get('total_symbols', 0)
            )
        else:
            self.workspace_context.total_files = index_data.get('total_files', 0)
            self.workspace_context.total_symbols = index_data.get('total_symbols', 0)
    
    def add_file_context(self, file_index: FileIndex):
        """Добавить контекст файла"""
        context = FileContext(
            path=file_index.path,
            symbols=list(file_index.symbols.values()),
            imports=file_index.imports,
            exports=file_index.exports,
            dependencies=file_index.dependencies
        )
        
        self.file_contexts[file_index.path] = context
        
        # Обновляем граф импортов
        if self.workspace_context:
            self.workspace_context.import_graph[file_index.path] = file_index.imports
    
    async def get_completion_context(self, file_path: str, line: int, column: int) -> CompletionContext:
        """Получить контекст для автодополнения"""
        context = CompletionContext(
            current_file=file_path,
            line=line,
            column=column
        )
        
        # Получаем контекст текущего файла
        if file_path in self.file_contexts:
            file_ctx = self.file_contexts[file_path]
            
            # Локальные символы
            for symbol in file_ctx.symbols:
                if symbol.line <= line:  # Символы определенные до текущей позиции
                    context.local_symbols.append(symbol)
            
            # Импортированные символы
            # TODO: Разрешить импорты и добавить символы
            
        # Добавляем недавно использованные символы
        for symbol_name, count in sorted(
            self.symbol_usage_stats.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]:
            # TODO: Найти символ по имени и добавить в recent_symbols
            pass
        
        return context
    
    def get_file_context(self, file_path: str) -> Optional[FileContext]:
        """Получить контекст файла"""
        return self.file_contexts.get(file_path)
    
    def get_related_files(self, file_path: str, max_files: Optional[int] = None) -> List[str]:
        """Получить связанные файлы"""
        if max_files is None:
            max_files = self.config.max_related_files
        
        related = set()
        
        if file_path in self.file_contexts:
            context = self.file_contexts[file_path]
            
            # Добавляем файлы из импортов
            for imp in context.imports:
                # TODO: Разрешить импорт в путь к файлу
                pass
            
            # Добавляем файлы, которые импортируют текущий
            for other_path, other_ctx in self.file_contexts.items():
                if file_path in other_ctx.dependencies:
                    related.add(other_path)
        
        # Сортируем по релевантности
        # TODO: Реализовать более умную сортировку
        related_list = list(related)[:max_files]
        
        return related_list
    
    def record_edit(self, file_path: str, line: int, text: str, edit_type: str = "change"):
        """Записать изменение для контекста"""
        edit = {
            'file': file_path,
            'line': line,
            'text': text[:100],  # Ограничиваем размер
            'type': edit_type,
            'timestamp': None  # TODO: Добавить timestamp
        }
        
        self.recent_edits.append(edit)
    
    def record_symbol_usage(self, symbol_name: str):
        """Записать использование символа"""
        self.symbol_usage_stats[symbol_name] = self.symbol_usage_stats.get(symbol_name, 0) + 1
    
    def build_file_tree(self, root_path: Path) -> Dict[str, Any]:
        """Построить дерево файлов"""
        tree = {}
        
        for file_path in self.file_contexts:
            path = Path(file_path)
            
            # Получаем относительный путь
            try:
                rel_path = path.relative_to(root_path)
            except ValueError:
                continue
            
            # Строим дерево
            current = tree
            parts = rel_path.parts
            
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Это файл
                    current[part] = {
                        'type': 'file',
                        'symbols': len(self.file_contexts[file_path].symbols),
                        'size': path.stat().st_size if path.exists() else 0
                    }
                else:
                    # Это директория
                    if part not in current:
                        current[part] = {
                            'type': 'directory',
                            'children': {}
                        }
                    current = current[part]['children']
        
        return tree
    
    def export_context(self, file_path: str, format: str = "markdown") -> str:
        """Экспортировать контекст для AI"""
        if format == "markdown":
            return self._export_markdown_context(file_path)
        elif format == "json":
            return self._export_json_context(file_path)
        else:
            return self._export_xml_context(file_path)
    
    def _export_markdown_context(self, file_path: str) -> str:
        """Экспорт контекста в Markdown формате"""
        lines = []
        
        # Заголовок
        lines.append(f"# Контекст для файла: {Path(file_path).name}")
        lines.append("")
        
        # Информация о файле
        if file_path in self.file_contexts:
            ctx = self.file_contexts[file_path]
            lines.append("## Текущий файл")
            lines.append(f"- Путь: `{file_path}`")
            lines.append(f"- Символов: {len(ctx.symbols)}")
            lines.append(f"- Импортов: {len(ctx.imports)}")
            lines.append(f"- Экспортов: {len(ctx.exports)}")
            lines.append("")
            
            # Символы
            if ctx.symbols:
                lines.append("### Символы в файле:")
                for symbol in ctx.symbols[:20]:  # Ограничиваем количество
                    icon = self._get_symbol_icon(symbol.kind)
                    lines.append(f"- {icon} `{symbol.name}` ({symbol.kind}) - строка {symbol.line + 1}")
                    if symbol.docstring:
                        lines.append(f"  > {symbol.docstring.split(chr(10))[0][:80]}...")
                lines.append("")
        
        # Связанные файлы
        related = self.get_related_files(file_path)
        if related:
            lines.append("## Связанные файлы")
            for rel_path in related[:5]:
                lines.append(f"- `{Path(rel_path).name}`")
            lines.append("")
        
        # Недавние изменения
        if self.recent_edits:
            lines.append("## Недавние изменения")
            for edit in list(self.recent_edits)[-5:]:
                lines.append(f"- {edit['type']} в `{Path(edit['file']).name}` строка {edit['line']}")
            lines.append("")
        
        # Структура проекта (если включена)
        if self.config.include_file_tree and self.workspace_context:
            lines.append("## Структура проекта")
            lines.append("```")
            lines.extend(self._format_file_tree(
                self.build_file_tree(Path(self.workspace_context.root_path))
            ))
            lines.append("```")
        
        return "\n".join(lines)
    
    def _export_json_context(self, file_path: str) -> str:
        """Экспорт контекста в JSON формате"""
        data = {
            'current_file': file_path,
            'file_context': None,
            'related_files': [],
            'recent_edits': list(self.recent_edits),
            'workspace': None
        }
        
        if file_path in self.file_contexts:
            ctx = self.file_contexts[file_path]
            data['file_context'] = {
                'path': ctx.path,
                'symbols_count': len(ctx.symbols),
                'imports': ctx.imports,
                'exports': ctx.exports,
                'symbols': [
                    {
                        'name': s.name,
                        'kind': s.kind,
                        'line': s.line,
                        'signature': s.signature
                    }
                    for s in ctx.symbols[:50]
                ]
            }
        
        data['related_files'] = self.get_related_files(file_path)
        
        if self.workspace_context:
            data['workspace'] = {
                'root': self.workspace_context.root_path,
                'total_files': self.workspace_context.total_files,
                'total_symbols': self.workspace_context.total_symbols
            }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _export_xml_context(self, file_path: str) -> str:
        """Экспорт контекста в XML формате"""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<context>')
        lines.append(f'  <current_file>{file_path}</current_file>')
        
        if file_path in self.file_contexts:
            ctx = self.file_contexts[file_path]
            lines.append('  <file_info>')
            lines.append(f'    <symbols_count>{len(ctx.symbols)}</symbols_count>')
            lines.append(f'    <imports_count>{len(ctx.imports)}</imports_count>')
            lines.append(f'    <exports_count>{len(ctx.exports)}</exports_count>')
            lines.append('  </file_info>')
            
            lines.append('  <symbols>')
            for symbol in ctx.symbols[:20]:
                lines.append(f'    <symbol kind="{symbol.kind}" line="{symbol.line}">')
                lines.append(f'      <name>{symbol.name}</name>')
                if symbol.signature:
                    lines.append(f'      <signature>{symbol.signature}</signature>')
                lines.append('    </symbol>')
            lines.append('  </symbols>')
        
        lines.append('</context>')
        
        return '\n'.join(lines)
    
    def _get_symbol_icon(self, kind: str) -> str:
        """Получить иконку для типа символа"""
        icons = {
            'class': '🏛️',
            'function': '🔧',
            'method': '⚙️',
            'variable': '📦',
            'import': '📥',
            'module': '📁',
            'constant': '🔒',
            'property': '🏷️'
        }
        return icons.get(kind, '•')
    
    def _format_file_tree(self, tree: Dict[str, Any], prefix: str = "") -> List[str]:
        """Форматировать дерево файлов"""
        lines = []
        items = sorted(tree.items())
        
        for i, (name, info) in enumerate(items):
            is_last = i == len(items) - 1
            
            if isinstance(info, dict) and info.get('type') == 'directory':
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}{name}/")
                
                if 'children' in info:
                    extension = "    " if is_last else "│   "
                    lines.extend(self._format_file_tree(
                        info['children'], 
                        prefix + extension
                    ))
            else:
                symbol_info = f" ({info.get('symbols', 0)} symbols)" if info.get('symbols') else ""
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}{name}{symbol_info}")
        
        return lines