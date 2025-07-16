#!/usr/bin/env python3
"""
Демонстрация работы LSP сервера с фокусом на файлы проекта
"""

import asyncio
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from enhanced_lsp_server import EnhancedPythonLSPServer, PythonASTAnalyzer
from lsprotocol import types
import json


async def demo_project_indexing():
    """Демонстрация индексации только файлов проекта"""
    print("\n🎯 Демонстрация работы Enhanced Python LSP Server")
    print("=" * 70)
    
    # Создаем LSP сервер
    server = EnhancedPythonLSPServer("demo-server", "v1.0.0")
    
    # Получаем список Python файлов в проекте (исключая venv)
    project_path = Path(__file__).parent.parent
    python_files = []
    
    for file in project_path.rglob("*.py"):
        # Исключаем файлы из виртуального окружения
        if "venv" not in str(file) and "__pycache__" not in str(file):
            python_files.append(file)
    
    print(f"\n📂 Найдено Python файлов в проекте: {len(python_files)}")
    
    # Индексируем каждый файл
    for file in python_files:
        print(f"\n📄 Индексация: {file.name}")
        await server.index_file(file)
        
        # Показываем информацию о файле
        if str(file) in server.project_index:
            file_index = server.project_index[str(file)]
            print(f"   ✓ Символов: {len(file_index.symbols)}")
            print(f"   ✓ Импортов: {len(file_index.imports)}")
            
            # Показываем найденные символы
            if file_index.symbols:
                print(f"   📍 Найденные символы:")
                for name, symbol in list(file_index.symbols.items())[:5]:
                    print(f"      - {symbol.kind.name}: {name}")
                    if symbol.signature:
                        print(f"        {symbol.signature}")
                        
    return server


async def demo_context_generation():
    """Демонстрация генерации контекста для LLM"""
    print("\n\n🧠 Демонстрация генерации контекста для LLM")
    print("=" * 70)
    
    server = EnhancedPythonLSPServer("context-demo", "v1.0.0")
    demo_file = Path(__file__).parent.parent / "demo_lsp_test.py"
    
    if demo_file.exists():
        await server.index_file(demo_file)
        
        # Симулируем позицию курсора в классе CryptoTrader
        uri = f"file://{demo_file}"
        
        # Позиция в методе get_balance (строка ~35)
        position = types.Position(line=35, character=10)
        
        print(f"\n📍 Генерация контекста для позиции:")
        print(f"   Файл: {demo_file.name}")
        print(f"   Строка: {position.line + 1}")
        print(f"   Символ: {position.character}")
        
        # Генерируем контекст
        context = server.generate_llm_context(uri, position)
        
        print(f"\n📝 Сгенерированный контекст для LLM:")
        print("-" * 70)
        print(context)
        print("-" * 70)
        
        # Показываем статистику контекста
        print(f"\n📊 Статистика контекста:")
        context_window = server.get_context_window(uri)
        print(f"   - Связанных файлов: {len(context_window.related_files)}")
        print(f"   - Последних изменений: {len(context_window.recent_edits)}")
        print(f"   - Макс. токенов: {context_window.max_tokens}")


async def demo_symbol_search():
    """Демонстрация поиска символов"""
    print("\n\n🔍 Демонстрация поиска символов")
    print("=" * 70)
    
    server = EnhancedPythonLSPServer("search-demo", "v1.0.0")
    
    # Индексируем демо файл
    demo_file = Path(__file__).parent.parent / "demo_lsp_test.py"
    if demo_file.exists():
        await server.index_file(demo_file)
        
        # Ищем символы
        search_queries = ["CryptoTrader", "get_balance", "calculate_position_size"]
        
        for query in search_queries:
            print(f"\n🔎 Поиск: '{query}'")
            
            if query in server.symbol_index:
                symbols = server.symbol_index[query]
                for symbol in symbols:
                    print(f"   ✓ Найдено: {symbol.kind.name} '{symbol.name}'")
                    print(f"     Файл: {Path(symbol.file_path).name}")
                    print(f"     Строка: {symbol.line + 1}")
                    if symbol.signature:
                        print(f"     Сигнатура: {symbol.signature}")
                    if symbol.docstring:
                        print(f"     Документация: {symbol.docstring[:60]}...")
            else:
                print(f"   ✗ Не найдено")


async def demo_import_analysis():
    """Демонстрация анализа импортов"""
    print("\n\n📦 Демонстрация анализа импортов")
    print("=" * 70)
    
    server = EnhancedPythonLSPServer("import-demo", "v1.0.0")
    
    # Индексируем файлы проекта
    project_path = Path(__file__).parent.parent
    for file in project_path.rglob("*.py"):
        if "venv" not in str(file) and "__pycache__" not in str(file):
            await server.index_file(file)
    
    print("\n📊 Граф импортов проекта:")
    for file_path, imports in server.import_graph.items():
        if imports and "venv" not in file_path:
            print(f"\n📄 {Path(file_path).name}:")
            for imp in sorted(imports)[:5]:  # Показываем первые 5
                print(f"   ← {imp}")
            if len(imports) > 5:
                print(f"   ... и еще {len(imports) - 5} импортов")


async def demo_completion_context():
    """Демонстрация контекста для автодополнения"""
    print("\n\n💡 Демонстрация контекста для автодополнения")
    print("=" * 70)
    
    server = EnhancedPythonLSPServer("completion-demo", "v1.0.0")
    demo_file = Path(__file__).parent.parent / "demo_lsp_test.py"
    
    if demo_file.exists():
        await server.index_file(demo_file)
        
        # Симулируем автодополнение
        uri = f"file://{demo_file}"
        
        # Позиция после "trader." (где должно появиться автодополнение)
        position = types.Position(line=65, character=12)  # После "trader."
        
        print(f"\n📍 Контекст для автодополнения:")
        print(f"   Позиция: строка {position.line + 1}, символ {position.character}")
        print(f"   Ожидается: методы класса CryptoTrader")
        
        # Генерируем предложения для автодополнения
        print(f"\n🔤 Доступные методы:")
        
        # Ищем методы класса CryptoTrader
        for symbol_name, symbols in server.symbol_index.items():
            for symbol in symbols:
                if symbol.parent == "CryptoTrader" and symbol.kind == types.SymbolKind.Method:
                    print(f"   • {symbol.name}")
                    if symbol.signature:
                        print(f"     {symbol.signature}")
                    if symbol.docstring:
                        print(f"     📝 {symbol.docstring[:60]}...")


async def main():
    """Главная функция демонстрации"""
    print("\n" + "🚀 " * 20)
    print("ДЕМОНСТРАЦИЯ ENHANCED PYTHON LSP SERVER")
    print("🚀 " * 20)
    
    try:
        # Запускаем демонстрации
        server = await demo_project_indexing()
        await demo_context_generation()
        await demo_symbol_search()
        await demo_import_analysis()
        await demo_completion_context()
        
        # Финальная статистика
        print("\n\n📊 ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 70)
        print(f"✓ Файлов проиндексировано: {server.stats['indexed_files']}")
        print(f"✓ Символов найдено: {server.stats['total_symbols']}")
        print(f"✓ Уникальных символов: {len(server.symbol_index)}")
        print(f"✓ Файлов с импортами: {len(server.import_graph)}")
        
        print("\n✨ LSP сервер работает корректно и готов к использованию!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
