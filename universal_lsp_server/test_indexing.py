#!/usr/bin/env python3
"""
Расширенный тест индексации и функциональности LSP сервера
"""

import asyncio
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from enhanced_lsp_server import EnhancedPythonLSPServer, PythonASTAnalyzer, ContextWindow
from lsprotocol import types


async def test_workspace_indexing():
    """Тест индексации рабочего пространства"""
    print("\n🔍 Тестирование индексации проекта...")
    
    # Создаем LSP сервер
    server = EnhancedPythonLSPServer("test-indexing", "v1.0.0")
    
    # Индексируем родительскую директорию (crypto_ai_trading)
    project_path = Path(__file__).parent.parent
    print(f"📂 Индексация проекта: {project_path}")
    
    await server.index_workspace(project_path)
    
    print(f"\n📊 Статистика индексации:")
    print(f"  - Файлов проиндексировано: {server.stats['indexed_files']}")
    print(f"  - Символов найдено: {server.stats['total_symbols']}")
    print(f"  - Файлов в индексе: {len(server.project_index)}")
    
    # Показываем проиндексированные файлы
    print(f"\n📄 Проиндексированные файлы:")
    for file_path, file_index in server.project_index.items():
        print(f"  - {Path(file_path).name}")
        print(f"    Символов: {len(file_index.symbols)}")
        print(f"    Импортов: {len(file_index.imports)}")
        
    # Показываем найденные символы
    print(f"\n🔤 Найденные символы (первые 10):")
    symbol_count = 0
    for symbol_name, symbol_list in server.symbol_index.items():
        if symbol_count >= 10:
            break
        for symbol in symbol_list:
            print(f"  - {symbol.kind.name}: {symbol_name}")
            if symbol.signature:
                print(f"    {symbol.signature}")
            symbol_count += 1
            if symbol_count >= 10:
                break
                
    return server


async def test_context_generation():
    """Тест генерации контекста для LLM"""
    print("\n🧠 Тестирование генерации контекста...")
    
    # Создаем сервер и индексируем
    server = EnhancedPythonLSPServer("test-context", "v1.0.0")
    demo_file = Path(__file__).parent.parent / "demo_lsp_test.py"
    
    if demo_file.exists():
        await server.index_file(demo_file)
        
        # Генерируем контекст для позиции в файле
        uri = f"file://{demo_file}"
        position = types.Position(line=30, character=0)  # Позиция в классе CryptoTrader
        
        context = server.generate_llm_context(uri, position)
        
        print(f"\n📝 Сгенерированный контекст для LLM:")
        print("-" * 60)
        print(context)
        print("-" * 60)
    else:
        print("❌ Демо файл не найден")


async def test_symbol_analysis():
    """Тест анализа символов в конкретном файле"""
    print("\n🔬 Тестирование анализа символов...")
    
    # Анализируем сам LSP сервер
    lsp_file = Path(__file__).parent / "enhanced_lsp_server.py"
    
    if lsp_file.exists():
        content = lsp_file.read_text(encoding='utf-8')
        
        # Подсчитываем статистику
        lines = content.split('\n')
        classes = len([l for l in lines if l.strip().startswith('class ')])
        functions = len([l for l in lines if l.strip().startswith('def ') or l.strip().startswith('async def ')])
        
        print(f"\n📈 Статистика файла {lsp_file.name}:")
        print(f"  - Строк кода: {len(lines)}")
        print(f"  - Классов: ~{classes}")
        print(f"  - Функций/методов: ~{functions}")
        
        # Анализируем с помощью AST
        import ast
        tree = ast.parse(content, filename=str(lsp_file))
        analyzer = PythonASTAnalyzer(str(lsp_file))
        analyzer.visit(tree)
        
        print(f"\n🎯 Точный анализ AST:")
        print(f"  - Символов найдено: {len(analyzer.symbols)}")
        print(f"  - Импортов: {len(analyzer.imports)}")
        
        # Показываем классы
        print(f"\n📦 Найденные классы:")
        for name, symbol in analyzer.symbols.items():
            if symbol.kind == types.SymbolKind.Class:
                print(f"  - {name}")
                if symbol.docstring:
                    print(f"    📝 {symbol.docstring[:60]}...")


async def test_import_graph():
    """Тест построения графа импортов"""
    print("\n🕸️  Тестирование графа импортов...")
    
    server = EnhancedPythonLSPServer("test-imports", "v1.0.0")
    project_path = Path(__file__).parent.parent
    
    await server.index_workspace(project_path)
    
    print(f"\n📊 Граф импортов:")
    for file_path, imports in server.import_graph.items():
        if imports:
            print(f"\n  📄 {Path(file_path).name}:")
            for imp in list(imports)[:5]:  # Показываем первые 5
                print(f"    ⬅️  {imp}")


async def test_context_window():
    """Тест работы с контекстным окном"""
    print("\n🪟 Тестирование контекстного окна...")
    
    server = EnhancedPythonLSPServer("test-window", "v1.0.0")
    
    # Создаем контекстное окно
    test_uri = "file:///test/example.py"
    context_window = server.get_context_window(test_uri)
    
    print(f"\n📋 Контекстное окно создано:")
    print(f"  - Текущий файл: {context_window.current_file}")
    print(f"  - Макс. токенов: {context_window.max_tokens}")
    print(f"  - Связанных файлов: {len(context_window.related_files)}")
    print(f"  - Последних изменений: {len(context_window.recent_edits)}")


async def main():
    """Главная функция для запуска всех тестов"""
    print("=" * 80)
    print("🚀 Расширенное тестирование Enhanced Python LSP Server")
    print("=" * 80)
    
    try:
        # Запускаем тесты последовательно
        server = await test_workspace_indexing()
        await test_symbol_analysis()
        await test_context_generation()
        await test_import_graph()
        await test_context_window()
        
        print("\n" + "=" * 80)
        print("✅ Все тесты успешно завершены!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
