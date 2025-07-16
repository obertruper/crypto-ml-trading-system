#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы Enhanced Python LSP Server
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from enhanced_lsp_server import EnhancedPythonLSPServer, PythonASTAnalyzer
from pathlib import Path


def test_ast_analyzer():
    """Тест анализатора AST"""
    print("🧪 Тестирование AST анализатора...")
    
    test_code = '''
class TestClass:
    """Тестовый класс"""
    
    def __init__(self, name: str):
        self.name = name
        
    def greet(self) -> str:
        """Приветствие"""
        return f"Hello, {self.name}!"

def test_function(x: int, y: int) -> int:
    """Тестовая функция"""
    return x + y
'''
    
    # Создаем временный файл
    test_file = Path("test_temp.py")
    test_file.write_text(test_code)
    
    try:
        # Анализируем файл
        import ast
        tree = ast.parse(test_code, filename=str(test_file))
        analyzer = PythonASTAnalyzer(str(test_file))
        analyzer.visit(tree)
        
        print(f"✅ Найдено символов: {len(analyzer.symbols)}")
        for name, symbol in analyzer.symbols.items():
            print(f"  - {symbol.kind.name}: {name}")
            if symbol.signature:
                print(f"    Сигнатура: {symbol.signature}")
                
    finally:
        # Удаляем временный файл
        if test_file.exists():
            test_file.unlink()
            

def test_lsp_server():
    """Тест создания LSP сервера"""
    print("\n🧪 Тестирование LSP сервера...")
    
    try:
        server = EnhancedPythonLSPServer("test-server", "v1.0.0")
        print("✅ LSP сервер создан успешно")
        print(f"  - Название: {server.name}")
        print(f"  - Версия: {server.version}")
        print(f"  - Расширения для индексации: {server.index_extensions}")
        print(f"  - Лимит токенов: {server.context_token_limit}")
        
    except Exception as e:
        print(f"❌ Ошибка создания сервера: {e}")
        

def test_dependencies():
    """Проверка установленных зависимостей"""
    print("\n🧪 Проверка зависимостей...")
    
    dependencies = [
        'pygls',
        'lsprotocol',
        'aiofiles',
        'watchdog',
        'jedi',
        'rope',
        'pylint',
        'black',
        'isort',
        'docstring_parser',
        'tiktoken',
        'jsonschema',
        'yaml'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            print(f"✅ {dep}")
        except ImportError:
            missing.append(dep)
            print(f"❌ {dep} - не установлен")
            
    if missing:
        print(f"\n⚠️  Отсутствуют зависимости: {', '.join(missing)}")
        print("   Выполните: pip install -r requirements.txt")
    else:
        print("\n✅ Все зависимости установлены")
        

def main():
    """Главная функция тестирования"""
    print("=" * 60)
    print("Enhanced Python LSP Server - Тестирование")
    print("=" * 60)
    
    test_dependencies()
    test_ast_analyzer()
    test_lsp_server()
    
    print("\n" + "=" * 60)
    print("Тестирование завершено!")
    print("=" * 60)
    

if __name__ == "__main__":
    main()
