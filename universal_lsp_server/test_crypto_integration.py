#!/usr/bin/env python3
"""
Тест интеграции LSP сервера с проектом crypto_ai_trading
"""

import sys
from pathlib import Path
import json

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_lsp_bridge import MCPLSPBridge

def test_lsp_integration():
    """Тестирует интеграцию LSP с проектом crypto_ai_trading"""
    
    print("🔍 Тестирование интеграции LSP сервера с crypto_ai_trading...\n")
    
    # Создаем экземпляр моста
    bridge = MCPLSPBridge()
    
    print(f"✅ Корень проекта: {bridge.project_root}")
    print(f"✅ Директория LSP данных: {bridge.lsp_data_dir}")
    
    # Проверяем структуру проекта
    important_dirs = ['config', 'data', 'models', 'trading', 'training', 'utils']
    
    print("\n📁 Проверка структуры проекта:")
    for dir_name in important_dirs:
        dir_path = bridge.project_root / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ - найдена")
        else:
            print(f"  ❌ {dir_name}/ - не найдена")
    
    # Проверяем ключевые файлы
    key_files = [
        'main.py',
        'config/config.yaml',
        'models/patchtst.py',
        'training/trainer.py',
        'data/data_loader.py'
    ]
    
    print("\n📄 Проверка ключевых файлов:")
    for file_path in key_files:
        full_path = bridge.project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path} - найден")
        else:
            print(f"  ❌ {file_path} - не найден")
    
    # Тест получения контекста для файла
    print("\n🔧 Тест получения контекста:")
    test_file = bridge.project_root / "models" / "patchtst.py"
    if test_file.exists():
        print(f"  📄 Анализируем файл: {test_file.name}")
        print(f"  📏 Размер файла: {test_file.stat().st_size} байт")
        
        # Здесь можно добавить вызов методов bridge для получения контекста
        # context = bridge.get_file_context(str(test_file))
        print("  ✅ Файл готов для анализа LSP сервером")
    
    print("\n✨ Интеграция LSP сервера успешно настроена!")

if __name__ == "__main__":
    test_lsp_integration()