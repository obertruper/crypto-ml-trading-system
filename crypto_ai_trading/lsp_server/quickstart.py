#!/usr/bin/env python3
"""
Скрипт быстрого старта Enhanced Python LSP Server
"""

import subprocess
import sys
import os

def main():
    print("🚀 Enhanced Python LSP Server - Быстрый старт")
    print("=" * 60)
    
    # Проверяем виртуальное окружение
    venv_path = os.path.join(os.path.dirname(__file__), 'venv')
    if not os.path.exists(venv_path):
        print("❌ Виртуальное окружение не найдено!")
        print("   Выполните: python3 -m venv venv")
        sys.exit(1)
        
    # Проверяем установленные зависимости
    print("📦 Проверка зависимостей...")
    python_path = os.path.join(venv_path, 'bin', 'python')
    
    try:
        result = subprocess.run(
            [python_path, '-c', 'import pygls, lsprotocol; print("✅ Основные зависимости установлены")'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("❌ Зависимости не установлены!")
            print("   Выполните: ./venv/bin/pip install -r requirements.txt")
            sys.exit(1)
        print(result.stdout.strip())
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)
        
    print("\n📋 Конфигурация LSP сервера:")
    print(f"   - Путь: {os.path.abspath('enhanced_lsp_server.py')}")
    print(f"   - Python: {python_path}")
    print(f"   - Режим: stdio")
    
    print("\n🔧 Для интеграции с VS Code:")
    print("   1. Скопируйте .vscode/settings.json в корень вашего проекта")
    print("   2. Перезапустите VS Code")
    
    print("\n▶️  Команда для запуска:")
    print(f"   {python_path} enhanced_lsp_server.py --stdio")
    
    print("\n✨ LSP сервер готов к использованию!")
    print("=" * 60)

if __name__ == "__main__":
    main()
