#!/usr/bin/env python3
"""
Быстрый запуск LSP сервера без установки
Просто запустите: ./quickstart.py start
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Добавляем текущую директорию в Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Проверка и установка зависимостей"""
    try:
        import pygls
        import jedi
        import yaml
        print("✅ Все зависимости установлены")
        return True
    except ImportError:
        print("📦 Установка зависимостей...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Зависимости установлены")
        return True

def start_server(args):
    """Запуск LSP сервера"""
    print(f"🚀 Запуск LSP сервера на порту {args.port}...")
    
    # Импортируем после проверки зависимостей
    from lsp_server.cli import start_command
    
    # Запускаем сервер
    start_command(
        host=args.host,
        port=args.port,
        project=args.project or os.getcwd(),
        debug=args.debug
    )

def main():
    parser = argparse.ArgumentParser(description="Universal LSP Server - Быстрый запуск")
    
    subparsers = parser.add_subparsers(dest="command", help="Команды")
    
    # Команда start
    start_parser = subparsers.add_parser("start", help="Запустить сервер")
    start_parser.add_argument("--port", type=int, default=3000, help="Порт сервера")
    start_parser.add_argument("--host", default="127.0.0.1", help="Хост сервера")
    start_parser.add_argument("--project", help="Путь к проекту (по умолчанию текущая директория)")
    start_parser.add_argument("--debug", action="store_true", help="Debug режим")
    
    # Команда check
    check_parser = subparsers.add_parser("check", help="Проверить зависимости")
    
    args = parser.parse_args()
    
    if not args.command:
        print("Universal LSP Server - Быстрый запуск")
        print("\nИспользование:")
        print("  ./quickstart.py start       - Запустить сервер")
        print("  ./quickstart.py check       - Проверить зависимости")
        print("\nДля подробной справки: ./quickstart.py --help")
        return
    
    if args.command == "check" or args.command == "start":
        if not check_dependencies():
            sys.exit(1)
    
    if args.command == "start":
        start_server(args)

if __name__ == "__main__":
    # Делаем файл исполняемым
    if os.name != 'nt':  # Unix/Linux/macOS
        os.chmod(__file__, 0o755)
    
    main()