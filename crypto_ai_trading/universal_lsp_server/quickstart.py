#!/usr/bin/env python3
"""
Быстрый запуск Universal LSP Server
Можно использовать без установки пакета
"""

import sys
import os
from pathlib import Path

# Добавляем текущую директорию в PYTHONPATH
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Импортируем и запускаем CLI
from lsp_server.cli import main

if __name__ == "__main__":
    main()