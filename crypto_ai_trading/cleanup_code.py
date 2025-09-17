#!/usr/bin/env python3
"""
Скрипт для поиска и удаления неиспользуемого кода
"""
import ast
import os
from pathlib import Path
from typing import Set, Dict, List
import re

def find_unused_imports(file_path: str) -> List[str]:
    """Найти неиспользуемые импорты в файле"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Синтаксическая ошибка в {file_path}")
        return []

    imported_names = set()
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_modules.add(name)
                imported_names.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names.add(name)

    # Проверяем использование
    unused = []
    for name in imported_names:
        # Простая проверка - ищем имя в коде
        pattern = r'\b' + re.escape(name) + r'\b'
        # Исключаем строку импорта из поиска
        code_without_imports = '\n'.join([line for line in content.split('\n')
                                         if not line.strip().startswith('import ')
                                         and not line.strip().startswith('from ')])

        if not re.search(pattern, code_without_imports):
            unused.append(name)

    return unused

def find_duplicate_functions():
    """Найти дублирующиеся функции между файлами"""
    duplicates = {}

    # Сравниваем StagedTrainingManager и StagedTrainer
    print("\n📊 Сравнение StagedTrainingManager (main.py) и StagedTrainer (staged_trainer.py):")
    print("-" * 60)

    # Методы в StagedTrainingManager
    manager_methods = [
        'get_stage_config',
        'run_staged_training',
        '_recursive_update'
    ]

    # Методы в StagedTrainer
    trainer_methods = [
        '_create_stage_config',
        'train',
        '_configure_losses'
    ]

    print("StagedTrainingManager методы:")
    for m in manager_methods:
        print(f"  - {m}")

    print("\nStagedTrainer методы:")
    for m in trainer_methods:
        print(f"  - {m}")

    print("\n🔍 Анализ:")
    print("- get_stage_config и _create_stage_config делают похожую работу")
    print("- run_staged_training и train оба запускают поэтапное обучение")
    print("⚠️ Рекомендация: удалить StagedTrainingManager и использовать только StagedTrainer")

    return duplicates

def check_file_sizes():
    """Проверить размеры файлов"""
    print("\n📏 Размеры основных файлов:")
    print("-" * 60)

    files = [
        'main.py',
        'training/staged_trainer.py',
        'training/optimized_trainer.py',
        'models/patchtst_unified.py',
        'data/feature_engineering.py'
    ]

    for file_path in files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"{file_path:40} {lines:6} строк, {size:8.2f} KB")

    print("\n⚠️ main.py слишком большой (1878 строк)! Нужен рефакторинг.")

def find_test_files():
    """Найти тестовые файлы"""
    print("\n🧪 Тестовые файлы (можно удалить):")
    print("-" * 60)

    test_patterns = [
        'test_*.py',
        'debug_*.py',
        'check_*.py',
        'fix_*.py',
        'temp_*.py'
    ]

    test_files = []
    for pattern in test_patterns:
        test_files.extend(Path('.').glob(pattern))

    for file in sorted(test_files):
        size = os.path.getsize(file) / 1024
        print(f"  {file}: {size:.2f} KB")

    total_size = sum(os.path.getsize(f) for f in test_files) / 1024
    print(f"\n  Всего: {len(test_files)} файлов, {total_size:.2f} KB")

    return test_files

def main():
    print("=" * 60)
    print("🔍 АНАЛИЗ КОДА НА ДУБЛИКАТЫ И НЕИСПОЛЬЗУЕМЫЕ ЭЛЕМЕНТЫ")
    print("=" * 60)

    # 1. Проверяем неиспользуемые импорты в main.py
    print("\n📦 Неиспользуемые импорты в main.py:")
    print("-" * 60)
    unused = find_unused_imports('main.py')
    if unused:
        for name in unused:
            print(f"  - {name}")
    else:
        print("  Все импорты используются")

    # 2. Находим дублирующиеся функции
    find_duplicate_functions()

    # 3. Проверяем размеры файлов
    check_file_sizes()

    # 4. Находим тестовые файлы
    test_files = find_test_files()

    # 5. Рекомендации
    print("\n✅ РЕКОМЕНДАЦИИ:")
    print("-" * 60)
    print("1. Удалить StagedTrainingManager из main.py (строки 104-220)")
    print("2. Заменить использование StagedTrainingManager на StagedTrainer")
    print("3. Удалить тестовые файлы после проверки")
    print("4. Разделить main.py на модули:")
    print("   - data_preparation.py")
    print("   - model_factory.py")
    print("   - inference.py")
    print("5. Оставить в main.py только точку входа и аргументы")

    print("\n" + "=" * 60)
    print("✅ Анализ завершен!")
    print("=" * 60)

if __name__ == "__main__":
    main()