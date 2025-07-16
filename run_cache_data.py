#!/usr/bin/env python3
"""
Скрипт для запуска кэширования данных на сервере
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    """Парсинг аргументов"""
    parser = argparse.ArgumentParser(description="Run Data Caching")
    
    parser.add_argument('--all-symbols', action='store_true',
                       help='Кэшировать данные по всем символам')
    
    parser.add_argument('--test-symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Символы для тестового режима')
    
    parser.add_argument('--force-refresh', action='store_true',
                       help='Принудительно обновить кеш')
    
    parser.add_argument('--skip-sequences', action='store_true',
                       help='Не создавать последовательности')
    
    return parser.parse_args()


def main():
    """Главная функция"""
    args = parse_args()
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║               Data Caching for Transformer v3            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    # Определяем режим
    if args.all_symbols:
        print("🌍 Режим: Кэширование ВСЕХ символов")
        print("⚠️  Внимание: Это займет много времени и места!")
        confirm = input("Продолжить? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Отменено пользователем")
            return
    else:
        print(f"🧪 Режим: Кэширование тестовых символов: {args.test_symbols}")
    
    if args.force_refresh:
        print("🔄 Принудительное обновление кеша")
    
    if args.skip_sequences:
        print("⏩ Пропуск создания последовательностей")
    
    print()
    
    # Формируем команду
    cmd_parts = ["python", "transformer_v3/cache_all_data.py"]
    
    if args.all_symbols:
        cmd_parts.append("--all-symbols")
    else:
        cmd_parts.extend(["--test-symbols"] + args.test_symbols)
    
    if args.force_refresh:
        cmd_parts.append("--force-refresh")
    
    if args.skip_sequences:
        cmd_parts.append("--skip-sequences")
    
    # Запускаем
    cmd = " ".join(cmd_parts)
    print(f"🚀 Выполняю: {cmd}")
    print()
    
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("\n🎉 Кэширование завершено успешно!")
        print("📁 Кеш сохранен в: cache/transformer_v3/")
        print("🚀 Теперь обучение может работать автономно")
    else:
        print(f"\n❌ Кэширование завершилось с ошибкой (код: {exit_code})")


if __name__ == "__main__":
    main()