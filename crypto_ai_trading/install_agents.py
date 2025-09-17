#!/usr/bin/env python3
"""
🚀 АВТОМАТИЧЕСКАЯ УСТАНОВКА АГЕНТОВ ДЛЯ CRYPTO AI TRADING

Этот скрипт автоматически создает всех 8 агентов в правильном формате.
Агенты создаются как Markdown файлы в директории .claude/agents/
"""

import os
import sys
from pathlib import Path
import shutil

def check_claude_code_directory():
    """Проверяет, что мы в директории с Claude Code проектом"""
    current_dir = Path.cwd()
    
    # Проверяем наличие типичных файлов проекта
    indicators = [
        'config/config.yaml',
        'models/patchtst_unified.py',
        'data/feature_engineering.py',
        'CLAUDE.md'
    ]
    
    missing = []
    for indicator in indicators:
        if not (current_dir / indicator).exists():
            missing.append(indicator)
    
    if missing:
        print("⚠️  Предупреждение: не найдены некоторые файлы проекта:")
        for file in missing:
            print(f"   - {file}")
        print("\nВозможно, вы не в корне проекта crypto_ai_trading")
        
        response = input("\nПродолжить установку агентов? (y/N): ")
        if response.lower() not in ['y', 'yes', 'да']:
            print("Установка отменена.")
            return False
    
    return True

def create_agents_directory():
    """Создает директорию для агентов"""
    agents_dir = Path('.claude/agents')
    agents_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Создана директория: {agents_dir}")
    return agents_dir

def check_existing_agents():
    """Проверяет существующих агентов"""
    agents_dir = Path('.claude/agents')
    if not agents_dir.exists():
        return []
    
    existing = list(agents_dir.glob('*.md'))
    if existing:
        print(f"📋 Найдено существующих агентов: {len(existing)}")
        for agent in existing:
            print(f"   - {agent.name}")
        
        response = input("\nПерезаписать существующих агентов? (y/N): ")
        if response.lower() not in ['y', 'yes', 'да']:
            print("Установка отменена.")
            return None
    
    return existing

def install_agents():
    """Основная функция установки агентов"""
    
    print("🤖 АВТОМАТИЧЕСКАЯ УСТАНОВКА АГЕНТОВ")
    print("=" * 50)
    
    # Проверяем, что мы в правильной директории
    if not check_claude_code_directory():
        return False
    
    # Проверяем существующих агентов
    existing = check_existing_agents()
    if existing is None:
        return False
    
    # Создаем директорию
    agents_dir = create_agents_directory()
    
    # Список агентов для установки
    agents = [
        'crypto-architect.md',
        'crypto-data-engineer.md', 
        'crypto-trainer.md',
        'crypto-backtester.md',
        'crypto-debugger.md',
        'crypto-reviewer.md',
        'crypto-researcher.md',
        'crypto-devops.md'
    ]
    
    installed = 0
    errors = 0
    
    for agent_file in agents:
        source = Path(__file__).parent / '.claude' / 'agents' / agent_file
        target = agents_dir / agent_file
        
        try:
            if source.exists():
                shutil.copy2(source, target)
                print(f"✅ Установлен: {agent_file}")
                installed += 1
            else:
                print(f"❌ Не найден исходный файл: {agent_file}")
                errors += 1
        except Exception as e:
            print(f"❌ Ошибка установки {agent_file}: {e}")
            errors += 1
    
    # Результаты
    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТЫ УСТАНОВКИ:")
    print(f"✅ Установлено агентов: {installed}")
    if errors > 0:
        print(f"❌ Ошибок: {errors}")
    
    if installed > 0:
        print(f"\n🎉 АГЕНТЫ ГОТОВЫ К ИСПОЛЬЗОВАНИЮ!")
        print(f"📁 Расположение: {agents_dir.absolute()}")
        print(f"\n🚀 ИСПОЛЬЗОВАНИЕ:")
        print("   /agents - для просмотра агентов")
        print("   Агенты автоматически загрузятся в Claude Code")
        print("\n📋 ПРИМЕРЫ ВЫЗОВА:")
        print("   Используй задачи с ключевыми словами, и Claude выберет подходящего агента:")
        print("   - 'оптимизируй модель' → crypto-architect")
        print("   - 'проблемы с данными' → crypto-data-engineer")
        print("   - 'улучши обучение' → crypto-trainer")
        print("   - 'протестируй стратегию' → crypto-backtester")
        print("   - 'ошибка CUDA' → crypto-debugger")
    
    return installed > 0

def list_agents():
    """Показывает список установленных агентов"""
    agents_dir = Path('.claude/agents')
    
    if not agents_dir.exists():
        print("❌ Директория агентов не найдена. Запустите установку первыми.")
        return
    
    agents = list(agents_dir.glob('*.md'))
    
    if not agents:
        print("❌ Агенты не установлены. Запустите установку.")
        return
    
    print(f"📋 УСТАНОВЛЕННЫЕ АГЕНТЫ ({len(agents)}):")
    print("=" * 40)
    
    for agent_file in sorted(agents):
        name = agent_file.stem
        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                description = "Описание не найдено"
                
                # Ищем описание в YAML frontmatter
                in_frontmatter = False
                for line in lines:
                    line = line.strip()
                    if line == '---':
                        in_frontmatter = not in_frontmatter
                        continue
                    
                    if in_frontmatter and line.startswith('description:'):
                        description = line.split(':', 1)[1].strip().strip('"')
                        break
            
            print(f"🤖 {name}")
            print(f"   {description}")
            print()
        
        except Exception as e:
            print(f"❌ Ошибка чтения {agent_file}: {e}")

def main():
    """Главная функция"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--list', '-l']:
            list_agents()
            return
        elif sys.argv[1] in ['--help', '-h']:
            print("🤖 УСТАНОВКА АГЕНТОВ ДЛЯ CRYPTO AI TRADING")
            print("\nИспользование:")
            print("  python install_agents.py          - установить всех агентов")
            print("  python install_agents.py --list   - показать установленных агентов")
            print("  python install_agents.py --help   - эта справка")
            return
    
    install_agents()

if __name__ == "__main__":
    main()