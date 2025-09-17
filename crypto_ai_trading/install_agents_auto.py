#!/usr/bin/env python3
"""
🚀 АВТОМАТИЧЕСКАЯ УСТАНОВКА АГЕНТОВ (БЕЗ ВЗАИМОДЕЙСТВИЯ)

Быстрая установка всех 8 агентов для проекта crypto_ai_trading
"""

import os
import sys
from pathlib import Path
import shutil

def install_agents_auto():
    """Автоматическая установка агентов без запросов"""
    
    print("🤖 АВТОМАТИЧЕСКАЯ УСТАНОВКА АГЕНТОВ")
    print("=" * 50)
    
    # Создаем директорию
    agents_dir = Path('.claude/agents')
    agents_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Директория: {agents_dir}")
    
    # Список агентов
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
    
    # Копируем файлы
    for agent_file in agents:
        source = Path(__file__).parent / '.claude' / 'agents' / agent_file
        target = agents_dir / agent_file
        
        try:
            if source.exists():
                shutil.copy2(source, target)
                print(f"✅ {agent_file}")
                installed += 1
            else:
                print(f"❌ Не найден: {agent_file}")
        except Exception as e:
            print(f"❌ Ошибка {agent_file}: {e}")
    
    # Результат
    print(f"\n📊 РЕЗУЛЬТАТ: Установлено {installed} агентов")
    
    if installed > 0:
        print(f"\n🎉 АГЕНТЫ ГОТОВЫ!")
        print(f"📁 Расположение: {agents_dir.absolute()}")
        print(f"\n🚀 ПРОВЕРКА:")
        print("   Введите команду: /agents")
        print("   Агенты должны появиться в списке")
        
        return True
    
    return False

def list_agents():
    """Список установленных агентов"""
    agents_dir = Path('.claude/agents')
    
    if not agents_dir.exists():
        print("❌ Агенты не найдены")
        return
    
    agents = list(agents_dir.glob('*.md'))
    print(f"\n📋 НАЙДЕНО АГЕНТОВ: {len(agents)}")
    
    for agent in sorted(agents):
        name = agent.stem
        print(f"🤖 {name}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_agents()
    else:
        if install_agents_auto():
            list_agents()