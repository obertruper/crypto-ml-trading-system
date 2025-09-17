#!/usr/bin/env python3
"""
🤖 АВТОМАТИЧЕСКАЯ НАСТРОЙКА АГЕНТОВ ДЛЯ CRYPTO AI TRADING

Этот скрипт помогает создать всех 8 специализированных агентов для проекта.
Поскольку Claude Code требует интерактивного создания агентов через /agents,
скрипт выводит готовые конфигурации для копирования.
"""

import json
import os
from pathlib import Path

# Определяем путь к конфигурациям агентов
AGENTS_DIR = Path(__file__).parent / ".claude" / "agents"

def load_agent_configs():
    """Загружает конфигурации всех агентов из JSON файлов"""
    
    agents = []
    
    if not AGENTS_DIR.exists():
        print(f"❌ Директория {AGENTS_DIR} не найдена!")
        return agents
    
    # Список файлов агентов в правильном порядке
    agent_files = [
        "crypto-architect.json",
        "crypto-data-engineer.json", 
        "crypto-trainer.json",
        "crypto-backtester.json",
        "crypto-debugger.json",
        "crypto-reviewer.json",
        "crypto-researcher.json",
        "crypto-devops.json"
    ]
    
    for filename in agent_files:
        file_path = AGENTS_DIR / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    agents.append(config)
                    print(f"✅ Загружен: {config['name']}")
            except Exception as e:
                print(f"❌ Ошибка загрузки {filename}: {e}")
        else:
            print(f"⚠️  Файл не найден: {filename}")
    
    return agents

def display_creation_instructions():
    """Выводит пошаговые инструкции по созданию агентов"""
    
    print("\n" + "="*80)
    print("🚀 ПОШАГОВОЕ СОЗДАНИЕ АГЕНТОВ В CLAUDE CODE")
    print("="*80)
    
    print("""
📋 ДЛЯ СОЗДАНИЯ КАЖДОГО АГЕНТА:

1. Введите команду: /agents
2. Выберите "Create new agent" 
3. Скопируйте данные из секций ниже для каждого агента:
   - Name (имя)
   - Description (описание)
   - System Prompt (системный промпт)

4. Нажмите "Create Agent"

⚠️  ВАЖНО: Создавайте агентов в указанном порядке для лучшей организации!
""")

def print_agent_config(agent_config, index):
    """Выводит конфигурацию агента для создания"""
    
    print(f"\n{'='*20} АГЕНТ {index}/8 {'='*20}")
    print(f"🤖 {agent_config['name']}")
    print("="*60)
    
    print(f"\n📝 NAME:")
    print(f"{agent_config['name']}")
    
    print(f"\n📝 DESCRIPTION:")
    print(f"{agent_config['description']}")
    
    print(f"\n📝 SYSTEM PROMPT:")
    print("-" * 40)
    print(agent_config['system_prompt'])
    print("-" * 40)
    
    print(f"\n✅ После создания этого агента нажмите Enter для следующего...")
    input()

def main():
    """Основная функция"""
    
    print("🤖 НАСТРОЙКА АГЕНТОВ ДЛЯ CRYPTO AI TRADING SYSTEM")
    print("="*60)
    
    # Загружаем конфигурации
    agents = load_agent_configs()
    
    if not agents:
        print("❌ Не найдено конфигураций агентов!")
        return
    
    print(f"\n✅ Найдено {len(agents)} агентов для создания")
    
    # Показываем общие инструкции
    display_creation_instructions()
    
    print("Нажмите Enter чтобы начать пошаговое создание агентов...")
    input()
    
    # Выводим каждого агента по очереди
    for i, agent in enumerate(agents, 1):
        print_agent_config(agent, i)
    
    print("\n🎉 ВСЕ АГЕНТЫ ГОТОВЫ К СОЗДАНИЮ!")
    print("\n📋 КРАТКАЯ СВОДКА СОЗДАННЫХ АГЕНТОВ:")
    for i, agent in enumerate(agents, 1):
        print(f"{i}. {agent['name']} - {agent['description']}")
    
    print(f"\n🚀 ПОСЛЕ СОЗДАНИЯ ВСЕХ АГЕНТОВ:")
    print("- Используйте AGENT_USAGE_GUIDE.md для работы с агентами")
    print("- Пример: @crypto-architect оптимизируй модель для RTX 5090")
    print("- Проверьте созданных агентов командой: /agents")

def show_quick_reference():
    """Показывает быструю справку по агентам"""
    
    agents = load_agent_configs()
    
    print("\n📚 БЫСТРАЯ СПРАВКА ПО АГЕНТАМ:")
    print("="*50)
    
    for agent in agents:
        name = agent['name']
        desc = agent['description']
        print(f"• {name:<25} - {desc}")
    
    print("\nИспользование: @agent-name ваша задача")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_quick_reference()
    else:
        main()