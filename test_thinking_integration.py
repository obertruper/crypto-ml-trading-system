#!/usr/bin/env python3
"""
Тестирование thinking интеграции для ML Crypto Trading
"""

import asyncio
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

from universal_lsp_server.thinking_lsp_integration import ThinkingLSPIntegration
from universal_lsp_server.mcp_lsp_bridge import MCPLSPBridge

async def test_thinking_analysis():
    """Тестирует анализ важных файлов проекта с thinking"""
    
    print("🧠 Тестирование Sequential Thinking для ML Crypto Trading\n")
    
    # Инициализируем интеграцию
    thinking = ThinkingLSPIntegration()
    bridge = MCPLSPBridge()
    
    # Важные файлы для анализа
    important_files = [
        "train_universal_transformer.py",
        "models/patchtst.py", 
        "data/feature_engineering.py",
        "trading/signals.py"
    ]
    
    for file_name in important_files:
        file_path = Path(__file__).parent / file_name
        
        if not file_path.exists():
            print(f"❌ Файл не найден: {file_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"📄 Анализ файла: {file_name}")
        print(f"{'='*60}")
        
        try:
            # Получаем контекст через bridge
            context = bridge.get_file_context(str(file_path))
            print(f"\n📊 Контекст файла:")
            print(f"   - Импорты: {len(context.get('imports', []))}")
            print(f"   - Функции: {len(context.get('functions', []))}")
            print(f"   - Классы: {len(context.get('classes', []))}")
            
            # Анализируем с thinking
            print(f"\n🤔 Запуск Sequential Thinking анализа...")
            analysis = await thinking.analyze_file_with_thinking(str(file_path))
            
            # Выводим результаты мышления
            if 'thinking_steps' in analysis:
                for step in analysis['thinking_steps']:
                    print(f"\n💭 Шаг {step['step_number']}: {step['thought']}")
                    if 'conclusions' in step:
                        for conclusion in step['conclusions']:
                            print(f"   ✓ {conclusion}")
            
            # Итоговые рекомендации
            if 'recommendations' in analysis:
                print(f"\n📝 Рекомендации:")
                for rec in analysis['recommendations']:
                    print(f"   • {rec}")
                    
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()

async def test_project_overview():
    """Тестирует получение обзора проекта"""
    
    print("\n\n🏗️ ОБЗОР ПРОЕКТА ML CRYPTO TRADING")
    print("="*60)
    
    bridge = MCPLSPBridge()
    
    # Статистика проекта
    project_root = Path(__file__).parent
    # Исключаем venv и другие системные папки
    py_files = [f for f in project_root.rglob("*.py") 
                if 'venv' not in str(f) and '__pycache__' not in str(f)]
    yaml_files = [f for f in project_root.rglob("*.yaml") 
                  if 'venv' not in str(f)]
    
    print(f"\n📊 Статистика:")
    print(f"   - Python файлов: {len(py_files)}")
    print(f"   - YAML конфигураций: {len(yaml_files)}")
    print(f"   - Размер проекта: {sum(f.stat().st_size for f in py_files) / 1024 / 1024:.1f} MB")
    
    # Анализ изменений
    recent_changes = bridge.get_recent_changes(limit=10)  # Последние 10 изменений
    if recent_changes:
        print(f"\n📝 Последние изменения:")
        for change in recent_changes[:5]:
            print(f"   - {change.path}: {change.change_type} ({change.timestamp})")

async def main():
    """Главная функция"""
    
    # Тест 1: Анализ файлов с thinking
    await test_thinking_analysis()
    
    # Тест 2: Обзор проекта
    await test_project_overview()
    
    print("\n\n✅ Тестирование завершено!")
    print("\n💡 Подсказка: Используйте thinking анализ перед важными изменениями в коде")

if __name__ == "__main__":
    # Проверяем наличие необходимых файлов
    if not Path("universal_lsp_server/thinking_lsp_integration.py").exists():
        print("❌ Файлы thinking интеграции не найдены!")
        print("   Убедитесь, что вы находитесь в корне проекта")
        sys.exit(1)
        
    asyncio.run(main())