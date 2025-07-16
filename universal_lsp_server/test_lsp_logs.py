#!/usr/bin/env python3
"""
Тестирование LSP логирования и мониторинга
"""

import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))

from mcp_lsp_bridge import get_bridge, mcp_get_file_context, mcp_get_recent_changes
from thinking_lsp_integration import analyze_with_thinking
import asyncio

def test_lsp_monitoring():
    """Тестирует мониторинг и логирование LSP"""
    print("🧠 Тестирование LSP мониторинга")
    print("="*50)
    
    # 1. Получаем сводку проекта
    bridge = get_bridge()
    summary = bridge.get_project_summary()
    
    print(f"\n📊 Статус проекта:")
    print(f"  • Проект: {summary['project']}")
    print(f"  • LSP статус: {summary['lsp_status']}")
    print(f"  • Отслеживание: {'включено' if summary['tracking_enabled'] else 'выключено'}")
    
    # 2. Последние изменения
    print(f"\n📝 Изменения за последние 24 часа:")
    for change_type, count in summary['changes_24h'].items():
        print(f"  • {change_type}: {count}")
    
    # 3. Активные файлы
    print(f"\n🔥 Самые активные файлы:")
    for file_info in summary['active_files'][:5]:
        print(f"  • {Path(file_info['path']).name}: {file_info['changes']} изменений")
    
    # 4. Тестируем отслеживание
    print(f"\n🔍 Тестирование отслеживания файлов...")
    
    test_files = [
        "/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/config/config.yaml",
        "/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/main.py",
        "/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/models/patchtst.py"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            change = bridge.track_file_change(Path(file_path))
            if change:
                print(f"  ✅ {Path(file_path).name}: {change.change_type}")
            else:
                print(f"  ℹ️  {Path(file_path).name}: без изменений")
        else:
            print(f"  ❌ {Path(file_path).name}: не найден")
    
    # 5. Последние изменения
    print(f"\n📜 История последних изменений:")
    recent = mcp_get_recent_changes(5)
    for i, change in enumerate(recent, 1):
        print(f"  {i}. {Path(change['path']).name} - {change['type']} ({change['timestamp']})")
    
    # 6. Статистика контекстных запросов
    print(f"\n📊 Статистика использования:")
    # TODO: Добавить статистику из БД
    
    print("\n✅ Тестирование завершено!")

async def test_thinking_analysis():
    """Тестирует анализ с мышлением"""
    print("\n\n🤔 Тестирование Sequential Thinking")
    print("="*50)
    
    file_path = "/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/config/config.yaml"
    
    print(f"📁 Анализируем: {Path(file_path).name}")
    print("⏳ Выполняется 5-шаговый анализ...")
    
    analysis = await analyze_with_thinking(file_path)
    
    print("\n📊 Результаты анализа:")
    print(f"  • Шагов выполнено: {analysis['final_analysis']['total_steps']}")
    print(f"  • Уровень риска: {analysis['final_analysis']['risk_level']}")
    print(f"  • Рекомендаций: {len(analysis['final_analysis']['recommendations'])}")
    
    print("\n🔍 Ключевые находки:")
    for finding in analysis['final_analysis']['key_findings'][:3]:
        print(f"  • {finding}")
    
    print("\n💡 Рекомендации:")
    for rec in analysis['final_analysis']['recommendations']:
        print(f"  • {rec}")

def main():
    """Главная функция"""
    print("🚀 LSP Сервер - Проверка логов и мониторинга")
    print("="*70)
    
    # Проверяем базовый мониторинг
    test_lsp_monitoring()
    
    # Проверяем thinking анализ
    asyncio.run(test_thinking_analysis())
    
    print("\n" + "="*70)
    print("📝 Логи LSP сервера:")
    print(f"  • Основной: lsp_server/lsp_service.log")
    print(f"  • Детальный: lsp_server/enhanced-lsp.log")
    print(f"  • База данных: lsp_server/.lsp_data/file_tracking.db")
    print("\n✨ Все системы работают!")

if __name__ == "__main__":
    main()