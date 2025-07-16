#!/usr/bin/env python3
"""
Автоматический хук для Claude Code
Интегрирует LSP с Sequential Thinking
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Добавляем путь к LSP серверу
sys.path.insert(0, str(Path(__file__).parent.parent / "lsp_server"))

from mcp_lsp_bridge import get_bridge
from thinking_lsp_integration import analyze_with_thinking
import asyncio

class ClaudeThinkingHook:
    """Хук для автоматического анализа с мышлением"""
    
    def __init__(self):
        self.bridge = get_bridge()
        self.last_analyzed_file = None
        self.analysis_cache = {}
        
    def before_file_operation(self, file_path: str, operation: str) -> Dict[str, Any]:
        """Вызывается перед любой операцией с файлом"""
        print(f"\n🧠 Анализирую файл перед операцией '{operation}': {file_path}\n")
        
        # Получаем контекст через LSP
        context = self.bridge.get_file_context(file_path)
        
        # Если файл важный, делаем глубокий анализ
        if self._is_important_file(file_path):
            print("📊 Выполняю глубокий анализ с последовательным мышлением...\n")
            analysis = asyncio.run(analyze_with_thinking(file_path))
            self.analysis_cache[file_path] = analysis
            
            # Выводим ключевые выводы
            print("🔍 Ключевые находки:")
            for finding in analysis["final_analysis"]["key_findings"][:3]:
                print(f"  • {finding}")
                
            print("\n💡 Рекомендации:")
            for rec in analysis["final_analysis"]["recommendations"]:
                print(f"  • {rec}")
                
            print(f"\n⚠️  Уровень риска: {analysis['final_analysis']['risk_level']}")
            
        else:
            print("ℹ️  Базовый контекст получен")
            
        print("\n" + "="*50 + "\n")
        
        return {
            "context": context,
            "analysis": self.analysis_cache.get(file_path),
            "should_proceed": True
        }
        
    def after_file_operation(self, file_path: str, operation: str, success: bool):
        """Вызывается после операции с файлом"""
        if success:
            # Отслеживаем изменение
            change = self.bridge.track_file_change(Path(file_path))
            if change:
                print(f"✅ Изменение записано: {change.change_type}")
                
    def _is_important_file(self, file_path: str) -> bool:
        """Определяет, является ли файл важным для глубокого анализа"""
        path = Path(file_path)
        
        # Важные файлы
        important_files = [
            "config.yaml",
            "main.py",
            "patchtst.py",
            "trainer.py",
            "data_loader.py"
        ]
        
        # Важные директории
        important_dirs = ["models", "trading", "training"]
        
        return (
            path.name in important_files or
            any(dir_name in path.parts for dir_name in important_dirs)
        )

# Глобальный экземпляр хука
_hook = ClaudeThinkingHook()

def claude_before_edit(file_path: str) -> Dict[str, Any]:
    """Хук перед редактированием файла"""
    return _hook.before_file_operation(file_path, "edit")

def claude_before_read(file_path: str) -> Dict[str, Any]:
    """Хук перед чтением файла"""
    return _hook.before_file_operation(file_path, "read")

def claude_after_edit(file_path: str, success: bool):
    """Хук после редактирования файла"""
    _hook.after_file_operation(file_path, "edit", success)

def claude_get_project_context() -> Dict[str, Any]:
    """Получает общий контекст проекта"""
    summary = _hook.bridge.get_project_summary()
    recent_changes = _hook.bridge.get_recent_changes(5)
    
    return {
        "project_summary": summary,
        "recent_changes": recent_changes,
        "thinking_enabled": True,
        "lsp_active": True
    }

if __name__ == "__main__":
    # Тест хука
    print("🧪 Тестирование хука Claude Thinking\n")
    
    test_file = "/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/config/config.yaml"
    
    # Тест перед чтением
    result = claude_before_read(test_file)
    print(f"\nРезультат анализа: {result['should_proceed']}")
    
    # Тест контекста проекта
    project = claude_get_project_context()
    print(f"\nСтатус проекта: LSP={project['lsp_active']}, Thinking={project['thinking_enabled']}")