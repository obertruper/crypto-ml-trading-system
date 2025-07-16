#!/usr/bin/env python3
"""
Интеграция LSP с Sequential Thinking для Claude Code
Создает думающую модель с глубоким анализом кода
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from mcp_lsp_bridge import get_bridge, MCPLSPBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThinkingStep:
    """Шаг размышления"""
    step_number: int
    thought: str
    analysis: Dict[str, Any]
    conclusions: List[str]
    next_actions: List[str]

class ThinkingLSPIntegration:
    """Интеграция LSP с последовательным мышлением"""
    
    def __init__(self):
        self.bridge = get_bridge()
        self.project_root = Path("/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading")
        self.thinking_steps: List[ThinkingStep] = []
        
    async def analyze_file_with_thinking(self, file_path: str) -> Dict[str, Any]:
        """Анализирует файл с последовательным мышлением"""
        logger.info(f"🧠 Начинаю анализ файла с мышлением: {file_path}")
        
        # Шаг 1: Получаем базовый контекст
        step1 = await self._think_step_1_context(file_path)
        self.thinking_steps.append(step1)
        
        # Шаг 2: Анализируем зависимости
        step2 = await self._think_step_2_dependencies(file_path, step1)
        self.thinking_steps.append(step2)
        
        # Шаг 3: Понимаем назначение
        step3 = await self._think_step_3_purpose(file_path, step1, step2)
        self.thinking_steps.append(step3)
        
        # Шаг 4: Выявляем потенциальные проблемы
        step4 = await self._think_step_4_issues(file_path, step1, step2, step3)
        self.thinking_steps.append(step4)
        
        # Шаг 5: Формируем рекомендации
        step5 = await self._think_step_5_recommendations(file_path, self.thinking_steps)
        self.thinking_steps.append(step5)
        
        return self._compile_analysis()
        
    async def _think_step_1_context(self, file_path: str) -> ThinkingStep:
        """Шаг 1: Получение контекста файла"""
        context = self.bridge.get_file_context(file_path)
        
        analysis = {
            "file_exists": context["exists"],
            "file_type": context.get("file_type", "unknown"),
            "last_modified": context.get("last_modified"),
            "related_files_count": len(context.get("related_files", [])),
            "recent_changes": len(context.get("recent_changes", []))
        }
        
        conclusions = []
        if context["exists"]:
            conclusions.append(f"Файл существует и имеет тип {analysis['file_type']}")
        else:
            conclusions.append("Файл не существует")
            
        if analysis["related_files_count"] > 0:
            conclusions.append(f"Файл связан с {analysis['related_files_count']} другими файлами")
            
        if analysis["recent_changes"] > 0:
            conclusions.append("Файл недавно изменялся")
            
        next_actions = ["Проанализировать зависимости", "Понять назначение файла"]
        
        return ThinkingStep(
            step_number=1,
            thought="Получаю базовую информацию о файле",
            analysis=analysis,
            conclusions=conclusions,
            next_actions=next_actions
        )
        
    async def _think_step_2_dependencies(self, file_path: str, prev_step: ThinkingStep) -> ThinkingStep:
        """Шаг 2: Анализ зависимостей"""
        context = self.bridge.get_file_context(file_path)
        
        analysis = {
            "imports": context.get("imports", []),
            "exports": context.get("exports", []),
            "related_files": context.get("related_files", []),
            "dependency_graph": self._build_dependency_graph(file_path, context)
        }
        
        conclusions = []
        if analysis["imports"]:
            conclusions.append(f"Файл импортирует {len(analysis['imports'])} модулей")
            
        if analysis["exports"]:
            conclusions.append(f"Файл экспортирует {len(analysis['exports'])} символов")
            
        if analysis["related_files"]:
            conclusions.append("Найдены связанные файлы: " + ", ".join(analysis["related_files"][:3]))
            
        next_actions = ["Определить роль в проекте", "Проверить консистентность"]
        
        return ThinkingStep(
            step_number=2,
            thought="Анализирую зависимости и связи",
            analysis=analysis,
            conclusions=conclusions,
            next_actions=next_actions
        )
        
    async def _think_step_3_purpose(self, file_path: str, step1: ThinkingStep, step2: ThinkingStep) -> ThinkingStep:
        """Шаг 3: Понимание назначения"""
        path = Path(file_path)
        context = self.bridge.get_file_context(file_path)
        
        analysis = {
            "directory": path.parent.name,
            "filename": path.name,
            "usage_hints": context.get("usage_hints", []),
            "inferred_purpose": self._infer_purpose(path, context)
        }
        
        conclusions = []
        if analysis["inferred_purpose"]:
            conclusions.append(f"Назначение: {analysis['inferred_purpose']}")
            
        if analysis["usage_hints"]:
            conclusions.extend(analysis["usage_hints"][:2])
            
        next_actions = ["Проверить на проблемы", "Сформировать рекомендации"]
        
        return ThinkingStep(
            step_number=3,
            thought="Определяю назначение и роль файла",
            analysis=analysis,
            conclusions=conclusions,
            next_actions=next_actions
        )
        
    async def _think_step_4_issues(self, file_path: str, *prev_steps) -> ThinkingStep:
        """Шаг 4: Выявление потенциальных проблем"""
        context = self.bridge.get_file_context(file_path)
        
        issues = []
        
        # Проверка на частые изменения
        recent_changes = context.get("recent_changes", [])
        if len(recent_changes) > 3:
            issues.append("Файл часто изменяется, возможна нестабильность")
            
        # Проверка на отсутствие связей
        if not context.get("related_files"):
            issues.append("Файл изолирован, нет явных связей с другими модулями")
            
        # Специфичные проверки для разных типов
        path = Path(file_path)
        if context.get("file_type") == "yaml" and path.name == "config.yaml":
            issues.append("Изменения конфигурации влияют на весь проект")
            
        analysis = {
            "issues_found": len(issues),
            "severity": "high" if len(issues) > 2 else "medium" if issues else "low",
            "issues": issues
        }
        
        conclusions = []
        if issues:
            conclusions.append(f"Найдено {len(issues)} потенциальных проблем")
            conclusions.extend(issues[:2])
        else:
            conclusions.append("Проблем не обнаружено")
            
        next_actions = ["Сформировать итоговые рекомендации"]
        
        return ThinkingStep(
            step_number=4,
            thought="Проверяю на потенциальные проблемы",
            analysis=analysis,
            conclusions=conclusions,
            next_actions=next_actions
        )
        
    async def _think_step_5_recommendations(self, file_path: str, all_steps: List[ThinkingStep]) -> ThinkingStep:
        """Шаг 5: Формирование рекомендаций"""
        recommendations = []
        
        # Анализируем все предыдущие шаги
        for step in all_steps[:-1]:
            if "config.yaml" in file_path and step.step_number == 3:
                recommendations.append("Проверьте все зависимые модули после изменения конфигурации")
                
            if step.step_number == 4 and step.analysis.get("severity") == "high":
                recommendations.append("Требуется внимательная проверка изменений")
                
        # Добавляем специфичные рекомендации
        context = self.bridge.get_file_context(file_path)
        if context.get("file_type") == "python":
            recommendations.append("Запустите тесты после изменений")
            recommendations.append("Проверьте импорты и экспорты")
            
        analysis = {
            "total_recommendations": len(recommendations),
            "priority": "high" if len(recommendations) > 3 else "medium",
            "recommendations": recommendations
        }
        
        conclusions = [
            f"Сформировано {len(recommendations)} рекомендаций",
            "Анализ завершен"
        ]
        
        return ThinkingStep(
            step_number=5,
            thought="Формирую итоговые рекомендации",
            analysis=analysis,
            conclusions=conclusions,
            next_actions=[]
        )
        
    def _build_dependency_graph(self, file_path: str, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Строит граф зависимостей"""
        graph = {
            "imports_from": context.get("imports", []),
            "imported_by": [],  # TODO: реализовать обратный поиск
            "related_to": context.get("related_files", [])
        }
        return graph
        
    def _infer_purpose(self, path: Path, context: Dict[str, Any]) -> str:
        """Определяет назначение файла"""
        if path.name == "config.yaml":
            return "Главный конфигурационный файл проекта"
        elif "models" in path.parts:
            return "Определение ML модели"
        elif "data" in path.parts:
            return "Обработка и загрузка данных"
        elif "trading" in path.parts:
            return "Торговая логика и стратегии"
        elif "utils" in path.parts:
            return "Вспомогательные утилиты"
        else:
            return "Общего назначения"
            
    def _compile_analysis(self) -> Dict[str, Any]:
        """Компилирует результаты анализа"""
        return {
            "thinking_process": [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "conclusions": step.conclusions,
                    "next_actions": step.next_actions
                }
                for step in self.thinking_steps
            ],
            "final_analysis": {
                "total_steps": len(self.thinking_steps),
                "key_findings": self._extract_key_findings(),
                "recommendations": self._extract_recommendations(),
                "risk_level": self._assess_risk_level()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    def _extract_key_findings(self) -> List[str]:
        """Извлекает ключевые находки"""
        findings = []
        for step in self.thinking_steps:
            findings.extend(step.conclusions[:2])
        return findings[:5]  # Топ-5 находок
        
    def _extract_recommendations(self) -> List[str]:
        """Извлекает все рекомендации"""
        for step in self.thinking_steps:
            if step.step_number == 5:
                return step.analysis.get("recommendations", [])
        return []
        
    def _assess_risk_level(self) -> str:
        """Оценивает уровень риска"""
        for step in self.thinking_steps:
            if step.step_number == 4:
                return step.analysis.get("severity", "low")
        return "unknown"

async def analyze_with_thinking(file_path: str) -> Dict[str, Any]:
    """Главная функция для анализа с мышлением"""
    integration = ThinkingLSPIntegration()
    return await integration.analyze_file_with_thinking(file_path)

if __name__ == "__main__":
    # Тест
    import asyncio
    
    async def test():
        result = await analyze_with_thinking(
            "/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/config/config.yaml"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    asyncio.run(test())