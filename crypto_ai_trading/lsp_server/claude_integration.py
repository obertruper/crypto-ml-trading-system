#!/usr/bin/env python3
"""
Интеграционный модуль для работы LSP сервера с Claude Code
Предоставляет контекст и метаданные для улучшения понимания кода
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import time

class ClaudeLSPIntegration:
    """Интеграция LSP сервера с Claude Code"""
    
    def __init__(self):
        self.lsp_port = 2087
        self.project_root = Path("/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading")
        self.lsp_server_path = self.project_root / "lsp_server"
        
    def ensure_lsp_running(self) -> bool:
        """Проверяет и запускает LSP сервер если нужно"""
        # Проверяем, запущен ли сервер
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', self.lsp_port))
            sock.close()
            
            if result == 0:
                return True
                
        except:
            pass
            
        # Запускаем сервер
        print("🚀 Запуск LSP сервера...")
        subprocess.Popen([
            str(self.lsp_server_path / "venv" / "bin" / "python"),
            str(self.lsp_server_path / "start_lsp_service.py"),
            "start"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Ждем запуска
        time.sleep(3)
        return True
        
    def get_project_context(self) -> Dict[str, Any]:
        """Получает контекст проекта для Claude"""
        context = {
            "project_name": "crypto_ai_trading",
            "project_root": str(self.project_root),
            "main_components": {
                "models": {
                    "path": "models/",
                    "description": "ML модели (PatchTST, ансамбли)",
                    "key_files": ["patchtst.py", "ensemble.py"]
                },
                "data": {
                    "path": "data/",
                    "description": "Загрузка и обработка данных",
                    "key_files": ["data_loader.py", "feature_engineering.py"]
                },
                "trading": {
                    "path": "trading/",
                    "description": "Торговые стратегии и риск-менеджмент",
                    "key_files": ["signals.py", "risk_management.py", "position_manager.py"]
                },
                "training": {
                    "path": "training/",
                    "description": "Обучение и валидация моделей",
                    "key_files": ["trainer.py", "validator.py"]
                },
                "utils": {
                    "path": "utils/",
                    "description": "Утилиты и визуализация",
                    "key_files": ["metrics.py", "visualization.py", "logger.py"]
                }
            },
            "key_features": [
                "PatchTST архитектура для временных рядов",
                "100+ технических индикаторов",
                "Многозадачное обучение (цена + вероятности TP/SL)",
                "6 стратегий управления позициями",
                "Ансамблирование моделей",
                "PostgreSQL база данных на порту 5555"
            ],
            "current_focus": "Улучшение предсказаний и риск-менеджмента",
            "lsp_features": {
                "enabled": True,
                "port": self.lsp_port,
                "capabilities": [
                    "Автодополнение с контекстом",
                    "Навигация по определениям",
                    "Поиск использований",
                    "Анализ импортов и зависимостей",
                    "Генерация контекста для LLM"
                ]
            }
        }
        
        return context
        
    def get_file_context(self, file_path: str) -> Dict[str, Any]:
        """Получает контекст для конкретного файла"""
        # Здесь можно интегрироваться с LSP сервером через JSON-RPC
        # для получения детальной информации о файле
        
        file_info = {
            "path": file_path,
            "imports": [],
            "exports": [],
            "classes": [],
            "functions": [],
            "related_files": []
        }
        
        # TODO: Реализовать через LSP протокол
        
        return file_info
        
    def get_symbol_info(self, symbol_name: str) -> Optional[Dict[str, Any]]:
        """Получает информацию о символе"""
        # TODO: Интеграция с LSP для получения информации о символе
        pass
        
    def get_recommendations(self, context: str) -> List[str]:
        """Генерирует рекомендации на основе контекста"""
        recommendations = []
        
        if "model" in context.lower():
            recommendations.extend([
                "Используйте models/patchtst.py для работы с архитектурой модели",
                "Проверьте config/config.yaml для настройки параметров модели",
                "Для обучения используйте python main.py --mode train"
            ])
            
        if "data" in context.lower():
            recommendations.extend([
                "Данные загружаются через data/data_loader.py",
                "Признаки создаются в data/feature_engineering.py",
                "База данных PostgreSQL на порту 5555"
            ])
            
        if "strategy" in context.lower() or "trading" in context.lower():
            recommendations.extend([
                "Стратегии реализованы в trading/signals.py",
                "Риск-менеджмент в trading/risk_management.py",
                "6 различных стратегий управления позициями"
            ])
            
        return recommendations

def setup_claude_integration():
    """Настраивает интеграцию с Claude Code"""
    integration = ClaudeLSPIntegration()
    
    # Проверяем LSP сервер
    if integration.ensure_lsp_running():
        print("✅ LSP сервер запущен")
    else:
        print("❌ Не удалось запустить LSP сервер")
        return False
        
    # Получаем контекст проекта
    context = integration.get_project_context()
    
    # Сохраняем контекст для Claude
    context_file = integration.project_root / ".claude_context.json"
    with open(context_file, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Контекст проекта сохранен в {context_file}")
    
    # Выводим информацию
    print("\n📊 Проект crypto_ai_trading готов к работе с Claude Code!")
    print(f"📁 Корневая директория: {integration.project_root}")
    print(f"🔧 LSP сервер: порт {integration.lsp_port}")
    print("\n🚀 Ключевые команды:")
    print("  - python main.py --mode demo     # Демо режим")
    print("  - python main.py --mode train    # Обучение модели")
    print("  - python main.py --mode backtest # Бэктестинг")
    
    return True

if __name__ == "__main__":
    setup_claude_integration()