#!/usr/bin/env python3
"""
Настройка и запуск Sequential Thinking MCP для проекта ML Crypto Trading
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def setup_mcp_config():
    """Создаёт конфигурацию MCP для Claude Desktop"""
    
    # Путь к конфигу Claude Desktop
    config_dir = Path.home() / ".config" / "claude"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "claude_desktop_config.json"
    
    # Базовая конфигурация
    config = {
        "mcpServers": {
            "sequential-thinking": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
                "env": {}
            },
            "crypto-trading-context": {
                "command": "python3",
                "args": [
                    str(Path(__file__).parent / "universal_lsp_server" / "mcp_server.py")
                ],
                "env": {
                    "PROJECT_ROOT": str(Path(__file__).parent),
                    "THINKING_MODE": "enabled"
                }
            }
        }
    }
    
    # Если файл существует, добавляем к существующей конфигурации
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
            existing_config.update(config)
            config = existing_config
        except:
            pass
    
    # Сохраняем конфигурацию
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ MCP конфигурация сохранена в: {config_file}")
    return config_file

def create_mcp_server():
    """Создаёт локальный MCP сервер для проекта"""
    
    server_path = Path(__file__).parent / "universal_lsp_server" / "mcp_server.py"
    
    server_code = '''#!/usr/bin/env python3
"""
MCP Server для ML Crypto Trading Project
Предоставляет контекст и thinking capabilities
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from universal_lsp_server.mcp_lsp_bridge import MCPLSPBridge
from universal_lsp_server.thinking_lsp_integration import ThinkingLSPIntegration

class CryptoTradingMCPServer:
    """MCP сервер для crypto trading проекта"""
    
    def __init__(self):
        self.bridge = MCPLSPBridge()
        self.thinking = ThinkingLSPIntegration()
        self.context_cache = {}
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает запросы MCP"""
        
        method = request.get("method", "")
        params = request.get("params", {})
        
        if method == "tools/list":
            return await self.list_tools()
        elif method == "tools/call":
            return await self.call_tool(params)
        elif method == "resources/list":
            return await self.list_resources()
        elif method == "resources/read":
            return await self.read_resource(params)
        else:
            return {"error": f"Unknown method: {method}"}
    
    async def list_tools(self) -> Dict[str, Any]:
        """Список доступных инструментов"""
        return {
            "tools": [
                {
                    "name": "analyze_file",
                    "description": "Анализирует файл с последовательным мышлением",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Путь к файлу"}
                        },
                        "required": ["path"]
                    }
                },
                {
                    "name": "get_project_context",
                    "description": "Получает контекст проекта для файла",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Путь к файлу"}
                        },
                        "required": ["path"]
                    }
                },
                {
                    "name": "analyze_trading_strategy",
                    "description": "Анализирует торговую стратегию",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "strategy_name": {"type": "string"}
                        },
                        "required": ["strategy_name"]
                    }
                }
            ]
        }
    
    async def call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Вызывает инструмент"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "analyze_file":
            file_path = arguments.get("path")
            result = await self.thinking.analyze_file_with_thinking(file_path)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
            
        elif tool_name == "get_project_context":
            file_path = arguments.get("path")
            context = self.bridge.get_file_context(file_path)
            return {"content": [{"type": "text", "text": json.dumps(context, indent=2)}]}
            
        elif tool_name == "analyze_trading_strategy":
            strategy_name = arguments.get("strategy_name")
            # Анализ стратегии
            analysis = await self.analyze_strategy(strategy_name)
            return {"content": [{"type": "text", "text": json.dumps(analysis, indent=2)}]}
            
        return {"error": f"Unknown tool: {tool_name}"}
    
    async def list_resources(self) -> Dict[str, Any]:
        """Список доступных ресурсов"""
        return {
            "resources": [
                {
                    "uri": "crypto-trading://project-overview",
                    "name": "Project Overview",
                    "mimeType": "text/markdown"
                },
                {
                    "uri": "crypto-trading://important-files",
                    "name": "Important Files",
                    "mimeType": "application/json"
                }
            ]
        }
    
    async def read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Читает ресурс"""
        uri = params.get("uri", "")
        
        if uri == "crypto-trading://project-overview":
            overview = self.get_project_overview()
            return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": overview}]}
            
        elif uri == "crypto-trading://important-files":
            files = self.get_important_files()
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(files, indent=2)}]}
            
        return {"error": f"Unknown resource: {uri}"}
    
    def get_project_overview(self) -> str:
        """Возвращает обзор проекта"""
        return """# ML Crypto Trading Project
        
## Основные компоненты:
- **PatchTST модель** - архитектура для временных рядов
- **100+ технических индикаторов**
- **PostgreSQL база данных** на порту 5555
- **50 криптовалютных пар** с историческими данными

## Ключевые файлы:
- `train_universal_transformer.py` - главный файл обучения
- `models/patchtst.py` - архитектура модели
- `data/feature_engineering.py` - инженерия признаков
- `trading/signals.py` - торговые стратегии

## Текущий статус:
- ✅ База данных инициализирована
- ✅ Исторические данные загружены (4.7M записей)
- ⏳ Готово к обучению модели
"""
    
    def get_important_files(self) -> List[Dict[str, str]]:
        """Возвращает список важных файлов"""
        return [
            {"path": "train_universal_transformer.py", "description": "Главный файл обучения TFT"},
            {"path": "models/patchtst.py", "description": "Архитектура PatchTST"},
            {"path": "config/config.yaml", "description": "Конфигурация системы"},
            {"path": "data/feature_engineering.py", "description": "Расчёт технических индикаторов"},
            {"path": "trading/signals.py", "description": "Торговые стратегии"},
            {"path": "CLAUDE.md", "description": "Правила и документация проекта"}
        ]
    
    async def analyze_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Анализирует торговую стратегию"""
        # Здесь будет логика анализа стратегии
        return {
            "strategy": strategy_name,
            "analysis": "Strategy analysis would go here",
            "recommendations": []
        }

async def main():
    """Главная функция сервера"""
    server = CryptoTradingMCPServer()
    
    # Читаем запросы из stdin
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
                
            request = json.loads(line)
            response = await server.handle_request(request)
            
            print(json.dumps(response))
            sys.stdout.flush()
            
        except Exception as e:
            error_response = {"error": str(e)}
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # Создаём директорию если нет
    server_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем сервер
    with open(server_path, 'w') as f:
        f.write(server_code)
    
    # Делаем исполняемым
    os.chmod(server_path, 0o755)
    
    print(f"✅ MCP сервер создан: {server_path}")
    return server_path

def test_mcp_connection():
    """Тестирует подключение MCP"""
    try:
        # Тестируем sequential thinking
        result = subprocess.run(
            ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
            input='{"method": "tools/list"}\n',
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ Sequential Thinking MCP работает")
        else:
            print("❌ Ошибка Sequential Thinking MCP")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")

def main():
    """Основная функция настройки"""
    print("🚀 Настройка MCP для ML Crypto Trading проекта...")
    
    # 1. Создаём MCP сервер
    server_path = create_mcp_server()
    
    # 2. Настраиваем конфигурацию
    config_path = setup_mcp_config()
    
    # 3. Тестируем подключение
    print("\n🧪 Тестирование MCP...")
    test_mcp_connection()
    
    print("\n✅ Настройка завершена!")
    print("\n📝 Инструкции:")
    print("1. Перезапустите Claude Desktop")
    print("2. В настройках Developer включите MCP")
    print("3. Серверы будут доступны автоматически")
    print("\n🔧 Доступные MCP команды:")
    print("- analyze_file - анализ файла с thinking")
    print("- get_project_context - получение контекста") 
    print("- analyze_trading_strategy - анализ стратегии")

if __name__ == "__main__":
    main()