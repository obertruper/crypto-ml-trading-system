#!/usr/bin/env python3
"""
MCP-LSP Bridge для Claude Code
Автоматически предоставляет контекст через MCP перед любыми операциями
"""

import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
import sqlite3
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileChange:
    """Информация об изменении файла"""
    path: str
    timestamp: datetime
    change_type: str  # 'modified', 'created', 'deleted'
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    lines_added: int = 0
    lines_removed: int = 0
    symbols_affected: List[str] = None

class MCPLSPBridge:
    """Мост между MCP и LSP для Claude Code"""
    
    def __init__(self):
        # Автоматически определяем корень проекта
        current_dir = Path(__file__).parent
        self.project_root = current_dir.parent  # crypto_ai_trading
        self.lsp_data_dir = current_dir / ".lsp_data"
        self.lsp_data_dir.mkdir(exist_ok=True)
        
        # База данных для отслеживания изменений
        self.db_path = self.lsp_data_dir / "file_tracking.db"
        self.init_database()
        
        # Кеш контекста
        self.context_cache = {}
        self.last_context_update = datetime.now()
        
    def init_database(self):
        """Инициализирует базу данных для отслеживания файлов"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_states (
                path TEXT PRIMARY KEY,
                last_hash TEXT,
                last_modified TIMESTAMP,
                last_analyzed TIMESTAMP,
                symbols TEXT,
                imports TEXT,
                exports TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                timestamp TIMESTAMP,
                change_type TEXT,
                old_hash TEXT,
                new_hash TEXT,
                lines_added INTEGER,
                lines_removed INTEGER,
                symbols_affected TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                requested_file TEXT,
                context_files TEXT,
                tokens_used INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Вычисляет хеш файла"""
        if not file_path.exists():
            return ""
        
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
            
    def track_file_change(self, file_path: Path) -> Optional[FileChange]:
        """Отслеживает изменения в файле"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        str_path = str(file_path)
        current_hash = self.calculate_file_hash(file_path)
        
        # Получаем предыдущее состояние
        cursor.execute(
            "SELECT last_hash FROM file_states WHERE path = ?",
            (str_path,)
        )
        result = cursor.fetchone()
        
        change = None
        if result is None:
            # Новый файл
            change = FileChange(
                path=str_path,
                timestamp=datetime.now(),
                change_type='created',
                new_hash=current_hash
            )
        elif result[0] != current_hash:
            # Файл изменен
            change = FileChange(
                path=str_path,
                timestamp=datetime.now(),
                change_type='modified',
                old_hash=result[0],
                new_hash=current_hash
            )
            
        if change:
            # Сохраняем изменение
            cursor.execute("""
                INSERT INTO file_changes 
                (path, timestamp, change_type, old_hash, new_hash)
                VALUES (?, ?, ?, ?, ?)
            """, (
                change.path,
                change.timestamp,
                change.change_type,
                change.old_hash,
                change.new_hash
            ))
            
            # Обновляем состояние файла
            cursor.execute("""
                INSERT OR REPLACE INTO file_states
                (path, last_hash, last_modified, last_analyzed)
                VALUES (?, ?, ?, ?)
            """, (
                str_path,
                current_hash,
                datetime.now(),
                datetime.now()
            ))
            
        conn.commit()
        conn.close()
        
        return change
        
    def get_recent_changes(self, limit: int = 10) -> List[FileChange]:
        """Получает последние изменения файлов"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT path, timestamp, change_type, old_hash, new_hash,
                   lines_added, lines_removed, symbols_affected
            FROM file_changes
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        changes = []
        for row in cursor.fetchall():
            changes.append(FileChange(
                path=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                change_type=row[2],
                old_hash=row[3],
                new_hash=row[4],
                lines_added=row[5] or 0,
                lines_removed=row[6] or 0,
                symbols_affected=json.loads(row[7]) if row[7] else []
            ))
            
        conn.close()
        return changes
        
    def get_file_context(self, file_path: str) -> Dict[str, Any]:
        """Получает контекст для файла через LSP"""
        file_path = Path(file_path)
        
        # Отслеживаем изменение
        change = self.track_file_change(file_path)
        
        context = {
            "file": str(file_path),
            "exists": file_path.exists(),
            "relative_path": str(file_path.relative_to(self.project_root)),
            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None,
            "recent_changes": [],
            "related_files": [],
            "symbols": [],
            "imports": [],
            "exports": [],
            "usage_hints": []
        }
        
        if not file_path.exists():
            return context
            
        # Определяем тип файла и связанные файлы
        if file_path.suffix == '.py':
            context.update(self._get_python_context(file_path))
        elif file_path.suffix in ['.yaml', '.yml']:
            context.update(self._get_yaml_context(file_path))
        elif file_path.suffix == '.json':
            context.update(self._get_json_context(file_path))
            
        # Добавляем последние изменения
        recent_changes = self.get_recent_changes(5)
        context["recent_changes"] = [
            {
                "file": change.path,
                "type": change.change_type,
                "timestamp": change.timestamp.isoformat()
            }
            for change in recent_changes
        ]
        
        # Сохраняем запрос контекста
        self._log_context_request(file_path, context)
        
        return context
        
    def _get_python_context(self, file_path: Path) -> Dict[str, Any]:
        """Получает контекст для Python файла"""
        context = {
            "file_type": "python",
            "related_files": [],
            "symbols": [],
            "imports": [],
            "exports": []
        }
        
        # Анализируем импорты и символы
        try:
            import ast
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            # Импорты
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        context["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        context["imports"].append(node.module)
                        
            # Экспортируемые символы
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    context["exports"].append({
                        "name": node.name,
                        "type": "class" if isinstance(node, ast.ClassDef) else "function",
                        "line": node.lineno
                    })
                    
        except Exception as e:
            logger.error(f"Ошибка анализа Python файла: {e}")
            
        # Определяем связанные файлы на основе пути
        rel_path = file_path.relative_to(self.project_root)
        
        if "models" in rel_path.parts:
            context["related_files"] = [
                "config/config.yaml",
                "training/trainer.py",
                "utils/metrics.py"
            ]
            context["usage_hints"] = [
                "Это файл модели. Изменения могут повлиять на обучение.",
                "Проверьте config.yaml для параметров модели",
                "Используйте trainer.py для обучения"
            ]
            
        elif "data" in rel_path.parts:
            context["related_files"] = [
                "config/config.yaml",
                "utils/db_utils.py",
                "models/patchtst.py"
            ]
            context["usage_hints"] = [
                "Это файл обработки данных.",
                "База данных PostgreSQL на порту 5555",
                "Проверьте feature_engineering.py для признаков"
            ]
            
        elif "trading" in rel_path.parts:
            context["related_files"] = [
                "config/config.yaml",
                "models/patchtst.py",
                "utils/metrics.py"
            ]
            context["usage_hints"] = [
                "Это файл торговой логики.",
                "Используйте risk_management.py для управления рисками",
                "Проверьте backtesting для тестирования"
            ]
            
        return context
        
    def _get_yaml_context(self, file_path: Path) -> Dict[str, Any]:
        """Получает контекст для YAML файла"""
        context = {
            "file_type": "yaml",
            "related_files": [],
            "usage_hints": []
        }
        
        if file_path.name == "config.yaml":
            context["related_files"] = [
                "main.py",
                "models/patchtst.py",
                "training/trainer.py",
                "data/data_loader.py"
            ]
            context["usage_hints"] = [
                "Главный конфигурационный файл проекта",
                "Изменения влияют на все компоненты системы",
                "Параметры модели в разделе 'model'",
                "Параметры данных в разделе 'data'",
                "Риск-менеджмент в разделе 'risk_management'"
            ]
            
        return context
        
    def _get_json_context(self, file_path: Path) -> Dict[str, Any]:
        """Получает контекст для JSON файла"""
        return {
            "file_type": "json",
            "related_files": [],
            "usage_hints": ["JSON файл с данными или конфигурацией"]
        }
        
    def _log_context_request(self, file_path: Path, context: Dict[str, Any]):
        """Логирует запрос контекста"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO context_requests
            (timestamp, requested_file, context_files, tokens_used)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now(),
            str(file_path),
            json.dumps(context.get("related_files", [])),
            len(json.dumps(context))  # Примерная оценка токенов
        ))
        
        conn.commit()
        conn.close()
        
    def get_project_summary(self) -> Dict[str, Any]:
        """Получает общую сводку по проекту"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Статистика изменений
        cursor.execute("""
            SELECT COUNT(*), change_type
            FROM file_changes
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY change_type
        """)
        
        changes_24h = {}
        for count, change_type in cursor.fetchall():
            changes_24h[change_type] = count
            
        # Самые активные файлы
        cursor.execute("""
            SELECT path, COUNT(*) as change_count
            FROM file_changes
            WHERE timestamp > datetime('now', '-7 days')
            GROUP BY path
            ORDER BY change_count DESC
            LIMIT 10
        """)
        
        active_files = [
            {"path": row[0], "changes": row[1]}
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            "project": "crypto_ai_trading",
            "last_update": datetime.now().isoformat(),
            "changes_24h": changes_24h,
            "active_files": active_files,
            "lsp_status": "active",
            "tracking_enabled": True
        }

# Глобальный экземпляр моста
_bridge = None

def get_bridge() -> MCPLSPBridge:
    """Получает глобальный экземпляр моста"""
    global _bridge
    if _bridge is None:
        _bridge = MCPLSPBridge()
    return _bridge

def mcp_get_file_context(file_path: str) -> Dict[str, Any]:
    """MCP функция для получения контекста файла"""
    bridge = get_bridge()
    return bridge.get_file_context(file_path)

def mcp_get_project_summary() -> Dict[str, Any]:
    """MCP функция для получения сводки проекта"""
    bridge = get_bridge()
    return bridge.get_project_summary()

def mcp_get_recent_changes(limit: int = 10) -> List[Dict[str, Any]]:
    """MCP функция для получения последних изменений"""
    bridge = get_bridge()
    changes = bridge.get_recent_changes(limit)
    return [
        {
            "path": c.path,
            "timestamp": c.timestamp.isoformat(),
            "type": c.change_type
        }
        for c in changes
    ]

if __name__ == "__main__":
    # Тест
    bridge = get_bridge()
    
    # Получаем контекст для config.yaml
    context = bridge.get_file_context("/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/config/config.yaml")
    print("Контекст для config.yaml:")
    print(json.dumps(context, indent=2, ensure_ascii=False))
    
    # Получаем сводку проекта
    summary = bridge.get_project_summary()
    print("\nСводка проекта:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))