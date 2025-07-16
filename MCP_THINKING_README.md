# 🧠 MCP Sequential Thinking для ML Crypto Trading

## ✅ Статус установки

MCP Sequential Thinking успешно настроен и интегрирован в проект!

### Компоненты:
1. **Sequential Thinking MCP Server** - официальный сервер от Anthropic
2. **MCP-LSP Bridge** - мост между MCP и LSP для анализа кода
3. **Thinking LSP Integration** - 5-шаговый анализ файлов

## 🚀 Использование

### 1. В Claude Desktop (после перезапуска):
```
- Настройки → Developer → Enable MCP
- Доступные команды:
  - analyze_file - анализ файла с thinking
  - get_project_context - контекст проекта
  - analyze_trading_strategy - анализ стратегии
```

### 2. Из командной строки:
```bash
# Тестирование thinking анализа
python test_thinking_integration.py

# Запуск MCP сервера вручную
python universal_lsp_server/mcp_server.py

# Sequential Thinking через npx
npx -y @modelcontextprotocol/server-sequential-thinking
```

### 3. В коде Python:
```python
from universal_lsp_server.thinking_lsp_integration import analyze_with_thinking

# Анализ файла с thinking
result = await analyze_with_thinking("path/to/file.py")
```

## 📋 Процесс анализа (5 шагов)

1. **Контекст** - базовая информация о файле
2. **Зависимости** - импорты, экспорты, связи
3. **Назначение** - роль файла в проекте
4. **Проблемы** - потенциальные issues
5. **Рекомендации** - что нужно проверить

## 🔧 Конфигурация

Файл: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "crypto-trading-context": {
      "command": "python3",
      "args": ["/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/universal_lsp_server/mcp_server.py"],
      "env": {
        "PROJECT_ROOT": "/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM",
        "THINKING_MODE": "enabled"
      }
    }
  }
}
```

## 📁 Структура файлов

```
universal_lsp_server/
├── mcp_lsp_bridge.py         # Мост MCP-LSP
├── thinking_lsp_integration.py # 5-шаговый анализ
├── mcp_server.py             # MCP сервер проекта
└── .lsp_data/                # База данных изменений
```

## 💡 Советы

1. **Перед важными изменениями** - запускайте thinking анализ
2. **При рефакторинге** - используйте контекст зависимостей
3. **Для новых фич** - анализируйте связанные файлы

## 🛠️ Troubleshooting

Если MCP не работает:
1. Перезапустите Claude Desktop
2. Проверьте логи: `tail -f ~/.config/claude/logs/mcp.log`
3. Убедитесь что npx доступен: `which npx`
4. Проверьте конфиг: `cat ~/.config/claude/claude_desktop_config.json`

## 📚 Ссылки

- [MCP Documentation](https://modelcontextprotocol.io)
- [Sequential Thinking GitHub](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking)
- [Claude Code Docs](https://docs.anthropic.com/en/docs/claude-code)