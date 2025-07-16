# 🧠 MCP Sequential Thinking в Claude Code

## ✅ Установка завершена!

MCP серверы успешно добавлены в Claude Code для этого проекта.

## 📋 Установленные MCP серверы:

1. **sequential-thinking** - официальный Sequential Thinking от Anthropic
   - Scope: Local (только для этого проекта)
   - Команда: `npx -y @modelcontextprotocol/server-sequential-thinking`

2. **crypto-trading-mcp** - локальный MCP сервер проекта
   - Scope: Project (в файле `.mcp.json`)
   - Команда: `python3 universal_lsp_server/mcp_server.py`

## 🚀 Использование в Claude Code

### Доступные MCP инструменты:

После перезапуска Claude Code вам будут доступны новые инструменты:

1. **Sequential Thinking**:
   - Структурированное мышление по шагам
   - Анализ проблем и решений
   - Планирование изменений

2. **Crypto Trading MCP**:
   - `analyze_file` - анализ файла с thinking
   - `get_project_context` - контекст проекта
   - `analyze_trading_strategy` - анализ стратегий

### Как использовать:

1. **Перезапустите Claude Code** (важно!)
2. **Инструменты появятся автоматически** при работе с проектом
3. **Claude будет использовать их** для лучшего понимания кода

## 🔧 Управление MCP серверами

```bash
# Просмотр всех серверов
claude mcp list

# Информация о сервере
claude mcp get sequential-thinking
claude mcp get crypto-trading-mcp

# Удаление сервера
claude mcp remove sequential-thinking -s local
claude mcp remove crypto-trading-mcp -s project

# Добавление нового сервера
claude mcp add <name> <command> [args...]
```

## 📁 Конфигурационные файлы

1. **`.mcp.json`** (в корне проекта) - конфигурация для проекта
2. **Локальная конфигурация** - приватная для вас в этом проекте

## 🧪 Тестирование

```bash
# Тест thinking интеграции
python test_thinking_integration.py

# Проверка работы MCP сервера
python universal_lsp_server/mcp_server.py
```

## ⚡ Быстрый старт

1. Перезапустите Claude Code
2. Откройте любой Python файл проекта
3. Claude автоматически получит доступ к MCP инструментам
4. При анализе кода будет использоваться Sequential Thinking

## 🛠️ Troubleshooting

Если MCP не работает:

1. **Проверьте установку**:
   ```bash
   claude mcp list
   ```

2. **Перезапустите Claude Code**:
   - Закройте все окна
   - Откройте заново

3. **Проверьте логи**:
   ```bash
   # В режиме отладки
   claude --debug
   ```

4. **Переустановите сервер**:
   ```bash
   claude mcp remove sequential-thinking -s local
   claude mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
   ```

## 📚 Полезные ссылки

- [Claude Code MCP Docs](https://docs.anthropic.com/en/docs/claude-code/mcp)
- [Sequential Thinking GitHub](https://github.com/modelcontextprotocol/servers)
- [MCP Protocol Spec](https://modelcontextprotocol.io)

## 💡 Советы

1. **MCP серверы запускаются автоматически** при использовании
2. **Конфигурация сохраняется** в проекте (`.mcp.json`)
3. **Sequential Thinking** особенно полезен для сложных задач
4. **Используйте thinking** перед важными изменениями архитектуры