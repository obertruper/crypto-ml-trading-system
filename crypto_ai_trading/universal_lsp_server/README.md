# Universal LSP Server

Универсальный Language Server Protocol (LSP) сервер для улучшения работы с кодом в AI-ассистентах и IDE.

## 🚀 Особенности

- **Универсальность**: Работает с любым Python проектом без дополнительной настройки
- **Оптимизация для AI**: Специальные функции для работы с LLM (контекстные окна, форматирование)
- **Производительность**: Параллельная индексация, умное кеширование, инкрементальные обновления
- **Расширяемость**: Модульная архитектура, простое добавление новых возможностей
- **Кроссплатформенность**: Работает на macOS, Linux, Windows

## 📦 Установка

### Через pip (рекомендуется)

```bash
pip install universal-lsp-server
```

### Из исходников

```bash
git clone https://github.com/yourusername/universal-lsp-server
cd universal-lsp-server
pip install -e .
```

### Быстрая установка в проект

```bash
# Скачать и распаковать в текущий проект
curl -L https://github.com/yourusername/universal-lsp-server/archive/main.tar.gz | tar xz
cd universal-lsp-server-main
pip install -e .
```

## 🏃 Быстрый старт

### 1. Запуск сервера

```bash
# Простой запуск
lsp-server start

# С параметрами
lsp-server start --host 0.0.0.0 --port 3000 --project-root /path/to/project

# Через stdio (для IDE интеграции)
lsp-server start --stdio
```

### 2. Создание конфигурации

```bash
# Создать конфигурацию по умолчанию
lsp-server init

# Создать в определенном формате
lsp-server init --format yaml --output my-config.yaml
```

### 3. Проверка работы

```bash
# Проверить статус сервера
lsp-server check --host localhost --port 2087

# Проиндексировать проект
lsp-server index /path/to/project
```

## ⚙️ Конфигурация

### Через файл конфигурации

```yaml
# lsp-server.yaml
server:
  host: "127.0.0.1"
  port: 2087
  log_level: "INFO"

indexing:
  extensions: [".py", ".pyi"]
  exclude_dirs: ["__pycache__", ".git", "venv"]
  max_file_size: 1048576  # 1MB
  parallel_indexing: true

context:
  max_tokens: 100000
  max_related_files: 10
  context_format: "markdown"  # или "json", "xml"

features:
  completion: true
  hover: true
  definition: true
  references: true
  diagnostics: true
```

### Через переменные окружения

```bash
export LSP_HOST=0.0.0.0
export LSP_PORT=3000
export LSP_LOG_LEVEL=DEBUG
export LSP_PROJECT_ROOT=/path/to/project
export LSP_MAX_TOKENS=50000

lsp-server start
```

### Через аргументы командной строки

```bash
lsp-server start \
  --host 0.0.0.0 \
  --port 3000 \
  --log-level DEBUG \
  --project-root /path/to/project \
  --config my-config.yaml
```

## 🛠️ Использование в проекте

### Python API

```python
from lsp_server import UniversalLSPServer, Config

# Создать конфигурацию
config = Config()
config.server.port = 3000
config.indexing.extensions = [".py", ".pyx"]

# Создать и запустить сервер
server = UniversalLSPServer(config)
server.start_tcp("localhost", 3000)
```

### Интеграция с IDE

#### VS Code

Добавьте в `.vscode/settings.json`:

```json
{
  "python.languageServer": "None",
  "lsp.servers": {
    "universal-lsp": {
      "command": ["lsp-server", "start", "--stdio"],
      "filetypes": ["python"],
      "initializationOptions": {
        "maxTokens": 100000,
        "excludeDirs": ["tests", "docs"]
      }
    }
  }
}
```

#### Neovim

Добавьте в конфигурацию:

```lua
require'lspconfig'.universal_lsp.setup{
  cmd = {"lsp-server", "start", "--stdio"},
  filetypes = {"python"},
  root_dir = require'lspconfig'.util.root_pattern(".git", "setup.py", "pyproject.toml"),
}
```

## 🤖 Интеграция с AI-ассистентами

### Получение контекста для LLM

```bash
# Через команду
curl -X POST http://localhost:2087/execute \
  -H "Content-Type: application/json" \
  -d '{"command": "universal-lsp.getContext", "arguments": ["/path/to/file.py", "markdown"]}'
```

### Использование в Claude Code

LSP сервер автоматически предоставляет расширенный контекст для Claude Code, включая:

- Структуру проекта
- Связанные файлы
- Историю изменений
- Символы и их документацию

## 🚀 Установка как системный сервис

### macOS (launchd)

```bash
lsp-server install
# Управление:
launchctl start com.universal.lsp
launchctl stop com.universal.lsp
```

### Linux (systemd)

```bash
lsp-server install
# Управление:
systemctl --user start universal-lsp
systemctl --user stop universal-lsp
systemctl --user status universal-lsp
```

### Ручной запуск

```bash
# Создать скрипт запуска
lsp-server init --format bash > start-lsp.sh
chmod +x start-lsp.sh

# Использование
./start-lsp.sh start
./start-lsp.sh stop
./start-lsp.sh status
```

## 📊 Мониторинг и отладка

### Просмотр логов

```bash
# Если запущен как сервис
tail -f ~/Library/Logs/UniversalLSP/lsp-server.log  # macOS
journalctl --user -u universal-lsp -f              # Linux

# Если запущен вручную
tail -f lsp-server.log
```

### Получение статистики

```bash
# Через API
curl http://localhost:2087/stats

# Через команду
echo '{"command": "universal-lsp.getStats"}' | nc localhost 2087
```

## 🔧 Расширенные возможности

### Кастомные анализаторы

```python
from lsp_server.indexer import ProjectIndexer

class MyCustomIndexer(ProjectIndexer):
    def analyze_custom_syntax(self, content: str):
        # Ваша логика анализа
        pass
```

### Дополнительные обработчики

```python
from lsp_server.handlers import setup_handlers

def setup_custom_handlers(server):
    @server.command("my-custom.command")
    async def my_command(arg1: str, arg2: int):
        return {"result": f"Processed {arg1} with {arg2}"}
```

## 🐛 Решение проблем

### Сервер не запускается

1. Проверьте, не занят ли порт: `lsof -i :2087`
2. Проверьте права доступа к проекту
3. Посмотрите логи на наличие ошибок

### Медленная индексация

1. Увеличьте количество воркеров: `--max-workers 8`
2. Исключите большие директории в конфигурации
3. Включите кеширование если отключено

### Высокое потребление памяти

1. Уменьшите `max_tokens` в конфигурации
2. Ограничьте количество файлов для индексации
3. Включите инкрементальную синхронизацию

## 📝 Лицензия

MIT License. См. файл LICENSE для подробностей.

## 🤝 Вклад в проект

Мы приветствуем ваш вклад! Пожалуйста, см. CONTRIBUTING.md для деталей.

## 📞 Поддержка

- Документация: https://universal-lsp.readthedocs.io
- Issues: https://github.com/yourusername/universal-lsp-server/issues
- Discussions: https://github.com/yourusername/universal-lsp-server/discussions