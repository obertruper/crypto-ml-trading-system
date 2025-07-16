# 🚀 Universal LSP Server

Универсальный Language Server Protocol (LSP) сервер для Python проектов с оптимизацией для работы с AI ассистентами.

## ✨ Особенности

- 🔍 **Полная поддержка LSP**: автодополнение, hover, определения, ссылки, символы
- 🤖 **AI-оптимизация**: экспорт контекста для Claude, ChatGPT и других AI
- ⚡ **Высокая производительность**: параллельная индексация, умное кеширование
- 🔧 **Гибкая конфигурация**: YAML, переменные окружения, CLI
- 🌍 **Универсальность**: работает с любым Python проектом

## 🚀 Быстрый старт (30 секунд)

```bash
# 1. Распаковать архив в ваш проект
tar -xzf universal_lsp_server.tar.gz
cd universal_lsp_server

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Запустить сервер
./quickstart.py start
```

## 📋 Установка как пакет

```bash
# Установка в режиме разработки
pip install -e universal_lsp_server/

# Или обычная установка
pip install universal_lsp_server/

# Запуск через CLI
lsp-server start
```

## 🎯 Использование

### Основные команды

```bash
# Запуск сервера
lsp-server start --port 3000

# Создание конфигурации
lsp-server init

# Проверка окружения
lsp-server check

# Быстрый запуск без установки
./quickstart.py start
```

### Конфигурация

Создайте файл `lsp_config.yaml`:

```yaml
server:
  host: "127.0.0.1"
  port: 3000
  
indexing:
  parallel: true
  exclude_patterns:
    - "__pycache__"
    - ".git"
    - "*.pyc"
    
ai_export:
  format: "markdown"
  include_docstrings: true
  max_context_size: 100000
```

### Переменные окружения

```bash
export LSP_PORT=3000
export LSP_HOST=0.0.0.0
export LSP_PROJECT_ROOT=/path/to/project
```

## 🔧 Интеграция с IDE

### VS Code
```json
{
  "python.lsp.server": "custom",
  "python.lsp.serverPath": "lsp-server",
  "python.lsp.serverArgs": ["start", "--stdio"]
}
```

### Neovim
```lua
require'lspconfig'.pylsp.setup{
  cmd = {"lsp-server", "start", "--stdio"}
}
```

## 📦 Структура

```
universal_lsp_server/
├── lsp_server/          # Основной пакет
├── quickstart.py        # Быстрый запуск
├── setup.py            # Установка
├── requirements.txt    # Зависимости
└── README.md          # Документация
```

## 🤝 Лицензия

MIT License - используйте свободно!