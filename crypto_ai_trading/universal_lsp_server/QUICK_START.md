# 🚀 Быстрый старт Universal LSP Server

## Установка в любой проект за 30 секунд

### 1. Скачайте и распакуйте архив

```bash
# В корне вашего проекта
curl -L -o universal_lsp_server.tar.gz [URL_TO_ARCHIVE]
tar -xzf universal_lsp_server.tar.gz
cd universal_lsp_server
```

### 2. Установите зависимости

```bash
pip install -r requirements.txt
```

### 3. Запустите сервер

```bash
# Способ 1: Через quickstart скрипт (без установки)
./quickstart.py start

# Способ 2: Установить и запустить
pip install -e .
lsp-server start

# Способ 3: Прямой запуск Python модуля
python -m lsp_server.cli start
```

## Проверка работы

```bash
# В другом терминале
curl http://localhost:2087/stats

# Или проиндексируйте текущий проект
./quickstart.py index .
```

## Минимальная конфигурация

Создайте файл `lsp-server.yaml`:

```yaml
server:
  port: 2087
  log_level: "INFO"

indexing:
  extensions: [".py"]
  exclude_dirs: ["venv", "__pycache__"]

features:
  completion: true
  hover: true
  definition: true
```

## Использование без установки

```python
# В вашем Python скрипте
import sys
sys.path.insert(0, './universal_lsp_server')

from lsp_server import UniversalLSPServer, Config

config = Config()
server = UniversalLSPServer(config)
# Теперь можно использовать API сервера
```

## Интеграция с VS Code

Добавьте в `.vscode/settings.json`:

```json
{
  "python.languageServer": "None",
  "lsp-client.servers": [
    {
      "name": "Universal LSP",
      "command": ["./universal_lsp_server/quickstart.py", "start", "--stdio"],
      "languages": ["python"],
      "rootPatterns": [".git", "pyproject.toml", "setup.py"]
    }
  ]
}
```

## Запуск как фоновый процесс

```bash
# Запуск
nohup ./quickstart.py start > lsp.log 2>&1 &
echo $! > lsp.pid

# Остановка
kill $(cat lsp.pid)
rm lsp.pid
```

## Переменные окружения

```bash
export LSP_PORT=3000
export LSP_LOG_LEVEL=DEBUG
export LSP_PROJECT_ROOT=/path/to/project

./quickstart.py start
```

## Получение помощи

```bash
# Список всех команд
./quickstart.py --help

# Помощь по конкретной команде
./quickstart.py start --help
./quickstart.py index --help
```

## Устранение проблем

1. **Порт занят**: Измените порт через `--port 3000` или в конфиге
2. **Нет прав на запуск**: `chmod +x quickstart.py`
3. **Модуль не найден**: Убедитесь что установлены все зависимости

---

Готово! LSP сервер запущен и готов к работе 🎉