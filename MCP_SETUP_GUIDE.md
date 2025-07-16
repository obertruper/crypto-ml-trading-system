# Руководство по настройке MCP (Model Context Protocol) для Claude Desktop

## 📍 Расположение конфигурационного файла

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

## 🚀 Установленные MCP серверы

### 1. **Brave Search** - Веб-поиск
```json
"brave-search": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-brave-search"],
  "env": {
    "BRAVE_API_KEY": "YOUR_BRAVE_API_KEY_HERE"
  }
}
```
**Настройка**: 
- Получите API ключ на https://brave.com/search/api/
- Замените `YOUR_BRAVE_API_KEY_HERE` на ваш ключ

### 2. **Fetch** - Загрузка веб-контента
```json
"fetch": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-fetch"]
}
```
**Возможности**: Загрузка и обработка веб-страниц

### 3. **Puppeteer** - Управление браузером
```json
"puppeteer": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
}
```
**Возможности**: 
- Автоматизация браузера
- Скриншоты веб-страниц
- Взаимодействие с динамическими сайтами

### 4. **Filesystem** - Работа с файлами
```json
"filesystem": {
  "command": "npx",
  "args": [
    "-y", 
    "@modelcontextprotocol/server-filesystem",
    "/Users/ruslan"
  ]
}
```
**Возможности**: Доступ к файловой системе

## 🎨 Дополнительные MCP серверы для медиа

### Для работы с изображениями:

#### 1. **Screenshot Tool**
```json
"screenshot": {
  "command": "npx",
  "args": ["-y", "mcp-server-screenshot"]
}
```

#### 2. **Image Analysis** (через Replicate)
```json
"replicate": {
  "command": "npx",
  "args": ["-y", "mcp-server-replicate"],
  "env": {
    "REPLICATE_API_TOKEN": "YOUR_REPLICATE_TOKEN"
  }
}
```

### Для видео:

#### **VideoDB** - Анализ и обработка видео
```json
"videodb": {
  "command": "npx",
  "args": ["-y", "mcp-server-videodb"],
  "env": {
    "VIDEODB_API_KEY": "YOUR_VIDEODB_KEY"
  }
}
```

## 🔧 Установка и настройка

### Предварительные требования:
1. **Node.js** должен быть установлен
   ```bash
   # Проверка установки
   node --version
   npm --version
   
   # Установка через Homebrew (если нет)
   brew install node
   ```

2. **Claude Desktop** должен быть перезапущен после изменения конфигурации

### Шаги настройки:

1. **Откройте конфигурационный файл**:
   ```bash
   open ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. **Добавьте нужные серверы** в секцию `mcpServers`

3. **Получите API ключи** для сервисов:
   - Brave Search: https://brave.com/search/api/
   - Replicate: https://replicate.com/account/api-tokens
   - VideoDB: https://videodb.io/

4. **Перезапустите Claude Desktop**

## 🔍 Проверка работы

### Логи MCP серверов:
```bash
# Просмотр логов
tail -f ~/Library/Logs/Claude/mcp*.log
```

### Тестирование:
1. Откройте Claude Desktop
2. Попробуйте команды:
   - "Найди информацию о..." (Brave Search)
   - "Сделай скриншот сайта..." (Puppeteer)
   - "Покажи файлы в папке..." (Filesystem)

## 🛠️ Решение проблем

### Если сервер не работает:

1. **Проверьте синтаксис JSON**:
   ```bash
   python3 -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. **Проверьте установку Node.js**:
   ```bash
   which node
   which npx
   ```

3. **Проверьте логи**:
   ```bash
   cat ~/Library/Logs/Claude/mcp-server-*.log
   ```

### Частые ошибки:
- **"Command not found"** - установите Node.js
- **"Invalid API key"** - проверьте правильность ключей
- **"Permission denied"** - проверьте права доступа к папкам

## 📚 Полезные ссылки

- [Официальная документация MCP](https://modelcontextprotocol.io/)
- [Список всех MCP серверов](https://mcpservers.org/)
- [GitHub репозиторий MCP](https://github.com/modelcontextprotocol/servers)
- [Сообщество Claude MCP](https://www.claudemcp.com/)

## 💡 Примеры использования

### Веб-поиск:
```
"Найди последние новости о криптовалютах используя Brave Search"
```

### Работа с браузером:
```
"Сделай скриншот сайта example.com используя Puppeteer"
```

### Анализ изображений:
```
"Проанализируй изображение на скриншоте"
```

## 🌟 Альтернативные MCP серверы

### Для веб-поиска:
- **DuckDuckGo Search**: Приватный поиск без отслеживания
- **Exa Search**: Поисковик оптимизированный для AI
- **Tavily**: Поиск + извлечение данных для AI агентов

### Для работы с медиа:
- **PiAPI**: Генерация изображений через Midjourney/Flux
- **EverArt**: AI генерация изображений
- **ComputerVision**: Распознавание и редактирование изображений

### Для веб-скрейпинга:
- **Firecrawl**: Извлечение веб-данных
- **WebScraping.AI**: Профессиональный веб-скрейпинг
- **Fetch**: Базовая загрузка веб-контента (уже установлен)

## 🔐 Безопасность

- API ключи хранятся локально в конфигурационном файле
- Серверы работают с правами текущего пользователя
- Рекомендуется ограничивать доступ filesystem сервера конкретными папками