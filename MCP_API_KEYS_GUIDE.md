# Руководство по получению API ключей для MCP серверов

## 🔑 Brave Search API
1. Перейдите на https://brave.com/search/api/
2. Нажмите "Get Started" 
3. Зарегистрируйтесь или войдите
4. Создайте новое приложение
5. Скопируйте API ключ
6. Вставьте в конфигурацию: `"BRAVE_API_KEY": "ваш_ключ"`

**Бесплатный план**: 2000 запросов в месяц

## 🎨 Replicate (для работы с изображениями)
1. Перейдите на https://replicate.com/
2. Зарегистрируйтесь через GitHub
3. Перейдите в Account → API tokens
4. Создайте новый токен
5. Вставьте в конфигурацию: `"REPLICATE_API_TOKEN": "ваш_токен"`

**Бесплатный план**: $0.10 кредитов для начала

## 🎬 VideoDB
1. Перейдите на https://videodb.io/
2. Зарегистрируйтесь
3. В дашборде найдите API Keys
4. Создайте новый ключ
5. Вставьте в конфигурацию: `"VIDEODB_API_KEY": "ваш_ключ"`

## 🔍 Альтернативные поисковые системы

### DuckDuckGo
- Не требует API ключа! 
- Работает сразу после установки

### Exa Search
1. https://exa.ai/
2. Sign up → API Keys
3. Вставьте: `"EXA_API_KEY": "ваш_ключ"`

### Tavily
1. https://tavily.com/
2. Register → API Keys
3. Вставьте: `"TAVILY_API_KEY": "ваш_ключ"`

**Бесплатный план**: 1000 запросов в месяц

## 🌐 Веб-скрейпинг

### Firecrawl
1. https://firecrawl.dev/
2. Sign up → Dashboard → API Keys
3. Вставьте: `"FIRECRAWL_API_KEY": "ваш_ключ"`

**Бесплатный план**: 500 страниц в месяц

### WebScraping.AI
1. https://webscraping.ai/
2. Register → API Management
3. Вставьте: `"WEBSCRAPING_AI_KEY": "ваш_ключ"`

**Бесплатный план**: 5000 API кредитов

## 🐙 GitHub
1. Перейдите на https://github.com/settings/tokens
2. Generate new token (classic)
3. Выберите нужные права доступа:
   - `repo` - полный доступ к репозиториям
   - `read:user` - чтение профиля
4. Вставьте: `"GITHUB_PERSONAL_ACCESS_TOKEN": "ваш_токен"`

## 💬 Slack
1. https://api.slack.com/apps
2. Create New App → From scratch
3. OAuth & Permissions → Bot Token Scopes
4. Добавьте необходимые разрешения
5. Install to Workspace
6. Скопируйте Bot User OAuth Token
7. Вставьте: `"SLACK_BOT_TOKEN": "xoxb-ваш-токен"`

## 📺 YouTube
1. https://console.cloud.google.com/
2. Создайте новый проект
3. Enable YouTube Data API v3
4. Create Credentials → API Key
5. Вставьте: `"YOUTUBE_API_KEY": "ваш_ключ"`

**Бесплатная квота**: 10,000 единиц в день

## 💡 Советы по безопасности

1. **Никогда не коммитьте API ключи в Git**
2. Используйте переменные окружения для чувствительных данных
3. Регулярно ротируйте ключи
4. Ограничивайте права доступа до минимально необходимых
5. Мониторьте использование через дашборды провайдеров

## 🚀 Быстрый старт без API ключей

Эти серверы работают без ключей:
- `filesystem` - работа с файлами
- `fetch` - базовая загрузка веб-страниц
- `puppeteer` - управление браузером
- `sequential-thinking` - последовательное мышление
- `duckduckgo-search` - приватный поиск (если доступен)