# 🚀 Настройка автозапуска LSP сервера

## macOS (через launchd)

### 1. Создайте plist файл:
```bash
sudo nano ~/Library/LaunchAgents/com.crypto.lsp.plist
```

### 2. Вставьте конфигурацию:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.crypto.lsp</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/lsp_server/start_lsp_auto.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/lsp_server/lsp_launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/lsp_server/lsp_launchd_error.log</string>
</dict>
</plist>
```

### 3. Загрузите сервис:
```bash
launchctl load ~/Library/LaunchAgents/com.crypto.lsp.plist
```

### 4. Проверьте статус:
```bash
launchctl list | grep crypto.lsp
```

## 🔧 Управление сервисом

### Остановить:
```bash
launchctl unload ~/Library/LaunchAgents/com.crypto.lsp.plist
```

### Перезапустить:
```bash
launchctl unload ~/Library/LaunchAgents/com.crypto.lsp.plist
launchctl load ~/Library/LaunchAgents/com.crypto.lsp.plist
```

### Удалить из автозапуска:
```bash
launchctl unload ~/Library/LaunchAgents/com.crypto.lsp.plist
rm ~/Library/LaunchAgents/com.crypto.lsp.plist
```

## 📋 Альтернативный вариант - добавить в .zshrc

Добавьте в ~/.zshrc:
```bash
# Автозапуск LSP для crypto_ai_trading
if ! pgrep -f "enhanced_lsp_server.py" > /dev/null; then
    cd "/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/lsp_server" && ./start_lsp_auto.sh &
fi
```

## ✅ После настройки

LSP сервер будет:
- Запускаться автоматически при входе в систему
- Предоставлять контекст для всех запросов
- Работать в фоне без вмешательства

Теперь Claude всегда имеет доступ к:
- **LSP контексту** - локальная информация о проекте
- **Context7** - актуальная документация
- **Sequential Thinking** - глубокий анализ