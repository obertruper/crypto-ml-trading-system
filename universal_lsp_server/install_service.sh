#!/bin/bash

# Скрипт для установки LSP сервера как системного сервиса на macOS

LSP_DIR="/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading/lsp_server"
PLIST_FILE="com.crypto.lsp.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "🔧 Установка LSP сервиса..."

# Создаем директорию LaunchAgents если её нет
mkdir -p "$LAUNCH_AGENTS_DIR"

# Копируем plist файл
cp "$LSP_DIR/$PLIST_FILE" "$LAUNCH_AGENTS_DIR/"

# Загружаем сервис
launchctl load "$LAUNCH_AGENTS_DIR/$PLIST_FILE"

echo "✅ LSP сервис установлен и запущен!"
echo ""
echo "📋 Команды управления:"
echo "  Остановить: launchctl unload ~/Library/LaunchAgents/$PLIST_FILE"
echo "  Запустить:  launchctl load ~/Library/LaunchAgents/$PLIST_FILE"
echo "  Удалить:    launchctl unload ~/Library/LaunchAgents/$PLIST_FILE && rm ~/Library/LaunchAgents/$PLIST_FILE"
echo ""
echo "📄 Логи:"
echo "  stdout: $LSP_DIR/lsp_stdout.log"
echo "  stderr: $LSP_DIR/lsp_stderr.log"
echo "  service: $LSP_DIR/lsp_service.log"