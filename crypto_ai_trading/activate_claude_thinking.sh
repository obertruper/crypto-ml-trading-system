#!/bin/bash

# Скрипт активации Claude Thinking LSP для проекта crypto_ai_trading

echo "🧠 Активация Claude Thinking LSP..."
echo "=================================="

PROJECT_ROOT="/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading"
LSP_DIR="$PROJECT_ROOT/lsp_server"

# Проверяем LSP сервер
echo "🔍 Проверка LSP сервера..."
cd "$LSP_DIR"
STATUS=$(./venv/bin/python start_lsp_service.py status 2>&1)

if [[ $STATUS == *"запущен"* ]]; then
    echo "✅ LSP сервер уже работает"
else
    echo "🚀 Запуск LSP сервера..."
    ./venv/bin/python start_lsp_service.py start &
    sleep 3
fi

# Тестируем интеграцию
echo ""
echo "🧪 Тестирование интеграции..."
cd "$PROJECT_ROOT"

python3 -c "
import sys
sys.path.insert(0, 'lsp_server')
from mcp_lsp_bridge import get_bridge

bridge = get_bridge()
summary = bridge.get_project_summary()

print('✅ MCP Bridge активен')
print(f'📊 Проект: {summary[\"project\"]}')
print(f'🔄 Изменений за 24ч: {sum(summary[\"changes_24h\"].values())}')
print(f'📁 Активных файлов: {len(summary[\"active_files\"])}')
"

echo ""
echo "🎯 Claude Thinking LSP активирован!"
echo ""
echo "📋 Доступные функции:"
echo "  • Автоматический контекст перед операциями"
echo "  • 5-шаговый анализ важных файлов"
echo "  • История всех изменений"
echo "  • Умные рекомендации"
echo ""
echo "📝 Файл правил: $PROJECT_ROOT/CLAUDE.md"
echo "🔧 LSP логи: $LSP_DIR/lsp_service.log"
echo ""
echo "✨ Теперь Claude Code думает перед действием!"