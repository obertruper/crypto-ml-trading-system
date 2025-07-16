#!/bin/bash
# Копирует SSH ключ в буфер обмена

echo "Копирую ваш публичный SSH ключ в буфер обмена..."
cat ~/.ssh/id_rsa.pub | pbcopy
echo "✓ SSH ключ скопирован в буфер обмена!"
echo ""
echo "Теперь вы можете вставить его в веб-терминал Vast.ai"
echo "используя Cmd+V (Mac) или Ctrl+V (Windows/Linux)"