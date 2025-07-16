#!/bin/bash

echo "🔐 Настройка Tailscale для удаленного доступа"
echo "=============================================="
echo ""

echo "📱 ШАГ 1: На Linux системе:"
echo "1. Откройте в браузере: https://login.tailscale.com/a/a39b16201cb61"
echo "2. Войдите в свой аккаунт (или создайте новый)"
echo "3. После авторизации выполните:"
echo "   sudo tailscale up"
echo ""

echo "💻 ШАГ 2: На вашем Mac:"
echo "1. Установите Tailscale: https://tailscale.com/download/mac"
echo "2. Войдите в тот же аккаунт"
echo "3. Оба устройства появятся в одной сети"
echo ""

echo "🚀 ШАГ 3: Подключение через Tailscale:"
echo "После настройки вы получите IP адреса вида 100.x.x.x"
echo "Например:"
echo "   ssh obertruper@100.64.0.1"
echo ""

echo "✅ Преимущества Tailscale:"
echo "- Работает через любые NAT и файрволы"
echo "- Шифрованное соединение"
echo "- Не нужно открывать порты"
echo "- Работает из любой сети"
echo ""

echo "📝 Альтернативный вариант - SSH туннель:"
echo "Если у вас есть общий сервер (например, VPS), можно:"
echo "1. С Linux: ssh -R 2222:localhost:22 user@vps"
echo "2. С Mac: ssh -p 2222 obertruper@vps"