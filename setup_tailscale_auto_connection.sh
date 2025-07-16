#!/bin/bash

echo "🔧 Настройка автоматического Tailscale подключения"
echo "================================================="

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "\n${BLUE}📱 Текущий статус Tailscale:${NC}"
tailscale status

echo -e "\n${YELLOW}🖥️  Для подключения с Mac:${NC}"
echo "1. Установите Tailscale на Mac:"
echo "   brew install tailscale"
echo "   или скачайте с https://tailscale.com/download/mac"
echo ""
echo "2. Запустите Tailscale на Mac:"
echo "   tailscale up"
echo ""
echo "3. После входа в аккаунт, Mac появится в сети"
echo ""

echo -e "${GREEN}✅ Linux готов к подключениям:${NC}"
echo "IP адрес: 100.118.184.106"
echo "Hostname: obertruper-system-product-name"
echo ""

echo -e "${BLUE}🔐 SSH через Tailscale:${NC}"
echo "С Mac выполните:"
echo "ssh obertruper@100.118.184.106"
echo "или"
echo "ssh obertruper@obertruper-system-product-name"
echo ""

echo -e "${YELLOW}⚡ Автоматическое подключение:${NC}"
echo "Tailscale автоматически:"
echo "- Поддерживает соединение между устройствами"
echo "- Работает через NAT и файрволы"
echo "- Шифрует весь трафик"
echo "- Переподключается при смене сети"
echo ""

echo -e "${GREEN}📝 Добавьте в ~/.ssh/config на Mac:${NC}"
cat << 'EOF'
Host linux-ts
    HostName 100.118.184.106
    User obertruper
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    
Host linux-ts-name
    HostName obertruper-system-product-name
    User obertruper
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
EOF

echo -e "\n${BLUE}🚀 После настройки на Mac:${NC}"
echo "ssh linux-ts          # подключение по IP"
echo "ssh linux-ts-name     # подключение по имени"