#!/bin/bash

# Скрипт настройки автоматической двусторонней синхронизации
# Настраивает git hooks и cron задачи

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo "======================================"
    echo "$1"
    echo "======================================"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Проверка что мы в правильной директории
if [ ! -f "main.py" ] || [ ! -d ".git" ]; then
    print_error "Запустите из корня проекта crypto_ai_trading!"
    exit 1
fi

print_header "🔧 Настройка автоматической синхронизации Git"

# 1. Настройка git hooks
print_info "Настройка Git hooks..."

# Активируем локальные hooks
git config core.hooksPath .githooks
print_success "Git hooks активированы"

# 2. Настройка git для автоматического pull при push
print_info "Настройка Git для автоматической синхронизации..."

# Включаем автоматический rebase при pull
git config pull.rebase true
print_success "Автоматический rebase включен"

# Настраиваем push для текущей ветки
git config push.default current
print_success "Push настроен для текущей ветки"

# 3. Создание systemd сервиса для автосинхронизации (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_info "Создание systemd сервиса для автосинхронизации..."
    
    SERVICE_FILE="/tmp/crypto-git-sync.service"
    TIMER_FILE="/tmp/crypto-git-sync.timer"
    
    # Создаем service файл
    cat > $SERVICE_FILE << EOF
[Unit]
Description=Git sync for crypto_ai_trading
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/scripts/git_sync.sh --once
User=$USER

[Install]
WantedBy=multi-user.target
EOF

    # Создаем timer файл (каждые 5 минут)
    cat > $TIMER_FILE << EOF
[Unit]
Description=Run Git sync every 5 minutes
Requires=crypto-git-sync.service

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
Unit=crypto-git-sync.service

[Install]
WantedBy=timers.target
EOF

    print_info "Для активации автоматической синхронизации выполните:"
    echo "sudo cp $SERVICE_FILE /etc/systemd/system/"
    echo "sudo cp $TIMER_FILE /etc/systemd/system/"
    echo "sudo systemctl daemon-reload"
    echo "sudo systemctl enable crypto-git-sync.timer"
    echo "sudo systemctl start crypto-git-sync.timer"
    
# 4. Создание cron задачи (альтернатива)
else
    print_info "Настройка cron для автосинхронизации..."
    
    CRON_CMD="*/5 * * * * cd $(pwd) && ./scripts/git_sync.sh --once >> $(pwd)/logs/git_sync.log 2>&1"
    
    # Проверяем, есть ли уже такая задача
    if ! crontab -l 2>/dev/null | grep -q "crypto_ai_trading.*git_sync"; then
        (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
        print_success "Cron задача добавлена (каждые 5 минут)"
    else
        print_warning "Cron задача уже существует"
    fi
fi

# 5. Создание git алиасов для удобства
print_info "Создание git алиасов..."

git config alias.sync '!bash ./scripts/git_sync.sh'
git config alias.autopush '!git add . && git commit -m "Auto-commit: $(date)" && git push'
git config alias.status-all '!git fetch && git status && echo "=== Remote changes ===" && git log HEAD..origin/main --oneline'

print_success "Git алиасы созданы:"
echo "  - git sync       # Интерактивная синхронизация"
echo "  - git autopush   # Быстрый коммит и push"  
echo "  - git status-all # Полный статус с удаленными изменениями"

# 6. Настройка автоматического fetch
print_info "Настройка автоматического fetch..."

# Включаем фоновый fetch каждые 3 минуты
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
git config fetch.prune true

# 7. Создаем лог директорию
mkdir -p logs
print_success "Директория для логов создана"

print_header "✅ Настройка завершена!"

echo ""
print_info "Теперь доступны следующие команды:"
echo ""
echo "1. ${GREEN}./scripts/git_sync.sh${NC}         - Интерактивная синхронизация"
echo "2. ${GREEN}./scripts/git_sync.sh --auto${NC}   - Автоматическая синхронизация каждые 5 минут"
echo "3. ${GREEN}./scripts/git_sync.sh --once${NC}   - Однократная синхронизация"
echo "4. ${GREEN}git sync${NC}                        - Алиас для интерактивной синхронизации"
echo "5. ${GREEN}git autopush${NC}                    - Быстрый коммит и отправка"
echo ""
print_warning "Git hooks активированы:"
echo "  - pre-commit:  Проверка синхронизации перед коммитом"
echo "  - post-merge:  Обновление зависимостей после pull"
echo ""

# Проверяем текущий статус
read -p "Хотите проверить текущий статус синхронизации? (y/N): " check_status

if [[ $check_status =~ ^[Yy]$ ]]; then
    ./scripts/git_sync.sh
fi