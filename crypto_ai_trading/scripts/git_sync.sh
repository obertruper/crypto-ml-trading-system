#!/bin/bash

# Скрипт для двусторонней синхронизации с GitHub
# Автор: Claude для crypto_ai_trading

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода с цветом
print_status() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Проверка, что мы в правильной директории
if [ ! -f "main.py" ] || [ ! -d ".git" ]; then
    print_error "Запустите скрипт из корня проекта crypto_ai_trading!"
    exit 1
fi

# Функция для проверки изменений
check_changes() {
    # Проверяем локальные изменения
    LOCAL_CHANGES=$(git status --porcelain | wc -l)
    
    # Проверяем удаленные изменения
    git fetch origin main --quiet
    REMOTE_CHANGES=$(git log HEAD..origin/main --oneline | wc -l)
    
    echo "$LOCAL_CHANGES $REMOTE_CHANGES"
}

# Функция синхронизации
sync_repository() {
    print_status "🔄 Начинаем синхронизацию с GitHub..."
    
    # 1. Сохраняем текущие изменения
    STASH_OUTPUT=$(git stash push -m "Auto-stash before sync $(date '+%Y-%m-%d %H:%M:%S')" 2>&1)
    STASHED=false
    if [[ $STASH_OUTPUT == *"Saved working directory"* ]]; then
        STASHED=true
        print_success "Локальные изменения временно сохранены"
    fi
    
    # 2. Получаем изменения с GitHub
    print_status "📥 Получаем изменения с GitHub..."
    if git pull origin main --rebase; then
        print_success "Изменения с GitHub получены"
    else
        print_error "Ошибка при получении изменений"
        if [ "$STASHED" = true ]; then
            git stash pop
        fi
        return 1
    fi
    
    # 3. Восстанавливаем локальные изменения
    if [ "$STASHED" = true ]; then
        print_status "📤 Восстанавливаем локальные изменения..."
        if git stash pop; then
            print_success "Локальные изменения восстановлены"
        else
            print_warning "Конфликт при восстановлении изменений"
            print_warning "Разрешите конфликты вручную и запустите: git add . && git rebase --continue"
            return 1
        fi
    fi
    
    # 4. Добавляем все изменения
    if [ $(git status --porcelain | wc -l) -gt 0 ]; then
        print_status "📝 Подготовка изменений к отправке..."
        git add .
        
        # Создаем коммит
        COMMIT_MSG="🔄 Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')

Changes synced from local development"
        
        git commit -m "$COMMIT_MSG" --quiet
        print_success "Создан коммит с локальными изменениями"
    fi
    
    # 5. Отправляем на GitHub
    print_status "📤 Отправляем изменения на GitHub..."
    if git push origin main; then
        print_success "Изменения успешно отправлены на GitHub"
    else
        print_error "Ошибка при отправке изменений"
        return 1
    fi
    
    print_success "✨ Синхронизация завершена успешно!"
}

# Функция для интерактивного режима
interactive_sync() {
    clear
    echo "======================================"
    echo "🔄 Git Sync для crypto_ai_trading"
    echo "======================================"
    
    # Проверяем статус
    read LOCAL_CHANGES REMOTE_CHANGES <<< $(check_changes)
    
    echo ""
    if [ $LOCAL_CHANGES -gt 0 ]; then
        print_warning "Локальные изменения: $LOCAL_CHANGES файлов"
        git status --short
    else
        print_success "Нет локальных изменений"
    fi
    
    echo ""
    if [ $REMOTE_CHANGES -gt 0 ]; then
        print_warning "Изменения на GitHub: $REMOTE_CHANGES коммитов"
        git log HEAD..origin/main --oneline
    else
        print_success "Нет изменений на GitHub"
    fi
    
    if [ $LOCAL_CHANGES -eq 0 ] && [ $REMOTE_CHANGES -eq 0 ]; then
        print_success "🎉 Всё синхронизировано!"
        return 0
    fi
    
    echo ""
    echo "Выберите действие:"
    echo "1) 🔄 Синхронизировать (pull + push)"
    echo "2) 📥 Только получить с GitHub (pull)"
    echo "3) 📤 Только отправить на GitHub (push)"
    echo "4) 📊 Показать детальный статус"
    echo "5) ❌ Отмена"
    echo ""
    read -p "Ваш выбор (1-5): " choice
    
    case $choice in
        1)
            sync_repository
            ;;
        2)
            print_status "📥 Получаем изменения с GitHub..."
            if git pull origin main; then
                print_success "Изменения получены"
            else
                print_error "Ошибка при получении изменений"
            fi
            ;;
        3)
            if [ $LOCAL_CHANGES -gt 0 ]; then
                print_status "📤 Отправляем изменения на GitHub..."
                git add .
                read -p "Введите сообщение коммита: " commit_msg
                git commit -m "$commit_msg"
                git push origin main
                print_success "Изменения отправлены"
            else
                print_warning "Нет локальных изменений для отправки"
            fi
            ;;
        4)
            echo ""
            print_status "📊 Детальный статус:"
            git status
            echo ""
            print_status "📝 Последние коммиты:"
            git log --oneline -10
            ;;
        5)
            print_warning "Отменено"
            ;;
        *)
            print_error "Неверный выбор"
            ;;
    esac
}

# Функция для автоматической синхронизации
auto_sync() {
    while true; do
        clear
        print_status "🤖 Автоматическая синхронизация каждые 5 минут"
        print_status "Нажмите Ctrl+C для остановки"
        echo ""
        
        # Проверяем и синхронизируем
        read LOCAL_CHANGES REMOTE_CHANGES <<< $(check_changes)
        
        if [ $LOCAL_CHANGES -gt 0 ] || [ $REMOTE_CHANGES -gt 0 ]; then
            print_warning "Обнаружены изменения, синхронизируем..."
            sync_repository
        else
            print_success "Всё синхронизировано"
        fi
        
        # Ждем 5 минут
        print_status "Следующая проверка через 5 минут..."
        sleep 300
    done
}

# Главное меню
main() {
    if [ "$1" == "--auto" ]; then
        auto_sync
    elif [ "$1" == "--once" ]; then
        sync_repository
    else
        interactive_sync
    fi
}

# Запуск
main "$@"