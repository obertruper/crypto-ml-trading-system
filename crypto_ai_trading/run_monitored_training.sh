#!/bin/bash

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

clear

echo -e "${PURPLE}${BOLD}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}${BOLD}║           🔍 ЗАПУСК ОБУЧЕНИЯ С ПОЛНЫМ МОНИТОРИНГОМ              ║${NC}"
echo -e "${PURPLE}${BOLD}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Создание директории для логов
mkdir -p training_monitor_logs

echo -e "${CYAN}📁 Директория логов: training_monitor_logs/${NC}"
echo ""

# Проверка GPU
echo -e "${YELLOW}🎮 Проверка GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo -e "${RED}⚠️  GPU не обнаружен или nvidia-smi недоступен${NC}"
    echo ""
fi

# Очистка старых логов (опционально)
if [ -d "training_monitor_logs" ] && [ "$(ls -A training_monitor_logs)" ]; then
    echo -e "${YELLOW}Найдены старые логи. Очистить? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -f training_monitor_logs/*
        echo -e "${GREEN}✅ Старые логи удалены${NC}"
    fi
    echo ""
fi

# Запуск мониторинга реального времени
echo -e "${GREEN}${BOLD}🚀 Запуск мониторинга реального времени...${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Сохраняем время начала
START_TIME=$(date +%s)

# Запуск монитора реального времени с отображением процессов
python -u realtime_monitor_with_processes.py 2>&1 | tee training_monitor_logs/console_output_$(date +%Y%m%d_%H%M%S).log

# Проверка кода завершения
EXIT_CODE=${PIPESTATUS[0]}

# Вычисляем время работы
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✅ Обучение завершено успешно!${NC}"
else
    echo -e "${RED}${BOLD}❌ Обучение завершилось с ошибкой (код: $EXIT_CODE)${NC}"
fi

echo ""
echo -e "${PURPLE}${BOLD}📊 РЕЗУЛЬТАТЫ МОНИТОРИНГА:${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Время работы
echo -e "${YELLOW}⏱️  Время работы: ${ELAPSED_MIN} мин ${ELAPSED_SEC} сек${NC}"

# Показ файлов логов
if [ -d "training_monitor_logs" ]; then
    echo ""
    echo -e "${YELLOW}📁 Созданные файлы:${NC}"
    ls -lah training_monitor_logs/*.log 2>/dev/null | tail -5 | awk '{print "  • " $9 " (" $5 ")"}'
fi

# Проверка на схлопывания
LATEST_LOG=$(ls -t training_monitor_logs/console_output_*.log 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    COLLAPSE_COUNT=$(grep -c "СХЛОПЫВАНИЕ\|схлопывается" "$LATEST_LOG" 2>/dev/null || echo "0")
    echo ""
    if [ "$COLLAPSE_COUNT" -gt "0" ]; then
        echo -e "${RED}${BOLD}🚨 ОБНАРУЖЕНО СХЛОПЫВАНИЙ: $COLLAPSE_COUNT${NC}"
        echo -e "${YELLOW}Места схлопывания:${NC}"
        grep -n "СХЛОПЫВАНИЕ\|схлопывается" "$LATEST_LOG" | head -5 | awk -F: '{print "  Строка " $1 ": " substr($0, index($0,$2))}'
    else
        echo -e "${GREEN}✅ Схлопываний не обнаружено${NC}"
    fi

    # Проверка на ошибки
    ERROR_COUNT=$(grep -c "ERROR\|Traceback\|❌" "$LATEST_LOG" 2>/dev/null || echo "0")
    WARNING_COUNT=$(grep -c "WARNING\|⚠️" "$LATEST_LOG" 2>/dev/null || echo "0")

    echo ""
    if [ "$ERROR_COUNT" -gt "0" ]; then
        echo -e "${RED}❌ Ошибок: $ERROR_COUNT${NC}"
    fi
    if [ "$WARNING_COUNT" -gt "0" ]; then
        echo -e "${YELLOW}⚠️  Предупреждений: $WARNING_COUNT${NC}"
    fi

    # Статистика Loss
    echo ""
    LAST_LOSS=$(grep "Loss:" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oP 'Loss:\s*\K[0-9.]+' || echo "N/A")
    if [ "$LAST_LOSS" != "N/A" ]; then
        echo -e "${CYAN}📉 Последний Loss: $LAST_LOSS${NC}"
    fi

    # Скорость обучения
    LAST_SPEED=$(grep "samples/s" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oP '\d+(?=/s)' || echo "N/A")
    if [ "$LAST_SPEED" != "N/A" ]; then
        echo -e "${CYAN}⚡ Последняя скорость: ${LAST_SPEED} samples/s${NC}"
    fi
fi

echo ""
echo -e "${PURPLE}${BOLD}🔍 КОМАНДЫ ДЛЯ АНАЛИЗА:${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  # Просмотр полного лога:"
echo "  less $LATEST_LOG"
echo ""
echo "  # Поиск схлопываний:"
echo "  grep -n 'СХЛОПЫВАНИЕ\|FLAT.*9[0-9]%' $LATEST_LOG"
echo ""
echo "  # Поиск ошибок:"
echo "  grep -n 'ERROR\|Traceback' $LATEST_LOG"
echo ""
echo "  # График прогресса Loss:"
echo "  grep 'Loss:' $LATEST_LOG | grep -oP 'Loss:\s*\K[0-9.]+' | gnuplot -e 'plot \"-\" with lines'"
echo ""

# Если есть критические проблемы
if [ "$COLLAPSE_COUNT" -gt "0" ] || [ "$ERROR_COUNT" -gt "0" ]; then
    echo -e "${RED}${BOLD}⚠️  ВНИМАНИЕ: Обнаружены проблемы во время обучения!${NC}"
    echo -e "${YELLOW}Рекомендуется проанализировать лог для выявления причин.${NC}"
    echo ""
fi

echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"