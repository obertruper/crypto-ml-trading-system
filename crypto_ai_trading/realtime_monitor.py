#!/usr/bin/env python3
"""
Простой монитор реального времени для отслеживания обучения
"""
import subprocess
import sys
import time
import re
from datetime import datetime
import signal
import os

# Цвета
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Глобальные переменные для статистики
stats = {
    'start_time': time.time(),
    'epochs': 0,
    'batches': 0,
    'loss_values': [],
    'collapse_detected': False,
    'last_batch_time': time.time(),
    'samples_per_second': 0,
    'gpu_memory': '',
    'warnings': 0,
    'errors': 0
}

def signal_handler(sig, frame):
    """Обработка Ctrl+C"""
    print(f"\n\n{Colors.YELLOW}🛑 Остановка мониторинга...{Colors.RESET}")
    print_summary()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def parse_line(line):
    """Парсинг строки вывода для извлечения информации"""

    # Детекция схлопывания
    if "СХЛОПЫВАНИЕ" in line or "схлопывается" in line or "FLAT: 9" in line or "FLAT: 10" in line:
        stats['collapse_detected'] = True
        print(f"\n{Colors.RED}{Colors.BOLD}🚨 ОБНАРУЖЕНО СХЛОПЫВАНИЕ МОДЕЛИ!{Colors.RESET}")
        print(f"{Colors.RED}{line}{Colors.RESET}")
        return True

    # Парсинг прогресса обучения (ищем паттерн Training: XX%)
    training_match = re.search(r'Training:\s*(\d+)%.*loss=([0-9.]+).*samples/s=(\d+)', line)
    if training_match:
        percent = training_match.group(1)
        loss = float(training_match.group(2))
        samples_s = int(training_match.group(3))

        stats['loss_values'].append(loss)
        stats['samples_per_second'] = samples_s
        stats['batches'] += 1

        # Форматированный вывод прогресса
        bar_length = 40
        filled = int(bar_length * int(percent) / 100)
        bar = '█' * filled + '░' * (bar_length - filled)

        # Цвет в зависимости от loss
        loss_color = Colors.GREEN if loss < 1.5 else Colors.YELLOW if loss < 2.0 else Colors.RED

        output = f"\r{Colors.CYAN}Training: {Colors.BOLD}{percent:>3}%{Colors.RESET} "
        output += f"{Colors.CYAN}[{bar}]{Colors.RESET} "
        output += f"Loss: {loss_color}{loss:.4f}{Colors.RESET} "
        output += f"Speed: {Colors.GREEN}{samples_s:,}/s{Colors.RESET}"

        # Проверка на малый loss
        if loss < 0.0001:
            output += f" {Colors.RED}⚠️ ОЧЕНЬ МАЛЫЙ LOSS!{Colors.RESET}"

        print(output, end='', flush=True)
        return True

    # Парсинг GPU памяти
    gpu_match = re.search(r'gpu_mem=([0-9.]+)/([0-9.]+)GB', line)
    if gpu_match:
        stats['gpu_memory'] = f"{gpu_match.group(1)}/{gpu_match.group(2)}GB"

    # Детекция эпох
    if "Epoch" in line and "/" in line:
        stats['epochs'] += 1
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}{line.strip()}{Colors.RESET}")
        return True

    # Детекция warnings
    if "WARNING" in line or "⚠️" in line:
        stats['warnings'] += 1
        print(f"\n{Colors.YELLOW}{line.strip()}{Colors.RESET}")
        return True

    # Детекция errors
    if "ERROR" in line or "❌" in line or "Traceback" in line:
        stats['errors'] += 1
        print(f"\n{Colors.RED}{Colors.BOLD}{line.strip()}{Colors.RESET}")
        return True

    # Детекция важных событий
    if any(keyword in line for keyword in ["✅", "🚀", "📊", "🔍", "Saving", "Loading", "Best model"]):
        print(f"\n{Colors.GREEN}{line.strip()}{Colors.RESET}")
        return True

    # Детекция распределения классов
    if "LONG=" in line or "SHORT=" in line or "FLAT=" in line:
        # Извлекаем проценты
        flat_match = re.search(r'FLAT=([0-9.]+)%', line)
        if flat_match:
            flat_percent = float(flat_match.group(1))
            if flat_percent > 90:
                print(f"\n{Colors.RED}{Colors.BOLD}⚠️ КРИТИЧНО: FLAT={flat_percent}%{Colors.RESET}")
            elif flat_percent > 70:
                print(f"\n{Colors.YELLOW}⚠️ Внимание: FLAT={flat_percent}%{Colors.RESET}")
            else:
                print(f"\n{Colors.CYAN}{line.strip()}{Colors.RESET}")
        return True

    return False

def print_summary():
    """Вывод итоговой статистики"""
    elapsed = time.time() - stats['start_time']

    print(f"\n\n{Colors.BOLD}{Colors.MAGENTA}{'='*70}")
    print(f"📊 ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*70}{Colors.RESET}")

    print(f"⏱️  Время работы: {elapsed:.1f} сек")
    print(f"🔄 Эпох пройдено: {stats['epochs']}")
    print(f"📦 Батчей обработано: {stats['batches']}")

    if stats['loss_values']:
        avg_loss = sum(stats['loss_values']) / len(stats['loss_values'])
        min_loss = min(stats['loss_values'])
        max_loss = max(stats['loss_values'])
        print(f"📉 Loss: avg={avg_loss:.4f}, min={min_loss:.4f}, max={max_loss:.4f}")

    if stats['gpu_memory']:
        print(f"🎮 GPU память: {stats['gpu_memory']}")

    if stats['samples_per_second']:
        print(f"⚡ Скорость: {stats['samples_per_second']:,} samples/s")

    print(f"⚠️  Warnings: {stats['warnings']}")
    print(f"❌ Errors: {stats['errors']}")

    if stats['collapse_detected']:
        print(f"\n{Colors.RED}{Colors.BOLD}🚨 БЫЛО ОБНАРУЖЕНО СХЛОПЫВАНИЕ МОДЕЛИ!{Colors.RESET}")

    print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}\n")

def main():
    """Основная функция мониторинга"""
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"🔍 МОНИТОР ОБУЧЕНИЯ РЕАЛЬНОГО ВРЕМЕНИ")
    print(f"{'='*70}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}📌 Следим за:")
    print(f"  • Прогрессом обучения")
    print(f"  • Loss значениями")
    print(f"  • Схлопыванием модели (>90% FLAT)")
    print(f"  • Ошибками и предупреждениями")
    print(f"  • Скоростью обучения{Colors.RESET}\n")

    print(f"{Colors.GREEN}🚀 Запуск обучения...{Colors.RESET}\n")
    print(f"{Colors.CYAN}{'─'*70}{Colors.RESET}\n")

    # Запуск процесса обучения
    cmd = [sys.executable, "main.py", "--mode", "staged"]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )

        # Чтение вывода построчно
        for line in process.stdout:
            # Парсим и выводим отформатированную информацию
            if not parse_line(line):
                # Если строка не была распознана, проверяем стоит ли её показать
                line_stripped = line.strip()
                if line_stripped and not any(skip in line for skip in ['matplotlib', 'UserWarning', 'import']):
                    # Показываем неразпознанные но потенциально важные строки
                    if len(line_stripped) > 10:  # Игнорируем очень короткие строки
                        print(f"{Colors.WHITE}{line_stripped}{Colors.RESET}")

        # Ждем завершения процесса
        process.wait()

        if process.returncode == 0:
            print(f"\n\n{Colors.GREEN}{Colors.BOLD}✅ Обучение завершено успешно!{Colors.RESET}")
        else:
            print(f"\n\n{Colors.RED}{Colors.BOLD}❌ Обучение завершилось с ошибкой (код {process.returncode}){Colors.RESET}")

    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Ошибка мониторинга: {e}{Colors.RESET}")

    finally:
        print_summary()

if __name__ == "__main__":
    main()