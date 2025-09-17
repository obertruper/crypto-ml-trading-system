#!/usr/bin/env python3
"""
Монитор с отображением процессов и обучения одновременно
"""
import subprocess
import sys
import time
import re
from datetime import datetime
import signal
import os
import threading
import psutil
import shutil

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
    CLEAR_LINE = '\033[K'
    CURSOR_UP = '\033[A'

# Глобальные переменные
training_process = None
stop_monitoring = False
terminal_width = shutil.get_terminal_size().columns

def signal_handler(sig, frame):
    """Обработка Ctrl+C"""
    global stop_monitoring, training_process
    stop_monitoring = True
    if training_process:
        training_process.terminate()
    print(f"\n{Colors.YELLOW}🛑 Остановка...{Colors.RESET}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_python_processes():
    """Получение списка Python процессов связанных с обучением"""
    processes = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                # Ищем python процессы
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''

                    # Фильтруем только процессы связанные с обучением
                    if any(x in cmdline for x in ['main.py', 'train', 'model', 'dataset', 'loader']):
                        mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                        cpu = proc.cpu_percent(interval=0.1)

                        # Сокращаем командную строку
                        if 'main.py' in cmdline:
                            short_cmd = "main.py --mode staged"
                        elif 'DataLoader' in cmdline:
                            short_cmd = "DataLoader Worker"
                        else:
                            # Берем только имя файла
                            parts = cmdline.split()
                            for part in parts:
                                if '.py' in part:
                                    short_cmd = os.path.basename(part)
                                    break
                            else:
                                short_cmd = proc.info['name']

                        processes.append({
                            'pid': proc.info['pid'],
                            'cmd': short_cmd,
                            'cpu': cpu,
                            'mem_mb': mem_mb
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass

    return sorted(processes, key=lambda x: x['cpu'], reverse=True)[:5]  # Топ 5 процессов

def get_gpu_info():
    """Получение информации о GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                return {
                    'util': int(parts[0]),
                    'mem_used': int(parts[1]),
                    'mem_total': int(parts[2]),
                    'temp': int(parts[3])
                }
    except Exception:
        pass
    return None

def monitor_system():
    """Поток для мониторинга системы"""
    global stop_monitoring

    while not stop_monitoring:
        try:
            # Очищаем область для системной информации
            print(f"\033[s", end='')  # Сохраняем позицию курсора

            # GPU информация
            gpu_info = get_gpu_info()
            if gpu_info:
                gpu_bar_length = 20
                gpu_filled = int(gpu_bar_length * gpu_info['util'] / 100)
                gpu_bar = '█' * gpu_filled + '░' * (gpu_bar_length - gpu_filled)

                mem_percent = (gpu_info['mem_used'] / gpu_info['mem_total']) * 100
                mem_filled = int(gpu_bar_length * mem_percent / 100)
                mem_bar = '█' * mem_filled + '░' * (gpu_bar_length - mem_filled)

                # Цвет температуры
                temp_color = Colors.GREEN if gpu_info['temp'] < 70 else Colors.YELLOW if gpu_info['temp'] < 80 else Colors.RED

                print(f"\r{Colors.CYAN}GPU: [{gpu_bar}] {gpu_info['util']:3d}% | "
                      f"MEM: [{mem_bar}] {gpu_info['mem_used']}/{gpu_info['mem_total']}MB | "
                      f"{temp_color}TEMP: {gpu_info['temp']}°C{Colors.RESET}{Colors.CLEAR_LINE}")

            # CPU и память системы
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()

            cpu_bar_length = 20
            cpu_filled = int(cpu_bar_length * cpu_percent / 100)
            cpu_bar = '█' * cpu_filled + '░' * (cpu_bar_length - cpu_filled)

            mem_filled = int(cpu_bar_length * mem.percent / 100)
            mem_bar = '█' * mem_filled + '░' * (cpu_bar_length - mem_filled)

            print(f"\r{Colors.BLUE}CPU: [{cpu_bar}] {cpu_percent:5.1f}% | "
                  f"RAM: [{mem_bar}] {mem.used//1024//1024}MB/{mem.total//1024//1024}MB{Colors.RESET}{Colors.CLEAR_LINE}")

            # Python процессы
            processes = get_python_processes()
            if processes:
                print(f"\r{Colors.MAGENTA}Процессы:{Colors.RESET}{Colors.CLEAR_LINE}")
                for proc in processes:
                    # Цвет для CPU использования
                    cpu_color = Colors.RED if proc['cpu'] > 80 else Colors.YELLOW if proc['cpu'] > 50 else Colors.GREEN
                    print(f"\r  {Colors.WHITE}[{proc['pid']:5d}] {proc['cmd'][:30]:30s} "
                          f"{cpu_color}CPU: {proc['cpu']:5.1f}%{Colors.RESET} "
                          f"MEM: {proc['mem_mb']:7.1f}MB{Colors.CLEAR_LINE}")

            print(f"\033[u", end='')  # Восстанавливаем позицию курсора

            time.sleep(2)  # Обновляем каждые 2 секунды

        except Exception as e:
            # Игнорируем ошибки чтобы не засорять вывод
            pass

def parse_training_line(line):
    """Парсинг строки обучения"""
    # Детекция схлопывания
    if "СХЛОПЫВАНИЕ" in line or "схлопывается" in line or "FLAT: 9" in line or "FLAT: 10" in line:
        print(f"\n{Colors.RED}{Colors.BOLD}🚨 ОБНАРУЖЕНО СХЛОПЫВАНИЕ МОДЕЛИ!{Colors.RESET}")
        print(f"{Colors.RED}{line}{Colors.RESET}")
        return True

    # Парсинг прогресса обучения
    training_match = re.search(r'Training:\s*(\d+)%.*loss=([0-9.]+).*samples/s=(\d+)', line)
    if training_match:
        percent = training_match.group(1)
        loss = float(training_match.group(2))
        samples_s = int(training_match.group(3))

        bar_length = 40
        filled = int(bar_length * int(percent) / 100)
        bar = '█' * filled + '░' * (bar_length - filled)

        loss_color = Colors.GREEN if loss < 1.5 else Colors.YELLOW if loss < 2.0 else Colors.RED

        # Оставляем место для системной информации сверху
        print(f"\n\n\n\n\n\n\n", end='')  # Пропускаем строки для системной информации

        output = f"\r{Colors.CYAN}Training: {Colors.BOLD}{percent:>3}%{Colors.RESET} "
        output += f"{Colors.CYAN}[{bar}]{Colors.RESET} "
        output += f"Loss: {loss_color}{loss:.4f}{Colors.RESET} "
        output += f"Speed: {Colors.GREEN}{samples_s:,}/s{Colors.RESET}"

        if loss < 0.0001:
            output += f" {Colors.RED}⚠️ ОЧЕНЬ МАЛЫЙ LOSS!{Colors.RESET}"

        print(output, end='', flush=True)
        return True

    # Детекция эпох
    if "Epoch" in line and "/" in line:
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}{line.strip()}{Colors.RESET}")
        return True

    # Детекция важных событий
    if any(keyword in line for keyword in ["WARNING", "ERROR", "✅", "🚀", "📊", "🔍"]):
        if "WARNING" in line or "⚠️" in line:
            print(f"\n{Colors.YELLOW}{line.strip()}{Colors.RESET}")
        elif "ERROR" in line or "❌" in line:
            print(f"\n{Colors.RED}{Colors.BOLD}{line.strip()}{Colors.RESET}")
        else:
            print(f"\n{Colors.GREEN}{line.strip()}{Colors.RESET}")
        return True

    # Распределение классов
    if "LONG=" in line or "SHORT=" in line or "FLAT=" in line:
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

def main():
    """Основная функция"""
    global training_process, stop_monitoring

    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"🔍 МОНИТОР ОБУЧЕНИЯ С ОТОБРАЖЕНИЕМ ПРОЦЕССОВ")
    print(f"{'='*70}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}📌 Отслеживаем:")
    print(f"  • Системные ресурсы (GPU/CPU/RAM)")
    print(f"  • Python процессы")
    print(f"  • Прогресс обучения")
    print(f"  • Схлопывание модели{Colors.RESET}\n")

    # Запускаем поток мониторинга системы
    monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    monitor_thread.start()

    print(f"{Colors.GREEN}🚀 Запуск обучения...{Colors.RESET}\n")
    print(f"{Colors.CYAN}{'─'*70}{Colors.RESET}\n")

    # Оставляем место для системной информации
    print("\n" * 7)

    # Запуск процесса обучения
    cmd = [sys.executable, "main.py", "--mode", "staged"]

    try:
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )

        # Чтение вывода построчно
        for line in training_process.stdout:
            if stop_monitoring:
                break

            # Парсим строки обучения
            if not parse_training_line(line):
                # Показываем важные неразпознанные строки
                line_stripped = line.strip()
                if line_stripped and len(line_stripped) > 10:
                    if not any(skip in line for skip in ['matplotlib', 'UserWarning', 'import']):
                        print(f"\n{Colors.WHITE}{line_stripped}{Colors.RESET}")

        training_process.wait()
        stop_monitoring = True

        if training_process.returncode == 0:
            print(f"\n\n{Colors.GREEN}{Colors.BOLD}✅ Обучение завершено успешно!{Colors.RESET}")
        else:
            print(f"\n\n{Colors.RED}{Colors.BOLD}❌ Обучение завершилось с ошибкой (код {training_process.returncode}){Colors.RESET}")

    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Ошибка: {e}{Colors.RESET}")
    finally:
        stop_monitoring = True
        monitor_thread.join(timeout=1)

if __name__ == "__main__":
    main()