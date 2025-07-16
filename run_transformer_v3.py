#!/usr/bin/env python3
"""
Интерактивное меню для запуска Transformer v3
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Цвета для терминала
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def clear_screen():
    """Очистка экрана"""
    os.system('clear' if os.name == 'posix' else 'cls')


def print_header():
    """Печать заголовка"""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         TRANSFORMER v3 - ML CRYPTO TRADING               ║")
    print("║              Интерактивное меню запуска                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")


def print_menu():
    """Главное меню"""
    print(f"{Colors.YELLOW}═══ ГЛАВНОЕ МЕНЮ ═══{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}[1]{Colors.ENDC} 🚀 Полное обучение (Регрессия)")
    print(f"    └─ Предсказание expected returns для всех символов")
    
    print(f"\n{Colors.GREEN}[2]{Colors.ENDC} 🎯 Полное обучение (Классификация)")
    print(f"    └─ Предсказание profit/loss для всех символов")
    
    print(f"\n{Colors.GREEN}[3]{Colors.ENDC} 🧪 Тестовый режим (Быстрая проверка)")
    print(f"    └─ 2 символа, 10 эпох для проверки работоспособности")
    
    print(f"\n{Colors.GREEN}[4]{Colors.ENDC} ⚙️  Кастомные настройки")
    print(f"    └─ Выбор параметров обучения")
    
    print(f"\n{Colors.GREEN}[5]{Colors.ENDC} 📊 Мониторинг последнего обучения")
    print(f"    └─ TensorBoard для просмотра метрик")
    
    print(f"\n{Colors.GREEN}[6]{Colors.ENDC} 🗄️  Подготовка данных")
    print(f"    └─ Загрузка и обработка данных")
    
    print(f"\n{Colors.GREEN}[7]{Colors.ENDC} 📈 Просмотр результатов")
    print(f"    └─ Последние логи и графики")
    
    print(f"\n{Colors.RED}[0]{Colors.ENDC} ❌ Выход\n")


def run_command(cmd, description=""):
    """Запуск команды с отображением прогресса"""
    if description:
        print(f"\n{Colors.CYAN}▶ {description}{Colors.ENDC}")
    
    print(f"{Colors.YELLOW}Команда: {cmd}{Colors.ENDC}\n")
    
    try:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Читаем вывод построчно
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n{Colors.GREEN}✅ Успешно завершено!{Colors.ENDC}")
        else:
            print(f"\n{Colors.RED}❌ Ошибка! Код возврата: {process.returncode}{Colors.ENDC}")
            
        return process.returncode
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Прервано пользователем{Colors.ENDC}")
        process.terminate()
        return -1
    except Exception as e:
        print(f"\n{Colors.RED}❌ Ошибка: {e}{Colors.ENDC}")
        return -1


def custom_settings_menu():
    """Меню кастомных настроек"""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}⚙️  КАСТОМНЫЕ НАСТРОЙКИ{Colors.ENDC}\n")
    
    # Задача
    print(f"{Colors.YELLOW}Выберите тип задачи:{Colors.ENDC}")
    print("[1] Регрессия (expected returns)")
    print("[2] Классификация (profit/loss)")
    task_choice = input(f"\n{Colors.GREEN}Выбор (1-2): {Colors.ENDC}")
    task = "regression" if task_choice == "1" else "classification_binary"
    
    # Эпохи
    print(f"\n{Colors.YELLOW}Количество эпох:{Colors.ENDC}")
    print("[1] 50 (быстро)")
    print("[2] 100 (стандарт)")
    print("[3] 200 (долго)")
    print("[4] Свое значение")
    epoch_choice = input(f"\n{Colors.GREEN}Выбор (1-4): {Colors.ENDC}")
    
    if epoch_choice == "1":
        epochs = 50
    elif epoch_choice == "2":
        epochs = 100
    elif epoch_choice == "3":
        epochs = 200
    else:
        epochs = int(input(f"{Colors.GREEN}Введите количество эпох: {Colors.ENDC}"))
    
    # Размер батча
    print(f"\n{Colors.YELLOW}Размер батча:{Colors.ENDC}")
    print("[1] 32 (меньше памяти)")
    print("[2] 64 (стандарт)")
    print("[3] 128 (больше памяти)")
    print("[4] 256 (много памяти)")
    batch_choice = input(f"\n{Colors.GREEN}Выбор (1-4): {Colors.ENDC}")
    
    batch_sizes = {"1": 32, "2": 64, "3": 128, "4": 256}
    batch_size = batch_sizes.get(batch_choice, 64)
    
    # Ансамбль
    print(f"\n{Colors.YELLOW}Размер ансамбля:{Colors.ENDC}")
    print("[1] 1 модель (быстро)")
    print("[2] 3 модели (стандарт)")
    print("[3] 5 моделей (точнее)")
    ensemble_choice = input(f"\n{Colors.GREEN}Выбор (1-3): {Colors.ENDC}")
    
    ensemble_sizes = {"1": 1, "2": 3, "3": 5}
    ensemble_size = ensemble_sizes.get(ensemble_choice, 3)
    
    # Символы
    print(f"\n{Colors.YELLOW}Символы для обучения:{Colors.ENDC}")
    print("[1] Все доступные")
    print("[2] Топ-10 по объему")
    print("[3] Только BTC и ETH")
    print("[4] Выбрать вручную")
    symbol_choice = input(f"\n{Colors.GREEN}Выбор (1-4): {Colors.ENDC}")
    
    symbols_cmd = ""
    if symbol_choice == "2":
        symbols_cmd = "--limit-symbols 10"
    elif symbol_choice == "3":
        symbols_cmd = "--test-symbols BTCUSDT ETHUSDT"
    elif symbol_choice == "4":
        symbols = input(f"{Colors.GREEN}Введите символы через пробел: {Colors.ENDC}")
        symbols_cmd = f"--test-symbols {symbols}"
    
    # Формируем команду
    cmd = f"python transformer_v3/main.py --task {task} --epochs {epochs} --batch-size {batch_size} --ensemble-size {ensemble_size} {symbols_cmd}"
    
    print(f"\n{Colors.CYAN}Сформированная команда:{Colors.ENDC}")
    print(f"{Colors.YELLOW}{cmd}{Colors.ENDC}")
    
    confirm = input(f"\n{Colors.GREEN}Запустить? (y/n): {Colors.ENDC}")
    if confirm.lower() == 'y':
        run_command(cmd, "Запуск обучения с кастомными настройками")
    
    input(f"\n{Colors.YELLOW}Нажмите Enter для возврата в меню...{Colors.ENDC}")


def data_preparation_menu():
    """Меню подготовки данных"""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}🗄️  ПОДГОТОВКА ДАННЫХ{Colors.ENDC}\n")
    
    print("[1] Полный пайплайн (init → download → prepare)")
    print("[2] Только инициализация БД")
    print("[3] Только загрузка данных")
    print("[4] Только подготовка датасета")
    print("[5] Проверка данных в БД")
    print("[0] Назад")
    
    choice = input(f"\n{Colors.GREEN}Выбор: {Colors.ENDC}")
    
    if choice == "1":
        run_command("python run_futures_pipeline.py", "Запуск полного пайплайна подготовки данных")
    elif choice == "2":
        run_command("python init_database.py", "Инициализация базы данных")
    elif choice == "3":
        run_command("python download_data.py", "Загрузка данных с Bybit")
    elif choice == "4":
        run_command("python prepare_dataset.py", "Подготовка датасета с индикаторами")
    elif choice == "5":
        cmd = """psql -h localhost -p 5555 -U ruslan -d crypto_trading -c "
            SELECT symbol, COUNT(*) as records, 
                   MIN(timestamp) as first_date, 
                   MAX(timestamp) as last_date 
            FROM raw_market_data 
            GROUP BY symbol 
            ORDER BY records DESC 
            LIMIT 20;"
        """
        run_command(cmd, "Проверка данных в БД")
    
    if choice != "0":
        input(f"\n{Colors.YELLOW}Нажмите Enter для возврата в меню...{Colors.ENDC}")


def view_results():
    """Просмотр результатов последнего обучения"""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ{Colors.ENDC}\n")
    
    # Находим последнюю папку с логами
    log_base = Path("logs")
    if not log_base.exists():
        print(f"{Colors.RED}Папка с логами не найдена{Colors.ENDC}")
        input(f"\n{Colors.YELLOW}Нажмите Enter для возврата...{Colors.ENDC}")
        return
    
    # Ищем папки training_*
    training_dirs = sorted([d for d in log_base.iterdir() if d.is_dir() and d.name.startswith("training_")])
    
    if not training_dirs:
        print(f"{Colors.RED}Логи обучения не найдены{Colors.ENDC}")
        input(f"\n{Colors.YELLOW}Нажмите Enter для возврата...{Colors.ENDC}")
        return
    
    latest_dir = training_dirs[-1]
    print(f"{Colors.GREEN}Последнее обучение: {latest_dir.name}{Colors.ENDC}\n")
    
    # Показываем доступные файлы
    print(f"{Colors.YELLOW}Доступные результаты:{Colors.ENDC}")
    
    files_to_check = [
        ("final_report.txt", "📄 Итоговый отчет"),
        ("plots/training_progress.png", "📊 График обучения"),
        ("plots/final_evaluation.png", "📈 Финальная оценка"),
        ("training.log", "📝 Полный лог обучения")
    ]
    
    available_files = []
    for i, (file_path, description) in enumerate(files_to_check, 1):
        full_path = latest_dir / file_path
        if full_path.exists():
            print(f"[{i}] {description}")
            available_files.append((i, full_path))
    
    print(f"\n[0] Назад")
    
    choice = input(f"\n{Colors.GREEN}Что открыть? {Colors.ENDC}")
    
    if choice == "0":
        return
    
    try:
        choice_num = int(choice)
        for num, path in available_files:
            if num == choice_num:
                if path.suffix in ['.png', '.jpg']:
                    # Открываем изображение
                    if sys.platform == "darwin":  # macOS
                        subprocess.run(["open", str(path)])
                    elif sys.platform == "linux":
                        subprocess.run(["xdg-open", str(path)])
                    else:  # Windows
                        subprocess.run(["start", str(path)], shell=True)
                else:
                    # Показываем текстовый файл
                    if path.name == "training.log":
                        # Показываем последние 50 строк лога
                        subprocess.run(["tail", "-n", "50", str(path)])
                    else:
                        subprocess.run(["cat", str(path)])
                break
    except:
        pass
    
    input(f"\n{Colors.YELLOW}Нажмите Enter для возврата...{Colors.ENDC}")


def monitor_training():
    """Запуск мониторинга обучения"""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}📊 МОНИТОРИНГ ОБУЧЕНИЯ{Colors.ENDC}\n")
    
    print("[1] TensorBoard (детальные метрики)")
    print("[2] Monitor скрипт (текстовый мониторинг)")
    print("[0] Назад")
    
    choice = input(f"\n{Colors.GREEN}Выбор: {Colors.ENDC}")
    
    if choice == "1":
        # Находим последнюю папку tensorboard
        log_base = Path("logs")
        training_dirs = sorted([d for d in log_base.iterdir() if d.is_dir() and d.name.startswith("training_")])
        
        if training_dirs:
            latest_dir = training_dirs[-1]
            tb_dir = latest_dir / "tensorboard"
            if tb_dir.exists():
                print(f"\n{Colors.YELLOW}Запускаю TensorBoard...{Colors.ENDC}")
                print(f"Откройте в браузере: {Colors.CYAN}http://localhost:6006{Colors.ENDC}\n")
                run_command(f"tensorboard --logdir {tb_dir}", "TensorBoard")
            else:
                print(f"{Colors.RED}TensorBoard логи не найдены{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Логи обучения не найдены{Colors.ENDC}")
            
    elif choice == "2":
        run_command("python monitor_training.py", "Мониторинг обучения")
    
    if choice != "0":
        input(f"\n{Colors.YELLOW}Нажмите Enter для возврата...{Colors.ENDC}")


def main():
    """Главная функция"""
    while True:
        print_header()
        print_menu()
        
        choice = input(f"{Colors.GREEN}Выберите действие (0-7): {Colors.ENDC}")
        
        if choice == "0":
            print(f"\n{Colors.CYAN}До свидания! 👋{Colors.ENDC}\n")
            break
            
        elif choice == "1":
            # Полное обучение - регрессия с выбором кэша
            clear_screen()
            print(f"{Colors.CYAN}{Colors.BOLD}🚀 ПОЛНОЕ ОБУЧЕНИЕ (РЕГРЕССИЯ){Colors.ENDC}\n")
            print(f"{Colors.YELLOW}Выберите режим работы с данными:{Colors.ENDC}\n")
            print("[1] 📦 Использовать кэш последовательностей")
            print("    └─ Быстрый старт с готовыми данными (sequence_length=100)")
            print("[2] 🔄 Пересчитать последовательности") 
            print("    └─ Новые параметры (sequence_length=50, новые признаки)")
            print("[3] 🆕 Упрощенная архитектура без кэша")
            print("    └─ main_simplified.py с новыми параметрами")
            print("[0] ↩️  Назад в главное меню\n")
            
            cache_choice = input(f"{Colors.GREEN}Выберите действие (0-3): {Colors.ENDC}")
            
            if cache_choice == "1":
                cmd = "python transformer_v3/main.py --task regression"
                run_command(cmd, "Запуск с кэшем последовательностей")
            elif cache_choice == "2":
                # Удаляем кэш последовательностей и запускаем
                print(f"\n{Colors.YELLOW}Удаляю старый кэш последовательностей...{Colors.ENDC}")
                subprocess.run(["ssh", "-p", "42244", "root@84.68.60.115", "rm -f cache/transformer_v3/sequences_*.pkl"])
                cmd = "python transformer_v3/main.py --task regression"
                run_command(cmd, "Запуск с пересчетом последовательностей")
            elif cache_choice == "3":
                cmd = "python transformer_v3/main_simplified.py --task regression --no-cache"
                run_command(cmd, "Запуск упрощенной архитектуры")
            elif cache_choice == "0":
                continue
            else:
                print(f"\n{Colors.RED}Неверный выбор!{Colors.ENDC}")
                time.sleep(1)
                continue
                
            input(f"\n{Colors.YELLOW}Нажмите Enter для возврата в меню...{Colors.ENDC}")
            
        elif choice == "2":
            # Полное обучение - классификация
            cmd = "python transformer_v3/main.py --task classification_binary"
            run_command(cmd, "Запуск полного обучения (Классификация)")
            input(f"\n{Colors.YELLOW}Нажмите Enter для возврата в меню...{Colors.ENDC}")
            
        elif choice == "3":
            # Тестовый режим
            cmd = "python transformer_v3/main.py --test-mode"
            run_command(cmd, "Запуск тестового режима")
            input(f"\n{Colors.YELLOW}Нажмите Enter для возврата в меню...{Colors.ENDC}")
            
        elif choice == "4":
            # Кастомные настройки
            custom_settings_menu()
            
        elif choice == "5":
            # Мониторинг
            monitor_training()
            
        elif choice == "6":
            # Подготовка данных
            data_preparation_menu()
            
        elif choice == "7":
            # Просмотр результатов
            view_results()
            
        else:
            print(f"\n{Colors.RED}Неверный выбор!{Colors.ENDC}")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Прервано пользователем{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Ошибка: {e}{Colors.ENDC}\n")