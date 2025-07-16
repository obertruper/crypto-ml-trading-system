#!/usr/bin/env python3
"""
Удобный скрипт для запуска XGBoost v3.0 с поддержкой удаленных серверов
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
import threading

# Конфигурация серверов
SERVERS = {
    "uk": {
        "host": "ssh1.vast.ai",
        "port": 18645,
        "user": "root",
        "name": "UK EPYC 64-Core + 2×RTX 4090",
        "gpu": "2×RTX 4090 + 64 CPU cores + 251GB RAM",
        "features": ["GPU", "Мощный CPU", "Много RAM"],
        "performance": "Максимальная"
    },
    "vast1": {
        "host": "ssh3.vast.ai",
        "port": 17929,
        "user": "root",
        "name": "Vast.ai RTX 4090 (старый)",
        "gpu": "2 × RTX 4090 (24GB)",
        "features": ["GPU"],
        "performance": "Средняя"
    }
}

# Глобальная конфигурация оптимизаций
OPTIMIZATIONS = {
    "powerful_server": {
        "cpu_threshold": 64,
        "ram_threshold_gb": 100,
        "data_loader_workers": 50,
        "optuna_parallel_jobs": 32,
        "adasyn_neighbors": 15,
        "batch_size": 50
    },
    "normal_server": {
        "cpu_threshold": 8,
        "ram_threshold_gb": 16,
        "data_loader_workers": 10,
        "optuna_parallel_jobs": 1,
        "adasyn_neighbors": 5,
        "batch_size": 10
    }
}

def check_remote_connection(server_info):
    """Проверка подключения к серверу"""
    cmd = f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p {server_info['port']} {server_info['user']}@{server_info['host']} 'echo OK'"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def run_remote_command(server_info, command, monitor=True):
    """Запуск команды на удаленном сервере с мониторингом"""
    ssh_cmd = f"ssh -p {server_info['port']} {server_info['user']}@{server_info['host']}"
    
    # Пробрасываем порт БД
    ssh_cmd += " -R 5555:localhost:5555"
    
    # Проверяем характеристики сервера и применяем оптимизации
    check_cmd = f'{ssh_cmd} "nproc && free -g | grep Mem | awk \'{{print \\$2}}\'"'
    try:
        result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            cpu_count = int(lines[0])
            ram_gb = int(lines[1]) if len(lines) > 1 else 16
            
            # Устанавливаем переменные окружения для оптимизаций
            env_vars = ""
            if cpu_count >= OPTIMIZATIONS["powerful_server"]["cpu_threshold"]:
                print(f"🚀 Обнаружен мощный сервер: {cpu_count} CPU, {ram_gb}GB RAM")
                env_vars = "export SERVER_TYPE=powerful && "
            else:
                env_vars = "export SERVER_TYPE=normal && "
    except:
        env_vars = ""
    
    # Команда для выполнения (создаем папку logs если её нет)
    remote_cmd = f"cd /workspace && mkdir -p logs && {env_vars}{command}"
    
    full_cmd = f'{ssh_cmd} "{remote_cmd}"'
    
    print(f"\n🚀 Выполнение на сервере {server_info['name']}...")
    print(f"📍 Команда: {command}")
    print(f"🔧 Характеристики: {server_info['gpu']}")
    print("="*60)
    
    # Если включен мониторинг, запускаем в отдельном потоке проверку прогресса
    if monitor:
        monitor_thread = threading.Thread(
            target=monitor_remote_training, 
            args=(server_info,),
            daemon=True
        )
        monitor_thread.start()
    
    # Запускаем команду
    start_time = time.time()
    result = os.system(full_cmd)
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️ Время выполнения: {elapsed_time/60:.1f} минут")
    return result

def monitor_remote_training(server_info):
    """Мониторинг прогресса обучения на удаленном сервере"""
    ssh_cmd = f"ssh -p {server_info['port']} {server_info['user']}@{server_info['host']}"
    
    while True:
        time.sleep(30)  # Проверяем каждые 30 секунд
        
        # Проверяем последние строки лога
        check_cmd = f'{ssh_cmd} "tail -n 5 /workspace/logs/*/training.log 2>/dev/null | grep -E \'Epoch|Loss|Accuracy\'"'
        
        try:
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                print(f"\n📊 Прогресс: {result.stdout.strip()}")
        except:
            pass

def sync_to_server(server_info):
    """Синхронизация проекта на сервер с дополнительными файлами"""
    print(f"\n📤 Синхронизация проекта на {server_info['name']}...")
    
    # Синхронизируем основной проект
    rsync_cmd = f"rsync -avz -e 'ssh -p {server_info['port']}' "
    rsync_cmd += "--exclude='logs/' --exclude='__pycache__/' --exclude='.git/' --exclude='*.pyc' "
    rsync_cmd += f"/Users/ruslan/PycharmProjects/LLM\\ TRANSFORM/xgboost_v3/ "
    rsync_cmd += f"{server_info['user']}@{server_info['host']}:/workspace/xgboost_v3/"
    
    result = os.system(rsync_cmd)
    
    # Синхронизируем файл оптимизаций
    if result == 0:
        if os.path.exists(opt_file.replace("\\", "")):
            rsync_opt = f"rsync -avz -e 'ssh -p {server_info['port']}' "
            rsync_opt += f"{opt_file} "
            rsync_opt += f"{server_info['user']}@{server_info['host']}:/workspace/"
            os.system(rsync_opt)
    
    if result == 0:
        print("✅ Синхронизация завершена")
        
        # Проверяем установленные пакеты на сервере
        check_packages_on_server(server_info)
    else:
        print("❌ Ошибка синхронизации")
    return result == 0

def check_packages_on_server(server_info):
    """Проверка и установка необходимых пакетов на сервере"""
    ssh_cmd = f"ssh -p {server_info['port']} {server_info['user']}@{server_info['host']}"
    
    print("📦 Проверка зависимостей...")
    
    # Проверяем наличие основных пакетов
    check_cmd = f'{ssh_cmd} "cd /workspace && python3 -c \'import xgboost, optuna, psutil; print(\\\"✅ Все пакеты установлены\\\")\' 2>&1"'
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    
    if "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
        print("📦 Установка недостающих пакетов...")
        install_cmd = f'{ssh_cmd} "cd /workspace && pip install xgboost optuna psutil GPUtil -q"'
        os.system(install_cmd)

def download_results(server_info):
    """Скачивание результатов с сервера с отображением статистики"""
    print(f"\n📥 Загрузка результатов с {server_info['name']}...")
    
    # Создаем папку для результатов с временной меткой
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"/Users/ruslan/PycharmProjects/LLM TRANSFORM/logs_from_gpu/{server_info['name'].replace(' ', '_')}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    rsync_cmd = f"rsync -avz --progress -e 'ssh -p {server_info['port']}' "
    rsync_cmd += f"{server_info['user']}@{server_info['host']}:/workspace/logs/ "
    rsync_cmd += f"{result_dir.replace(' ', '\\ ')}/"
    
    result = os.system(rsync_cmd)
    
    if result == 0:
        print(f"✅ Результаты загружены в {result_dir}")
        
        # Показываем краткую статистику
        show_results_summary(result_dir)
    else:
        print("❌ Ошибка при загрузке результатов")

def show_results_summary(log_dir):
    """Показать краткую сводку результатов обучения"""
    print("\n📊 Сводка результатов:")
    
    # Ищем последний отчет
    import glob
    reports = glob.glob(f"{log_dir}/**/final_report.txt", recursive=True)
    
    if reports:
        latest_report = max(reports, key=os.path.getmtime)
        print(f"\n📄 Последний отчет: {os.path.basename(os.path.dirname(latest_report))}")
        
        # Читаем ключевые метрики
        with open(latest_report, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if any(key in line for key in ['Accuracy:', 'Precision:', 'Recall:', 'F1:', 'ROC-AUC:']):
                    print(f"   {line.strip()}")
    
    # Проверяем наличие моделей
    models = glob.glob(f"{log_dir}/**/*.json", recursive=True)
    if models:
        print(f"\n💾 Найдено моделей: {len(models)}")
        
    # Проверяем графики
    plots = glob.glob(f"{log_dir}/**/*.png", recursive=True)
    if plots:
        print(f"📊 Найдено графиков: {len(plots)}")

def show_server_performance_analysis():
    """Показать анализ производительности серверов"""
    print("\n📊 Анализ производительности серверов")
    print("="*60)
    
    for key, server in SERVERS.items():
        print(f"\n🖥️ {server['name']}")
        print(f"   Характеристики: {server['gpu']}")
        print(f"   Производительность: {server['performance']}")
        print(f"   Особенности: {', '.join(server['features'])}")
        
        # Проверяем доступность
        if check_remote_connection(server):
            # Получаем детальную информацию
            ssh_cmd = f"ssh -p {server['port']} {server['user']}@{server['host']}"
            
            # CPU и RAM
            cmd = f'{ssh_cmd} "nproc && free -g | grep Mem"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                print(f"   ✅ Доступен: {lines[0]} CPU")
                if len(lines) > 1:
                    mem_info = lines[1].split()
                    print(f"   💾 RAM: {mem_info[1]}GB всего, {mem_info[6]}GB свободно")
            
            # GPU информация
            gpu_cmd = f'{ssh_cmd} "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null"'
            gpu_result = subprocess.run(gpu_cmd, shell=True, capture_output=True, text=True)
            if gpu_result.returncode == 0 and gpu_result.stdout.strip():
                print("   🎮 GPU:")
                for gpu_line in gpu_result.stdout.strip().split('\n'):
                    print(f"      {gpu_line}")
        else:
            print("   ❌ Недоступен")
    
    print("\n💡 Рекомендации:")
    print("   • UK сервер оптимален для продакшн обучения (64 CPU + 251GB RAM)")
    print("   • GPU ускорение дает 10-15x прирост скорости")
    print("   • Для тестов достаточно любого сервера")

def show_optimization_settings():
    """Показать текущие настройки оптимизаций"""
    print("\n🔧 Настройки оптимизаций")
    print("="*60)
    
    # Определяем текущий тип системы
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    print(f"💻 Текущая система: {cpu_count} CPU")
    
    current_type = "powerful_server" if cpu_count >= 64 else "normal_server"
    print(f"📋 Используемый профиль: {current_type}\n")
    
    for profile_name, settings in OPTIMIZATIONS.items():
        print(f"{'🚀' if profile_name == 'powerful_server' else '💻'} Профиль: {profile_name}")
        print(f"   CPU порог: {settings['cpu_threshold']} ядер")
        print(f"   RAM порог: {settings['ram_threshold_gb']} GB")
        print(f"   DataLoader воркеры: {settings['data_loader_workers']}")
        print(f"   Optuna параллельность: {settings['optuna_parallel_jobs']}")
        print(f"   ADASYN соседи: {settings['adasyn_neighbors']}")
        print(f"   Размер батча: {settings['batch_size']}")
        print()
    
    print("💡 Автоматическая оптимизация:")
    print("   • При запуске на сервере автоматически определяются ресурсы")
    print("   • Настройки применяются динамически")
    print("   • Для мощных серверов включается параллельная Optuna")
    
    # Показываем текущие переменные окружения
    if os.environ.get('SERVER_TYPE'):
        print(f"\n🔧 Текущий SERVER_TYPE: {os.environ.get('SERVER_TYPE')}")

def quick_launch_best_server():
    """Быстрый запуск на лучшем доступном сервере"""
    print("\n⚡ Поиск лучшего доступного сервера...")
    
    # Проверяем все серверы и выбираем лучший
    best_server = None
    best_server_key = None
    
    # Приоритет серверов (UK сервер приоритетнее)
    priority_order = ["uk", "vast1"]
    
    for server_key in priority_order:
        server = SERVERS[server_key]
        if check_remote_connection(server):
            print(f"✅ Найден доступный сервер: {server['name']}")
            best_server = server
            best_server_key = server_key
            break
    
    if not best_server:
        print("❌ Нет доступных серверов. Запускаем локально...")
        # Локальный запуск с GPU тестом
        command = "python xgboost_v3/main.py --test-mode --optimize --ensemble-size 5 --gpu"
        print(f"\n🚀 Запуск: {command}\n")
        os.system(command)
        return
    
    print(f"\n🎯 Выбран сервер: {best_server['name']}")
    print(f"   Характеристики: {best_server['gpu']}")
    
    # Автоматическая синхронизация
    print("\n📤 Синхронизация проекта...")
    if not sync_to_server(best_server):
        print("❌ Ошибка синхронизации")
        return
    
    # Выбор команды на основе характеристик сервера
    if "64 CPU" in best_server['gpu']:
        # Мощный сервер - продакшн режим
        print("\n🚀 Запуск в ПРОДАКШН режиме (мощный сервер)")
        command = "python3 xgboost_v3/main.py --task classification_binary --optimize --ensemble-size 10 --gpu"
    else:
        # Обычный GPU сервер - быстрый тест
        print("\n⚡ Запуск в режиме быстрого теста")
        command = "python3 xgboost_v3/main.py --test-mode --optimize --ensemble-size 5 --gpu"
    
    # Запуск с мониторингом
    run_remote_command(best_server, command, monitor=True)
    
    # Предложение скачать результаты
    download_choice = input("\n📥 Скачать результаты? (y/n): ").strip().lower()
    if download_choice == 'y':
        download_results(best_server)

def main():
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        # Прямой запуск с параметрами
        args = ' '.join(sys.argv[1:])
        command = f"python xgboost_v3/main.py {args}"
        print(f"🚀 Запуск: {command}\n")
        os.system(command)
        return
    
    # Выбор режима запуска
    print("""
🤖 XGBoost v3.0.1 - ML система для криптотрейдинга
==========================================================
🆕 ИСПРАВЛЕНИЯ: Temporal признаки ≤ 2%, Technical ≥ 85%

Где запустить обучение?
1. 💻 Локально (CPU)
2. 🚀 На GPU сервере (Vast.ai) 
3. 📥 Скачать результаты с сервера
4. 📊 Анализ производительности серверов
5. 🔧 Настройки оптимизаций
6. ⚡ Быстрый запуск на лучшем сервере (рекомендуется)
7. 🧪 Запустить тесты исправлений
0. ❌ Выход

💡 Рекомендуется: Вариант 6 для автоматического выбора сервера
""")
    
    mode_choice = input("Ваш выбор (0-7): ").strip()
    
    if mode_choice == "0":
        print("Выход...")
        return
    elif mode_choice == "7":
        # Запуск тестов исправлений
        print("\n🧪 Запуск тестов исправлений Feature Selection...")
        test_command = "python xgboost_v3/test_feature_selection_fixes.py"
        print(f"🚀 Выполнение: {test_command}\n")
        os.system(test_command)
        return
    elif mode_choice == "3":
        # Скачивание результатов
        print("\nВыберите сервер:")
        server_map = {}
        for i, (key, server) in enumerate(SERVERS.items(), 1):
            print(f"{i}. {server['name']} ({server['gpu']})")
            server_map[str(i)] = key
            
        server_choice = input("Выберите сервер (1-2): ").strip()
        
        # Поддержка выбора по номеру
        if server_choice in server_map:
            server_choice = server_map[server_choice]
            
        if server_choice in SERVERS:
            download_results(SERVERS[server_choice])
        else:
            print("❌ Неверный выбор сервера")
        return
    elif mode_choice == "4":
        # Анализ производительности
        show_server_performance_analysis()
        return
    elif mode_choice == "5":
        # Настройки оптимизаций
        show_optimization_settings()
        return
    elif mode_choice == "6":
        # Быстрый запуск на лучшем сервере
        quick_launch_best_server()
        return
    elif mode_choice == "2":
        # Удаленный запуск
        print("\n🖥️ Доступные серверы:")
        available_servers = []
        server_map = {}  # Для поддержки выбора по номеру
        
        for i, (key, server) in enumerate(SERVERS.items(), 1):
            if check_remote_connection(server):
                print(f"✅ {i}. {server['name']} - {server['gpu']}")
                available_servers.append(key)
                server_map[str(i)] = key
            else:
                print(f"❌ {i}. {server['name']} - недоступен")
        
        if not available_servers:
            print("\n❌ Нет доступных серверов")
            return
            
        server_choice = input("\nВыберите сервер (1-2 или vast1/vast2): ").strip()
        
        # Поддержка выбора по номеру
        if server_choice in server_map:
            server_choice = server_map[server_choice]
            
        if server_choice not in available_servers:
            print("❌ Неверный выбор или сервер недоступен")
            return
            
        server_info = SERVERS[server_choice]
        
        # Синхронизация
        sync_choice = input("\n📤 Синхронизировать проект? (y/n): ").strip().lower()
        if sync_choice == 'y':
            if not sync_to_server(server_info):
                return
                
        # Выбор команды для удаленного запуска
        remote_mode = "remote"
    else:
        remote_mode = "local"
        server_info = None
    
    # Основное меню
    print(f"""
🤖 XGBoost v3.0 - ML система для криптотрейдинга
==========================================================
{'🚀 РЕЖИМ: GPU СЕРВЕР - ' + server_info['name'] if remote_mode == 'remote' else '💻 РЕЖИМ: ЛОКАЛЬНЫЙ ЗАПУСК'}

⚡ БЫСТРЫЙ СТАРТ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 🧪 ТЕСТ с исправлениями Feature Selection (НОВОЕ!)
   • ✅ Temporal признаки ≤ 2% (исправлено переобучение)
   • ✅ Technical признаки ≥ 85% 
   • ✅ Blacklist для dow_cos, dow_sin, is_weekend
   • ✅ Валидация важности после обучения
   • 📊 2 символа, 80 признаков, 5 моделей
   
2. ⚡ GPU ТЕСТ - Быстрая проверка (2-3 мин)
   • 🎮 RTX 4090 ускорение (10-15x)
   • 📊 BTCUSDT + ETHUSDT
   • 🔍 80 признаков с новой логикой отбора

ПОЛНОЕ ОБУЧЕНИЕ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. 🎯 ПРОДАКШН с исправлениями (CPU)
   • 📊 Все 51 символ фьючерсов
   • 🔍 120 признаков (85% technical)
   • 🤖 10 моделей в ансамбле
   • ⏱️ ~2-3 часа на CPU
   
4. 🚀 ПРОДАКШН на GPU (15-20 мин)
   • 🎮 2×RTX 4090 + 64 CPU
   • 📊 Все символы, полная оптимизация
   • 🔍 Параллельная Optuna (32 воркера)
   • ✅ Автоматическая валидация

СПЕЦИАЛЬНЫЕ РЕЖИМЫ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. 📈 Регрессия (expected returns)
   • Предсказание точной доходности
   • Оптимизация MAE вместо AUC
   
6. 🔧 Минимальный тест (отладка)
   • 2 символа, 3 модели, 20 Optuna
   • Для быстрой проверки изменений
   
7. 🔍 Анализ Feature Importance
   • Проверка важности признаков
   • Выявление переобучения на temporal
   
8. 📋 Кастомные параметры
0. ❌ Выход

🆕 НОВОЕ в v3.0.1:
• Исправлено переобучение на календарных эффектах
• Temporal признаки жестко ограничены 2%
• Добавлена валидация важности признаков
• Уменьшены дублирующие rolling windows

💡 Рекомендуется: Вариант 1 или 2 для проверки исправлений
""")
    
    choice = input("Ваш выбор (0-8): ").strip()
    
    # Определяем команду python в зависимости от режима
    python_cmd = "python3" if remote_mode == "remote" else "python"
    
    commands = {
        "1": f"{python_cmd} xgboost_v3/main.py --test-mode --optimize --ensemble-size 5",  # Тест с исправлениями
        "2": f"{python_cmd} xgboost_v3/main.py --test-mode --optimize --ensemble-size 5 --gpu",  # GPU тест
        "3": f"{python_cmd} xgboost_v3/main.py --task classification_binary --optimize --ensemble-size 10",  # Продакшн CPU
        "4": f"{python_cmd} xgboost_v3/main.py --task classification_binary --optimize --ensemble-size 10 --gpu",  # Продакшн GPU
        "5": f"{python_cmd} xgboost_v3/main.py --task regression --optimize --test-mode",  # Регрессия
        "6": f"{python_cmd} xgboost_v3/main.py --test-mode --optimize --ensemble-size 3",  # Минимальный тест
        "7": f"{python_cmd} xgboost_v3/check_feature_importance.py",  # Анализ важности признаков
    }
    
    if choice == "0":
        print("Выход...")
        return
        
    elif choice == "8":
        print("\n📋 Доступные параметры:")
        print("  --task [classification_binary|regression] - тип задачи")
        print("  --test-mode - быстрый тест на 2 символах")
        print("  --no-cache - не использовать кэш данных")
        print("  --optimize - запустить Optuna оптимизацию")
        print("  --ensemble-size N - размер ансамбля (по умолчанию 5)")
        print("  --gpu - использовать GPU для обучения")
        print("  --config PATH - путь к конфигурации YAML")
        print("\n🆕 Новые возможности v3.0.1:")
        print("  • Hierarchical feature selection: 85% technical, 2% temporal")
        print("  • Temporal blacklist: dow_cos, dow_sin, is_weekend исключены")
        print("  • Feature importance validation после обучения")
        print("  • Уменьшены rolling windows: [20, 60] вместо [5, 10, 20, 60]")
        print("  • Автоматическая проверка на переобучение")
        print("\n💡 Примеры команд:")
        print("  --test-mode --optimize --ensemble-size 3  # минимальный тест")
        print("  --task regression --optimize  # регрессия с оптимизацией")
        print("  --optimize --ensemble-size 10 --gpu  # продакшн на GPU")
        print("\nВведите параметры:")
        command = input(f"> {python_cmd} xgboost_v3/main.py ")
        command = f"{python_cmd} xgboost_v3/main.py {command}"
        
    elif choice in commands:
        command = commands[choice]
        print(f"\n📊 Конфигурация:")
        
        # Специальные описания для разных режимов
        if choice == "1":
            print("  🧪 ТЕСТ с исправлениями Feature Selection")
            print("  ✅ Temporal признаки жестко ограничены 2%")
            print("  ✅ Technical признаки приоритет 85%")
            print("  ✅ Blacklist: dow_cos, dow_sin, is_weekend")
            print("  ✅ Валидация важности после обучения")
            print("  📊 Ожидаемое время: 5-10 минут на CPU")
        elif choice == "2":
            print("  ⚡ GPU ТЕСТ - быстрая проверка исправлений")
            print("  🎮 GPU ускорение: 10-15x")
            print("  📊 2 символа для быстрой проверки")
            print("  ⏱️ Ожидаемое время: 2-3 минуты")
            if remote_mode == "remote" and "64 CPU" in server_info['gpu']:
                print("  ✨ + 64 CPU для параллельной Optuna")
        elif choice == "3":
            print("  🎯 ПРОДАКШН режим с исправлениями (CPU)")
            print("  📊 Все 51 символ фьючерсов")
            print("  🔍 120 признаков с новой логикой отбора")
            print("  🤖 10 моделей в ансамбле")
            print("  ⏱️ Ожидаемое время: 2-3 часа")
        elif choice == "4":
            print("  🚀 ПРОДАКШН на GPU - максимальная скорость")
            print("  🎮 2×RTX 4090 доступно")
            print("  📊 Все символы + полная оптимизация")
            print("  ⏱️ Ожидаемое время: 15-20 минут")
            if remote_mode == "remote" and "64 CPU" in server_info['gpu']:
                print("  ✨ 32 параллельных Optuna воркера")
                print("  ✨ 251GB RAM для больших батчей")
        elif choice == "5":
            print("  📈 РЕГРЕССИЯ - предсказание expected returns")
            print("  📊 Точная доходность вместо бинарной классификации")
            print("  🎯 Оптимизация MAE вместо AUC")
            print("  ⏱️ Ожидаемое время: 5-10 минут")
        elif choice == "6":
            print("  🔧 МИНИМАЛЬНЫЙ ТЕСТ для отладки")
            print("  📊 2 символа, 3 модели")
            print("  🎯 20 попыток Optuna (быстро)")
            print("  ⏱️ Ожидаемое время: 3-5 минут")
        elif choice == "7":
            print("  🔍 АНАЛИЗ Feature Importance")
            print("  📊 Проверка важности признаков обученных моделей")
            print("  ⚠️ Выявление переобучения на temporal")
            print("  ✅ Рекомендации по улучшению")
            
        if choice != "7":  # Не для анализа важности
            if "optimize" in command:
                if choice == "6":
                    print("  ✅ Optuna оптимизация: 20 попыток (минимальная)")
                else:
                    print("  ✅ Optuna оптимизация: 100 попыток")
            else:
                print("  ⚠️  Оптимизация отключена (базовые параметры)")
                
            if "test-mode" in command:
                print("  ✅ Тест режим: BTCUSDT, ETHUSDT")
            else:
                print("  ✅ Полный датасет: 51 символ фьючерсов")
                
            if "ensemble-size" in command:
                size = command.split("ensemble-size")[1].strip().split()[0]
                print(f"  ✅ Размер ансамбля: {size} моделей")
                
            if "regression" in command:
                print("  ✅ Режим: регрессия (expected returns)")
            else:
                print("  ✅ Режим: бинарная классификация (порог 1.5%)")
                
            if "--gpu" in command:
                print("  ✅ Устройство: GPU (RTX 4090)")
            else:
                print("  ✅ Устройство: CPU")
                
            # Feature selection с новыми правилами
            if choice in ["1", "2", "5", "6"]:
                print("  ✅ Feature Selection: топ-80 признаков")
                print("  ✅ Hierarchical отбор: 85% technical, 2% temporal")
            else:
                print("  ✅ Feature Selection: топ-120 признаков")
                print("  ✅ Hierarchical отбор: 85% technical, 2% temporal")
                
            print("  ✅ Балансировка: ADASYN")
            print("  ✅ Temporal blacklist: dow_cos, dow_sin, is_weekend")
            print("  ✅ Валидация важности: включена")
    else:
        print("❌ Неверный выбор")
        return
        
    # Выполнение команды
    if remote_mode == "remote":
        # Удаленный запуск
        print(f"\n🚀 Удаленный запуск на {server_info['name']}")
        run_remote_command(server_info, command)
        
        # Предложить скачать результаты
        download_choice = input("\n📥 Скачать результаты? (y/n): ").strip().lower()
        if download_choice == 'y':
            download_results(server_info)
    else:
        # Локальный запуск
        print(f"\n🚀 Запуск: {command}\n")
        print("="*60)
        
        # Запускаем команду
        result = os.system(command)
        
        # Показываем результат
        if result == 0:
            print("\n✅ Обучение завершено успешно!")
            print("📁 Результаты сохранены в папке logs/")
            print("📊 Проверьте файлы:")
            print("   • final_report.txt - итоговый отчет")
            print("   • metrics.json - детальные метрики")
            print("   • plots/ - графики обучения")
        else:
            print("\n❌ Произошла ошибка при обучении")
            print("📋 Проверьте логи в папке logs/")


if __name__ == "__main__":
    main()