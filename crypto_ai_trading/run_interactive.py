#!/usr/bin/env python3
"""
Интерактивное меню для управления Crypto AI Trading System
"""

import sys
import os
import subprocess
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Настройка логирования
import logging
log_dir = Path("logs/interactive")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"menu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
    ]
)
menu_logger = logging.getLogger('interactive_menu')

# Импорты для интерактивного меню
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from rich.markdown import Markdown

console = Console()

class CryptoTradingMenu:
    """Интерактивное меню для управления системой"""
    
    def __init__(self):
        self.console = Console()
        self.config_path = Path("config/config.yaml")
        self.config = self.load_config()
        self.logger = menu_logger
        self.logger.info(f"Запуск интерактивного меню. Лог файл: {log_file}")
        
        # Определяем режим работы (локальный или удаленный)
        self.is_local_mode = self._detect_local_mode()
        
        # Загружаем профили серверов
        self.profiles_path = Path("config/server_profiles.yaml")
        self.server_profiles = self.load_server_profiles()
        
        if self.is_local_mode:
            self.logger.info("🏠 Работа в локальном режиме")
            self.current_server_profile = "local"
            # Проверяем GPU
            self._check_local_gpu()
        else:
            # SSH параметры подключения из активного профиля или конфига
            if self.server_profiles:
                active_profile = self.server_profiles.get('active_profile')
                if active_profile and active_profile in self.server_profiles.get('profiles', {}):
                    profile = self.server_profiles['profiles'][active_profile]
                    self.ssh_host = profile['connection']['host']
                    self.ssh_port = str(profile['connection']['port'])
                    self.ssh_user = profile['connection']['user']
                    self.ssh_key_path = profile['connection']['key_path']
                    self.current_server_profile = active_profile
                    self.logger.info(f"Используется профиль сервера: {active_profile} - {profile['name']}")
                else:
                    self.logger.warning("Активный профиль сервера не найден, используются настройки из основного конфига")
                    self._load_config_ssh_params()
            else:
                self.logger.warning("Файл профилей серверов не найден, используются настройки из основного конфига")
                self._load_config_ssh_params()
            
    def _detect_local_mode(self) -> bool:
        """Определение локального режима работы"""
        # Проверяем переменную окружения
        if os.environ.get('CRYPTO_AI_LOCAL_MODE', '').lower() == 'true':
            return True
        
        # Проверяем наличие GPU локально
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if 'RTX 5090' in gpu_name or 'RTX 50' in gpu_name:
                    return True
        except:
            pass
            
        # Проверяем отсутствие SSH конфигурации
        remote_config = self.config.get('remote_server', {})
        if not remote_config.get('enabled', True):
            return True
            
        return False
    
    def _check_local_gpu(self):
        """Проверка локального GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"🎮 Обнаружен GPU: {self.gpu_name} ({self.gpu_memory:.1f} GB)")
            else:
                self.gpu_name = "CPU"
                self.gpu_memory = 0
                self.logger.warning("⚠️  GPU не обнаружен, будет использоваться CPU")
        except Exception as e:
            self.gpu_name = "Не определен"
            self.gpu_memory = 0
            self.logger.error(f"Ошибка проверки GPU: {e}")
    
    def _load_config_ssh_params(self):
        """Загрузка SSH параметров из основного конфига (fallback)"""
        remote_config = self.config.get('remote_server', {})
        direct_config = remote_config.get('direct_connection', {})
        
        self.ssh_host = direct_config.get('host', '109.198.107.223')
        self.ssh_port = str(direct_config.get('port', 48937))
        self.ssh_user = direct_config.get('user', 'root')
        self.ssh_key_path = direct_config.get('key_path', '~/.ssh/id_rsa')
        self.current_server_profile = None
        
    def load_server_profiles(self):
        """Загрузка профилей серверов"""
        try:
            if self.profiles_path.exists():
                with open(self.profiles_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            return None
        except Exception as e:
            self.logger.error(f"Ошибка загрузки профилей серверов: {e}")
            return None
        
    def load_config(self) -> Dict:
        """Загрузка конфигурации"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_config(self):
        """Сохранение конфигурации"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def get_ssh_command(self, command: Optional[str] = None) -> List[str]:
        """Получение SSH команды с правильными параметрами"""
        ssh_cmd = [
            "ssh",
            "-i", os.path.expanduser(self.ssh_key_path),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-p", self.ssh_port,
            f"{self.ssh_user}@{self.ssh_host}"
        ]
        if command:
            ssh_cmd.append(command)
        return ssh_cmd
    
    def get_scp_command(self, source: str, destination: str) -> List[str]:
        """Получение SCP команды с правильными параметрами"""
        return [
            "scp",
            "-i", os.path.expanduser(self.ssh_key_path),
            "-P", self.ssh_port,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            source,
            destination
        ]
    
    def display_main_menu(self):
        """Отображение главного меню"""
        self.console.clear()
        
        # Заголовок
        self.console.print(Panel.fit(
            "[bold cyan]🚀 Crypto AI Trading System[/bold cyan]\n"
            "[dim]Интерактивное управление[/dim]",
            border_style="blue"
        ))
        
        # Опции меню
        menu_options = Table(show_header=False, box=None)
        menu_options.add_column("Option", style="cyan", width=3)
        menu_options.add_column("Description", style="white")
        
        menu_options.add_row("1", "📊 Управление данными")
        menu_options.add_row("2", "🧠 Обучение моделей")
        menu_options.add_row("3", "📈 Мониторинг и логи")
        menu_options.add_row("4", "⚙️  Настройки системы")
        menu_options.add_row("5", "🔧 Утилиты и инструменты")
        menu_options.add_row("6", "📋 Информация о системе")
        menu_options.add_row("7", "🚀 Быстрый запуск")
        menu_options.add_row("0", "❌ Выход")
        
        self.console.print(menu_options)
        
    def data_management_menu(self):
        """Меню управления данными"""
        while True:
            self.console.clear()
            self.console.print(Panel("📊 Управление данными", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Проверить подключение к БД")
            options.add_row("2", "Загрузить данные из БД")
            options.add_row("3", "Создать/обновить признаки")
            options.add_row("4", "Статистика по данным")
            options.add_row("5", "Очистить кэш")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.check_database_connection()
            elif choice == "2":
                self.load_data_from_db()
            elif choice == "3":
                self.create_features()
            elif choice == "4":
                self.show_data_statistics()
            elif choice == "5":
                self.clear_cache()
            elif choice == "0":
                break
    
    def training_menu(self):
        """Меню обучения моделей"""
        while True:
            self.console.clear()
            self.console.print(Panel("🧠 Обучение моделей", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Быстрое обучение (demo)")
            options.add_row("2", "Полное обучение")
            options.add_row("3", "Продолжить обучение")
            options.add_row("4", "Настройки обучения")
            options.add_row("5", "Обучение на GPU (Vast.ai)")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.run_demo_training()
            elif choice == "2":
                self.run_full_training()
            elif choice == "3":
                self.resume_training()
            elif choice == "4":
                self.configure_training()
            elif choice == "5":
                self.run_gpu_training()
            elif choice == "0":
                break
    
    def monitoring_menu(self):
        """Меню мониторинга"""
        while True:
            self.console.clear()
            self.console.print(Panel("📈 Мониторинг и логи", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Мониторинг обучения в реальном времени")
            options.add_row("2", "Просмотр логов")
            options.add_row("3", "TensorBoard")
            options.add_row("4", "Результаты последнего обучения")
            options.add_row("5", "Очистить логи")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.monitor_training()
            elif choice == "2":
                self.view_logs()
            elif choice == "3":
                self.launch_tensorboard()
            elif choice == "4":
                self.show_training_results()
            elif choice == "5":
                self.clear_logs()
            elif choice == "0":
                break
    
    def settings_menu(self):
        """Меню настроек"""
        while True:
            self.console.clear()
            self.console.print(Panel("⚙️ Настройки системы", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Настройки модели")
            options.add_row("2", "Настройки данных")
            options.add_row("3", "Настройки риск-менеджмента")
            options.add_row("4", "Параметры бэктестинга")
            options.add_row("5", "🖥️ Управление профилями серверов")
            options.add_row("6", "Сохранить конфигурацию")
            options.add_row("7", "Загрузить конфигурацию")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.configure_model()
            elif choice == "2":
                self.configure_data()
            elif choice == "3":
                self.configure_risk_management()
            elif choice == "4":
                self.configure_backtesting()
            elif choice == "5":
                self.manage_server_profiles()
            elif choice == "6":
                self.save_config()
                self.console.print("[green]✅ Конфигурация сохранена[/green]")
                time.sleep(1)
            elif choice == "7":
                self.config = self.load_config()
                self.console.print("[green]✅ Конфигурация загружена[/green]")
                time.sleep(1)
            elif choice == "0":
                break
    
    def utilities_menu(self):
        """Меню утилит"""
        while True:
            self.console.clear()
            self.console.print(Panel("🔧 Утилиты и инструменты", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Проверить доступные символы")
            options.add_row("2", "Валидация данных")
            options.add_row("3", "Бэктестинг стратегии")
            options.add_row("4", "Генерация отчетов")
            options.add_row("5", "Экспорт модели")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.check_available_symbols()
            elif choice == "2":
                self.validate_data()
            elif choice == "3":
                self.run_backtesting()
            elif choice == "4":
                self.generate_reports()
            elif choice == "5":
                self.export_model()
            elif choice == "0":
                break
    
    # Реализация функций
    def check_database_connection(self):
        """Проверка подключения к БД"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Проверка подключения к БД...", total=None)
            
            try:
                result = subprocess.run(
                    ["python", "-c", """
import psycopg2
conn = psycopg2.connect(
    host='localhost',
    port=5555,
    database='crypto_trading',
    user='ruslan',
    password='ruslan'
)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM raw_market_data')
count = cursor.fetchone()[0]
print(f'✅ Подключение успешно! Записей в БД: {count:,}')
conn.close()
                    """],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.console.print(f"\n[green]{result.stdout.strip()}[/green]")
                else:
                    self.console.print(f"\n[red]❌ Ошибка подключения: {result.stderr}[/red]")
            except Exception as e:
                self.console.print(f"\n[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def load_data_from_db(self):
        """Загрузка данных из БД"""
        symbols = self.config['data']['symbols']
        self.console.print(f"\n[cyan]Будут загружены данные для {len(symbols)} символов[/cyan]")
        
        if Confirm.ask("Продолжить?"):
            result = subprocess.run(
                ["python", "run_full_pipeline.py", "--mode", "data"],
                text=True
            )
            if result.returncode == 0:
                self.console.print("[green]✅ Данные успешно загружены[/green]")
            else:
                self.console.print("[red]❌ Ошибка при загрузке данных[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def create_features(self):
        """Создание признаков"""
        self.console.print("\n[cyan]Создание технических индикаторов и признаков...[/cyan]")
        
        if Confirm.ask("Пересоздать кэш признаков?"):
            # Удаляем старый кэш
            cache_file = Path("cache/features_cache.pkl")
            if cache_file.exists():
                cache_file.unlink()
                self.console.print("[yellow]Старый кэш удален[/yellow]")
            
            # Запускаем создание признаков
            result = subprocess.run(
                ["python", "run_full_pipeline.py", "--mode", "data"],
                text=True
            )
            if result.returncode == 0:
                self.console.print("[green]✅ Признаки успешно созданы[/green]")
            else:
                self.console.print("[red]❌ Ошибка при создании признаков[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def show_data_statistics(self):
        """Показать статистику по данным"""
        try:
            result = subprocess.run(
                ["python", "-c", """
import pickle
import pandas as pd

try:
    with open('cache/features_cache.pkl', 'rb') as f:
        features = pickle.load(f)
    
    print(f'\\n📊 Статистика данных:')
    print(f'Всего записей: {len(features):,}')
    print(f'Период: {features.datetime.min()} - {features.datetime.max()}')
    print(f'Количество признаков: {len(features.columns)}')
    
    print(f'\\nРаспределение по символам:')
    for symbol, count in features.groupby('symbol').size().items():
        print(f'  {symbol}: {count:,} записей')
    
    print(f'\\nЦелевые переменные:')
    target_cols = [col for col in features.columns if col.startswith('future_return_')]
    for col in target_cols:
        print(f'  {col}: mean={features[col].mean():.4f}, std={features[col].std():.4f}')
except:
    print('❌ Кэш данных не найден. Сначала создайте признаки.')
                """],
                capture_output=True,
                text=True
            )
            self.console.print(result.stdout)
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def clear_cache(self):
        """Очистка кэша"""
        cache_dir = Path("cache")
        if cache_dir.exists():
            files = list(cache_dir.glob("*.pkl"))
            if files:
                self.console.print(f"\n[yellow]Найдено {len(files)} файлов кэша[/yellow]")
                if Confirm.ask("Удалить все файлы кэша?"):
                    for file in files:
                        file.unlink()
                    self.console.print("[green]✅ Кэш очищен[/green]")
            else:
                self.console.print("[yellow]Кэш пуст[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run_demo_training(self):
        """Быстрое демо обучение"""
        self.console.print("\n[cyan]Запуск демо обучения (5 эпох)...[/cyan]")
        
        # Используем количество эпох из конфигурации
        # Закомментировано для использования значения из config.yaml
        # original_epochs = self.config['model']['epochs']
        # self.config['model']['epochs'] = 5
        # self.save_config()
        
        try:
            # Запускаем обучение в фоне
            process = subprocess.Popen(
                ["python", "run_full_pipeline.py", "--mode", "train"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Показываем вывод в реальном времени
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                self.console.print("\n[green]✅ Демо обучение завершено[/green]")
            else:
                self.console.print("\n[red]❌ Ошибка при обучении[/red]")
        finally:
            # Восстанавливаем конфигурацию
            self.config['model']['epochs'] = original_epochs
            self.save_config()
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run_full_training(self):
        """Полное обучение"""
        self.console.print(f"\n[cyan]Запуск полного обучения ({self.config['model']['epochs']} эпох)...[/cyan]")
        
        if Confirm.ask("Это может занять много времени. Продолжить?"):
            # Запускаем обучение в фоне
            subprocess.Popen(
                ["python", "run_full_pipeline.py", "--mode", "train"],
                stdout=open("logs/training.log", "w"),
                stderr=subprocess.STDOUT
            )
            
            self.console.print("[green]✅ Обучение запущено в фоновом режиме[/green]")
            self.console.print("[yellow]Используйте 'Мониторинг обучения' для отслеживания прогресса[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def resume_training(self):
        """Продолжить обучение с последней сохраненной модели"""
        self.console.print("\n[cyan]Продолжение обучения с последней контрольной точки...[/cyan]")
        
        # Проверяем наличие сохраненных моделей
        models_dir = Path("models_saved")
        if not models_dir.exists() or not list(models_dir.glob("*.pth")):
            self.console.print("[yellow]⚠️ Сохраненные модели не найдены[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        # Находим последнюю модель
        latest_model = max(models_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime)
        self.console.print(f"[green]Найдена модель: {latest_model.name}[/green]")
        
        if Confirm.ask("\nПродолжить обучение с этой модели?"):
            # TODO: Реализовать продолжение обучения
            self.console.print("[yellow]Функция в разработке[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_training(self):
        """Настройка параметров обучения"""
        self.console.print("\n[cyan]⚙️ Настройки обучения:[/cyan]")
        
        # Показываем текущие настройки
        table = Table(title="Текущие параметры обучения")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="yellow")
        
        table.add_row("Эпохи", str(self.config['model']['epochs']))
        table.add_row("Batch size", str(self.config['model']['batch_size']))
        table.add_row("Learning rate", str(self.config['model']['learning_rate']))
        table.add_row("Early stopping", str(self.config['model']['early_stopping_patience']))
        table.add_row("Устройство", "GPU" if self.config['performance']['device'] == 'cuda' else "CPU")
        
        self.console.print(table)
        
        if Confirm.ask("\nИзменить параметры?"):
            self.config['model']['epochs'] = IntPrompt.ask("Количество эпох", default=self.config['model']['epochs'])
            self.config['model']['batch_size'] = IntPrompt.ask("Batch size", default=self.config['model']['batch_size'])
            self.config['model']['learning_rate'] = FloatPrompt.ask("Learning rate", default=self.config['model']['learning_rate'])
            self.config['model']['early_stopping_patience'] = IntPrompt.ask("Early stopping patience", default=self.config['model']['early_stopping_patience'])
            
            device_choice = Prompt.ask("Устройство (cpu/cuda)", default=self.config['performance']['device'])
            self.config['performance']['device'] = device_choice
            
            self.save_config()
            self.console.print("[green]✅ Параметры обучения обновлены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    
    
    def monitor_gpu(self):
        """Мониторинг GPU сервера"""
        self.console.print("\n[cyan]📊 Команды для мониторинга GPU:[/cyan]")
        self.console.print("\nПосле подключения к серверу используйте:")
        self.console.print("   • nvidia-smi -l 1        # Мониторинг GPU в реальном времени")
        self.console.print("   • htop                   # Мониторинг CPU и памяти")
        self.console.print("   • tmux attach -t training # Подключение к сессии обучения")
        self.console.print("   • tail -f logs/training_gpu.log # Просмотр логов")
        self.console.print("\n[yellow]Для TensorBoard откройте http://localhost:6006[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def show_gpu_instructions(self):
        """Показать инструкции по GPU"""
        self.console.print("\n[cyan]📋 Инструкция по работе с GPU сервером[/cyan]")
        
        instructions = """
1. **Первоначальная настройка:**
   - Создайте SSH ключ: ~/.ssh/vast_ai_key
   - Установите права: chmod 600 ~/.ssh/vast_ai_key
   
2. **Рабочий процесс:**
   - Синхронизируйте проект (опция 1)
   - Подключитесь к серверу (опция 2)
   - Запустите обучение (опция 3)
   
3. **Мониторинг:**
   - TensorBoard: http://localhost:6006
   - GPU статус: nvidia-smi -l 1
   - Логи: tail -f logs/training_gpu.log
   
4. **Управление tmux:**
   - Подключиться: tmux attach -t training
   - Отключиться: Ctrl+B, затем D
   - Список сессий: tmux ls
        """
        
        self.console.print(Markdown(instructions))
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_gpu_server(self):
        """Настройка GPU сервера"""
        self.console.print("\n[cyan]⚙️ Настройка GPU сервера[/cyan]")
        
        remote_config = self.config.setdefault('remote_server', {})
        
        # Проверяем текущие настройки
        direct_config = remote_config.get('direct_connection', {})
        proxy_config = remote_config.get('proxy_connection', {})
        
        self.console.print("\n[cyan]Текущие настройки подключения:[/cyan]")
        self.console.print("\n[bold]Прямое подключение:[/bold]")
        self.console.print(f"  Host: {direct_config.get('host', '109.198.107.223')}")
        self.console.print(f"  Port: {direct_config.get('port', 48937)}")
        
        if proxy_config:
            self.console.print("\n[bold]Подключение через прокси:[/bold]")
            self.console.print(f"  Host: {proxy_config.get('host', 'ssh8.vast.ai')}")
            self.console.print(f"  Port: {proxy_config.get('port', 13641)}")
        
        if Confirm.ask("\nИзменить настройки?"):
            # Настройка прямого подключения
            self.console.print("\n[cyan]Настройка прямого подключения:[/cyan]")
            direct_host = Prompt.ask("IP адрес сервера", default=direct_config.get('host', "109.198.107.223"))
            direct_port = IntPrompt.ask("SSH порт", default=direct_config.get('port', 48937))
            
            # Спрашиваем про прокси
            if Confirm.ask("\nНастроить альтернативное подключение через прокси?"):
                self.console.print("\n[cyan]Настройка подключения через прокси:[/cyan]")
                proxy_host = Prompt.ask("Прокси хост", default="ssh8.vast.ai")
                proxy_port = IntPrompt.ask("Прокси порт", default=13641)
                
                remote_config['proxy_connection'] = {
                    'host': proxy_host,
                    'port': proxy_port,
                    'user': 'root',
                    'key_path': '~/.ssh/vast_ai_key'
                }
            
            # Сохраняем настройки
            remote_config['enabled'] = True
            remote_config['direct_connection'] = {
                'host': direct_host,
                'port': direct_port,
                'user': 'root',
                'key_path': '~/.ssh/vast_ai_key'
            }
            
            # Спрашиваем предпочтительный способ
            self.console.print("\n[cyan]Выберите предпочтительный способ подключения:[/cyan]")
            self.console.print("1. Прямое подключение")
            self.console.print("2. Через прокси")
            pref = Prompt.ask("Выбор", default="1")
            remote_config['preferred_connection'] = 'direct' if pref == "1" else 'proxy'
            
            self.save_config()
            self.console.print("[green]✅ Настройки сохранены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def monitor_training(self):
        """Мониторинг обучения"""
        self.console.print("\n[cyan]Запуск мониторинга обучения...[/cyan]")
        
        try:
            subprocess.run(["python", "monitor_training.py"])
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Мониторинг остановлен[/yellow]")
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def view_logs(self):
        """Просмотр логов"""
        log_dir = Path("experiments/logs")
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if log_files:
                self.console.print("\n[cyan]Доступные логи:[/cyan]")
                for i, file in enumerate(log_files[:10], 1):
                    size = file.stat().st_size / 1024  # KB
                    mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    self.console.print(f"{i}. {file.name} ({size:.1f} KB, {mtime})")
                
                choice = IntPrompt.ask("\nВыберите файл (0 для отмены)", default=0)
                
                if 0 < choice <= len(log_files[:10]):
                    log_file = log_files[choice - 1]
                    
                    # Показываем последние строки
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        last_lines = lines[-50:] if len(lines) > 50 else lines
                        
                        self.console.print(f"\n[cyan]Последние строки из {log_file.name}:[/cyan]")
                        for line in last_lines:
                            print(line.rstrip())
            else:
                self.console.print("[yellow]Логи не найдены[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def launch_tensorboard(self):
        """Запуск TensorBoard"""
        tb_dir = Path("experiments/runs")
        if tb_dir.exists():
            self.console.print("\n[cyan]Запуск TensorBoard...[/cyan]")
            self.console.print("[yellow]Откройте http://localhost:6006 в браузере[/yellow]")
            self.console.print("[yellow]Нажмите Ctrl+C для остановки[/yellow]\n")
            
            try:
                subprocess.run(["tensorboard", "--logdir", str(tb_dir)])
            except KeyboardInterrupt:
                self.console.print("\n[yellow]TensorBoard остановлен[/yellow]")
        else:
            self.console.print("[yellow]Директория TensorBoard не найдена[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_model(self):
        """Настройка параметров модели"""
        self.console.print("\n[cyan]Настройки модели PatchTST:[/cyan]")
        
        model_config = self.config['model']
        
        # Показываем текущие настройки
        table = Table(title="Текущие параметры")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="yellow")
        
        table.add_row("d_model", str(model_config['d_model']))
        table.add_row("n_heads", str(model_config['n_heads']))
        table.add_row("e_layers", str(model_config['e_layers']))
        table.add_row("d_ff", str(model_config['d_ff']))
        table.add_row("dropout", str(model_config['dropout']))
        table.add_row("batch_size", str(model_config['batch_size']))
        table.add_row("learning_rate", str(model_config['learning_rate']))
        table.add_row("epochs", str(model_config['epochs']))
        
        self.console.print(table)
        
        if Confirm.ask("\nИзменить параметры?"):
            model_config['d_model'] = IntPrompt.ask("d_model", default=model_config['d_model'])
            model_config['n_heads'] = IntPrompt.ask("n_heads", default=model_config['n_heads'])
            model_config['e_layers'] = IntPrompt.ask("e_layers", default=model_config['e_layers'])
            model_config['d_ff'] = IntPrompt.ask("d_ff", default=model_config['d_ff'])
            model_config['dropout'] = FloatPrompt.ask("dropout", default=model_config['dropout'])
            model_config['batch_size'] = IntPrompt.ask("batch_size", default=model_config['batch_size'])
            model_config['learning_rate'] = FloatPrompt.ask("learning_rate", default=model_config['learning_rate'])
            model_config['epochs'] = IntPrompt.ask("epochs", default=model_config['epochs'])
            
            self.save_config()
            self.console.print("[green]✅ Параметры модели обновлены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_data(self):
        """Настройка параметров данных"""
        self.console.print("\n[cyan]Настройки данных:[/cyan]")
        
        data_config = self.config['data']
        
        # Показываем текущие символы
        self.console.print(f"\nТекущие символы ({len(data_config['symbols'])}):")
        for symbol in data_config['symbols'][:10]:
            self.console.print(f"  • {symbol}")
        if len(data_config['symbols']) > 10:
            self.console.print(f"  ... и еще {len(data_config['symbols']) - 10}")
        
        if Confirm.ask("\nИзменить список символов?"):
            self.console.print("\nВыберите опцию:")
            self.console.print("1. Топ-10 монет")
            self.console.print("2. Топ-20 монет")
            self.console.print("3. Все доступные символы")
            self.console.print("4. Ввести вручную")
            
            choice = Prompt.ask("Выбор")
            
            if choice == "1":
                data_config['symbols'] = [
                    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                    "ADAUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT"
                ]
            elif choice == "2":
                data_config['symbols'] = [
                    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                    "ADAUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
                    "MATICUSDT", "UNIUSDT", "ATOMUSDT", "XLMUSDT", "ETCUSDT",
                    "NEARUSDT", "ALGOUSDT", "FILUSDT", "ICPUSDT", "TRXUSDT"
                ]
            elif choice == "3":
                data_config['symbols'] = ["ALL"]
            elif choice == "4":
                symbols_str = Prompt.ask("Введите символы через запятую")
                data_config['symbols'] = [s.strip() for s in symbols_str.split(",")]
            
            self.save_config()
            self.console.print("[green]✅ Список символов обновлен[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def check_available_symbols(self):
        """Проверка доступных символов"""
        self.console.print("\n[cyan]Проверка доступных символов в БД...[/cyan]")
        
        try:
            result = subprocess.run(
                ["python", "-c", """
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host='localhost', port=5555, database='crypto_trading',
    user='ruslan', password='ruslan'
)

query = '''
SELECT DISTINCT symbol, COUNT(*) as records, 
       MIN(datetime) as first_date, MAX(datetime) as last_date
FROM raw_market_data
GROUP BY symbol
ORDER BY symbol
'''

df = pd.read_sql(query, conn)
conn.close()

print(f'\\nНайдено {len(df)} символов:\\n')
for _, row in df.iterrows():
    print(f"{row['symbol']:15} {row['records']:8,} записей  ({row['first_date'].strftime('%Y-%m-%d')} - {row['last_date'].strftime('%Y-%m-%d')})")
                """],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print(result.stdout)
            else:
                self.console.print(f"[red]❌ Ошибка: {result.stderr}[/red]")
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def validate_data(self):
        """Валидация данных"""
        self.console.print("\n[cyan]Валидация данных...[/cyan]")
        
        cache_file = Path("cache/features_cache.pkl")
        if not cache_file.exists():
            self.console.print("[yellow]⚠️ Кэш данных не найден[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                features = pickle.load(f)
            
            self.console.print(f"\n[green]✅ Данные загружены успешно[/green]")
            self.console.print(f"Размер: {features.shape}")
            self.console.print(f"Период: {features.datetime.min()} - {features.datetime.max()}")
            
            # Проверка на NaN
            nan_count = features.isna().sum().sum()
            if nan_count > 0:
                self.console.print(f"\n[yellow]⚠️ Найдено {nan_count} NaN значений[/yellow]")
            else:
                self.console.print(f"\n[green]✅ NaN значений нет[/green]")
            
            # Проверка целевых переменных
            target_cols = [col for col in features.columns if col.startswith('future_return_')]
            self.console.print(f"\n[cyan]Целевые переменные ({len(target_cols)}):[/cyan]")
            for col in target_cols:
                mean_val = features[col].mean()
                std_val = features[col].std()
                self.console.print(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}")
        
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка при валидации: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run_backtesting(self):
        """Запуск бэктестинга"""
        self.console.print("\n[cyan]Бэктестинг стратегии[/cyan]")
        
        models_dir = Path("models_saved")
        if not models_dir.exists() or not list(models_dir.glob("*.pth")):
            self.console.print("[yellow]⚠️ Сохраненные модели не найдены[/yellow]")
            self.console.print("Сначала обучите модель")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        # Выбор модели
        models = sorted(models_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
        self.console.print("\n[cyan]Доступные модели:[/cyan]")
        for i, model in enumerate(models[:5], 1):
            self.console.print(f"{i}. {model.name}")
        
        choice = IntPrompt.ask("\nВыберите модель (0 для отмены)", default=1)
        
        if 0 < choice <= len(models[:5]):
            selected_model = models[choice - 1]
            self.console.print(f"\n[yellow]Запуск бэктестинга с моделью {selected_model.name}...[/yellow]")
            
            if Confirm.ask("Запустить бэктестинг?"):
                subprocess.run([
                    "python", "run_full_pipeline.py", 
                    "--mode", "backtest",
                    "--model", str(selected_model)
                ])
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def generate_reports(self):
        """Генерация отчетов"""
        self.console.print("\n[cyan]Генерация отчетов[/cyan]")
        
        self.console.print("\n[cyan]Выберите тип отчета:[/cyan]")
        self.console.print("1. Отчет по обучению")
        self.console.print("2. Отчет по бэктестингу")
        self.console.print("3. Анализ признаков")
        self.console.print("4. Полный отчет")
        
        choice = Prompt.ask("Выбор", default="1")
        
        if choice == "1":
            # Отчет по обучению
            log_dir = Path("experiments/logs")
            if log_dir.exists():
                self.console.print("\n[yellow]Генерация отчета по обучению...[/yellow]")
                # TODO: Реализовать генерацию отчета
                self.console.print("[yellow]Функция в разработке[/yellow]")
        
        elif choice == "2":
            # Отчет по бэктестингу
            self.console.print("\n[yellow]Генерация отчета по бэктестингу...[/yellow]")
            # TODO: Реализовать
            self.console.print("[yellow]Функция в разработке[/yellow]")
        
        elif choice == "3":
            # Анализ признаков
            self.console.print("\n[yellow]Анализ важности признаков...[/yellow]")
            # TODO: Реализовать
            self.console.print("[yellow]Функция в разработке[/yellow]")
        
        elif choice == "4":
            # Полный отчет
            self.console.print("\n[yellow]Генерация полного отчета...[/yellow]")
            # TODO: Реализовать
            self.console.print("[yellow]Функция в разработке[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def export_model(self):
        """Экспорт модели"""
        self.console.print("\n[cyan]Экспорт модели[/cyan]")
        
        models_dir = Path("models_saved")
        if not models_dir.exists() or not list(models_dir.glob("*.pth")):
            self.console.print("[yellow]⚠️ Сохраненные модели не найдены[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        # Выбор модели
        models = sorted(models_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
        self.console.print("\n[cyan]Доступные модели:[/cyan]")
        for i, model in enumerate(models[:5], 1):
            size = model.stat().st_size / (1024 * 1024)  # MB
            self.console.print(f"{i}. {model.name} ({size:.1f} MB)")
        
        choice = IntPrompt.ask("\nВыберите модель для экспорта (0 для отмены)", default=1)
        
        if 0 < choice <= len(models[:5]):
            selected_model = models[choice - 1]
            
            self.console.print("\n[cyan]Формат экспорта:[/cyan]")
            self.console.print("1. ONNX (универсальный)")
            self.console.print("2. TorchScript")
            self.console.print("3. Копировать файл")
            
            format_choice = Prompt.ask("Выбор", default="3")
            
            if format_choice == "3":
                # Простое копирование
                export_dir = Path("exports")
                export_dir.mkdir(exist_ok=True)
                
                export_path = export_dir / f"model_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                
                import shutil
                shutil.copy2(selected_model, export_path)
                
                self.console.print(f"\n[green]✅ Модель экспортирована: {export_path}[/green]")
            else:
                self.console.print("[yellow]Функция в разработке[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def manage_server_profiles(self):
        """Управление профилями серверов"""
        self.console.print("\n[cyan]🖥️ Управление профилями серверов[/cyan]")
        
        # Проверяем наличие утилиты управления профилями
        profile_manager = Path("manage_servers.py")
        if not profile_manager.exists():
            self.console.print("[red]❌ Утилита manage_servers.py не найдена[/red]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        try:
            # Запускаем утилиту управления профилями
            self.console.print("[yellow]Запуск утилиты управления профилями серверов...[/yellow]")
            result = subprocess.run(["python", "manage_servers.py"], check=False)
            
            # После возврата из утилиты перезагружаем профили
            self.server_profiles = self.load_server_profiles()
            
            # Обновляем SSH параметры если профиль изменился
            if self.server_profiles:
                active_profile = self.server_profiles.get('active_profile')
                if active_profile and active_profile in self.server_profiles.get('profiles', {}):
                    profile = self.server_profiles['profiles'][active_profile]
                    
                    # Проверяем, изменились ли параметры подключения
                    old_host = self.ssh_host
                    new_host = profile['connection']['host']
                    
                    if old_host != new_host:
                        self.ssh_host = profile['connection']['host']
                        self.ssh_port = str(profile['connection']['port'])
                        self.ssh_user = profile['connection']['user']
                        self.ssh_key_path = profile['connection']['key_path']
                        self.current_server_profile = active_profile
                        
                        self.console.print(f"[green]✅ Профиль сервера обновлен: {profile['name']}[/green]")
                        self.logger.info(f"Профиль сервера изменен на: {active_profile}")
            
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка при запуске утилиты: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_risk_management(self):
        """Настройка риск-менеджмента"""
        self.console.print("\n[cyan]⚙️ Настройки риск-менеджмента:[/cyan]")
        
        risk_config = self.config['risk_management']
        
        # Показываем текущие настройки
        table = Table(title="Текущие параметры риск-менеджмента")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="yellow")
        
        table.add_row("Stop Loss %", f"{risk_config['stop_loss_pct']}%")
        table.add_row("Take Profit уровни", str(risk_config['take_profit_targets']))
        table.add_row("Частичные закрытия", str(risk_config['partial_close_sizes']))
        table.add_row("Макс риск на сделку", f"{risk_config['position_sizing']['max_risk_per_trade']}%")
        table.add_row("Метод размера позиции", risk_config['position_sizing']['method'])
        
        self.console.print(table)
        
        if Confirm.ask("\nИзменить параметры?"):
            risk_config['stop_loss_pct'] = FloatPrompt.ask("Stop Loss %", default=risk_config['stop_loss_pct'])
            
            # Take Profit уровни
            tp_str = Prompt.ask("Take Profit уровни (через запятую)", default=",".join(map(str, risk_config['take_profit_targets'])))
            risk_config['take_profit_targets'] = [float(x.strip()) for x in tp_str.split(",")]
            
            # Частичные закрытия
            pc_str = Prompt.ask("Частичные закрытия % (через запятую)", default=",".join(map(str, risk_config['partial_close_sizes'])))
            risk_config['partial_close_sizes'] = [int(x.strip()) for x in pc_str.split(",")]
            
            risk_config['position_sizing']['max_risk_per_trade'] = FloatPrompt.ask(
                "Макс риск на сделку %", 
                default=risk_config['position_sizing']['max_risk_per_trade']
            )
            
            self.save_config()
            self.console.print("[green]✅ Параметры риск-менеджмента обновлены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_backtesting(self):
        """Настройка параметров бэктестинга"""
        self.console.print("\n[cyan]⚙️ Настройки бэктестинга:[/cyan]")
        
        backtest_config = self.config['backtesting']
        
        # Показываем текущие настройки
        table = Table(title="Текущие параметры бэктестинга")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="yellow")
        
        table.add_row("Начальный капитал", f"${backtest_config['initial_capital']:,}")
        table.add_row("Комиссия", f"{backtest_config['commission']*100:.1f}%")
        table.add_row("Проскальзывание", f"{backtest_config['slippage']*100:.2f}%")
        
        self.console.print(table)
        
        if Confirm.ask("\nИзменить параметры?"):
            backtest_config['initial_capital'] = IntPrompt.ask(
                "Начальный капитал $", 
                default=backtest_config['initial_capital']
            )
            
            commission_pct = FloatPrompt.ask(
                "Комиссия %", 
                default=backtest_config['commission']*100
            )
            backtest_config['commission'] = commission_pct / 100
            
            slippage_pct = FloatPrompt.ask(
                "Проскальзывание %", 
                default=backtest_config['slippage']*100
            )
            backtest_config['slippage'] = slippage_pct / 100
            
            self.save_config()
            self.console.print("[green]✅ Параметры бэктестинга обновлены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def show_training_results(self):
        """Показать результаты последнего обучения"""
        self.console.print("\n[cyan]📊 Результаты последнего обучения[/cyan]")
        
        # Находим последнюю директорию с обучением
        exp_dir = Path("experiments/runs")
        if not exp_dir.exists():
            self.console.print("[yellow]⚠️ Результаты обучения не найдены[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        training_dirs = sorted(exp_dir.glob("training_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not training_dirs:
            self.console.print("[yellow]⚠️ Результаты обучения не найдены[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        latest_dir = training_dirs[0]
        self.console.print(f"\n[green]Последнее обучение: {latest_dir.name}[/green]")
        
        # Читаем финальный отчет если есть
        report_file = latest_dir / "final_report.txt"
        if report_file.exists():
            with open(report_file, 'r') as f:
                self.console.print("\n[cyan]Финальный отчет:[/cyan]")
                self.console.print(f.read())
        
        # Проверяем метрики
        metrics_files = list(latest_dir.glob("*_metrics.csv"))
        if metrics_files:
            self.console.print("\n[cyan]Файлы метрик:[/cyan]")
            for file in metrics_files:
                self.console.print(f"  • {file.name}")
        
        # Проверяем графики
        plots_dir = latest_dir / "plots"
        if plots_dir.exists():
            plots = list(plots_dir.glob("*.png"))
            if plots:
                self.console.print(f"\n[cyan]Найдено {len(plots)} графиков[/cyan]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run_gpu_training(self):
        """Обучение на GPU сервере"""
        while True:
            self.console.clear()
            self.console.print(Panel("🚀 Обучение на GPU сервере (Vast.ai)", style="cyan"))
            
            # Информация о сервере
            info_table = Table(show_header=False, box=None)
            info_table.add_column("Parameter", style="cyan")
            info_table.add_column("Value", style="white")
            
            if self.is_local_mode:
                info_table.add_row("📊 Информация о системе:", "🏠 Локальный режим")
                if hasattr(self, 'gpu_name'):
                    info_table.add_row("   • GPU:", f"{self.gpu_name}")
                    if self.gpu_memory > 0:
                        info_table.add_row("   • VRAM:", f"{self.gpu_memory:.1f} GB")
                else:
                    info_table.add_row("   • GPU:", "Не определен")
                
                # Локальная системная информация
                try:
                    import psutil
                    ram_total = psutil.virtual_memory().total / 1024**3
                    ram_available = psutil.virtual_memory().available / 1024**3
                    info_table.add_row("   • RAM:", f"{ram_total:.1f} GB (свободно {ram_available:.1f} GB)")
                    cpu_count = psutil.cpu_count(logical=False)
                    info_table.add_row("   • CPU:", f"{cpu_count} ядер")
                except:
                    pass
                    
                info_table.add_row("   • Ускорение:", "Максимальная производительность")
            else:
                info_table.add_row("📊 Информация о сервере:", "")
                info_table.add_row("   • GPU:", "2x RTX 4090 (82.6 TFLOPS)")
                info_table.add_row("   • VRAM:", "48 GB (2x24)")
                info_table.add_row("   • RAM:", "90 GB")
                info_table.add_row("   • Ускорение:", "10-20x по сравнению с CPU")
            
            # Показываем текущий способ подключения
            if self.is_local_mode:
                info_table.add_row("   • Подключение:", "Локальная работа")
            else:
                remote_config = self.config.get('remote_server', {})
                preferred = remote_config.get('preferred_connection', 'direct')
                conn_type = "Прямое подключение" if preferred == 'direct' else "Через прокси"
                info_table.add_row("   • Подключение:", conn_type)
            
            self.console.print(info_table)
            
            # Проверяем состояние системы
            server_status = {'connected': True, 'project_exists': True}  # Дефолтные значения для локального режима
            
            if self.is_local_mode:
                with self.console.status("[cyan]Проверка локальной системы...[/cyan]"):
                    # Проверяем БД
                    try:
                        from data.data_loader import CryptoDataLoader
                        loader = CryptoDataLoader(self.config)
                        self.console.print("[green]✅ База данных доступна[/green]")
                    except Exception as e:
                        self.console.print(f"[red]❌ Ошибка подключения к БД: {e}[/red]")
                    
                    # Проверяем GPU
                    if hasattr(self, 'gpu_name') and self.gpu_name != "CPU":
                        self.console.print(f"[green]✅ GPU доступен: {self.gpu_name}[/green]")
                    else:
                        self.console.print("[yellow]⚠️  GPU не обнаружен[/yellow]")
            else:
                with self.console.status("[cyan]Проверка подключения к серверу...[/cyan]"):
                    server_status = self._check_server_status()
                
                if server_status['connected']:
                    self.console.print("[green]✅ Сервер доступен[/green]")
                    if server_status['project_exists']:
                        self.console.print("[green]✅ Проект синхронизирован[/green]")
                    else:
                        self.console.print("[yellow]⚠️  Проект не найден на сервере[/yellow]")
                else:
                    self.console.print("[red]❌ Сервер недоступен[/red]")
            
            # Меню действий
            self.console.print("\n[bold cyan]Выберите действие:[/bold cyan]")
            action_table = Table(show_header=False, box=None)
            action_table.add_column("Option", style="cyan", width=5)
            action_table.add_column("Description", style="white")
            
            action_table.add_row("1", "📤 Синхронизировать проект")
            action_table.add_row("2", "🚀 Запустить обучение")
            action_table.add_row("3", "📊 Мониторинг с браузером")
            action_table.add_row("4", "📋 Проверить логи")
            action_table.add_row("5", "🔧 Настройки сервера")
            action_table.add_row("6", "🔌 Настроить туннель БД")
            action_table.add_row("0", "Назад")
            
            self.console.print(action_table)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.logger.info("GPU меню: выбрана синхронизация")
                if self.is_local_mode:
                    self.console.print("\n[yellow]ℹ️  В локальном режиме синхронизация не требуется[/yellow]")
                    Prompt.ask("\nНажмите Enter для продолжения")
                else:
                    self.sync_to_gpu_server()
            elif choice == "2":
                self.logger.info("GPU меню: выбран запуск обучения")
                if self.is_local_mode:
                    self.launch_gpu_training()
                else:
                    if not server_status['project_exists']:
                        self.console.print("\n[yellow]⚠️  Сначала нужно синхронизировать проект![/yellow]")
                        Prompt.ask("\nНажмите Enter для продолжения")
                    else:
                        self.launch_gpu_training()
            elif choice == "3":
                self.logger.info("GPU меню: выбран мониторинг")
                self.monitor_with_browser()
            elif choice == "4":
                self.logger.info("GPU меню: выбраны логи")
                self.check_gpu_logs()
            elif choice == "5":
                self.logger.info("GPU меню: выбраны настройки")
                self.configure_gpu_server()
            elif choice == "6":
                self.logger.info("GPU меню: выбран туннель БД")
                self.setup_db_tunnel()
            elif choice == "0":
                self.logger.info("GPU меню: выход")
                break
    
    def _check_server_status(self):
        """Проверка состояния сервера"""
        try:
            # Получаем параметры подключения из конфига
            remote_config = self.config.get('remote_server', {})
            
            # Определяем способ подключения - пробуем новые серверы
            connections = [
                ("109.198.107.223", 48937),  # Основной сервер
                ("ssh1.vast.ai", 30421)       # Альтернативный сервер
            ]
            
            for host, port in connections:
                # Проверяем доступность сервера
                result = subprocess.run(
                    ["ssh", "-p", str(port), f"root@{host}",
                     "-i", os.path.expanduser("~/.ssh/id_rsa"),
                     "-o", "ConnectTimeout=5",
                     "-o", "StrictHostKeyChecking=no",
                     "-o", "UserKnownHostsFile=/dev/null",
                     "test -d /root/crypto_ai_trading && echo 'PROJECT_EXISTS' || echo 'NO_PROJECT'"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    project_exists = "PROJECT_EXISTS" in result.stdout
                    return {'connected': True, 'project_exists': project_exists, 'host': host, 'port': port}
            
            # Если ничего не сработало
            return {'connected': False, 'project_exists': False}
        except Exception as e:
            self.logger.error(f"Ошибка проверки сервера: {e}")
            return {'connected': False, 'project_exists': False}
    
    def sync_to_gpu_server(self):
        """Синхронизация проекта с GPU сервером"""
        self.console.print("\n[cyan]📤 Синхронизация проекта с GPU сервером...[/cyan]")
        
        script_path = "scripts/sync_to_vast.sh"
        if Path(script_path).exists():
            subprocess.run(["bash", script_path])
            self.console.print("\n[green]✅ Синхронизация завершена[/green]")
        else:
            self.console.print(f"[red]❌ Скрипт {script_path} не найден[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def launch_gpu_training(self):
        """Запуск обучения на GPU"""
        if self.is_local_mode:
            self.console.print("\n[cyan]🚀 Запуск локального обучения[/cyan]")
            self._launch_local_training()
            return
        
        self.console.print("\n[cyan]🚀 Запуск обучения на GPU сервере[/cyan]")
        
        # Проверяем наличие кэша локально
        cache_file = Path("cache/features_cache.pkl")
        if not cache_file.exists():
            self.console.print("[red]❌ Файл кэша не найден локально![/red]")
            self.console.print("[yellow]Сначала создайте кэш через 'Управление данными' -> 'Создать/обновить признаки'[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        # Проверяем наличие кэша на сервере
        with self.console.status("[cyan]Проверка данных на сервере...[/cyan]"):
            result = subprocess.run(
                self.get_ssh_command("test -f /root/crypto_ai_trading/cache/features_cache.pkl && echo 'EXISTS' || echo 'NOT_EXISTS'"),
                capture_output=True,
                text=True
            )
            
        if "NOT_EXISTS" in result.stdout:
            self.console.print("[yellow]📤 Кэш не найден на сервере. Копирование...[/yellow]")
            self.console.print(f"[dim]Размер файла: {cache_file.stat().st_size / (1024*1024):.1f} MB[/dim]")
            
            # Создаем директорию и копируем файл
            subprocess.run(self.get_ssh_command("mkdir -p /root/crypto_ai_trading/cache"))
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Копирование кэша...", total=100)
                result = subprocess.run(
                    self.get_scp_command(str(cache_file), f"{self.ssh_user}@{self.ssh_host}:/root/crypto_ai_trading/cache/"),
                    capture_output=True,
                    text=True
                )
                progress.update(task, completed=100)
            
            if result.returncode == 0:
                self.console.print("[green]✅ Кэш скопирован на сервер[/green]")
            else:
                self.console.print("[red]❌ Ошибка копирования кэша[/red]")
                Prompt.ask("\nНажмите Enter для продолжения")
                return
        else:
            self.console.print("[green]✅ Кэш найден на сервере[/green]")
        
        # Выбор режима обучения
        self.console.print("\n[bold cyan]Выберите режим обучения:[/bold cyan]")
        self.console.print("1. Демо (5 эпох) - ~15-20 минут")
        self.console.print("2. Стандартное (50 эпох) - ~2-3 часа")
        self.console.print("3. Полное (100 эпох) - ~5-6 часов")
        self.console.print("4. Пользовательское")
        
        choice = Prompt.ask("Выбор", default="1")
        
        if choice == "1":
            epochs = 5
        elif choice == "2":
            epochs = 50
        elif choice == "3":
            epochs = 100
        elif choice == "4":
            epochs = IntPrompt.ask("Количество эпох", default=10)
        else:
            return
        
        # Выбор источника данных
        self.console.print("\n[bold cyan]Выберите источник данных:[/bold cyan]")
        self.console.print("1. Использовать только кэш (рекомендуется для GPU сервера)")
        self.console.print("2. Использовать базу данных (требует SSH туннель)")
        
        data_source = Prompt.ask("Выбор", default="1")
        
        use_cache_only = True
        if data_source == "2":
            use_cache_only = False
            # Проверяем SSH туннель
            self.console.print("\n[yellow]Проверка SSH туннеля к локальной БД...[/yellow]")
            tunnel_check = subprocess.run(
                self.get_ssh_command("nc -zv localhost 5555 2>&1 | grep -q 'succeeded' && echo 'TUNNEL_OK' || echo 'NO_TUNNEL'"),
                capture_output=True,
                text=True
            )
            
            if "NO_TUNNEL" in tunnel_check.stdout:
                self.console.print("[yellow]⚠️ SSH туннель к БД не обнаружен[/yellow]")
                if Confirm.ask("Создать туннель сейчас?"):
                    self.setup_db_tunnel()
                    # После создания туннеля продолжаем
                else:
                    self.console.print("[red]❌ Без туннеля невозможно использовать БД[/red]")
                    Prompt.ask("\nНажмите Enter для продолжения")
                    return
        
        self.console.print(f"\n[yellow]Запуск обучения на {epochs} эпох в tmux сессии...[/yellow]")
        self.console.print(f"[dim]Источник данных: {'Кэш' if use_cache_only else 'База данных'}[/dim]")
        
        # Изменяем количество эпох в конфиге на сервере
        self.console.print("[dim]Обновление конфигурации на сервере...[/dim]")
        subprocess.run(
            self.get_ssh_command(f"cd /root/crypto_ai_trading && python3 -c \"import yaml; config = yaml.safe_load(open('config/config.yaml')); config['model']['epochs'] = {epochs}; yaml.dump(config, open('config/config.yaml', 'w'), default_flow_style=False)\"")
        )
        
        # Запускаем обучение в tmux
        session_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Команда для запуска в зависимости от источника данных
        if use_cache_only:
            # Режим только с кэшем - используем существующий run_full_pipeline.py с переменной окружения
            training_cmd = f"""cd /root/crypto_ai_trading && \
export USE_CACHE_ONLY=1 && \
export CUDA_VISIBLE_DEVICES=0,1 && \
export PYTHONUNBUFFERED=1 && \
python3 run_full_pipeline.py --mode train 2>&1 | tee logs/training_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"""
        else:
            # Режим с базой данных
            training_cmd = f"""cd /root/crypto_ai_trading && \
export CUDA_VISIBLE_DEVICES=0,1 && \
export PYTHONUNBUFFERED=1 && \
python3 run_full_pipeline.py --mode train 2>&1 | tee logs/training_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"""
        
        # Создаем tmux сессию
        self.console.print(f"[dim]Создание tmux сессии: {session_name}[/dim]")
        result = subprocess.run(
            self.get_ssh_command(f"tmux new-session -d -s {session_name} '{training_cmd}'"),
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            self.console.print(f"\n[green]✅ Обучение запущено в tmux сессии: {session_name}[/green]")
            self.console.print("\n[cyan]Полезные команды:[/cyan]")
            self.console.print(f"   • Подключиться к сессии: ssh -p {self.ssh_port} {self.ssh_user}@{self.ssh_host} tmux attach -t {session_name}")
            self.console.print(f"   • Отключиться от сессии: Ctrl+B, затем D")
            self.console.print(f"   • Список сессий: ssh -p {self.ssh_port} {self.ssh_user}@{self.ssh_host} tmux ls")
            
            # Ждем немного, чтобы обучение начало записывать логи
            time.sleep(3)
            
            # Предлагаем мониторинг
            self.console.print("\n[bold cyan]Выберите действие:[/bold cyan]")
            self.console.print("1. Открыть мониторинг в реальном времени")
            self.console.print("2. Открыть TensorBoard в браузере")
            self.console.print("3. Вернуться в меню")
            
            monitor_choice = Prompt.ask("Выбор", default="1")
            
            if monitor_choice == "1":
                self.monitor_realtime_logs(session_name)
            elif monitor_choice == "2":
                self.monitor_with_browser()
        else:
            self.console.print(f"\n[red]❌ Ошибка создания tmux сессии[/red]")
            if result.stderr:
                self.console.print(f"[red]{result.stderr}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def monitor_with_browser(self):
        """Мониторинг с автоматическим запуском браузера"""
        self.logger.info("Запуск monitor_with_browser")
        try:
            self.console.print("\n[cyan]📊 Запуск мониторинга обучения...[/cyan]")
            
            # Проверяем, запущен ли TensorBoard на сервере
            self.console.print("[yellow]Проверка TensorBoard на сервере...[/yellow]")
            
            result = subprocess.run(
                self.get_ssh_command("pgrep -f tensorboard"),
                capture_output=True
            )
            
            if result.returncode != 0:
                self.console.print("[yellow]TensorBoard не запущен. Запускаем...[/yellow]")
                
                # Сначала проверяем, где есть логи
                check_dirs = subprocess.run(
                    self.get_ssh_command("cd /root/crypto_ai_trading && find . -name 'events.out.tfevents*' | head -5"),
                    capture_output=True,
                    text=True
                )
                
                if check_dirs.stdout.strip():
                    self.console.print(f"[dim]Найдены логи: {check_dirs.stdout.strip().split()[0]}[/dim]")
                
                # Запускаем TensorBoard на порту 6007 (6006 занят Caddy)
                subprocess.run(
                    self.get_ssh_command("cd /root/crypto_ai_trading && pkill -f tensorboard; nohup tensorboard --logdir ./experiments/runs --bind_all --port 6007 > logs/tensorboard.log 2>&1 &"),
                    capture_output=True
                )
                time.sleep(3)
                self.console.print("[green]✅ TensorBoard запущен на порту 6007[/green]")
            
            # Запускаем SSH с пробросом портов
            self.console.print("[yellow]Открываем туннель к серверу...[/yellow]")
            
            import threading
            import webbrowser
            import time as time_module
            
            def open_browser():
                time_module.sleep(3)  # Даем время на установку туннеля
                self.console.print("\n[green]Открываем TensorBoard в браузере...[/green]")
                self.console.print("[yellow]TensorBoard доступен на: http://localhost:6007[/yellow]")
                webbrowser.open('http://localhost:6007')
            
            # Запускаем браузер в отдельном потоке
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.start()
            
            # Определяем способ подключения
            env = os.environ.copy()
            
            # Проверяем предпочтительный способ подключения из конфига
            remote_config = self.config.get('remote_server', {})
            preferred = remote_config.get('preferred_connection', 'direct')
            
            if preferred == 'proxy' and 'proxy_connection' in remote_config:
                env['VAST_CONNECTION_MODE'] = '2'  # Прокси подключение
                self.console.print("[dim]Используется подключение через прокси[/dim]")
            else:
                env['VAST_CONNECTION_MODE'] = '1'  # Прямое подключение
                self.console.print("[dim]Используется прямое подключение[/dim]")
            
            script_path = "scripts/connect_vast.sh"
            if Path(script_path).exists():
                try:
                    subprocess.run(["bash", script_path], env=env)
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Мониторинг остановлен[/yellow]")
                except Exception as e:
                    self.console.print(f"[red]❌ Ошибка: {e}[/red]")
            else:
                self.console.print(f"[red]❌ Скрипт {script_path} не найден[/red]")
                
        except Exception as e:
            self.logger.error(f"Ошибка в monitor_with_browser: {e}", exc_info=True)
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def monitor_realtime_logs(self, session_name=None):
        """Мониторинг логов в реальном времени"""
        self.console.print("\n[cyan]📋 Мониторинг логов в реальном времени[/cyan]")
        
        # Выбираем источник логов
        self.console.print("\n[bold cyan]Выберите источник логов:[/bold cyan]")
        self.console.print("1. Последний лог файл")
        self.console.print("2. Вывод из tmux сессии")
        self.console.print("3. Все логи обучения")
        
        choice = Prompt.ask("Выбор", default="1")
        
        try:
            if choice == "1":
                # Находим последний лог файл
                result = subprocess.run(
                    self.get_ssh_command("cd /root/crypto_ai_trading && ls -t logs/training_gpu_*.log 2>/dev/null | head -1"),
                    capture_output=True,
                    text=True
                )
                
                if result.stdout.strip():
                    log_file = result.stdout.strip()
                    self.console.print(f"\n[yellow]Мониторинг файла: {log_file}[/yellow]")
                    self.console.print("[dim]Нажмите Ctrl+C для остановки[/dim]\n")
                    
                    # Запускаем tail -f с полным путем
                    subprocess.run(self.get_ssh_command(f"cd /root/crypto_ai_trading && tail -f {log_file}"))
                else:
                    self.console.print("[red]❌ Лог файлы не найдены[/red]")
                    
            elif choice == "2":
                # Список tmux сессий
                if not session_name:
                    result = subprocess.run(
                        self.get_ssh_command("tmux ls 2>/dev/null"),
                        capture_output=True,
                        text=True
                    )
                    
                    if result.stdout:
                        self.console.print("\n[cyan]Активные tmux сессии:[/cyan]")
                        sessions = []
                        for line in result.stdout.strip().split('\n'):
                            session = line.split(':')[0]
                            sessions.append(session)
                            self.console.print(f"  • {session}")
                        
                        if len(sessions) == 1:
                            session_name = sessions[0]
                        else:
                            session_name = Prompt.ask("\nВведите имя сессии", default=sessions[0])
                    else:
                        self.console.print("[yellow]⚠️ Нет активных tmux сессий[/yellow]")
                        return
                
                self.console.print(f"\n[yellow]Подключение к сессии: {session_name}[/yellow]")
                self.console.print("[dim]Для отключения: Ctrl+B, затем D[/dim]\n")
                
                # Подключаемся к tmux сессии
                ssh_cmd = self.get_ssh_command()
                ssh_cmd.insert(1, "-t")  # Добавляем -t после ssh для интерактивного режима
                ssh_cmd.append(f"tmux attach -t {session_name}")
                subprocess.run(ssh_cmd)
                
            elif choice == "3":
                # Мониторинг всех логов
                self.console.print("\n[yellow]Мониторинг всех логов обучения[/yellow]")
                self.console.print("[dim]Нажмите Ctrl+C для остановки[/dim]\n")
                
                subprocess.run(
                    self.get_ssh_command("cd /root/crypto_ai_trading && tail -f logs/training*.log logs/*.log 2>/dev/null")
                )
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Мониторинг остановлен[/yellow]")
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def check_gpu_logs(self):
        """Проверить логи на GPU сервере"""
        self.console.print("\n[cyan]📋 Логи с GPU сервера[/cyan]")
        
        # Меню выбора
        self.console.print("\n[bold cyan]Выберите действие:[/bold cyan]")
        self.console.print("1. Показать последние 50 строк")
        self.console.print("2. Мониторинг в реальном времени")
        self.console.print("3. Скачать лог файл")
        self.console.print("4. Показать список всех логов")
        
        choice = Prompt.ask("Выбор", default="1")
        
        if choice == "1":
            # Последние строки
            result = subprocess.run(
                self.get_ssh_command("cd /root/crypto_ai_trading && ls -t logs/training_gpu_*.log 2>/dev/null | head -1 | xargs tail -n 50"),
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                self.console.print("\n[yellow]Последние 50 строк лога:[/yellow]")
                self.console.print(result.stdout)
            else:
                self.console.print("[red]❌ Логи не найдены[/red]")
                
        elif choice == "2":
            # Мониторинг в реальном времени
            self.monitor_realtime_logs()
            return  # monitor_realtime_logs уже показывает prompt
            
        elif choice == "3":
            # Скачать лог
            result = subprocess.run(
                self.get_ssh_command("cd /root/crypto_ai_trading && ls -t logs/training_gpu_*.log 2>/dev/null | head -1"),
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                remote_file = result.stdout.strip()
                local_file = f"logs/downloaded_{Path(remote_file).name}"
                
                self.console.print(f"\n[yellow]Скачивание {remote_file}...[/yellow]")
                
                Path("logs").mkdir(exist_ok=True)
                result = subprocess.run(
                    self.get_scp_command(f"{self.ssh_user}@{self.ssh_host}:{remote_file}", local_file),
                    capture_output=True
                )
                
                if result.returncode == 0:
                    self.console.print(f"[green]✅ Лог скачан: {local_file}[/green]")
                else:
                    self.console.print("[red]❌ Ошибка скачивания[/red]")
            else:
                self.console.print("[red]❌ Лог файлы не найдены[/red]")
                
        elif choice == "4":
            # Список логов
            result = subprocess.run(
                self.get_ssh_command("cd /root/crypto_ai_trading && ls -lah logs/*.log 2>/dev/null | tail -20"),
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                self.console.print("\n[cyan]Логи на сервере:[/cyan]")
                self.console.print(result.stdout)
            else:
                self.console.print("[yellow]Логи не найдены[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_gpu_server(self):
        """Настройка GPU сервера"""
        self.console.print("\n[cyan]⚙️ Настройка GPU сервера[/cyan]")
        
        remote_config = self.config.setdefault('remote_server', {})
        
        # Проверяем текущие настройки
        direct_config = remote_config.get('direct_connection', {})
        proxy_config = remote_config.get('proxy_connection', {})
        
        self.console.print("\n[cyan]Текущие настройки подключения:[/cyan]")
        self.console.print("\n[bold]Прямое подключение:[/bold]")
        self.console.print(f"  Host: {direct_config.get('host', '109.198.107.223')}")
        self.console.print(f"  Port: {direct_config.get('port', 48937)}")
        
        if proxy_config:
            self.console.print("\n[bold]Подключение через прокси:[/bold]")
            self.console.print(f"  Host: {proxy_config.get('host', 'ssh8.vast.ai')}")
            self.console.print(f"  Port: {proxy_config.get('port', 13641)}")
        
        if Confirm.ask("\nИзменить настройки?"):
            # Настройка прямого подключения
            self.console.print("\n[cyan]Настройка прямого подключения:[/cyan]")
            direct_host = Prompt.ask("IP адрес сервера", default=direct_config.get('host', "109.198.107.223"))
            direct_port = IntPrompt.ask("SSH порт", default=direct_config.get('port', 48937))
            
            # Спрашиваем про прокси
            if Confirm.ask("\nНастроить альтернативное подключение через прокси?"):
                self.console.print("\n[cyan]Настройка подключения через прокси:[/cyan]")
                proxy_host = Prompt.ask("Прокси хост", default="ssh8.vast.ai")
                proxy_port = IntPrompt.ask("Прокси порт", default=13641)
                
                remote_config['proxy_connection'] = {
                    'host': proxy_host,
                    'port': proxy_port,
                    'user': 'root',
                    'key_path': '~/.ssh/vast_ai_key'
                }
            
            # Сохраняем настройки
            remote_config['enabled'] = True
            remote_config['direct_connection'] = {
                'host': direct_host,
                'port': direct_port,
                'user': 'root',
                'key_path': '~/.ssh/vast_ai_key'
            }
            
            # Спрашиваем предпочтительный способ
            self.console.print("\n[cyan]Выберите предпочтительный способ подключения:[/cyan]")
            self.console.print("1. Прямое подключение")
            self.console.print("2. Через прокси")
            pref = Prompt.ask("Выбор", default="1")
            remote_config['preferred_connection'] = 'direct' if pref == "1" else 'proxy'
            
            self.save_config()
            self.console.print("[green]✅ Настройки сохранены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def setup_db_tunnel(self):
        """Настройка туннеля к локальной БД"""
        self.logger.info("Запуск setup_db_tunnel")
        try:
            self.console.print("\n[cyan]🔌 Настройка туннеля к локальной БД[/cyan]")
            
            # Проверяем, запущена ли локальная БД
            self.console.print("[yellow]Проверка локальной PostgreSQL...[/yellow]")
            
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 5555))
            sock.close()
            
            if result != 0:
                self.console.print("[red]❌ PostgreSQL не запущен на порту 5555[/red]")
                self.console.print("[yellow]Запустите БД командой:[/yellow]")
                self.console.print("   docker-compose up -d postgres")
                Prompt.ask("\nНажмите Enter для продолжения")
                return
            
            self.console.print("[green]✅ Локальная БД доступна[/green]")
            
            # Проверяем существующие туннели
            result = subprocess.run(
                ["pgrep", "-f", "ssh.*-R.*5555"],
                capture_output=True
            )
            
            if result.returncode == 0:
                self.console.print("[yellow]⚠️  Туннель уже существует[/yellow]")
                if Confirm.ask("Пересоздать туннель?"):
                    subprocess.run(["pkill", "-f", "ssh.*-R.*5555"])
                    time.sleep(1)
                else:
                    Prompt.ask("\nНажмите Enter для продолжения")
                    return
            
            # Запускаем скрипт настройки туннеля
            script_path = "scripts/setup_db_tunnel.sh"
            if Path(script_path).exists():
                self.console.print("[yellow]🚇 Создание SSH туннеля...[/yellow]")
                result = subprocess.run(
                    ["bash", script_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.console.print("[green]✅ Туннель успешно создан![/green]")
                    self.console.print("\n[cyan]Теперь GPU сервер может использовать БД[/cyan]")
                    self.console.print("При запуске обучения БД будет доступна автоматически")
                else:
                    self.console.print("[red]❌ Ошибка создания туннеля[/red]")
                    if result.stderr:
                        self.console.print(f"[red]{result.stderr}[/red]")
            else:
                self.console.print(f"[red]❌ Скрипт {script_path} не найден[/red]")
                
        except Exception as e:
            self.logger.error(f"Ошибка в setup_db_tunnel: {e}", exc_info=True)
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def clear_logs(self):
        """Очистка логов"""
        self.console.print("\n[cyan]🗑️ Очистка логов[/cyan]")
        
        log_dirs = [
            Path("logs"),
            Path("experiments/logs"),
            Path("experiments/runs")
        ]
        
        total_size = 0
        file_count = 0
        
        for log_dir in log_dirs:
            if log_dir.exists():
                for file in log_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
                        file_count += 1
        
        if file_count > 0:
            size_mb = total_size / (1024 * 1024)
            self.console.print(f"\nНайдено {file_count} файлов ({size_mb:.1f} MB)")
            
            if Confirm.ask("Удалить все логи?"):
                for log_dir in log_dirs:
                    if log_dir.exists():
                        import shutil
                        shutil.rmtree(log_dir)
                        log_dir.mkdir(exist_ok=True)
                
                self.console.print("[green]✅ Логи очищены[/green]")
        else:
            self.console.print("[yellow]Логи не найдены[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def show_system_info(self):
        """Показать информацию о системе"""
        self.console.print(Panel("📋 Информация о системе", style="cyan"))
        
        # Версия Python
        import sys
        self.console.print(f"\n[cyan]Python:[/cyan] {sys.version.split()[0]}")
        
        # Установленные пакеты
        try:
            import torch
            self.console.print(f"[cyan]PyTorch:[/cyan] {torch.__version__}")
        except:
            pass
        
        try:
            import pandas as pd
            self.console.print(f"[cyan]Pandas:[/cyan] {pd.__version__}")
        except:
            pass
        
        # Статус GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.console.print(f"\n[green]✅ GPU доступен:[/green] {torch.cuda.get_device_name(0)}")
            else:
                self.console.print("\n[yellow]⚠️ GPU не доступен, используется CPU[/yellow]")
        except:
            pass
        
        # Структура проекта
        self.console.print("\n[cyan]Структура проекта:[/cyan]")
        dirs = ["data", "models", "training", "trading", "utils", "config", "cache", "experiments"]
        for dir_name in dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*.py")))
                self.console.print(f"  📁 {dir_name}/ ({file_count} файлов)")
        
        # Последние модели
        models_dir = Path("models_saved")
        if models_dir.exists():
            models = sorted(models_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
            if models:
                self.console.print(f"\n[cyan]Последние модели:[/cyan]")
                for model in models[:3]:
                    size = model.stat().st_size / (1024 * 1024)  # MB
                    mtime = datetime.fromtimestamp(model.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    self.console.print(f"  • {model.name} ({size:.1f} MB, {mtime})")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def quick_start(self):
        """Быстрый запуск"""
        self.console.print("\n[cyan]🚀 Быстрый запуск системы[/cyan]")
        
        steps = [
            ("Проверка БД", lambda: subprocess.run(["python", "-c", "from data.data_loader import CryptoDataLoader; print('OK')"], capture_output=True).returncode == 0),
            ("Проверка данных", lambda: Path("cache/features_cache.pkl").exists()),
            ("Проверка моделей", lambda: Path("models").exists()),
        ]
        
        all_ok = True
        for step_name, check_func in steps:
            try:
                if check_func():
                    self.console.print(f"✅ {step_name}")
                else:
                    self.console.print(f"❌ {step_name}")
                    all_ok = False
            except:
                self.console.print(f"❌ {step_name}")
                all_ok = False
        
        if all_ok:
            self.console.print("\n[green]Система готова к работе![/green]")
            
            if Confirm.ask("\nЗапустить полный пайплайн?"):
                self.console.print("\n[cyan]Выберите режим:[/cyan]")
                self.console.print("1. Только подготовка данных")
                self.console.print("2. Демо обучение (5 эпох)")
                self.console.print("3. Полный пайплайн (данные + обучение + бэктест)")
                
                mode_choice = Prompt.ask("Выбор", default="2")
                
                if mode_choice == "1":
                    subprocess.run(["python", "run_full_pipeline.py", "--mode", "data"])
                elif mode_choice == "2":
                    # Для демо обучения используем эпохи из конфигурации
                    # Закомментировано для гибкости
                    # original_epochs = self.config['model']['epochs']
                    # self.config['model']['epochs'] = 5
                    # self.save_config()
                    subprocess.run(["python", "run_full_pipeline.py", "--mode", "train"])
                elif mode_choice == "3":
                    subprocess.run(["python", "run_full_pipeline.py", "--mode", "full"])
        else:
            self.console.print("\n[yellow]⚠️ Требуется настройка системы[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def _launch_local_training(self):
        """Запуск обучения на локальной системе с GPU"""
        self.console.print("\n[cyan]🚀 Запуск локального обучения[/cyan]")
        
        # Проверяем наличие GPU
        if not hasattr(self, 'gpu_name') or self.gpu_name == "CPU":
            self.console.print("[yellow]⚠️  GPU не обнаружен, обучение будет выполняться на CPU[/yellow]")
            if not Confirm.ask("Продолжить без GPU?"):
                return
        
        # Проверяем наличие кэша данных
        cache_file = Path("cache/features_cache.pkl")
        if not cache_file.exists():
            self.console.print("[red]❌ Файл кэша данных не найден![/red]")
            self.console.print("[yellow]Сначала создайте кэш через 'Управление данными' -> 'Создать/обновить признаки'[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        # Выбор режима обучения
        self.console.print("\n[bold cyan]Выберите режим обучения:[/bold cyan]")
        self.console.print("1. Демо (5 эпох) - быстрый тест")
        self.console.print("2. Стандартное (50 эпох) - оптимальный баланс")
        self.console.print("3. Полное (100 эпох) - максимальное качество")
        self.console.print("4. Пользовательское количество эпох")
        
        choice = Prompt.ask("Выбор", default="1")
        
        if choice == "1":
            epochs = 5
        elif choice == "2":
            epochs = 50
        elif choice == "3":
            epochs = 100
        elif choice == "4":
            epochs = IntPrompt.ask("Количество эпох", default=10)
        else:
            return
        
        # Обновляем конфигурацию с количеством эпох
        self.config['model']['epochs'] = epochs
        self.save_config()
        
        # Информация о ресурсах
        self.console.print(f"\n[cyan]Конфигурация обучения:[/cyan]")
        self.console.print(f"  • Эпохи: {epochs}")
        if hasattr(self, 'gpu_name'):
            self.console.print(f"  • GPU: {self.gpu_name}")
            if self.gpu_memory > 0:
                self.console.print(f"  • VRAM: {self.gpu_memory:.1f} GB")
        self.console.print(f"  • Batch size: {self.config['model']['batch_size']}")
        self.console.print(f"  • Learning rate: {self.config['model']['learning_rate']}")
        
        # Запуск обучения
        if Confirm.ask("\nЗапустить обучение?"):
            self.console.print("\n[yellow]Запуск обучения...[/yellow]")
            
            # Создаем директорию для логов если не существует
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Имя лог файла
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"training_local_{timestamp}.log"
            
            try:
                # Запускаем обучение с выводом в реальном времени
                self.console.print(f"[dim]Лог сохраняется в: {log_file}[/dim]\n")
                
                # Используем Popen для вывода в реальном времени
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        ["python", "main.py", "--mode", "train"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,  # Построчная буферизация
                        env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Отключаем буферизацию Python
                    )
                    
                    # Читаем и выводим построчно
                    for line in process.stdout:
                        # Записываем в файл
                        f.write(line)
                        f.flush()
                        # Выводим на консоль с форматированием
                        line = line.rstrip()
                        if "Epoch" in line and "/" in line:
                            self.console.print(f"[cyan]{line}[/cyan]")
                        elif "Loss:" in line:
                            self.console.print(f"[yellow]{line}[/yellow]")
                        elif "✅" in line or "Сохранен" in line:
                            self.console.print(f"[green]{line}[/green]")
                        elif "❌" in line or "Ошибка" in line:
                            self.console.print(f"[red]{line}[/red]")
                        elif "GPU" in line or "CUDA" in line:
                            self.console.print(f"[magenta]{line}[/magenta]")
                        else:
                            self.console.print(line)
                    
                    # Ждем завершения процесса
                    return_code = process.wait()
                    
                if return_code == 0:
                    self.console.print("\n[green]✅ Обучение успешно завершено![/green]")
                    
                    # Проверяем сохраненные модели
                    models_dir = Path("models_saved")
                    if models_dir.exists():
                        latest_models = sorted(models_dir.glob("*.pth"), 
                                             key=lambda x: x.stat().st_mtime, 
                                             reverse=True)[:3]
                        if latest_models:
                            self.console.print("\n[cyan]Сохраненные модели:[/cyan]")
                            for model in latest_models:
                                size_mb = model.stat().st_size / (1024 * 1024)
                                self.console.print(f"  • {model.name} ({size_mb:.1f} MB)")
                else:
                    self.console.print(f"\n[red]❌ Обучение завершилось с ошибкой (код: {return_code})[/red]")
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]⚠️  Обучение прервано пользователем[/yellow]")
                if process:
                    process.terminate()
            except Exception as e:
                self.console.print(f"\n[red]❌ Ошибка при запуске обучения: {e}[/red]")
                self.logger.error(f"Ошибка в _launch_local_training: {e}", exc_info=True)
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run(self):
        """Главный цикл меню"""
        while True:
            self.display_main_menu()
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]", default="0")
            
            if choice == "1":
                self.data_management_menu()
            elif choice == "2":
                self.training_menu()
            elif choice == "3":
                self.monitoring_menu()
            elif choice == "4":
                self.settings_menu()
            elif choice == "5":
                self.utilities_menu()
            elif choice == "6":
                self.show_system_info()
            elif choice == "7":
                self.quick_start()
            elif choice == "0":
                if Confirm.ask("\n[yellow]Вы уверены, что хотите выйти?[/yellow]"):
                    self.console.print("\n[cyan]До свидания! 👋[/cyan]")
                    break
            else:
                self.console.print("[red]Неверный выбор. Попробуйте снова.[/red]")
                time.sleep(1)


def main():
    """Точка входа"""
    menu_logger.info("="*60)
    menu_logger.info("Запуск Crypto AI Trading System")
    menu_logger.info(f"Файл логов: {log_file}")
    menu_logger.info("="*60)
    
    try:
        menu = CryptoTradingMenu()
        menu.run()
    except KeyboardInterrupt:
        menu_logger.info("Программа прервана пользователем")
        console.print("\n\n[yellow]Программа прервана пользователем[/yellow]")
    except Exception as e:
        menu_logger.error(f"Критическая ошибка: {e}", exc_info=True)
        console.print(f"\n[red]Критическая ошибка: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        menu_logger.info("Завершение работы")


if __name__ == "__main__":
    main()