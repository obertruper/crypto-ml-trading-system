#!/usr/bin/env python3
"""
Детальный просмотр логов текущего обучения
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

console = Console()

def find_training_logs():
    """Найти все файлы логов связанные с обучением"""
    log_files = []
    
    # Основные директории логов
    log_dirs = [
        Path("logs"),
        Path("experiments/logs"),
        Path("models_saved"),
        Path(".")
    ]
    
    for log_dir in log_dirs:
        if log_dir.exists():
            # Ищем все .log файлы
            for log_file in log_dir.rglob("*.log"):
                if 'venv' not in str(log_file):
                    stat = log_file.stat()
                    log_files.append({
                        'path': log_file,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime)
                    })
    
    # Сортируем по времени изменения
    log_files.sort(key=lambda x: x['modified'], reverse=True)
    return log_files

def get_process_info():
    """Получить информацию о процессах обучения"""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        
        processes = []
        for line in result.stdout.split('\n'):
            if 'python main.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    processes.append({
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'time': parts[9],
                        'cmd': ' '.join(parts[10:])
                    })
        
        return processes
    except:
        return []

def tail_file(file_path, lines=50):
    """Читать последние строки файла"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.readlines()
            return content[-lines:]
    except:
        return []

def create_logs_table(log_files):
    """Создать таблицу с файлами логов"""
    table = Table(title="📁 Файлы логов")
    table.add_column("№", style="cyan")
    table.add_column("Файл", style="green")
    table.add_column("Размер", style="yellow")
    table.add_column("Изменен", style="blue")
    
    for i, log in enumerate(log_files[:10]):
        size_mb = log['size'] / 1024 / 1024
        table.add_row(
            str(i + 1),
            str(log['path']),
            f"{size_mb:.2f} MB",
            log['modified'].strftime("%Y-%m-%d %H:%M:%S")
        )
    
    return table

def create_process_table(processes):
    """Создать таблицу процессов"""
    table = Table(title="⚙️ Процессы обучения")
    table.add_column("PID", style="cyan")
    table.add_column("CPU %", style="green")
    table.add_column("MEM %", style="yellow")
    table.add_column("Время", style="blue")
    table.add_column("Команда", style="white")
    
    for proc in processes[:5]:
        table.add_row(
            proc['pid'],
            proc['cpu'],
            proc['mem'],
            proc['time'],
            proc['cmd'][:50] + "..."
        )
    
    return table

def main():
    """Основная функция"""
    console.print("[bold green]🔍 Детальный просмотр логов обучения[/bold green]\n")
    
    # Находим логи
    log_files = find_training_logs()
    if not log_files:
        console.print("[red]❌ Файлы логов не найдены[/red]")
        return
    
    # Показываем таблицу логов
    console.print(create_logs_table(log_files))
    
    # Показываем процессы
    processes = get_process_info()
    if processes:
        console.print("\n")
        console.print(create_process_table(processes))
    
    # Выбор файла для просмотра
    console.print("\n[yellow]Введите номер файла для просмотра (или Enter для авто-выбора):[/yellow]")
    choice = input("Выбор: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(log_files):
        selected_log = log_files[int(choice) - 1]['path']
    else:
        selected_log = log_files[0]['path']
    
    console.print(f"\n[cyan]📄 Просмотр: {selected_log}[/cyan]")
    
    # Live обновление
    with Live(refresh_per_second=0.5) as live:
        while True:
            try:
                # Читаем последние строки
                lines = tail_file(selected_log, 100)
                
                # Форматируем контент
                content = ''.join(lines)
                
                # Подсвечиваем важные строки
                highlighted = content
                highlighted = highlighted.replace("ERROR", "[red]ERROR[/red]")
                highlighted = highlighted.replace("WARNING", "[yellow]WARNING[/yellow]")
                highlighted = highlighted.replace("INFO", "[green]INFO[/green]")
                highlighted = highlighted.replace("train_loss", "[cyan]train_loss[/cyan]")
                highlighted = highlighted.replace("val_loss", "[magenta]val_loss[/magenta]")
                highlighted = highlighted.replace("✅", "[green]✅[/green]")
                highlighted = highlighted.replace("📊", "[blue]📊[/blue]")
                
                panel = Panel(
                    highlighted,
                    title=f"📜 {selected_log} (последние 100 строк)",
                    border_style="blue"
                )
                
                live.update(panel)
                time.sleep(2)
                
            except KeyboardInterrupt:
                break
    
    console.print("\n[yellow]✨ Мониторинг завершен[/yellow]")

if __name__ == "__main__":
    main()