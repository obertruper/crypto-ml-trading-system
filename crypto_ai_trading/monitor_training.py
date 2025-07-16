#!/usr/bin/env python3
"""
Мониторинг процесса обучения в реальном времени
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import glob

console = Console()

def find_latest_training_dir():
    """Найти последнюю директорию с обучением"""
    base_dir = Path("experiments/runs")
    if not base_dir.exists():
        return None
    
    training_dirs = sorted(base_dir.glob("training_*"), key=lambda x: x.stat().st_mtime)
    if not training_dirs:
        return None
    
    return training_dirs[-1]

def read_metrics_file(metrics_file):
    """Читать файл с метриками"""
    if not metrics_file.exists():
        return pd.DataFrame()
    
    try:
        return pd.read_csv(metrics_file)
    except:
        return pd.DataFrame()

def create_metrics_table(metrics_df):
    """Создать таблицу с метриками"""
    table = Table(title="📊 Метрики обучения")
    
    if metrics_df.empty:
        table.add_column("Статус", style="yellow")
        table.add_row("Ожидание начала обучения...")
        return table
    
    # Добавляем колонки
    table.add_column("Эпоха", justify="right", style="cyan")
    table.add_column("Train Loss", justify="right", style="green")
    table.add_column("Val Loss", justify="right", style="blue")
    table.add_column("Train MAE", justify="right", style="green")
    table.add_column("Val MAE", justify="right", style="blue")
    table.add_column("LR", justify="right", style="yellow")
    
    # Берем последние 5 эпох
    last_rows = metrics_df.tail(5)
    
    for _, row in last_rows.iterrows():
        table.add_row(
            str(int(row.get('epoch', 0))),
            f"{row.get('train_loss', 0):.4f}",
            f"{row.get('val_loss', 0):.4f}",
            f"{row.get('train_mae', 0):.4f}",
            f"{row.get('val_mae', 0):.4f}",
            f"{row.get('learning_rate', 0):.6f}"
        )
    
    return table

def create_training_info(training_dir):
    """Создать панель с информацией об обучении"""
    info_lines = []
    
    # Читаем конфигурацию если есть
    config_file = training_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            info_lines.append(f"🏗️ Модель: {config.get('model', {}).get('name', 'Unknown')}")
            info_lines.append(f"📦 Batch size: {config.get('model', {}).get('batch_size', 'Unknown')}")
            info_lines.append(f"🎯 Learning rate: {config.get('model', {}).get('learning_rate', 'Unknown')}")
            info_lines.append(f"📈 Эпох: {config.get('model', {}).get('epochs', 'Unknown')}")
    
    # Время обучения
    start_time = datetime.fromtimestamp(training_dir.stat().st_ctime)
    duration = datetime.now() - start_time
    info_lines.append(f"⏱️ Время обучения: {str(duration).split('.')[0]}")
    
    # Директория
    info_lines.append(f"📁 {training_dir.name}")
    
    return Panel("\n".join(info_lines), title="ℹ️ Информация об обучении")

def plot_metrics(metrics_df, save_path):
    """Построить графики метрик"""
    if metrics_df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Прогресс обучения', fontsize=16)
    
    # Loss
    if 'train_loss' in metrics_df and 'val_loss' in metrics_df:
        ax = axes[0, 0]
        ax.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train', color='blue')
        ax.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val', color='orange')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # MAE
    if 'train_mae' in metrics_df and 'val_mae' in metrics_df:
        ax = axes[0, 1]
        ax.plot(metrics_df['epoch'], metrics_df['train_mae'], label='Train', color='blue')
        ax.plot(metrics_df['epoch'], metrics_df['val_mae'], label='Val', color='orange')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('MAE')
        ax.set_title('Mean Absolute Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Learning Rate
    if 'learning_rate' in metrics_df:
        ax = axes[1, 0]
        ax.plot(metrics_df['epoch'], metrics_df['learning_rate'], color='green')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    
    # Best metrics
    ax = axes[1, 1]
    ax.axis('off')
    best_val_loss = metrics_df['val_loss'].min() if 'val_loss' in metrics_df else 0
    best_epoch = metrics_df['val_loss'].idxmin() if 'val_loss' in metrics_df else 0
    ax.text(0.1, 0.7, f"Лучший Val Loss: {best_val_loss:.4f}", fontsize=12)
    ax.text(0.1, 0.5, f"Лучшая эпоха: {best_epoch}", fontsize=12)
    ax.text(0.1, 0.3, f"Текущая эпоха: {len(metrics_df)}", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def monitor_training():
    """Основной цикл мониторинга"""
    console.print("[bold green]🚀 Запуск мониторинга обучения...[/bold green]")
    
    while True:
        # Находим последнюю директорию обучения
        training_dir = find_latest_training_dir()
        
        if not training_dir:
            console.print("[yellow]⚠️ Директория обучения не найдена. Ожидание...[/yellow]")
            time.sleep(5)
            continue
        
        # Ищем файл с метриками
        metrics_files = list(training_dir.glob("*_metrics.csv"))
        
        if not metrics_files:
            console.print(f"[yellow]⚠️ Файл метрик не найден в {training_dir}. Ожидание...[/yellow]")
            time.sleep(5)
            continue
        
        metrics_file = metrics_files[0]
        
        # Мониторинг в цикле
        with Live(refresh_per_second=0.5) as live:
            while True:
                # Читаем метрики
                metrics_df = read_metrics_file(metrics_file)
                
                # Создаем layout
                layout = Layout()
                layout.split_column(
                    Layout(create_training_info(training_dir), size=8),
                    Layout(create_metrics_table(metrics_df))
                )
                
                live.update(layout)
                
                # Обновляем график каждые 10 секунд
                if int(time.time()) % 10 == 0:
                    plot_path = training_dir / "training_progress.png"
                    plot_metrics(metrics_df, plot_path)
                
                # Проверяем, завершено ли обучение
                final_report = training_dir / "final_report.txt"
                if final_report.exists():
                    console.print("\n[bold green]✅ Обучение завершено![/bold green]")
                    console.print(f"[cyan]📄 Финальный отчет: {final_report}[/cyan]")
                    console.print(f"[cyan]📊 Графики: {training_dir}/training_progress.png[/cyan]")
                    return
                
                time.sleep(2)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Мониторинг остановлен пользователем[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Ошибка: {e}[/red]")