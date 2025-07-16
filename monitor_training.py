#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Мониторинг процесса обучения в реальном времени
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import argparse

class TrainingMonitor:
    """Класс для мониторинга обучения"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle(f'Мониторинг обучения - {log_dir}', fontsize=14)
        
    def find_latest_metrics_files(self):
        """Находит файлы с метриками"""
        metrics_files = []
        for file in os.listdir(self.log_dir):
            if file.endswith('_metrics.csv'):
                metrics_files.append(os.path.join(self.log_dir, file))
        return metrics_files
    
    def update_plots(self, frame):
        """Обновляет графики"""
        metrics_files = self.find_latest_metrics_files()
        
        if not metrics_files:
            return
        
        # Очищаем графики
        for ax in self.axes.flat:
            ax.clear()
        
        colors = ['blue', 'green', 'red', 'orange']
        
        for idx, metrics_file in enumerate(metrics_files):
            if not os.path.exists(metrics_file):
                continue
                
            try:
                # Читаем метрики
                df = pd.read_csv(metrics_file)
                if len(df) == 0:
                    continue
                
                model_name = os.path.basename(metrics_file).replace('_metrics.csv', '')
                color = colors[idx % len(colors)]
                
                # Loss
                self.axes[0, 0].plot(df['epoch'], df['loss'], 
                                   label=f'{model_name} (train)', color=color)
                self.axes[0, 0].plot(df['epoch'], df['val_loss'], 
                                   label=f'{model_name} (val)', color=color, linestyle='--')
                self.axes[0, 0].set_title('Loss')
                self.axes[0, 0].set_xlabel('Эпоха')
                self.axes[0, 0].set_ylabel('Loss')
                self.axes[0, 0].legend(fontsize=8)
                self.axes[0, 0].grid(True, alpha=0.3)
                
                # Accuracy
                self.axes[0, 1].plot(df['epoch'], df['accuracy'], 
                                   label=f'{model_name} (train)', color=color)
                self.axes[0, 1].plot(df['epoch'], df['val_accuracy'], 
                                   label=f'{model_name} (val)', color=color, linestyle='--')
                self.axes[0, 1].set_title('Accuracy')
                self.axes[0, 1].set_xlabel('Эпоха')
                self.axes[0, 1].set_ylabel('Accuracy')
                self.axes[0, 1].legend(fontsize=8)
                self.axes[0, 1].grid(True, alpha=0.3)
                
                # AUC
                if 'auc' in df.columns:
                    self.axes[1, 0].plot(df['epoch'], df['auc'], 
                                       label=f'{model_name} (train)', color=color)
                    self.axes[1, 0].plot(df['epoch'], df['val_auc'], 
                                       label=f'{model_name} (val)', color=color, linestyle='--')
                    self.axes[1, 0].set_title('AUC')
                    self.axes[1, 0].set_xlabel('Эпоха')
                    self.axes[1, 0].set_ylabel('AUC')
                    self.axes[1, 0].legend(fontsize=8)
                    self.axes[1, 0].grid(True, alpha=0.3)
                
                # Learning rate
                if 'lr' in df.columns:
                    self.axes[1, 1].plot(df['epoch'], df['lr'], 
                                       label=model_name, color=color)
                    self.axes[1, 1].set_title('Learning Rate')
                    self.axes[1, 1].set_xlabel('Эпоха')
                    self.axes[1, 1].set_ylabel('LR')
                    self.axes[1, 1].legend(fontsize=8)
                    self.axes[1, 1].grid(True, alpha=0.3)
                    self.axes[1, 1].set_yscale('log')
                
            except Exception as e:
                print(f"Ошибка чтения {metrics_file}: {e}")
                continue
        
        # Обновляем время
        self.fig.suptitle(f'Мониторинг обучения - {datetime.now().strftime("%H:%M:%S")}', 
                         fontsize=14)
        
        plt.tight_layout()
    
    def start_monitoring(self, interval=10):
        """Запускает мониторинг"""
        print(f"🔍 Мониторинг директории: {self.log_dir}")
        print(f"📊 Обновление каждые {interval} секунд")
        print("Нажмите Ctrl+C для остановки")
        
        ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                    interval=interval*1000, cache_frame_data=False)
        plt.show()


def find_latest_log_dir():
    """Находит последнюю директорию с логами"""
    log_dirs = []
    if os.path.exists('logs'):
        for dir_name in os.listdir('logs'):
            if dir_name.startswith('training_'):
                log_dirs.append(os.path.join('logs', dir_name))
    
    if log_dirs:
        return sorted(log_dirs)[-1]  # Последняя по времени
    return None


def main():
    parser = argparse.ArgumentParser(description='Мониторинг обучения ML модели')
    parser.add_argument('--log-dir', type=str, help='Директория с логами')
    parser.add_argument('--interval', type=int, default=10, 
                       help='Интервал обновления в секундах')
    
    args = parser.parse_args()
    
    # Определяем директорию
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = find_latest_log_dir()
        
    if not log_dir or not os.path.exists(log_dir):
        print("❌ Не найдена директория с логами")
        print("💡 Укажите директорию через --log-dir или запустите обучение")
        return
    
    # Запускаем мониторинг
    monitor = TrainingMonitor(log_dir)
    monitor.start_monitoring(args.interval)


if __name__ == "__main__":
    main()