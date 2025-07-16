#!/usr/bin/env python3
"""
Мониторинг текущего обучения XGBoost в реальном времени
"""

import os
import time
import subprocess
from datetime import datetime
from pathlib import Path

def get_latest_log_dir():
    """Найти последнюю директорию с логами"""
    logs_dir = Path("logs")
    xgboost_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("xgboost_training_")]
    if not xgboost_dirs:
        return None
    return max(xgboost_dirs, key=lambda d: d.stat().st_mtime)

def monitor_training():
    """Мониторинг обучения"""
    log_dir = get_latest_log_dir()
    if not log_dir:
        print("❌ Не найдены логи обучения")
        return
        
    log_file = log_dir / "training.log"
    print(f"📂 Мониторинг: {log_dir.name}")
    print("=" * 60)
    
    # Проверяем статус процесса
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    python_processes = [line for line in result.stdout.split('\n') if 'python' in line and 'train' in line]
    
    if python_processes:
        print("✅ Процесс обучения активен")
        for proc in python_processes:
            if 'train_xgboost' in proc:
                parts = proc.split()
                cpu = parts[2]
                mem = parts[3]
                print(f"   CPU: {cpu}%, Memory: {mem}%")
    else:
        print("⚠️ Процесс обучения не найден (возможно завершен)")
    
    print("=" * 60)
    
    # Следим за логом
    print("\n📊 Последние события:")
    print("-" * 60)
    
    try:
        # Используем tail -f для отслеживания в реальном времени
        subprocess.run(["tail", "-f", str(log_file)])
    except KeyboardInterrupt:
        print("\n\n✋ Мониторинг остановлен")
        
        # Показываем итоговую статистику
        print("\n📈 Итоговая статистика:")
        print("-" * 60)
        
        # Проверяем наличие финального отчета
        final_report = log_dir / "final_report.txt"
        if final_report.exists():
            print("✅ Обучение завершено!")
            print("\n📄 Финальный отчет:")
            with open(final_report, 'r') as f:
                print(f.read())
        else:
            # Считаем строки с ключевыми событиями
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            optuna_count = sum(1 for line in lines if "Optuna" in line)
            model_count = sum(1 for line in lines if "Обучение модели" in line)
            error_count = sum(1 for line in lines if "ERROR" in line or "❌" in line)
            
            print(f"📊 Optuna оптимизаций: {optuna_count}")
            print(f"🤖 Моделей обучено: {model_count}")
            print(f"❌ Ошибок: {error_count}")
            
            # Время работы
            start_time = datetime.strptime(log_dir.name.split('_')[2], '%Y%m%d')
            runtime = datetime.now() - start_time
            print(f"⏱️ Время работы: {runtime}")

if __name__ == "__main__":
    print("🔍 Мониторинг обучения XGBoost Enhanced v2.0")
    print("   Нажмите Ctrl+C для остановки мониторинга")
    print()
    monitor_training()