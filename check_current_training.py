#!/usr/bin/env python3
"""
Проверка текущего процесса обучения
"""

import os
import json
import pandas as pd
from datetime import datetime
import glob

print("="*80)
print("🔍 ПРОВЕРКА ТЕКУЩЕГО ОБУЧЕНИЯ")
print("="*80)

# Находим последнюю директорию с логами
log_dirs = glob.glob("logs/training_*")
if not log_dirs:
    print("❌ Нет директорий с логами обучения")
    exit()

# Сортируем по дате и берем последнюю
latest_log_dir = sorted(log_dirs)[-1]
print(f"\n📁 Анализируем: {latest_log_dir}")

# Проверяем время начала
dir_name = os.path.basename(latest_log_dir)
timestamp = dir_name.replace("training_", "")
start_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
duration = datetime.now() - start_time
print(f"⏰ Начало обучения: {start_time}")
print(f"⏱️  Длительность: {duration}")

# Читаем лог файл
log_file = os.path.join(latest_log_dir, "training.log")
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Ищем информацию о данных
    print("\n📊 ИНФОРМАЦИЯ О ДАННЫХ:")
    for line in lines:
        if "Загружено" in line and "записей" in line:
            print(f"  {line.strip()}")
        if "Распределение по символам" in line:
            idx = lines.index(line)
            # Показываем следующие строки с символами
            for i in range(1, 10):
                if idx + i < len(lines) and "записей" in lines[idx + i]:
                    print(f"  {lines[idx + i].strip()}")
        if "Статистика buy_return" in line or "Статистика sell_return" in line:
            idx = lines.index(line)
            print(f"\n{line.strip()}")
            for i in range(1, 4):
                if idx + i < len(lines):
                    print(f"  {lines[idx + i].strip()}")

# Проверяем метрики
print("\n📈 МЕТРИКИ ОБУЧЕНИЯ:")

# buy_return_predictor метрики
buy_metrics_file = os.path.join(latest_log_dir, "buy_return_predictor_metrics.csv")
if os.path.exists(buy_metrics_file):
    buy_df = pd.read_csv(buy_metrics_file)
    if len(buy_df) > 0:
        print("\n🟢 BUY_RETURN_PREDICTOR:")
        print(f"   Эпох обучено: {len(buy_df)}")
        
        # Последние 5 эпох
        print("\n   Последние 5 эпох:")
        print("   " + "-"*60)
        print(f"   {'Epoch':>6} {'Loss':>10} {'MAE':>10} {'Val Loss':>10} {'Val MAE':>10}")
        print("   " + "-"*60)
        
        for _, row in buy_df.tail(5).iterrows():
            print(f"   {row['epoch']:>6} {row['loss']:>10.4f} {row['mae']:>10.4f} {row['val_loss']:>10.4f} {row['val_mae']:>10.4f}")
        
        # Лучшие результаты
        best_val_loss_idx = buy_df['val_loss'].idxmin()
        best_row = buy_df.loc[best_val_loss_idx]
        print(f"\n   ✨ Лучший результат (эпоха {best_row['epoch']}):")
        print(f"      Val Loss: {best_row['val_loss']:.4f}")
        print(f"      Val MAE: {best_row['val_mae']:.4f}%")
        
        # Тренд
        if len(buy_df) > 5:
            recent_trend = buy_df['val_loss'].tail(5).diff().mean()
            if recent_trend < 0:
                print(f"   📉 Тренд: Улучшение (val_loss снижается)")
            elif recent_trend > 0:
                print(f"   📈 Тренд: Ухудшение (val_loss растет)")
            else:
                print(f"   ➡️  Тренд: Стабилизация")

# sell_return_predictor метрики
sell_metrics_file = os.path.join(latest_log_dir, "sell_return_predictor_metrics.csv")
if os.path.exists(sell_metrics_file):
    sell_df = pd.read_csv(sell_metrics_file)
    if len(sell_df) > 0:
        print("\n🔴 SELL_RETURN_PREDICTOR:")
        print(f"   Эпох обучено: {len(sell_df)}")
        
        # Аналогичный анализ для sell модели
        print("\n   Последние 5 эпох:")
        print("   " + "-"*60)
        print(f"   {'Epoch':>6} {'Loss':>10} {'MAE':>10} {'Val Loss':>10} {'Val MAE':>10}")
        print("   " + "-"*60)
        
        for _, row in sell_df.tail(5).iterrows():
            print(f"   {row['epoch']:>6} {row['loss']:>10.4f} {row['mae']:>10.4f} {row['val_loss']:>10.4f} {row['val_mae']:>10.4f}")

# Проверяем графики
plots_dir = os.path.join(latest_log_dir, "plots")
if os.path.exists(plots_dir):
    plot_files = glob.glob(os.path.join(plots_dir, "*.png"))
    if plot_files:
        print(f"\n🎨 ГРАФИКИ:")
        for plot in sorted(plot_files):
            print(f"   ✅ {os.path.basename(plot)}")

# Финальный отчет
final_report = os.path.join(latest_log_dir, "final_report.txt")
if os.path.exists(final_report):
    print("\n📄 ФИНАЛЬНЫЙ ОТЧЕТ НАЙДЕН!")
    with open(final_report, 'r') as f:
        report_lines = f.readlines()
    
    # Показываем ключевые метрики из отчета
    in_results = False
    for line in report_lines:
        if "РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ" in line:
            in_results = True
        if in_results and ("MAE:" in line or "RMSE:" in line or "R²:" in line or "Direction Accuracy:" in line):
            print(f"   {line.strip()}")

# Проверяем, идет ли еще обучение
if not os.path.exists(final_report):
    print("\n⏳ Статус: Обучение продолжается...")
    
    # Оценка прогресса
    if os.path.exists(buy_metrics_file):
        buy_df = pd.read_csv(buy_metrics_file)
        if len(buy_df) > 0:
            epochs_done = len(buy_df)
            print(f"   Прогресс buy_return_predictor: {epochs_done}/100 эпох")
else:
    print("\n✅ Статус: Обучение завершено!")

print("\n" + "="*80)