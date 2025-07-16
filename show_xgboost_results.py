#!/usr/bin/env python3
"""
Скрипт для просмотра результатов обучения XGBoost v3.0
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

def find_latest_results():
    """Находит последнюю папку с результатами"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("❌ Папка logs не найдена")
        return None
        
    # Ищем папки xgboost_v3_*
    result_dirs = list(logs_dir.glob("xgboost_v3_*"))
    if not result_dirs:
        print("❌ Результаты обучения не найдены")
        return None
        
    # Сортируем по времени модификации
    latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
    return latest_dir

def show_results(result_dir):
    """Показывает результаты обучения"""
    print(f"\n📁 Результаты из: {result_dir}")
    print("="*60)
    
    # Читаем итоговый отчет
    report_file = result_dir / "final_report.txt"
    if report_file.exists():
        print("\n📄 ИТОГОВЫЙ ОТЧЕТ:")
        print("-"*60)
        with open(report_file, 'r', encoding='utf-8') as f:
            print(f.read())
    
    # Читаем метрики
    metrics_file = result_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        print("\n📊 ДЕТАЛЬНЫЕ МЕТРИКИ:")
        print("-"*60)
        
        for model_type in ['buy', 'sell']:
            if model_type in metrics:
                print(f"\n🎯 Модель {model_type.upper()}:")
                m = metrics[model_type]
                print(f"  • ROC-AUC: {m.get('roc_auc', 0):.4f}")
                print(f"  • Accuracy: {m.get('accuracy', 0)*100:.1f}%")
                print(f"  • Precision: {m.get('precision', 0)*100:.1f}%")
                print(f"  • Recall: {m.get('recall', 0)*100:.1f}%")
                print(f"  • F1-Score: {m.get('f1', 0):.4f}")
                print(f"  • Порог: {m.get('threshold', 0):.4f}")
                
                # Матрица ошибок
                print(f"\n  📋 Матрица ошибок:")
                print(f"     True Negatives:  {m.get('true_negatives', 0):>6}")
                print(f"     False Positives: {m.get('false_positives', 0):>6}")
                print(f"     False Negatives: {m.get('false_negatives', 0):>6}")
                print(f"     True Positives:  {m.get('true_positives', 0):>6}")
    
    # Показываем список файлов
    print("\n📂 ДОСТУПНЫЕ ФАЙЛЫ:")
    print("-"*60)
    
    # Логи
    log_files = list(result_dir.glob("*.log"))
    if log_files:
        print("\n📝 Логи:")
        for f in log_files:
            size = f.stat().st_size / 1024  # KB
            print(f"  • {f.name} ({size:.1f} KB)")
    
    # Графики
    plots_dir = result_dir / "plots"
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        if plot_files:
            print("\n📊 Графики:")
            for f in plot_files:
                print(f"  • {f.name}")
    
    # Модели
    model_dirs = [d for d in result_dir.iterdir() if d.is_dir() and 'model' in d.name]
    if model_dirs:
        print("\n🤖 Сохраненные модели:")
        for d in model_dirs:
            model_files = list(d.glob("*.json")) + list(d.glob("*.pkl"))
            print(f"  • {d.name}/ ({len(model_files)} файлов)")

def compare_results():
    """Сравнивает несколько последних результатов"""
    logs_dir = Path("logs")
    result_dirs = sorted(logs_dir.glob("xgboost_v3_*"), 
                        key=lambda x: x.stat().st_mtime, 
                        reverse=True)[:5]
    
    if len(result_dirs) < 2:
        print("❌ Недостаточно результатов для сравнения")
        return
        
    print("\n📊 СРАВНЕНИЕ ПОСЛЕДНИХ РЕЗУЛЬТАТОВ:")
    print("="*60)
    print(f"{'Дата/Время':<20} {'ROC-AUC Buy':<12} {'ROC-AUC Sell':<12} {'Acc Buy':<10} {'Acc Sell':<10}")
    print("-"*60)
    
    for result_dir in result_dirs:
        # Извлекаем время из имени папки
        timestamp = result_dir.name.split('_')[2] + '_' + result_dir.name.split('_')[3]
        dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
        date_str = dt.strftime('%Y-%m-%d %H:%M')
        
        # Читаем метрики
        metrics_file = result_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            buy_auc = metrics.get('buy', {}).get('roc_auc', 0)
            sell_auc = metrics.get('sell', {}).get('roc_auc', 0)
            buy_acc = metrics.get('buy', {}).get('accuracy', 0)
            sell_acc = metrics.get('sell', {}).get('accuracy', 0)
            
            print(f"{date_str:<20} {buy_auc:<12.4f} {sell_auc:<12.4f} {buy_acc*100:<10.1f} {sell_acc*100:<10.1f}")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            compare_results()
            return
        elif sys.argv[1] == "--help":
            print("Использование:")
            print("  python show_xgboost_results.py          # Показать последние результаты")
            print("  python show_xgboost_results.py --compare # Сравнить последние 5 результатов")
            return
    
    # Находим последние результаты
    latest_dir = find_latest_results()
    if latest_dir:
        show_results(latest_dir)
        print("\n💡 Для сравнения результатов используйте: python show_xgboost_results.py --compare")
    else:
        print("💡 Сначала запустите обучение: python run_xgboost_v3.py")

if __name__ == "__main__":
    main()