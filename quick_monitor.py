#!/usr/bin/env python3
"""
Быстрый мониторинг обучения
"""
import os
import pandas as pd
import time
from datetime import datetime

def monitor():
    # Находим последнюю папку с логами
    log_dirs = [d for d in os.listdir('logs') if d.startswith('training_')]
    latest_dir = sorted(log_dirs)[-1]
    log_path = f'logs/{latest_dir}'
    
    print(f"📊 Мониторинг обучения: {log_path}\n")
    
    while True:
        try:
            # Очищаем экран
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)
            
            # Читаем метрики для каждой модели
            for model in ['buy_profit', 'buy_loss', 'sell_profit', 'sell_loss']:
                metrics_file = f'{log_path}/{model}_model_metrics.csv'
                
                if os.path.exists(metrics_file):
                    df = pd.read_csv(metrics_file)
                    if len(df) > 0:
                        last = df.iloc[-1]
                        print(f"\n📈 {model}_model (Эпоха {int(last['epoch'])})")
                        print(f"   Loss: {last['loss']:.4f} (val: {last['val_loss']:.4f})")
                        print(f"   Accuracy: {last['accuracy']:.2%} (val: {last['val_accuracy']:.2%})")
                        print(f"   AUC: {last['auc']:.4f} (val: {last['val_auc']:.4f})")
                        print(f"   Время: {last['time']:.1f}с")
                        
                        # Прогресс
                        if len(df) > 1:
                            prev = df.iloc[-2]
                            loss_change = last['val_loss'] - prev['val_loss']
                            print(f"   Изменение loss: {loss_change:+.4f}")
            
            # Проверяем общий прогресс
            log_file = f'{log_path}/training.log'
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        if 'Эпоха' in line and '/' in line:
                            print(f"\n🎯 {line.strip()}")
                            break
            
            print("\n[Обновление каждые 5 секунд. Ctrl+C для выхода]")
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n👋 Мониторинг остановлен")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor()