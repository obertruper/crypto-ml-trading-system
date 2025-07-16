#!/usr/bin/env python3
"""
Мониторинг прогресса обучения
"""

import time
import subprocess
import os

def get_training_stats():
    """Получение статистики обучения из логов"""
    try:
        # Читаем последние строки из training_log.txt
        result = subprocess.run(
            ['tail', '-1', 'training_log.txt'], 
            capture_output=True, text=True, cwd='/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading'
        )
        if result.returncode == 0 and 'Training:' in result.stdout:
            # Парсим прогресс бар
            line = result.stdout.strip()
            # Извлекаем loss
            if 'loss=' in line:
                parts = line.split('loss=')
                if len(parts) > 1:
                    loss_part = parts[1].split(',')[0]
                    avg_loss_part = parts[1].split('avg_loss=')[1].split(']')[0] if 'avg_loss=' in parts[1] else None
                    
                    # Извлекаем процент
                    if '%' in line:
                        percent = line.split('%')[0].split()[-1]
                        return {
                            'percent': f"{percent}%",
                            'loss': loss_part,
                            'avg_loss': avg_loss_part
                        }
        
        # Альтернативный способ - из Trainer логов
        result = subprocess.run(
            ['tail', '-50', './experiments/logs/Trainer_20250701.log'], 
            capture_output=True, text=True, cwd='/mnt/SSD/PYCHARMPRODJECT/LLM TRANSFORM/crypto_ai_trading'
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if 'Training:' in line and 'loss=' in line:
                    return {'status': 'training', 'last_log': line[-100:]}
                    
    except Exception as e:
        return {'error': str(e)}
    
    return {'status': 'unknown'}

def get_gpu_stats():
    """Получение статистики GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw', 
             '--format=csv,noheader,nounits'], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'gpu_util': f"{stats[0]}%",
                'memory': f"{float(stats[1])/1024:.1f}GB",
                'temp': f"{stats[2]}°C",
                'power': f"{float(stats[3]):.0f}W"
            }
    except:
        pass
    return None

def main():
    """Основной цикл мониторинга"""
    print("📊 Мониторинг обучения модели...")
    print("=" * 80)
    
    while True:
        os.system('clear')
        
        print("🔥 СТАТУС ОБУЧЕНИЯ CRYPTO AI TRADING")
        print("=" * 80)
        
        # GPU статистика
        gpu_stats = get_gpu_stats()
        if gpu_stats:
            print(f"GPU: {gpu_stats['gpu_util']} | Память: {gpu_stats['memory']} | {gpu_stats['temp']} | {gpu_stats['power']}")
        
        print("-" * 80)
        
        # Статистика обучения
        train_stats = get_training_stats()
        if 'percent' in train_stats:
            print(f"Прогресс: {train_stats['percent']}")
            print(f"Loss: {train_stats['loss']}")
            print(f"Avg Loss: {train_stats['avg_loss']}")
        elif 'last_log' in train_stats:
            print(f"Статус: {train_stats.get('status', 'обучение')}")
            print(f"Лог: {train_stats['last_log']}")
        else:
            print(f"Статус: {train_stats}")
        
        print("\n[Ctrl+C для выхода] Обновление через 5 сек...")
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✅ Мониторинг завершен")