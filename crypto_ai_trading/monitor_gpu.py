#!/usr/bin/env python3
"""
Мониторинг GPU во время обучения
"""
import subprocess
import time
import datetime

print("🔍 Мониторинг GPU RTX 5090")
print("="*80)

# Мониторим 10 секунд с интервалом 1 секунда
for i in range(10):
    # Получаем статистику GPU
    result = subprocess.run([
        'nvidia-smi', 
        '--query-gpu=timestamp,gpu_name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
        '--format=csv,noheader,nounits'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        data = result.stdout.strip()
        parts = data.split(', ')
        
        timestamp = parts[0]
        gpu_name = parts[1]
        temp = float(parts[2])
        gpu_util = float(parts[3])
        mem_util = float(parts[4])
        mem_used = float(parts[5])
        mem_total = float(parts[6])
        power = float(parts[7])
        
        # Вычисляем процент памяти
        mem_percent = (mem_used / mem_total) * 100
        
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Итерация {i+1}/10:")
        print(f"  🌡️  Температура: {temp}°C")
        print(f"  ⚡ GPU утилизация: {gpu_util}%")
        print(f"  💾 Память: {mem_used:.0f}/{mem_total:.0f} MB ({mem_percent:.1f}%)")
        print(f"  🔌 Потребление: {power:.1f}W / 600W")
        
        # Рекомендации
        if gpu_util < 80:
            print(f"  ⚠️  GPU утилизация низкая ({gpu_util}%)")
            
        if mem_percent > 90:
            print(f"  ⚠️  Память почти заполнена ({mem_percent:.1f}%)")
        
    time.sleep(1)

print("\n" + "="*80)
print("📊 АНАЛИЗ:")
print("- RTX 5090 имеет 32GB памяти")
print("- Оптимальная утилизация GPU: 85-95%")
print("- Текущий batch_size: 1024")
print("\n💡 РЕКОМЕНДАЦИИ для увеличения нагрузки:")
print("1. Увеличить batch_size до 2048 или 3072")
print("2. Включить gradient_accumulation_steps: 2")
print("3. Увеличить размер модели (d_model, d_ff)")
print("4. Использовать больше слоев (e_layers)")