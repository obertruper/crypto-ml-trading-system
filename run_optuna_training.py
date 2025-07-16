#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск улучшенного обучения XGBoost v2.0 с Optuna оптимизацией
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_training():
    """Запуск обучения с оптимальными параметрами"""
    
    print(f"\n{'='*60}")
    print(f"🚀 Enhanced XGBoost v2.0 с Optuna оптимизацией")
    print(f"📊 Включает:")
    print(f"   - 49 технических индикаторов")
    print(f"   - Взвешенные комбинации признаков")
    print(f"   - Скользящие статистики")
    print(f"   - Дивергенции и паттерны свечей")
    print(f"   - Volume profile")
    print(f"   - Обязательная Optuna оптимизация")
    print(f"   - Паттерн-анализ и фильтрация сигналов")
    print(f"🕐 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Формируем команду
    cmd = [
        'python', 'train_xgboost_enhanced_v2.py',
        '--task', 'classification_binary',
        '--ensemble_size', '3'
    ]
    
    # Запускаем обучение
    try:
        print(f"Выполняется команда: {' '.join(cmd)}\n")
        
        # Запускаем процесс
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Читаем вывод в реальном времени
        for line in process.stdout:
            print(line, end='')
        
        # Ждем завершения
        return_code = process.wait()
        
        if return_code == 0:
            print(f"\n✅ Обучение завершено успешно!")
            print(f"\n📊 Результаты сохранены в:")
            print(f"   - logs/xgboost_training_*/")
            print(f"   - trained_model/*_xgboost_v2_*.pkl")
        else:
            print(f"\n❌ Ошибка при обучении! Код возврата: {return_code}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Обучение прервано пользователем")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)

def main():
    """Основная функция"""
    print("\n🤖 Enhanced XGBoost v2.0 для криптотрейдинга")
    print("="*50)
    print("\nОсновные улучшения:")
    print("✅ Добавлены взвешенные комбинации признаков")
    print("✅ Скользящие статистики для ключевых индикаторов")
    print("✅ Дивергенции между ценой и индикаторами")
    print("✅ Паттерны свечей (hammer, doji, engulfing)")
    print("✅ Volume profile признаки")
    print("✅ Обязательная Optuna оптимизация")
    print("✅ Анализ выигрышных паттернов")
    print("✅ Фильтрация сигналов")
    
    input("\nНажмите Enter для начала обучения...")
    
    run_training()

if __name__ == "__main__":
    main()