#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска улучшенного обучения XGBoost с паттерн-анализом
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_training(task='classification_binary', ensemble_size=3, test_mode=False, use_cache=True):
    """Запуск обучения с улучшенными параметрами и кэшированием"""
    
    print(f"\n{'='*60}")
    print(f"🚀 Запуск улучшенного обучения XGBoost v2.0")
    print(f"📊 Режим: {task}")
    print(f"🎯 Размер ансамбля: {ensemble_size}")
    print(f"⚡ Тестовый режим: {test_mode}")
    print(f"💾 Использование кэша: {use_cache}")
    print(f"🕐 Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Формируем команду
    cmd = [
        'python', 'train_xgboost_enhanced_v2.py',
        '--task', task,
        '--ensemble_size', str(ensemble_size)
    ]
    
    if test_mode:
        cmd.append('--test_mode')
    
    if use_cache:
        cmd.append('--use-cache')
        print("📦 Кэширование включено - повторные запуски будут быстрее!")
    else:
        cmd.append('--no-cache')
        print("🔄 Кэширование отключено - загружаются свежие данные из БД")
    
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
    """Основная функция с меню выбора"""
    
    print("\n🤖 Улучшенное обучение XGBoost для криптотрейдинга")
    print("="*50)
    print("\nВыберите режим обучения:")
    print("1. Бинарная классификация (порог 1%) - рекомендуется")
    print("2. Мультиклассовая классификация (4 класса)")
    print("3. Регрессия (предсказание expected returns)")
    print("4. Тестовый запуск (2 символа, быстро)")
    print("0. Выход")
    
    choice = input("\nВаш выбор (0-4): ").strip()
    
    if choice == '0':
        print("👋 До свидания!")
        return
    elif choice == '1':
        task = 'classification_binary'
    elif choice == '2':
        task = 'classification_multiclass'
    elif choice == '3':
        task = 'regression'
    elif choice == '4':
        # Тестовый режим
        print("\n⚡ Запуск в тестовом режиме...")
        
        # Выбор использования кеша для тестового режима
        print("\n💾 Использовать кеш для тестового запуска?")
        print("1. Да (быстрее)")
        print("2. Нет (свежие данные)")
        test_cache = input("\nВаш выбор (1-2, по умолчанию 1): ").strip()
        
        use_cache = test_cache != '2'
        
        run_training('classification_binary', ensemble_size=2, test_mode=True, use_cache=use_cache)
        return
    else:
        print("❌ Неверный выбор!")
        return
    
    # Выбор размера ансамбля
    print(f"\nВыбран режим: {task}")
    ensemble_input = input("Размер ансамбля (1-5, по умолчанию 3): ").strip()
    
    if ensemble_input:
        try:
            ensemble_size = int(ensemble_input)
            if ensemble_size < 1 or ensemble_size > 5:
                print("⚠️ Размер ансамбля должен быть от 1 до 5. Использую 3.")
                ensemble_size = 3
        except ValueError:
            print("⚠️ Неверное значение. Использую размер 3.")
            ensemble_size = 3
    else:
        ensemble_size = 3
    
    # Выбор использования кеша
    print("\n💾 Использовать кеш для ускорения загрузки данных?")
    print("1. Да (рекомендуется для повторных запусков)")
    print("2. Нет (загрузить свежие данные из БД)")
    cache_choice = input("\nВаш выбор (1-2, по умолчанию 1): ").strip()
    
    if cache_choice == '2':
        use_cache = False
        print("🔄 Будут загружены свежие данные из БД")
    else:
        use_cache = True
        print("📦 Будет использован кеш (если доступен)")
    
    # Запуск обучения
    run_training(task, ensemble_size, test_mode=False, use_cache=use_cache)

if __name__ == "__main__":
    main()