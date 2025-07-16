#!/usr/bin/env python3
"""
Скрипт для изменения языка в Cursor IDE на русский
"""

import json
import os
import subprocess
import time

def change_cursor_language_to_russian():
    """Изменяет язык Cursor на русский"""
    
    # Путь к файлу настроек Cursor
    settings_path = os.path.expanduser("~/.config/Cursor/User/settings.json")
    
    # Проверяем существование файла
    if not os.path.exists(settings_path):
        print("❌ Файл настроек Cursor не найден!")
        return False
    
    # Читаем текущие настройки
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка чтения настроек: {e}")
        return False
    
    # Добавляем/обновляем настройку языка
    settings['locale'] = 'ru'
    
    # Сохраняем обновленные настройки
    try:
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        print("✅ Настройка языка успешно изменена на русский!")
    except Exception as e:
        print(f"❌ Ошибка сохранения настроек: {e}")
        return False
    
    print("\n📌 Теперь нужно:")
    print("1. Перезапустить Cursor")
    print("2. Установить расширение 'Russian Language Pack for Visual Studio Code':")
    print("   - Откройте панель расширений (Ctrl+Shift+X)")
    print("   - Найдите 'Russian Language Pack'")
    print("   - Установите расширение")
    print("   - Перезапустите Cursor")
    
    # Попытка найти процесс Cursor
    try:
        result = subprocess.run(['pgrep', '-f', 'cursor'], capture_output=True, text=True)
        if result.stdout.strip():
            print(f"\n⚠️  Cursor запущен (PID: {result.stdout.strip()})")
            print("Рекомендуется закрыть и снова открыть Cursor для применения изменений.")
    except:
        pass
    
    return True

if __name__ == "__main__":
    print("🔧 Изменение языка Cursor на русский...")
    change_cursor_language_to_russian()