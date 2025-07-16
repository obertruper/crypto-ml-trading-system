#!/usr/bin/env python3
"""
Тест Google Gemini API
"""

import google.generativeai as genai

# Ваш API ключ
API_KEY = 'AIzaSyB9YMfvx-cWJk50GUFptMgXpnBOKBSyWuw'

# Настройка API
genai.configure(api_key=API_KEY)

# Проверяем доступные модели
print("📋 Доступные модели:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  - {m.name}")

print("\n🤖 Тестируем Gemini...")

# Используем модель
model = genai.GenerativeModel('gemini-1.5-flash')

# Простой запрос
response = model.generate_content("Объясни в 2-3 предложениях что такое криптотрейдинг")
print(f"\nОтвет Gemini:\n{response.text}")

print("\n✅ Gemini API работает успешно!")