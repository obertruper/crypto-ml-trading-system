#!/usr/bin/env python3
"""
Тест Google Gemini API с новой библиотекой google-genai (2025)
"""

from google import genai

# Ваш API ключ
API_KEY = 'AIzaSyB9YMfvx-cWJk50GUFptMgXpnBOKBSyWuw'

# Создаем клиент с API ключом
client = genai.Client(api_key=API_KEY)

print("🚀 Тестируем новую библиотеку Google GenAI...")

try:
    # Простой запрос с Gemini 2.5 Flash
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Объясни в 2-3 предложениях что такое PatchTST модель для временных рядов"
    )
    
    print(f"\n✅ Ответ Gemini 2.5 Flash:\n{response.text}")
    
    # Пример с параметрами генерации
    print("\n🎯 Запрос с кастомными параметрами...")
    response2 = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Предложи 3 инновационные идеи для улучшения криптотрейдинг модели",
        config={
            "temperature": 0.9,  # Больше креативности
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": 500,
        }
    )
    
    print(f"\n✨ Креативный ответ:\n{response2.text}")
    
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    print("\n💡 Возможные причины:")
    print("1. Проверьте API ключ")
    print("2. Убедитесь что у вас есть доступ к Gemini API")
    print("3. Проверьте лимиты использования")

print("\n📚 Доступные модели в 2025:")
print("- gemini-2.5-flash (быстрая)")
print("- gemini-2.5-pro (мощная)")
print("- gemini-2.0-flash-thinking-exp (с расширенным мышлением)")