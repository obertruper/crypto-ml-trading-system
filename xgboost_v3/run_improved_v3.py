#!/usr/bin/env python3
"""
Запуск улучшенной версии XGBoost v3 с новыми параметрами
"""

import subprocess
import sys
import os

print("="*60)
print("🚀 УЛУЧШЕННАЯ ВЕРСИЯ XGBoost v3")
print("="*60)
print("\n📊 КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:")
print("  ✅ Порог классификации: 1.5% (реалистичный для крипты)")
print("  ✅ Упрощенная модель: max_depth 3-5, сильная регуляризация")
print("  ✅ Метрика оценки: AUC вместо logloss")
print("  ✅ Оптимизация порога: F1-score вместо gmean")
print("  ✅ Ограничение scale_pos_weight до 5")
print("  ✅ Меньший ансамбль: 3 модели вместо 5")
print("  ✅ Усиленный отбор технических признаков (85%)")

print("\n📈 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
print("  • ROC-AUC > 0.55 (реалистично для крипты)")
print("  • Меньше сигналов, но качественнее")
print("  • Лучшая генерализация на новых данных")
print("  • Устойчивость к рыночному шуму")

print("\n⚠️ ВНИМАНИЕ:")
print("  • Метрики будут ниже, но честнее")
print("  • Модель будет консервативнее")
print("  • Фокус на качестве, а не количестве сигналов")

print("\n" + "="*60)

# Проверка нового порога
print("\n🔍 Проверка баланса классов с новым порогом...")
subprocess.run([sys.executable, "xgboost_v3/check_new_threshold.py"])

print("\n" + "="*60)

# Запрос подтверждения
response = input("\n🚀 Запустить обучение с новыми параметрами? (y/n): ")
if response.lower() != 'y':
    print("❌ Отменено пользователем")
    sys.exit(0)

# Команда запуска
cmd = 'python xgboost_v3/main.py --test-mode --optimize --ensemble-size 3'

print("\n🔧 Запускаем обучение...")
print(f"📌 Команда: {cmd}\n")

# Запуск
try:
    subprocess.run(cmd, shell=True, check=True)
    print("\n✅ Обучение завершено успешно!")
    print("📁 Результаты сохранены в папке logs/")
    print("\n💡 Рекомендации:")
    print("  1. Проверьте final_report.txt для итоговых метрик")
    print("  2. Изучите feature importance в metrics.json")
    print("  3. Проанализируйте confusion matrix в plots/")
    print("  4. Сравните с предыдущими результатами")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Ошибка при выполнении: {e}")
    sys.exit(1)