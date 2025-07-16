#!/usr/bin/env python3
"""
Финальный запуск улучшенной модели XGBoost v3 с исправленными параметрами
"""

import subprocess
import sys
import os

print("="*60)
print("🚀 ФИНАЛЬНАЯ ВЕРСИЯ XGBoost v3")
print("="*60)
print("\n📊 КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ:")
print("  ✅ Порог классификации: 0.7% (баланс классов ~20/80)")
print("  ✅ Ограничена глубина: max_depth 3-6 (было 6-10)")
print("  ✅ Усилена регуляризация: gamma 1-5, min_child_weight 20-50")
print("  ✅ Метрика порога: gmean (сбалансированная)")
print("  ✅ Новые признаки: price_change_1,3,5,10, momentum, бинарные сигналы")
print("  ✅ Базовый час как технический признак")
print("  ✅ Отключен SMOTE, используется scale_pos_weight")
print("\n📈 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
print("  • ROC-AUC > 0.52 (было 0.50)")
print("  • Сбалансированные precision/recall")
print("  • Меньше false positives")
print("  • Стабильная модель без переобучения")
print("\n" + "="*60)

# Команда запуска без активации venv (предполагается, что уже активирован)
cmd = 'python xgboost_v3/main.py --test-mode --optimize --ensemble-size 5'

print("\n🔧 Запускаем обучение...")
print(f"📌 Команда: {cmd}\n")

# Запуск
try:
    subprocess.run(cmd, shell=True, check=True)
    print("\n✅ Обучение завершено успешно!")
    print("📁 Результаты сохранены в папке logs/")
    print("\n💡 Проверьте:")
    print("  • final_report.txt - итоговые метрики")
    print("  • metrics.json - детальная статистика")
    print("  • plots/ - графики обучения")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Ошибка при выполнении: {e}")
    sys.exit(1)