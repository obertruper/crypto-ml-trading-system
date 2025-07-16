#!/usr/bin/env python3
"""
Скрипт для анализа важности признаков обученных моделей XGBoost v3
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import glob

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.feature_importance_validator import FeatureImportanceValidator
from config.feature_mapping import get_feature_category, get_temporal_blacklist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_models(base_dir="logs"):
    """Найти последние обученные модели"""
    model_files = glob.glob(f"{base_dir}/**/classification_binary_model_*.pkl", recursive=True)
    
    if not model_files:
        logger.error("❌ Не найдено обученных моделей в папке logs/")
        return None, None
    
    # Группируем по директориям
    model_dirs = {}
    for model_file in model_files:
        dir_name = os.path.dirname(model_file)
        if dir_name not in model_dirs:
            model_dirs[dir_name] = []
        model_dirs[dir_name].append(model_file)
    
    # Находим последнюю директорию по времени модификации
    latest_dir = max(model_dirs.keys(), key=lambda d: os.path.getmtime(d))
    
    logger.info(f"📁 Найдена последняя модель: {os.path.basename(os.path.dirname(latest_dir))}")
    
    return latest_dir, model_dirs[latest_dir]


def load_models_and_metadata(model_dir):
    """Загрузить модели и метаданные"""
    models = {'buy': [], 'sell': []}
    metadata = {'buy': [], 'sell': []}
    
    # Загружаем модели для buy
    buy_dir = os.path.join(model_dir, "buy_models")
    if os.path.exists(buy_dir):
        for i in range(10):  # Проверяем до 10 моделей
            model_path = os.path.join(buy_dir, f"classification_binary_model_{i}.pkl")
            meta_path = os.path.join(buy_dir, f"classification_binary_model_{i}_metadata.json")
            
            if os.path.exists(model_path) and os.path.exists(meta_path):
                with open(model_path, 'rb') as f:
                    models['buy'].append(pickle.load(f))
                with open(meta_path, 'r') as f:
                    metadata['buy'].append(json.load(f))
    
    # Загружаем модели для sell
    sell_dir = os.path.join(model_dir, "sell_models")
    if os.path.exists(sell_dir):
        for i in range(10):  # Проверяем до 10 моделей
            model_path = os.path.join(sell_dir, f"classification_binary_model_{i}.pkl")
            meta_path = os.path.join(sell_dir, f"classification_binary_model_{i}_metadata.json")
            
            if os.path.exists(model_path) and os.path.exists(meta_path):
                with open(model_path, 'rb') as f:
                    models['sell'].append(pickle.load(f))
                with open(meta_path, 'r') as f:
                    metadata['sell'].append(json.load(f))
    
    logger.info(f"✅ Загружено моделей: Buy={len(models['buy'])}, Sell={len(models['sell'])}")
    
    return models, metadata


def analyze_feature_importance(models, metadata):
    """Анализ важности признаков"""
    print("\n" + "="*80)
    print("📊 АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("="*80)
    
    temporal_blacklist = get_temporal_blacklist()
    
    for direction in ['buy', 'sell']:
        if not models[direction]:
            continue
            
        print(f"\n🎯 Направление: {direction.upper()}")
        print("-"*40)
        
        # Собираем важности со всех моделей
        all_importances = {}
        feature_names = metadata[direction][0]['feature_names'] if metadata[direction] else []
        
        for i, (model, meta) in enumerate(zip(models[direction], metadata[direction])):
            if hasattr(model, 'feature_importances_'):
                for feat, imp in zip(feature_names, model.feature_importances_):
                    if feat not in all_importances:
                        all_importances[feat] = []
                    all_importances[feat].append(imp)
        
        # Усредняем важности
        avg_importances = [(feat, np.mean(imps)) for feat, imps in all_importances.items()]
        avg_importances.sort(key=lambda x: x[1], reverse=True)
        
        # Топ-20 признаков
        print("\n📈 Топ-20 самых важных признаков:")
        for i, (feat, imp) in enumerate(avg_importances[:20]):
            category = get_feature_category(feat)
            emoji = "🔴" if category == "temporal" else "🟢" if category == "technical" else "🔵"
            warning = " ⚠️ BLACKLISTED!" if feat in temporal_blacklist else ""
            print(f"{i+1:2d}. {emoji} {feat:40s} {imp:.4f} ({category}){warning}")
        
        # Анализ по категориям
        category_stats = {}
        total_importance = sum(imp for _, imp in avg_importances)
        
        for feat, imp in avg_importances:
            cat = get_feature_category(feat)
            if cat not in category_stats:
                category_stats[cat] = {'count': 0, 'importance': 0}
            category_stats[cat]['count'] += 1
            category_stats[cat]['importance'] += imp
        
        print(f"\n📊 Распределение важности по категориям:")
        for cat, stats in sorted(category_stats.items()):
            percentage = (stats['importance'] / total_importance * 100) if total_importance > 0 else 0
            status = "✅" if cat != "temporal" or percentage <= 3 else "❌ ПРОБЛЕМА!"
            print(f"   {cat:15s}: {stats['count']:3d} признаков, {percentage:5.1f}% важности {status}")
        
        # Проверка на проблемные temporal в топе
        top_10_temporal = [f for f, _ in avg_importances[:10] if get_feature_category(f) == "temporal"]
        if top_10_temporal:
            print(f"\n⚠️ ВНИМАНИЕ: Temporal признаки в топ-10: {', '.join(top_10_temporal)}")
            
        # Blacklisted признаки в топе
        blacklisted_in_top = [f for f, _ in avg_importances[:20] if f in temporal_blacklist]
        if blacklisted_in_top:
            print(f"\n❌ КРИТИЧНО: Blacklisted признаки в топ-20: {', '.join(blacklisted_in_top)}")


def run_validation(models, metadata):
    """Запуск валидации через FeatureImportanceValidator"""
    print("\n" + "="*80)
    print("🔍 ВАЛИДАЦИЯ ВАЖНОСТИ ПРИЗНАКОВ")
    print("="*80)
    
    validator = FeatureImportanceValidator(max_temporal_importance=3.0)
    
    # Преобразуем модели в формат для валидатора
    models_dict = {}
    for direction in ['buy', 'sell']:
        if models[direction]:
            # Создаем mock ансамбль
            class MockEnsemble:
                def __init__(self, models_list):
                    self.models = models_list
            
            models_dict[direction] = {
                'ensemble': MockEnsemble(models[direction])
            }
    
    # Получаем имена признаков
    feature_names = []
    if metadata['buy']:
        feature_names = metadata['buy'][0]['feature_names']
    elif metadata['sell']:
        feature_names = metadata['sell'][0]['feature_names']
    
    if models_dict and feature_names:
        validation_results = validator.validate_ensemble_importance(models_dict, feature_names)
        
        # Выводим рекомендации
        recommendations = validator.get_recommendations()
        if recommendations:
            print("\n💡 РЕКОМЕНДАЦИИ:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
    else:
        print("❌ Недостаточно данных для валидации")


def analyze_training_config(model_dir):
    """Анализ конфигурации обучения"""
    config_path = os.path.join(model_dir, "config.yaml")
    
    if os.path.exists(config_path):
        print("\n" + "="*80)
        print("⚙️ КОНФИГУРАЦИЯ ОБУЧЕНИЯ")
        print("="*80)
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Выводим ключевые параметры
        training = config.get('training', {})
        model = config.get('model', {})
        
        print(f"📋 Основные параметры:")
        print(f"   Задача: {training.get('task_type', 'н/д')}")
        print(f"   Порог классификации: {training.get('classification_threshold', 'н/д')}%")
        print(f"   Метод отбора признаков: {training.get('feature_selection_method', 'н/д')}")
        print(f"   Количество признаков: {training.get('feature_selection_top_k', 'н/д')}")
        print(f"   Размер ансамбля: {training.get('ensemble_size', 'н/д')}")
        print(f"   Балансировка: {training.get('balance_method', 'н/д')}")
        
        print(f"\n🤖 Параметры модели:")
        print(f"   max_depth: {model.get('max_depth', 'н/д')}")
        print(f"   learning_rate: {model.get('learning_rate', 'н/д')}")
        print(f"   n_estimators: {model.get('n_estimators', 'н/д')}")
        print(f"   GPU: {'Да' if model.get('tree_method') == 'gpu_hist' else 'Нет'}")


def main():
    """Основная функция"""
    print("""
🔍 Анализ важности признаков XGBoost v3.0
=========================================
""")
    
    # Ищем последние модели
    model_dir, model_files = find_latest_models()
    
    if not model_dir:
        return
    
    # Загружаем модели и метаданные
    models, metadata = load_models_and_metadata(model_dir)
    
    if not any(models.values()):
        logger.error("❌ Не удалось загрузить модели")
        return
    
    # Анализируем конфигурацию
    analyze_training_config(model_dir)
    
    # Анализируем важность признаков
    analyze_feature_importance(models, metadata)
    
    # Запускаем валидацию
    run_validation(models, metadata)
    
    # Финальная сводка
    print("\n" + "="*80)
    print("📝 ИТОГОВАЯ СВОДКА")
    print("="*80)
    
    # Проверяем наличие final_report.txt
    report_path = os.path.join(model_dir, "final_report.txt")
    if os.path.exists(report_path):
        print("\n📄 Метрики из финального отчета:")
        with open(report_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if any(metric in line for metric in ['ROC-AUC:', 'Accuracy:', 'Precision:', 'Recall:']):
                    print(f"   {line.strip()}")
    
    print("\n✅ Анализ завершен!")
    print("\n💡 Следующие шаги:")
    print("   1. Если temporal > 3% - запустите обучение с новыми исправлениями")
    print("   2. Используйте python run_xgboost_v3.py и выберите вариант 1 или 2")
    print("   3. После обучения снова запустите этот скрипт для проверки")


if __name__ == "__main__":
    main()