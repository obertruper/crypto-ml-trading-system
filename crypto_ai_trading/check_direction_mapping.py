#!/usr/bin/env python3
"""
Скрипт проверки правильности маппинга direction классов
LONG=0, SHORT=1, FLAT=2
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2
from sqlalchemy import create_engine

def check_database_mapping():
    """Проверка маппинга в базе данных"""
    print("=" * 80)
    print("ПРОВЕРКА МАППИНГА В БАЗЕ ДАННЫХ")
    print("=" * 80)

    # Подключение к БД
    engine = create_engine('postgresql://ruslan:BYYLuNwjx7GL@localhost:5555/crypto_trading')

    # Запрос небольшой выборки
    query = """
    SELECT
        direction_15m, direction_1h, direction_4h, direction_12h,
        future_return_15m, future_return_1h, future_return_4h, future_return_12h
    FROM processed_market_data
    WHERE datetime > NOW() - INTERVAL '1 day'
    LIMIT 1000
    """

    df = pd.read_sql(query, engine)

    if len(df) == 0:
        print("❌ Нет данных в БД за последние сутки")
        return

    print(f"✅ Загружено {len(df)} записей")
    print()

    # Проверка каждого таймфрейма
    for timeframe in ['15m', '1h', '4h', '12h']:
        dir_col = f'direction_{timeframe}'
        ret_col = f'future_return_{timeframe}'

        print(f"\n📊 {timeframe.upper()} ТАЙМФРЕЙМ:")
        print("-" * 40)

        # Распределение классов
        value_counts = df[dir_col].value_counts().sort_index()
        print(f"Распределение классов:")
        for cls, count in value_counts.items():
            pct = count / len(df) * 100
            if cls == 0:
                label = "LONG (рост)"
            elif cls == 1:
                label = "SHORT (падение)"
            elif cls == 2:
                label = "FLAT (боковик)"
            else:
                label = "UNKNOWN"
            print(f"  Класс {cls} ({label}): {count} ({pct:.1f}%)")

        # Проверка соответствия
        print(f"\nПроверка соответствия returns и direction:")

        # LONG должен соответствовать положительным returns
        long_mask = df[dir_col] == 0
        if long_mask.any():
            long_returns = df.loc[long_mask, ret_col]
            positive_pct = (long_returns > 0).mean() * 100
            avg_return = long_returns.mean()
            print(f"  LONG (0): {positive_pct:.1f}% положительных, средний return: {avg_return:.4f}")

        # SHORT должен соответствовать отрицательным returns
        short_mask = df[dir_col] == 1
        if short_mask.any():
            short_returns = df.loc[short_mask, ret_col]
            negative_pct = (short_returns < 0).mean() * 100
            avg_return = short_returns.mean()
            print(f"  SHORT (1): {negative_pct:.1f}% отрицательных, средний return: {avg_return:.4f}")

        # FLAT должен быть около 0
        flat_mask = df[dir_col] == 2
        if flat_mask.any():
            flat_returns = df.loc[flat_mask, ret_col]
            avg_return = abs(flat_returns.mean())
            std_return = flat_returns.std()
            print(f"  FLAT (2): средний |return|: {avg_return:.5f}, std: {std_return:.4f}")

    engine.dispose()

def check_model_predictions():
    """Проверка предсказаний модели"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ПРЕДСКАЗАНИЙ МОДЕЛИ")
    print("=" * 80)

    # Проверяем последнюю сохранённую модель
    model_path = Path('models_saved')
    best_model = model_path / 'best_model.pth'

    if not best_model.exists():
        print("❌ Модель best_model.pth не найдена")
        return

    # Загружаем чекпоинт
    checkpoint = torch.load(best_model, map_location='cpu')

    if 'direction_distribution' in checkpoint:
        print("\n📊 Распределение предсказаний direction из чекпоинта:")
        dist = checkpoint['direction_distribution']
        for tf, values in dist.items():
            print(f"\n{tf}:")
            for cls_name, pct in values.items():
                print(f"  {cls_name}: {pct:.1f}%")

    # Создаём тестовый вход
    print("\n🧪 Тестовый прогон модели:")
    print("-" * 40)

    # Симулируем логиты для проверки softmax
    test_logits = torch.tensor([
        [[2.0, -1.0, 0.0],   # Сильный LONG
         [-1.0, 2.0, 0.0],   # Сильный SHORT
         [0.0, 0.0, 2.0],    # Сильный FLAT
         [0.5, 0.5, 0.5]]    # Равномерное распределение
    ])

    probs = torch.softmax(test_logits[0], dim=-1)
    predictions = torch.argmax(probs, dim=-1)

    labels = ['LONG', 'SHORT', 'FLAT']
    scenarios = ['Сильный LONG', 'Сильный SHORT', 'Сильный FLAT', 'Равномерное']

    for i, scenario in enumerate(scenarios):
        pred_class = predictions[i].item()
        print(f"\n{scenario}:")
        print(f"  Логиты: {test_logits[0, i].tolist()}")
        print(f"  Вероятности: LONG={probs[i, 0]:.3f}, SHORT={probs[i, 1]:.3f}, FLAT={probs[i, 2]:.3f}")
        print(f"  Предсказание: {pred_class} ({labels[pred_class]})")

def check_config():
    """Проверка конфигурации"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА КОНФИГУРАЦИИ")
    print("=" * 80)

    import yaml
    config_path = 'config/config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n📝 Настройки из config.yaml:")

    # Class weights
    if 'loss' in config and 'class_weights' in config['loss']:
        weights = config['loss']['class_weights']
        print(f"\nClass weights:")
        print(f"  LONG (0): {weights[0]}")
        print(f"  SHORT (1): {weights[1]}")
        print(f"  FLAT (2): {weights[2]}")

        if weights != [1.0, 1.0, 1.0]:
            print("  ⚠️ Веса не нейтральные!")

    # Direction bias
    if 'model' in config and 'direction_bias' in config['model']:
        bias = config['model']['direction_bias']
        print(f"\nDirection bias:")
        print(f"  LONG (0): {bias[0]}")
        print(f"  SHORT (1): {bias[1]}")
        print(f"  FLAT (2): {bias[2]}")

        if bias != [0.0, 0.0, 0.0]:
            print("  ⚠️ Bias не нейтральный!")

    # Direction thresholds
    if 'targets' in config and 'direction_thresholds' in config['targets']:
        thresholds = config['targets']['direction_thresholds']
        print(f"\nПороги для определения direction:")
        for tf, threshold in thresholds.items():
            print(f"  {tf}: ±{threshold*100:.2f}%")

if __name__ == "__main__":
    print("🔍 ПРОВЕРКА МАППИНГА DIRECTION КЛАССОВ")
    print("Правильный маппинг: LONG=0, SHORT=1, FLAT=2")
    print()

    try:
        check_config()
    except Exception as e:
        print(f"❌ Ошибка проверки конфигурации: {e}")

    try:
        check_database_mapping()
    except Exception as e:
        print(f"❌ Ошибка проверки БД: {e}")

    try:
        check_model_predictions()
    except Exception as e:
        print(f"❌ Ошибка проверки модели: {e}")

    print("\n" + "=" * 80)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 80)