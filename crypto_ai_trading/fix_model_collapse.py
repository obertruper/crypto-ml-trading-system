"""
Скрипт для исправления коллапса модели в FLAT класс
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ModelFixer')

def fix_direction_heads(model_path: str = None):
    """Исправляет веса direction head для предотвращения коллапса"""

    logger.info("🔧 Запуск исправления коллапса модели")

    # Загрузка модели
    if model_path and Path(model_path).exists():
        logger.info(f"📥 Загрузка модели из {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        logger.warning("⚠️ Модель не найдена, создаю новую инициализацию")
        state_dict = None

    # Анализ и исправление direction heads
    fixed_weights = {}

    if state_dict:
        for key, value in state_dict.items():
            if 'direction' in key.lower() and 'head' in key.lower():
                logger.info(f"🔍 Найден direction head: {key}")
                logger.info(f"   Размер: {value.shape}")
                logger.info(f"   Статистика: min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}")

                # Принудительная реинициализация для выхода из коллапса
                if 'weight' in key:
                    # Используем Xavier инициализацию с большей дисперсией
                    fan_in, fan_out = value.shape[-2], value.shape[-1]
                    std = np.sqrt(6.0 / (fan_in + fan_out)) * 2.0  # Увеличиваем дисперсию
                    new_weight = torch.randn_like(value) * std

                    # Добавляем случайный шум для разнообразия
                    noise = torch.randn_like(value) * 0.1
                    new_weight = new_weight + noise

                    fixed_weights[key] = new_weight
                    logger.info(f"   ✅ Веса переинициализированы с std={std:.4f}")

                elif 'bias' in key:
                    # Устанавливаем bias для балансировки классов
                    # LONG и SHORT получают положительное смещение
                    # FLAT получает отрицательное
                    if value.shape[0] == 3:  # Проверка что это 3 класса
                        new_bias = torch.tensor([0.5, 0.5, -1.0])  # Смещение в сторону LONG/SHORT
                        fixed_weights[key] = new_bias
                        logger.info(f"   ✅ Bias установлен: {new_bias.tolist()}")
                    else:
                        fixed_weights[key] = value
                else:
                    fixed_weights[key] = value
            else:
                fixed_weights[key] = value

    # Сохранение исправленной модели
    save_path = f"models_saved/fixed_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    if state_dict:
        if isinstance(checkpoint, dict):
            checkpoint['model_state_dict'] = fixed_weights
            torch.save(checkpoint, save_path)
        else:
            torch.save(fixed_weights, save_path)
        logger.info(f"💾 Исправленная модель сохранена: {save_path}")

    return save_path

def update_config_for_anti_collapse():
    """Обновляет конфигурацию для предотвращения коллапса"""

    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Агрессивные настройки против коллапса
    updates = {
        'loss': {
            'class_weights': [5.0, 5.0, 0.2],  # Сильное увеличение весов LONG/SHORT
            'use_weighted_sampling': True,
            'entropy_regularization': 2.0,  # Максимальная регуляризация энтропии
            'min_entropy': 1.0,
            'label_smoothing': 0.2,
            'use_focal_loss': True,
            'focal_alpha': 0.7,
            'focal_gamma': 5.0,  # Сильный фокус на сложных примерах
            'wrong_direction_penalty': 5.0,
            'auto_adjust_on_collapse': True,
            'collapse_threshold': 0.5,  # Быстрое реагирование на коллапс
            'min_flat_ratio': 0.2,  # Минимум FLAT
            'max_flat_ratio': 0.5,  # Максимум FLAT
            'force_diversity': True,  # Новый параметр
            'diversity_weight': 1.5,  # Вес для принудительного разнообразия
        },
        'model': {
            'learning_rate': 0.0001,  # Выше LR для быстрого выхода
            'dropout': 0.2,  # Меньше dropout для лучшего обучения
            'temperature': 5.0,  # Высокая температура для разнообразия
            'gradient_clip': 1.0,  # Больше градиенты
            'gradient_accumulation_steps': 2,  # Меньше накопление
            'batch_size': 8192,  # Больше батч для статистики
            'direction_head_init': {
                'bias_init': 'anti_collapse',  # Специальный режим
                'weight_scale': 3.0,
                'method': 'xavier_uniform',
                'add_noise': True,
                'noise_scale': 0.1,
            },
            'use_anti_collapse': True,
            'anti_collapse_strength': 2.0,  # Сила противодействия коллапсу
        }
    }

    # Обновляем конфигурацию
    for section, params in updates.items():
        if section in config:
            config[section].update(params)
        else:
            config[section] = params

    # Сохраняем обновленную конфигурацию
    backup_path = config_path.with_suffix('.yaml.bak')
    config_path.rename(backup_path)
    logger.info(f"📦 Создана резервная копия: {backup_path}")

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("✅ Конфигурация обновлена для предотвращения коллапса")

    return config

def check_data_balance():
    """Проверяет баланс классов в данных"""

    import pandas as pd
    from sqlalchemy import create_engine

    logger.info("📊 Проверка баланса классов в данных")

    try:
        # Подключение к БД
        engine = create_engine('postgresql://ruslan:ruslan@localhost:5555/crypto_trading')

        # Проверка распределения direction классов
        query = """
        SELECT
            COUNT(*) FILTER (WHERE direction_15m = 0) as long_count,
            COUNT(*) FILTER (WHERE direction_15m = 1) as short_count,
            COUNT(*) FILTER (WHERE direction_15m = 2) as flat_count,
            COUNT(*) as total
        FROM processed_market_data
        WHERE direction_15m IS NOT NULL
        LIMIT 100000
        """

        result = pd.read_sql(query, engine)

        if not result.empty:
            row = result.iloc[0]
            total = row['total']
            if total > 0:
                long_pct = (row['long_count'] / total) * 100
                short_pct = (row['short_count'] / total) * 100
                flat_pct = (row['flat_count'] / total) * 100

                logger.info(f"📈 Распределение классов в данных:")
                logger.info(f"   LONG:  {row['long_count']:,} ({long_pct:.1f}%)")
                logger.info(f"   SHORT: {row['short_count']:,} ({short_pct:.1f}%)")
                logger.info(f"   FLAT:  {row['flat_count']:,} ({flat_pct:.1f}%)")

                if flat_pct > 80:
                    logger.warning("⚠️ FLAT доминирует в данных! Требуется балансировка")
                    return False

        engine.dispose()
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка проверки данных: {e}")
        return False

def main():
    """Главная функция исправления"""

    logger.info("="*60)
    logger.info("🚀 Запуск процедуры исправления коллапса модели")
    logger.info("="*60)

    # 1. Проверка баланса данных
    data_ok = check_data_balance()
    if not data_ok:
        logger.warning("⚠️ Проблемы с балансом данных")

    # 2. Обновление конфигурации
    config = update_config_for_anti_collapse()

    # 3. Исправление весов модели (если есть)
    model_paths = [
        "models_saved/best_model.pth",
        "models_saved/checkpoint_latest.pth"
    ]

    fixed_model = None
    for path in model_paths:
        if Path(path).exists():
            fixed_model = fix_direction_heads(path)
            break

    if not fixed_model:
        logger.info("🆕 Модель будет создана заново с правильной инициализацией")

    logger.info("="*60)
    logger.info("✅ Процедура исправления завершена!")
    logger.info("🎯 Рекомендации:")
    logger.info("   1. Запустите обучение: python main.py --mode train")
    logger.info("   2. Следите за энтропией предсказаний (должна быть > 0.5)")
    logger.info("   3. Проверяйте распределение классов каждую эпоху")
    logger.info("="*60)

    return fixed_model

if __name__ == "__main__":
    main()