#!/usr/bin/env python3
"""
Анализ обработанных данных и запуск обучения модели Transformer
"""

import psycopg2
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from train_advanced import run_training_advanced

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_risk_profile():
    """Анализ эффективности текущего риск-профиля"""
    
    # Подключение к БД
    conn = psycopg2.connect(
        host='localhost',
        port=5555,
        database='crypto_trading',
        user='ruslan',
        password='ruslan'
    )
    cursor = conn.cursor()
    
    logger.info("📊 Анализ риск-профиля для 1000PEPEUSDT")
    
    # Получаем статистику по меткам
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(buy_profit_target) as buy_profits,
            SUM(buy_loss_target) as buy_losses,
            SUM(sell_profit_target) as sell_profits,
            SUM(sell_loss_target) as sell_losses
        FROM processed_market_data
        WHERE symbol = '1000PEPEUSDT'
    """)
    
    total, buy_p, buy_l, sell_p, sell_l = cursor.fetchone()
    
    # Считаем win rate
    buy_wr = (buy_p / (buy_p + buy_l) * 100) if (buy_p + buy_l) > 0 else 0
    sell_wr = (sell_p / (sell_p + sell_l) * 100) if (sell_p + sell_l) > 0 else 0
    
    # Ожидаемая доходность
    # Risk/Reward = 1:5.3 (1.1% loss vs 5.8% profit)
    buy_expected = (buy_wr/100 * 5.8) - ((100-buy_wr)/100 * 1.1)
    sell_expected = (sell_wr/100 * 5.8) - ((100-sell_wr)/100 * 1.1)
    
    logger.info(f"\n📈 Статистика меток:")
    logger.info(f"Всего записей: {total:,}")
    logger.info(f"\n🟢 BUY сигналы:")
    logger.info(f"  Прибыльных: {buy_p} ({buy_wr:.2f}%)")
    logger.info(f"  Убыточных: {buy_l} ({100-buy_wr:.2f}%)")
    logger.info(f"  Ожидаемая доходность: {buy_expected:.2f}%")
    
    logger.info(f"\n🔴 SELL сигналы:")
    logger.info(f"  Прибыльных: {sell_p} ({sell_wr:.2f}%)")
    logger.info(f"  Убыточных: {sell_l} ({100-sell_wr:.2f}%)")
    logger.info(f"  Ожидаемая доходность: {sell_expected:.2f}%")
    
    # Анализ распределения сигналов по времени
    cursor.execute("""
        SELECT 
            DATE_TRUNC('day', datetime) as day,
            SUM(buy_profit_target + buy_loss_target) as buy_signals,
            SUM(sell_profit_target + sell_loss_target) as sell_signals
        FROM processed_market_data
        WHERE symbol = '1000PEPEUSDT'
        GROUP BY day
        ORDER BY day DESC
        LIMIT 30
    """)
    
    recent_days = cursor.fetchall()
    avg_buy_signals = np.mean([d[1] for d in recent_days])
    avg_sell_signals = np.mean([d[2] for d in recent_days])
    
    logger.info(f"\n📅 Среднее количество сигналов в день (последние 30 дней):")
    logger.info(f"  BUY: {avg_buy_signals:.1f}")
    logger.info(f"  SELL: {avg_sell_signals:.1f}")
    
    cursor.close()
    conn.close()
    
    return {
        'buy_win_rate': buy_wr,
        'sell_win_rate': sell_wr,
        'buy_expected': buy_expected,
        'sell_expected': sell_expected,
        'total_records': total
    }

def optimize_model_config(risk_stats):
    """Оптимизация конфигурации модели на основе анализа"""
    
    # Базовая конфигурация
    config = {
        'epochs': 50,  # Начнем с 50 эпох
        'batch_size': 64,
        'learning_rate': 0.001,
        'hidden_size': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout_rate': 0.2,
        'early_stopping_patience': 10
    }
    
    # Корректировка на основе win rate
    avg_wr = (risk_stats['buy_win_rate'] + risk_stats['sell_win_rate']) / 2
    
    if avg_wr < 5:  # Очень низкий win rate
        logger.info("⚠️ Низкий win rate - увеличиваем сложность модели")
        config['hidden_size'] = 512
        config['num_layers'] = 6
        config['epochs'] = 100
        config['dropout_rate'] = 0.3
    elif avg_wr < 10:
        logger.info("📊 Средний win rate - стандартная конфигурация")
        config['hidden_size'] = 384
        config['num_layers'] = 5
        config['epochs'] = 75
    else:
        logger.info("✅ Хороший win rate - оптимальная конфигурация")
        config['hidden_size'] = 256
        config['num_layers'] = 4
        config['epochs'] = 50
    
    return config

def prepare_training_config(model_config):
    """Подготовка полной конфигурации для обучения"""
    
    # Загружаем базовую конфигурацию
    with open('config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Обновляем параметры модели
    training_config = {
        'database': base_config['database'],
        'model': {
            'architecture': 'transformer',
            'epochs': model_config['epochs'],
            'batch_size': model_config['batch_size'],
            'learning_rate': model_config['learning_rate'],
            'sequence_length': 60,
            'prediction_horizon': 100,
            'hidden_size': model_config['hidden_size'],
            'num_heads': model_config['num_heads'],
            'num_layers': model_config['num_layers'],
            'dropout_rate': model_config['dropout_rate'],
            'early_stopping_patience': model_config['early_stopping_patience']
        },
        'training': {
            'symbol': '1000PEPEUSDT',
            'validation_split': 0.2,
            'test_split': 0.1,
            'shuffle': False,  # Важно для временных рядов
            'use_class_weights': True,
            'save_best_only': True
        },
        'paths': base_config['paths']
    }
    
    return training_config

def main():
    """Основная функция"""
    
    logger.info("🚀 Запуск анализа и обучения модели")
    
    # 1. Анализ риск-профиля
    risk_stats = analyze_risk_profile()
    
    # 2. Оптимизация параметров модели
    model_config = optimize_model_config(risk_stats)
    
    logger.info(f"\n🔧 Оптимизированная конфигурация модели:")
    for key, value in model_config.items():
        logger.info(f"  {key}: {value}")
    
    # 3. Подготовка конфигурации
    training_config = prepare_training_config(model_config)
    
    # 4. Запуск обучения
    logger.info(f"\n🎯 Запуск обучения Transformer модели...")
    logger.info(f"  Символ: 1000PEPEUSDT")
    logger.info(f"  Записей: {risk_stats['total_records']:,}")
    logger.info(f"  Эпох: {model_config['epochs']}")
    logger.info(f"  Архитектура: {model_config['num_layers']} слоев, {model_config['hidden_size']} размер")
    
    # Сохраняем конфигурацию
    with open('training_config_optimized.yaml', 'w') as f:
        yaml.dump(training_config, f, default_flow_style=False)
    
    # Запускаем обучение
    results = run_training_advanced(training_config)
    
    if results['success']:
        logger.info(f"\n✅ Обучение завершено успешно!")
        logger.info(f"📊 Результаты:")
        logger.info(f"  Лучшая точность: {results['best_accuracy']:.2%}")
        logger.info(f"  Финальная loss: {results['final_loss']:.4f}")
        logger.info(f"  Модели сохранены в: {results['model_paths']}")
    else:
        logger.error(f"❌ Ошибка обучения: {results.get('error')}")

if __name__ == "__main__":
    main()