#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки соответствия названий индикаторов в коде и БД
"""

import psycopg2
import pandas as pd
import json
import yaml

# Загрузка конфигурации
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def check_indicators():
    """Проверяет соответствие индикаторов между кодом и БД"""
    
    # Индикаторы из кода train_xgboost_enhanced_v2.py
    TECHNICAL_INDICATORS = [
        'ema_15', 'adx_val', 'adx_plus_di', 'adx_minus_di',
        'macd_val', 'macd_signal', 'macd_hist', 'sar',
        'ichimoku_conv', 'ichimoku_base', 'aroon_up', 'aroon_down',
        'rsi_val', 'stoch_k', 'stoch_d', 'cci_val', 'roc_val',
        'williams_r', 'awesome_osc', 'ultimate_osc',
        'atr_val', 'bb_position', 'bb_width', 'donchian_position',
        'keltner_position', 'ulcer_index', 'mass_index',
        'obv_val', 'obv_signal', 'cmf_val', 'force_index',
        'eom_val', 'vpt_val', 'nvi_val', 'vwap_val',
        'ema_50', 'ema_200', 'trix_val', 'trix_signal',
        'vortex_pos', 'vortex_neg', 'vortex_ratio',
        'price_change_1', 'price_change_4', 'price_change_16',
        'volatility_4', 'volatility_16', 'volume_ratio'
    ]
    
    # Подключение к БД
    db_config = config['database'].copy()
    if not db_config.get('password'):
        db_config.pop('password', None)
    
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    print("📊 Проверка соответствия индикаторов между кодом и БД\n")
    
    # 1. Получаем пример JSON из БД
    query = """
    SELECT technical_indicators
    FROM processed_market_data
    WHERE symbol = 'BTCUSDT' 
    AND technical_indicators IS NOT NULL
    LIMIT 1
    """
    
    cursor.execute(query)
    result = cursor.fetchone()
    
    if not result:
        print("❌ Не найдены данные в БД")
        return
    
    # Индикаторы из БД
    db_indicators = list(result[0].keys())
    
    # 2. Получаем все уникальные индикаторы из БД
    query2 = """
    SELECT DISTINCT jsonb_object_keys(technical_indicators) as indicator_name
    FROM processed_market_data
    WHERE symbol = 'BTCUSDT'
    ORDER BY indicator_name
    """
    
    cursor.execute(query2)
    all_db_indicators = [row[0] for row in cursor.fetchall()]
    
    print(f"✅ В БД найдено {len(all_db_indicators)} уникальных индикаторов")
    print(f"📋 В коде определено {len(TECHNICAL_INDICATORS)} индикаторов\n")
    
    # 3. Сравниваем списки
    # Индикаторы в коде, но не в БД
    not_in_db = set(TECHNICAL_INDICATORS) - set(all_db_indicators)
    if not_in_db:
        print("❌ Индикаторы в КОДЕ, но НЕ в БД:")
        for ind in sorted(not_in_db):
            print(f"   - {ind}")
            # Пытаемся найти похожие
            similar = [db_ind for db_ind in all_db_indicators if ind.replace('_val', '') in db_ind or db_ind in ind]
            if similar:
                print(f"     💡 Возможно в БД: {similar}")
    
    # Индикаторы в БД, но не в коде
    not_in_code = set(all_db_indicators) - set(TECHNICAL_INDICATORS)
    if not_in_code:
        print("\n✅ Индикаторы в БД, но НЕ в коде:")
        for ind in sorted(not_in_code):
            print(f"   + {ind}")
    
    # 4. Создаем маппинг для исправления
    print("\n🔧 Предлагаемые исправления:")
    mappings = {}
    
    # Автоматический маппинг
    for code_ind in not_in_db:
        # Убираем _val и ищем
        base_name = code_ind.replace('_val', '')
        if base_name in all_db_indicators:
            mappings[code_ind] = base_name
            print(f"   '{code_ind}' → '{base_name}'")
        # Специальные случаи
        elif code_ind == 'bb_position' and 'bollinger_position' in all_db_indicators:
            mappings[code_ind] = 'bollinger_position'
            print(f"   '{code_ind}' → 'bollinger_position'")
        elif code_ind == 'bb_width' and 'bollinger_width' in all_db_indicators:
            mappings[code_ind] = 'bollinger_width'
            print(f"   '{code_ind}' → 'bollinger_width'")
    
    # 5. Генерируем исправленный список
    print("\n📝 Исправленный список TECHNICAL_INDICATORS:")
    corrected_indicators = []
    for ind in TECHNICAL_INDICATORS:
        if ind in mappings:
            corrected_indicators.append(mappings[ind])
        elif ind in all_db_indicators:
            corrected_indicators.append(ind)
        else:
            print(f"   ⚠️ Не найден маппинг для: {ind}")
    
    print("\nTECHNICAL_INDICATORS = [")
    for i in range(0, len(corrected_indicators), 4):
        chunk = corrected_indicators[i:i+4]
        print("    " + ", ".join(f"'{ind}'" for ind in chunk) + ",")
    print("]")
    
    cursor.close()
    conn.close()
    
    print(f"\n✅ Итого будет использовано {len(corrected_indicators)} индикаторов из {len(all_db_indicators)} доступных")

if __name__ == "__main__":
    check_indicators()