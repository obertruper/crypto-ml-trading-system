# Руководство по использованию паттернов оценки технических индикаторов

## Обзор системы

Система паттернов оценки состоит из трех основных компонентов:

1. **`technical_indicators_patterns.py`** - Основные паттерны для 49 технических индикаторов
2. **`signal_quality_criteria.py`** - Критерии оценки качества сигналов и структурные паттерны  
3. **`indicators_usage_guide.md`** - Данное руководство по использованию

## Структура индикаторов (49 штук)

### 1. ТРЕНДОВЫЕ ИНДИКАТОРЫ (13 штук)
- **EMA 15** - Экспоненциальная скользящая средняя
- **ADX** - Индекс направленного движения + DI+, DI-
- **MACD** - Схождение-расхождение скользящих средних + Signal + Histogram
- **CCI** - Индекс товарного канала
- **Ichimoku** - Конверсионная и базовая линии
- **SAR** - Параболический SAR
- **Aroon** - Aroon Up/Down
- **DPO** - Детрендованный ценовой осциллятор
- **Vortex** - Индикатор вихря VI+/VI-

### 2. ОСЦИЛЛЯТОРЫ (6 штук)
- **RSI** - Индекс относительной силы
- **Stochastic** - Стохастический осциллятор %K/%D
- **Williams %R** - Процент Уильямса
- **ROC** - Скорость изменения
- **Ultimate Oscillator** - Окончательный осциллятор

### 3. ИНДИКАТОРЫ ВОЛАТИЛЬНОСТИ (6 штук)
- **ATR** - Средний истинный диапазон
- **Bollinger Bands** - Полосы Боллинджера (Upper/Lower/Basis)
- **Donchian Channel** - Канал Дончиана (Upper/Lower)

### 4. ОБЪЕМНЫЕ ИНДИКАТОРЫ (5 штук)
- **OBV** - Объемный баланс
- **CMF** - Денежный поток Чайкина
- **MFI** - Индекс денежного потока
- **Volume SMA** - Скользящая средняя объема
- **Volume Ratio** - Отношение текущего объема к среднему

### 5. ПРОИЗВОДНЫЕ ИНДИКАТОРЫ (8 штук)
- **MACD Signal Ratio** - Отношение MACD к сигнальной линии
- **ADX Difference** - Разность +DI и -DI
- **BB Position** - Позиция цены в полосах Боллинджера
- **RSI Distance** - Отклонение RSI от центра (50)
- **Stoch Difference** - Разность %K и %D
- **Vortex Ratio** - Отношение VI+ к VI-
- **Ichimoku Difference** - Разность конверсионной и базовой линий
- **ATR Normalized** - Нормализованный ATR

### 6. ВРЕМЕННЫЕ ПРИЗНАКИ (3 штуки)
- **Hour** - Час торговли
- **Day of Week** - День недели
- **Is Weekend** - Флаг выходных

### 7. ЦЕНОВЫЕ ПАТТЕРНЫ (8 штук)
- **Price Change 1/4/16** - Изменение цены на 1, 4, 16 баров
- **Volatility 4/16** - Волатильность на 4, 16 баров

## Использование системы

### Базовое использование

```python
from technical_indicators_patterns import TechnicalIndicatorPatterns
from signal_quality_criteria import SignalQualityEvaluator

# Инициализация
patterns = TechnicalIndicatorPatterns()
evaluator = SignalQualityEvaluator()

# Данные индикаторов (из prepare_dataset.py)
indicators = {
    'ema_15': 100.5,
    'adx_val': 28,
    'rsi_val': 35,
    'macd_val': 0.0015,
    'volume_ratio': 1.8,
    # ... остальные индикаторы
}

# Получение комбинированного сигнала
signal_result = patterns.get_combined_signal(indicators, current_price=100.0)

# Оценка качества сигнала
quality_result = evaluator.evaluate_signal_quality(signal_result)

print(f"Действие: {signal_result['action']}")
print(f"Качество: {quality_result['quality_level'].name}")
print(f"Рекомендация: {quality_result['recommendation']}")
```

### Детальная оценка по категориям

```python
# Сигналы по категориям
category_signals = patterns.get_category_signals(indicators, current_price)

for category, data in category_signals.items():
    if data['signals']:
        print(f"\n{category.upper()}:")
        print(f"Консенсус: {data['consensus'].name}")
        print(f"Качество: {data['avg_quality']:.2f}")
        
        for signal in data['signals']:
            evaluation = patterns.evaluate_indicator(
                signal['indicator'], 
                signal['value'], 
                current_price
            )
            print(f"  {signal['indicator']}: {evaluation['recommendation']}")
```

### Оценка отдельного индикатора

```python
# Оценка RSI
rsi_evaluation = patterns.evaluate_indicator('rsi_val', 35, current_price)

print(f"RSI сигнал: {rsi_evaluation['signal_strength'].name}")
print(f"Качество: {rsi_evaluation['quality']:.2f}")
print(f"Рекомендация: {rsi_evaluation['recommendation']}")
print(f"В оптимальном диапазоне: {rsi_evaluation['in_optimal_range']}")
```

## Критерии оценки качества

### Уровни качества сигналов

1. **EXCELLENT (5)** - Более 90% уверенности
   - Все фильтры пройдены
   - Высокая согласованность индикаторов
   - Подтверждение объемом
   - Оптимальные рыночные условия

2. **GOOD (4)** - 70-90% уверенности
   - Большинство фильтров пройдено
   - Хорошая согласованность
   - Достаточное подтверждение

3. **MODERATE (3)** - 50-70% уверенности
   - Базовые критерии выполнены
   - Средняя согласованность
   - Требует дополнительного подтверждения

4. **WEAK (2)** - 30-50% уверенности
   - Слабые сигналы
   - Низкая согласованность
   - Не рекомендуется для торговли

5. **POOR (1)** - Менее 30% уверенности
   - Противоречивые сигналы
   - Высокий риск
   - Пропустить сигнал

### Фильтры качества

```python
# Проверка фильтров
filters = quality_result['filters_passed']

required_filters = [
    'min_agreement',      # Минимальная согласованность индикаторов (60%)
    'volume_confirm',     # Подтверждение объемом (объем > 1.2x среднего)
    'market_suitable',    # Подходящие рыночные условия
    'time_suitable',      # Подходящее время торговли
    'risk_reward_ok'      # Приемлемое соотношение риск/прибыль
]

if filters['all_passed']:
    print("✅ Все фильтры пройдены - сигнал готов к исполнению")
else:
    failed = [f for f in required_filters if not filters[f]]
    print(f"❌ Не пройдены фильтры: {', '.join(failed)}")
```

## Паттерны индикаторов по группам

### Трендовые индикаторы - Оптимальные комбинации

```python
# 1. Сильный восходящий тренд
strong_uptrend_conditions = {
    'adx_val': '>25',           # Сильный тренд
    'adx_plus_di': '>adx_minus_di + 5',  # +DI доминирует
    'ema_15': 'price > ema * 1.002',     # Цена выше EMA
    'macd_val': '>0 and rising',         # MACD выше нуля и растет
    'sar': 'below_price',               # SAR под ценой
    'ichimoku_conv': '>ichimoku_base'   # Конверсия выше базы
}

# 2. Разворот тренда
trend_reversal_conditions = {
    'adx_val': '<20',           # Слабый тренд
    'macd_hist': 'divergence',  # Дивергенция MACD
    'rsi_val': '>70 or <30',    # Экстремальные значения RSI
    'cci_val': '>100 or <-100', # Экстремальные значения CCI
    'volume_ratio': '>1.5'      # Повышенный объем
}
```

### Осцилляторы - Сигналы разворота

```python
# Перепроданность (сигнал на покупку)
oversold_signals = {
    'rsi_val': '<30',           # RSI в зоне перепроданности
    'stoch_k': '<20',           # Stochastic перепродан
    'williams_r': '<-80',       # Williams %R перепродан
    'mfi': '<20',               # MFI показывает отток капитала
    'volume_ratio': '>1.2'      # Подтверждение объемом
}

# Перекупленность (сигнал на продажу)
overbought_signals = {
    'rsi_val': '>70',           # RSI в зоне перекупленности
    'stoch_k': '>80',           # Stochastic перекуплен
    'williams_r': '>-20',       # Williams %R перекуплен
    'mfi': '>80',               # MFI показывает избыток покупок
    'volume_ratio': '>1.2'      # Подтверждение объемом
}
```

### Волатильность - Оптимальные условия

```python
# Идеальная волатильность для торговли
optimal_volatility = {
    'atr_norm': '0.005 < value < 0.02',    # 0.5% - 2% от цены
    'bb_position': '0.2 < value < 0.8',    # Не на краях полос
    'volatility_16': 'stable_trend',        # Стабильная долгосрочная волатильность
    'donchian_upper': 'not_at_extreme'      # Не на экстремальных уровнях
}

# Избегать торговли при экстремальной волатильности
avoid_conditions = {
    'atr_norm': '>0.03',        # Слишком высокая волатильность
    'bb_position': '>0.95 or <0.05',  # Цена на краях полос
    'volatility_4': 'spike',    # Внезапные всплески
}
```

### Объемные индикаторы - Подтверждение движений

```python
# Сильное подтверждение объемом
volume_confirmation = {
    'volume_ratio': '>1.5',     # Объем в 1.5+ раза выше среднего
    'obv': 'confirming_price',  # OBV подтверждает ценовое движение
    'cmf': '>0.1 for buy, <-0.1 for sell',  # Положительный/отрицательный CMF
    'mfi': 'confirming_direction'  # MFI согласуется с направлением
}

# Дивергенция объема (сигнал ослабления)
volume_divergence = {
    'volume_ratio': '<0.8',     # Падающий объем
    'obv': 'diverging_from_price',  # OBV расходится с ценой
    'cmf': 'weakening',         # CMF ослабевает
}
```

## Структурные паттерны

### Определение уровней поддержки/сопротивления

```python
from signal_quality_criteria import StructuralPatternDetector

# Получение уровней
sr_levels = StructuralPatternDetector.detect_support_resistance(
    highs_list, lows_list, closes_list, window=20
)

# Использование в торговле
current_price = closes_list[-1]

for support in sr_levels['support']:
    distance_to_support = abs(current_price - support['level']) / current_price
    
    if distance_to_support < 0.01:  # Менее 1% от уровня
        print(f"🔵 Цена у поддержки {support['level']:.2f}")
        print(f"   Сила: {support['strength']}, Касаний: {support['touches']}")
        
        if support['strength'] > 5:
            print("   ⚡ Сильный уровень - возможен отскок")

for resistance in sr_levels['resistance']:
    distance_to_resistance = abs(current_price - resistance['level']) / current_price
    
    if distance_to_resistance < 0.01:  # Менее 1% от уровня
        print(f"🔴 Цена у сопротивления {resistance['level']:.2f}")
        print(f"   Сила: {resistance['strength']}, Касаний: {resistance['touches']}")
        
        if resistance['strength'] > 5:
            print("   ⚡ Сильный уровень - возможен отскок")
```

### Графические паттерны

```python
# Поиск паттернов
patterns = StructuralPatternDetector.detect_chart_patterns(ohlc_df)

for pattern in patterns:
    print(f"\n📊 Найден паттерн: {pattern['type']}")
    print(f"   Уверенность: {pattern['confidence']:.1%}")
    
    if pattern['type'] == 'double_top':
        print(f"   Сопротивление: {pattern['resistance']:.2f}")
        print(f"   Цель: {pattern['target']:.2f}")
        print("   📉 Медвежий сигнал при пробое поддержки")
        
    elif pattern['type'] == 'double_bottom':
        print(f"   Поддержка: {pattern['support']:.2f}")
        print(f"   Цель: {pattern['target']:.2f}")
        print("   📈 Бычий сигнал при пробое сопротивления")
        
    elif pattern['type'] in ['bull_flag', 'bear_flag']:
        print(f"   Размер импульса: {pattern['impulse_size']:.1%}")
        print(f"   Размер коррекции: {pattern['correction_size']:.1%}")
        print(f"   Цель: {pattern['target']:.2f}")
```

## Межрыночные корреляции

```python
from signal_quality_criteria import IntermarketCorrelationAnalyzer

# Расчет корреляций
correlations = IntermarketCorrelationAnalyzer.calculate_correlations(
    symbol_data_dict, window=100
)

# Поиск опережающих индикаторов
leading_indicators = IntermarketCorrelationAnalyzer.find_leading_indicators(
    'BTCUSDT', correlations, threshold=0.7
)

for indicator in leading_indicators:
    print(f"📊 {indicator['symbol']}: корреляция {indicator['correlation']:.2f}")
    print(f"   Тип: {indicator['type']}, Сила: {indicator['strength']}")

# Поиск дивергенций
divergences = IntermarketCorrelationAnalyzer.detect_divergences(
    btc_data, eth_data, window=20
)

for div in divergences[-5:]:  # Последние 5 дивергенций
    print(f"🔄 {div['type']} дивергенция")
    print(f"   Время: {div['timestamp']}")
    print(f"   Изменения: {div['symbol1_change']:.1%} vs {div['symbol2_change']:.1%}")
```

## Практические рекомендации

### 1. Приоритет индикаторов по надежности

**Высокая надежность (вес 0.8-1.0):**
- ADX для определения силы тренда
- Volume Ratio для подтверждения
- MACD для направления тренда
- RSI для точек входа
- Parabolic SAR для трейлинг-стопов

**Средняя надежность (вес 0.6-0.7):**
- Stochastic для краткосрочных сигналов
- Bollinger Bands для волатильности
- Williams %R для перекупленности/перепроданности
- Ichimoku для общего направления

**Вспомогательные (вес 0.3-0.5):**
- Временные факторы
- Производные индикаторы
- Межрыночные корреляции

### 2. Стратегии по рыночным условиям

**Сильный тренд (ADX > 25):**
- Использовать трендовые индикаторы
- Избегать осцилляторы
- Фокус на MACD, EMA, SAR
- Высокий вес объемных индикаторов

**Боковое движение (ADX < 20):**
- Использовать осцилляторы
- RSI, Stochastic, Williams %R
- Bollinger Bands для границ диапазона
- Низкий вес трендовых индикаторов

**Высокая волатильность (ATR > 2%):**
- Увеличить стоп-лоссы
- Снизить размер позиций
- Избегать торговли при экстремальных значениях
- Фокус на структурных уровнях

### 3. Комбинации для максимальной точности

**Идеальный сигнал на покупку:**
```python
perfect_buy_signal = {
    'adx_val': '>25',                    # Сильный тренд
    'adx_plus_di': '>adx_minus_di + 5',  # Бычий тренд
    'rsi_val': '30-50',                  # Коррекция, но не перепроданность
    'macd_hist': '>0 and rising',        # Бычий момент
    'volume_ratio': '>1.5',              # Подтверждение объемом
    'price_above_ema': True,             # Цена выше EMA
    'bb_position': '0.2-0.6',            # Не у границ полос
    'time_favorable': True               # Активные часы торговли
}
```

**Идеальный сигнал на продажу:**
```python
perfect_sell_signal = {
    'adx_val': '>25',                    # Сильный тренд
    'adx_minus_di': '>adx_plus_di + 5',  # Медвежий тренд
    'rsi_val': '50-70',                  # Коррекция, но не перекупленность
    'macd_hist': '<0 and falling',       # Медвежий момент
    'volume_ratio': '>1.5',              # Подтверждение объемом
    'price_below_ema': True,             # Цена ниже EMA
    'bb_position': '0.4-0.8',            # Не у границ полос
    'time_favorable': True               # Активные часы торговли
}
```

### 4. Управление рисками на основе индикаторов

```python
def calculate_position_size(signal_quality, account_balance, base_risk=0.02):
    """
    Рассчитывает размер позиции на основе качества сигнала
    """
    risk_multipliers = {
        SignalQuality.EXCELLENT: 1.0,    # 2% риска
        SignalQuality.GOOD: 0.75,        # 1.5% риска
        SignalQuality.MODERATE: 0.5,     # 1% риска
        SignalQuality.WEAK: 0.25,        # 0.5% риска
        SignalQuality.POOR: 0.0          # Не торговать
    }
    
    risk_percent = base_risk * risk_multipliers.get(signal_quality, 0)
    return account_balance * risk_percent

def adjust_stop_loss(atr_value, base_sl=0.011):
    """
    Корректирует стоп-лосс на основе волатильности
    """
    if atr_value > 0.02:      # Высокая волатильность
        return base_sl * 1.5
    elif atr_value < 0.005:   # Низкая волатильность
        return base_sl * 0.7
    else:
        return base_sl
```

Эта система паттернов предоставляет комплексный подход к анализу всех 49 технических индикаторов с четкими критериями качества и практическими рекомендациями для интеграции в ML-модель трейдинга.