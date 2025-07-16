#!/usr/bin/env python3
"""
Изолированный скрипт для тестирования модели PatchTST
НЕ влияет на процесс обучения - полностью независим!
"""

import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Импорты проекта
from models.patchtst_unified import UnifiedPatchTSTForTrading as UnifiedPatchTST
from trading.model_adapter import ModelOutputAdapter
from trading.backtester import Backtester
from trading.signals import SignalGenerator
from trading.risk_manager import RiskManager
from data.data_loader import CryptoDataLoader

print("🚀 Запуск изолированного тестирования модели...")
print("=" * 80)

# Загрузка конфигурации
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Устройство: {device}")

# 1. Загрузка модели
print("\n📥 Загрузка обученной модели...")
checkpoint_path = 'models_saved/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

# Создание модели - UnifiedPatchTSTForTrading принимает config целиком
model = UnifiedPatchTST(checkpoint['config'])

# Загрузка весов
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✅ Модель загружена. Эпоха: {checkpoint.get('epoch', 'N/A')}")
print(f"📊 Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

# 2. Загрузка тестовых данных
print("\n📊 Загрузка тестовых данных...")
data_loader = CryptoDataLoader(config)

# Используем кэшированные данные если есть
test_data_path = Path('data/processed/test_data.parquet')
if test_data_path.exists():
    test_df = pd.read_parquet(test_data_path)
    print(f"✅ Загружено {len(test_df)} записей из кэша")
else:
    print("⚠️ Кэшированные данные не найдены, загружаем из БД...")
    test_df = data_loader.load_test_data()

# Ограничимся последними 30 днями для быстрого теста
test_df_recent = test_df.sort_values('datetime').tail(30 * 24 * 4)  # 30 дней * 24 часа * 4 (15-мин интервалы)
print(f"📅 Тестирование на последних {len(test_df_recent)} записях")

# 3. Создание адаптера и получение предсказаний
print("\n🔮 Генерация предсказаний модели...")
adapter = ModelOutputAdapter()

# Подготовка данных для модели
# Берем признаки (все колонки кроме целевых и мета-данных)
feature_columns = [col for col in test_df_recent.columns 
                  if not col.startswith(('future_', 'direction_', 'volatility_', 'volume_change_', 'price_range_'))
                  and col not in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

# Создаем батчи для предсказаний
batch_size = 1024
predictions_list = []
symbols_list = []

print(f"📊 Найдено признаков: {len(feature_columns)}")

# Модель обучена на 240 признаках, берем первые 240
if len(feature_columns) > 240:
    print(f"⚠️ Модель обучена на 240 признаках, обрезаем с {len(feature_columns)} до 240")
    feature_columns = feature_columns[:240]
    
print(f"📦 Обработка батчей (размер батча: {batch_size})...")

with torch.no_grad():
    for i in range(0, len(test_df_recent), batch_size):
        batch_df = test_df_recent.iloc[i:i+batch_size]
        
        # Подготовка признаков с обработкой типов
        try:
            # Преобразуем в float, заменяя нечисловые значения на 0
            features_np = batch_df[feature_columns].values.astype(np.float32)
        except:
            # Если есть проблемы с типами, обрабатываем поколоночно
            features_list = []
            for col in feature_columns:
                try:
                    col_values = pd.to_numeric(batch_df[col], errors='coerce').fillna(0).values
                    features_list.append(col_values)
                except:
                    # Пропускаем проблемную колонку
                    print(f"⚠️ Пропущена колонка {col}")
                    features_list.append(np.zeros(len(batch_df)))
            features_np = np.column_stack(features_list).astype(np.float32)
            
        features = torch.tensor(features_np, dtype=torch.float32).to(device)
        
        # Добавляем размерность для sequence length если нужно
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # [batch, 1, features]
        
        # Предсказание
        outputs = model(features)
        
        predictions_list.append(outputs.cpu())
        symbols_list.extend(batch_df['symbol'].tolist())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"   Обработано: {min(i + batch_size, len(test_df_recent))}/{len(test_df_recent)}")

# Объединяем предсказания
all_predictions = torch.cat(predictions_list, dim=0)
print(f"✅ Получено предсказаний: {all_predictions.shape}")

# 4. Адаптация предсказаний
print("\n🔄 Адаптация предсказаний для торговых сигналов...")
adapted_predictions = adapter.adapt_model_outputs(all_predictions, symbols_list)

# Статистика по символам
print(f"\n📊 Статистика предсказаний по символам:")
for symbol, preds in adapted_predictions.items():
    print(f"   {symbol}: направление={preds['direction']}, уверенность={preds['confidence']:.3f}")

# 5. Генерация торговых сигналов
print("\n📈 Генерация торговых сигналов...")
signal_generator = SignalGenerator(config)

# Модифицируем SignalGenerator чтобы принимать наши адаптированные предсказания
# Переопределяем метод _extract_symbol_predictions
original_extract = signal_generator._extract_symbol_predictions
signal_generator._extract_symbol_predictions = lambda preds, symbol: preds.get(symbol)

# Генерируем сигналы
signals = signal_generator.generate_signals(
    predictions=adapted_predictions,
    market_data=test_df_recent,
    features=test_df_recent
)

print(f"✅ Сгенерировано {len(signals)} торговых сигналов")

# 6. Запуск бэктестинга
print("\n💰 Запуск бэктестинга...")
backtester = Backtester(config)

# Подготовка данных для бэктестера
market_data = test_df_recent[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()

# Запускаем бэктестинг
try:
    results = backtester.run_backtest(
        market_data=market_data,
        features=test_df_recent,
        model_predictions=adapted_predictions
    )
    
    # 7. Отображение результатов
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 80)
    
    print(f"\n💵 Финансовые показатели:")
    print(f"   Начальный капитал: ${results['initial_capital']:,.2f}")
    print(f"   Финальный капитал: ${results['final_capital']:,.2f}")
    print(f"   Общая доходность: {results['total_return_pct']:.2f}%")
    
    print(f"\n📈 Метрики производительности:")
    print(f"   Коэффициент Шарпа: {results['sharpe_ratio']:.2f}")
    print(f"   Максимальная просадка: {results['max_drawdown_pct']:.2f}%")
    print(f"   Win Rate: {results['win_rate_pct']:.2f}%")
    print(f"   Profit Factor: {results.get('profit_factor', 0):.2f}")
    
    print(f"\n🔢 Статистика торговли:")
    print(f"   Всего сделок: {results['total_trades']}")
    print(f"   Прибыльных: {results.get('winning_trades', 0)}")
    print(f"   Убыточных: {results.get('losing_trades', 0)}")
    
    # Сохраняем результаты
    output_dir = Path('experiments/backtest_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'test_results_{timestamp}.yaml'
    
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\n💾 Результаты сохранены в: {results_file}")
    
    # Оценка готовности к production
    print("\n" + "=" * 80)
    print("🎯 ОЦЕНКА ГОТОВНОСТИ К PRODUCTION:")
    print("=" * 80)
    
    min_sharpe = config['validation']['min_sharpe_ratio']
    min_win_rate = config['validation']['min_win_rate'] * 100
    max_dd = config['validation']['max_drawdown'] * 100
    
    checks = {
        'Sharpe Ratio': (results['sharpe_ratio'] >= min_sharpe, f">= {min_sharpe}"),
        'Win Rate': (results['win_rate_pct'] >= min_win_rate, f">= {min_win_rate}%"),
        'Max Drawdown': (results['max_drawdown_pct'] <= max_dd, f"<= {max_dd}%"),
        'Положительная доходность': (results['total_return_pct'] > 0, "> 0%")
    }
    
    all_passed = True
    for metric, (passed, threshold) in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {metric}: {status} (требуется {threshold})")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ МОДЕЛЬ ГОТОВА К ЗАПУСКУ НА ВСЕ МОНЕТЫ!")
        print("   Все критерии валидации пройдены успешно.")
    else:
        print("⚠️ МОДЕЛЬ ТРЕБУЕТ ДОПОЛНИТЕЛЬНОЙ ОПТИМИЗАЦИИ")
        print("   Некоторые критерии не выполнены.")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Ошибка при бэктестинге: {str(e)}")
    print("   Проверьте совместимость адаптера с бэктестером")
    
    # Минимальная проверка предсказаний
    print("\n📊 Анализ качества предсказаний:")
    
    # Считаем статистику по направлениям
    long_count = sum(1 for p in adapted_predictions.values() if p['direction'] == 'long')
    short_count = len(adapted_predictions) - long_count
    avg_confidence = np.mean([p['confidence'] for p in adapted_predictions.values()])
    
    print(f"   Long сигналов: {long_count} ({long_count/len(adapted_predictions)*100:.1f}%)")
    print(f"   Short сигналов: {short_count} ({short_count/len(adapted_predictions)*100:.1f}%)")
    print(f"   Средняя уверенность: {avg_confidence:.3f}")
    
    if avg_confidence > 0.6:
        print("\n✅ Модель демонстрирует хорошую уверенность в предсказаниях")
    else:
        print("\n⚠️ Модель показывает низкую уверенность, требуется доработка")

print("\n🏁 Тестирование завершено!")