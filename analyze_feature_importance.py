#!/usr/bin/env python3
"""
Анализ важности признаков для XGBoost v3
Сравнение разных запусков и выявление проблем
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def load_run_data(log_dir: Path) -> dict:
    """Загружает данные из директории с логами"""
    data = {
        'dir': log_dir,
        'timestamp': log_dir.name.split('_')[-2] + '_' + log_dir.name.split('_')[-1]
    }
    
    # Загружаем метрики
    metrics_file = log_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data['metrics'] = json.load(f)
    
    # Загружаем отчет
    report_file = log_dir / 'final_report.txt'
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            data['report'] = f.read()
    
    # Извлекаем топ признаки из отчета
    if 'report' in data:
        lines = data['report'].split('\n')
        features = []
        in_features_section = False
        
        for line in lines:
            if 'ТОП-' in line and 'ПРИЗНАКОВ' in line:
                in_features_section = True
                continue
            if in_features_section and line.strip() == '':
                break
            if in_features_section and '. ' in line:
                feature = line.split('. ', 1)[1].strip()
                features.append(feature)
        
        data['top_features'] = features[:20]
    
    return data

def categorize_feature(feature: str) -> str:
    """Категоризирует признак"""
    # Технические индикаторы
    technical_indicators = [
        'rsi', 'macd', 'bb_', 'adx', 'atr', 'stoch', 'williams', 'mfi', 
        'cci', 'cmf', 'obv', 'ema', 'sma', 'vwap', 'sar', 'ich_'
    ]
    
    # BTC корреляция
    btc_features = ['btc_', 'correlation']
    
    # Временные признаки
    time_features = ['hour', 'dow', 'day', 'week']
    
    # Категориальные признаки
    categorical_features = ['is_', 'market_regime']
    
    feature_lower = feature.lower()
    
    for indicator in technical_indicators:
        if indicator in feature_lower:
            return 'technical'
    
    for btc in btc_features:
        if btc in feature_lower:
            return 'btc_related'
    
    for time in time_features:
        if time in feature_lower:
            return 'temporal'
    
    for cat in categorical_features:
        if cat in feature_lower:
            return 'categorical'
    
    return 'other'

def analyze_feature_distribution(features: list) -> dict:
    """Анализирует распределение признаков по категориям"""
    categories = {}
    
    for feature in features:
        category = categorize_feature(feature)
        categories[category] = categories.get(category, 0) + 1
    
    total = len(features)
    distribution = {
        cat: {'count': count, 'percentage': (count/total)*100}
        for cat, count in categories.items()
    }
    
    return distribution

def compare_runs(runs: list) -> pd.DataFrame:
    """Сравнивает несколько запусков"""
    comparison_data = []
    
    for run in runs:
        row = {
            'timestamp': run['timestamp'],
            'directory': run['dir'].name
        }
        
        # Метрики
        if 'metrics' in run:
            if 'buy' in run['metrics']:
                row['buy_roc_auc'] = run['metrics']['buy'].get('roc_auc', 0)
                row['buy_precision'] = run['metrics']['buy'].get('precision', 0)
            if 'sell' in run['metrics']:
                row['sell_roc_auc'] = run['metrics']['sell'].get('roc_auc', 0)
                row['sell_precision'] = run['metrics']['sell'].get('precision', 0)
        
        # Распределение признаков
        if 'top_features' in run:
            distribution = analyze_feature_distribution(run['top_features'])
            for cat, data in distribution.items():
                row[f'{cat}_features_%'] = data['percentage']
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def plot_feature_comparison(runs: list, save_path: str = None):
    """Визуализирует сравнение признаков между запусками"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Анализ важности признаков XGBoost v3', fontsize=16)
    
    # 1. Распределение категорий признаков
    ax = axes[0, 0]
    
    categories_data = []
    for run in runs:
        if 'top_features' in run:
            distribution = analyze_feature_distribution(run['top_features'])
            for cat, data in distribution.items():
                categories_data.append({
                    'run': run['timestamp'],
                    'category': cat,
                    'percentage': data['percentage']
                })
    
    if categories_data:
        df_cat = pd.DataFrame(categories_data)
        df_pivot = df_cat.pivot(index='run', columns='category', values='percentage').fillna(0)
        df_pivot.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Распределение категорий признаков по запускам')
        ax.set_ylabel('Процент признаков')
        ax.set_xlabel('Запуск')
        ax.legend(title='Категория', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    # 2. ROC-AUC по запускам
    ax = axes[0, 1]
    
    roc_data = []
    for run in runs:
        if 'metrics' in run:
            if 'buy' in run['metrics']:
                roc_data.append({
                    'run': run['timestamp'],
                    'model': 'buy',
                    'roc_auc': run['metrics']['buy'].get('roc_auc', 0)
                })
            if 'sell' in run['metrics']:
                roc_data.append({
                    'run': run['timestamp'],
                    'model': 'sell',
                    'roc_auc': run['metrics']['sell'].get('roc_auc', 0)
                })
    
    if roc_data:
        df_roc = pd.DataFrame(roc_data)
        df_roc_pivot = df_roc.pivot(index='run', columns='model', values='roc_auc')
        df_roc_pivot.plot(kind='bar', ax=ax)
        ax.set_title('ROC-AUC по запускам')
        ax.set_ylabel('ROC-AUC')
        ax.set_xlabel('Запуск')
        ax.axhline(y=0.8, color='r', linestyle='--', label='Целевой уровень')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    # 3. Топ признаки последнего запуска
    ax = axes[1, 0]
    
    if runs and 'top_features' in runs[-1]:
        features = runs[-1]['top_features'][:10]
        categories = [categorize_feature(f) for f in features]
        colors = {
            'technical': 'green',
            'btc_related': 'blue',
            'temporal': 'red',
            'categorical': 'orange',
            'other': 'gray'
        }
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, range(len(features), 0, -1), 
                color=[colors.get(cat, 'gray') for cat in categories])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Важность (ранг)')
        ax.set_title(f'Топ-10 признаков последнего запуска ({runs[-1]["timestamp"]})')
        ax.invert_xaxis()
        
        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat) 
                          for cat, color in colors.items()]
        ax.legend(handles=legend_elements, loc='lower right')
    
    # 4. Рекомендации
    ax = axes[1, 1]
    ax.axis('off')
    
    recommendations = """
РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ:

1. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ должны составлять >70%
   Текущий статус: {}

2. ВРЕМЕННЫЕ ПРИЗНАКИ должны быть <20%
   Текущий статус: {}

3. Целевой ROC-AUC > 0.80
   Текущий статус: Buy={:.3f}, Sell={:.3f}

4. ДЕЙСТВИЯ:
   • Использовать hierarchical feature selection
   • Применить веса к категориям признаков
   • Увеличить регуляризацию (alpha, lambda)
   • Уменьшить max_depth до 6-8
    """.format(
        "✅ OK" if runs and 'top_features' in runs[-1] and 
        analyze_feature_distribution(runs[-1]['top_features']).get('technical', {}).get('percentage', 0) > 70
        else "❌ Требует исправления",
        
        "✅ OK" if runs and 'top_features' in runs[-1] and 
        analyze_feature_distribution(runs[-1]['top_features']).get('temporal', {}).get('percentage', 0) < 20
        else "❌ Требует исправления",
        
        runs[-1]['metrics']['buy'].get('roc_auc', 0) if runs and 'metrics' in runs[-1] and 'buy' in runs[-1]['metrics'] else 0,
        runs[-1]['metrics']['sell'].get('roc_auc', 0) if runs and 'metrics' in runs[-1] and 'sell' in runs[-1]['metrics'] else 0
    )
    
    ax.text(0.05, 0.95, recommendations, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Анализ важности признаков XGBoost')
    parser.add_argument('--compare-runs', action='store_true',
                       help='Сравнить несколько последних запусков')
    parser.add_argument('--last-n', type=int, default=3,
                       help='Количество последних запусков для сравнения')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Путь для сохранения графика')
    
    args = parser.parse_args()
    
    # Находим директории с логами
    log_base = Path('logs')
    xgboost_dirs = sorted([d for d in log_base.glob('xgboost_v3_*') if d.is_dir()])
    
    if not xgboost_dirs:
        print("Не найдены логи XGBoost v3")
        return
    
    # Загружаем данные
    runs = []
    for log_dir in xgboost_dirs[-args.last_n:]:
        print(f"Загружаю данные из: {log_dir}")
        run_data = load_run_data(log_dir)
        runs.append(run_data)
    
    # Анализируем
    if args.compare_runs:
        comparison_df = compare_runs(runs)
        print("\n📊 СРАВНЕНИЕ ЗАПУСКОВ:")
        print(comparison_df.to_string())
        
        # Визуализация
        plot_path = args.save_plot or 'feature_importance_analysis.png'
        plot_feature_comparison(runs, plot_path)
    
    # Детальный анализ последнего запуска
    if runs:
        last_run = runs[-1]
        print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПОСЛЕДНЕГО ЗАПУСКА ({last_run['timestamp']}):")
        
        if 'top_features' in last_run:
            distribution = analyze_feature_distribution(last_run['top_features'])
            print("\nРаспределение категорий признаков:")
            for cat, data in sorted(distribution.items(), 
                                   key=lambda x: x[1]['percentage'], 
                                   reverse=True):
                print(f"  {cat:15s}: {data['count']:2d} признаков ({data['percentage']:5.1f}%)")
        
        if 'metrics' in last_run:
            print("\nМетрики модели:")
            for model in ['buy', 'sell']:
                if model in last_run['metrics']:
                    metrics = last_run['metrics'][model]
                    print(f"  {model.upper()}:")
                    print(f"    ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                    print(f"    Precision: {metrics.get('precision', 0):.4f}")
                    print(f"    Recall: {metrics.get('recall', 0):.4f}")

if __name__ == "__main__":
    main()