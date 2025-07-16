#!/usr/bin/env python3
"""
🚀 ГЛАВНЫЙ МОДУЛЬ ЗАПУСКА ML TRADING SYSTEM

Объединяет все улучшения и решает проблему ROC-AUC 0.5:
1. Продвинутые целевые переменные с адаптивными порогами
2. Confidence-based предсказания
3. Ансамбль стратегий (trend, reversion, breakout, momentum)  
4. Walk-forward анализ
5. Учет рыночного режима

БЫСТРЫЙ СТАРТ:
python run_ml_trading.py --mode test
python run_ml_trading.py --mode full --symbols BTCUSDT ETHUSDT
"""

import sys
import os
import argparse
import logging
import yaml
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLTradingPipeline:
    """Главный пайплайн ML торговой системы"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.start_time = datetime.now()
        
    def run_pipeline(self, mode: str, symbols: List[str] = None, 
                    target_horizon: str = '1hour', cv_splits: int = 5):
        """
        Запускает полный пайплайн ML системы
        
        Args:
            mode: 'test' или 'full'
            symbols: Список символов
            target_horizon: Временной горизонт
            cv_splits: Количество fold для кросс-валидации
        """
        
        logger.info(self._get_header())
        
        if mode == 'test':
            symbols = ['BTCUSDT']
            limit = 50000
            cv_splits = 3
            logger.info("🧪 ТЕСТОВЫЙ РЕЖИМ")
        else:
            symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
            limit = None
            logger.info("🚀 ПОЛНЫЙ РЕЖИМ")
            
        logger.info(f"Символы: {symbols}")
        logger.info(f"Горизонт: {target_horizon}")
        logger.info(f"CV splits: {cv_splits}")
        
        try:
            # ШАГ 1: Создание продвинутых целевых переменных
            self._step1_create_advanced_targets(symbols, limit)
            
            # ШАГ 2: Обучение продвинутых моделей
            model_results = self._step2_train_advanced_models(symbols, target_horizon, cv_splits)
            
            # ШАГ 3: Анализ результатов
            self._step3_analyze_results(model_results)
            
            # ШАГ 4: Генерация отчета
            self._step4_generate_report(model_results, mode, symbols)
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"\n✅ ПАЙПЛАЙН ЗАВЕРШЕН ЗА {elapsed:.1f} СЕКУНД")
            
        except Exception as e:
            logger.error(f"❌ ОШИБКА В ПАЙПЛАЙНЕ: {e}")
            raise
    
    def _step1_create_advanced_targets(self, symbols: List[str], limit: int = None):
        """Шаг 1: Создание продвинутых целевых переменных"""
        logger.info("\n" + "="*60)
        logger.info("📊 ШАГ 1: СОЗДАНИЕ ПРОДВИНУТЫХ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ")
        logger.info("="*60)
        
        try:
            from advanced_trading_system import MultiHorizonModel
            
            # Создаем систему
            system = MultiHorizonModel(self.config_path)
            
            # Создаем таблицу
            system.create_advanced_targets_table()
            
            # Генерируем таргеты
            system.generate_and_save_targets(symbols=symbols, limit=limit)
            
            logger.info("✅ Продвинутые целевые переменные созданы")
            
        except Exception as e:
            logger.error(f"❌ Ошибка в создании целевых переменных: {e}")
            raise
    
    def _step2_train_advanced_models(self, symbols: List[str], 
                                   target_horizon: str, cv_splits: int) -> Dict:
        """Шаг 2: Обучение продвинутых моделей"""
        logger.info("\n" + "="*60)
        logger.info("🤖 ШАГ 2: ОБУЧЕНИЕ ПРОДВИНУТЫХ МОДЕЛЕЙ")
        logger.info("="*60)
        
        try:
            from train_advanced_models import AdvancedTrainingSystem
            import joblib
            
            # Создаем систему обучения
            training_system = AdvancedTrainingSystem(self.config_path)
            
            # Загружаем данные
            logger.info("📥 Загрузка данных для обучения...")
            df = training_system.load_advanced_data(
                symbols=symbols,
                target_horizon=target_horizon
            )
            
            if len(df) == 0:
                raise ValueError("❌ Нет данных для обучения!")
            
            # Обучаем модели
            logger.info("🚀 Начало обучения...")
            final_model, cv_results = training_system.train_with_time_series_cv(
                df, n_splits=cv_splits
            )
            
            # Сохраняем модель
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = Path(f'ml_models_{timestamp}')
            model_dir.mkdir(exist_ok=True)
            
            joblib.dump(final_model, model_dir / 'ensemble_model.pkl')
            
            # Сохраняем результаты
            import pandas as pd
            pd.DataFrame(cv_results).to_csv(model_dir / 'cv_results.csv', index=False)
            
            logger.info(f"✅ Модель сохранена в {model_dir}")
            
            return {
                'model': final_model,
                'cv_results': cv_results,
                'model_dir': model_dir,
                'data_size': len(df)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка в обучении моделей: {e}")
            raise
    
    def _step3_analyze_results(self, model_results: Dict):
        """Шаг 3: Анализ результатов"""
        logger.info("\n" + "="*60)
        logger.info("📈 ШАГ 3: АНАЛИЗ РЕЗУЛЬТАТОВ")
        logger.info("="*60)
        
        cv_results = model_results['cv_results']
        
        # Вычисляем средние метрики
        import pandas as pd
        df_results = pd.DataFrame(cv_results)
        
        metrics_summary = {}
        for metric in ['accuracy', 'high_confidence_accuracy', 'coverage', 'roc_auc']:
            mean_val = df_results[metric].mean()
            std_val = df_results[metric].std()
            metrics_summary[metric] = {'mean': mean_val, 'std': std_val}
        
        # Логируем результаты
        logger.info("📊 ИТОГОВЫЕ МЕТРИКИ:")
        logger.info("-" * 40)
        
        for metric, stats in metrics_summary.items():
            logger.info(f"{metric:25}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        # Анализ улучшений
        overall_roc_auc = metrics_summary['roc_auc']['mean']
        high_conf_accuracy = metrics_summary['high_confidence_accuracy']['mean']
        coverage = metrics_summary['coverage']['mean']
        
        logger.info("\n🎯 АНАЛИЗ УЛУЧШЕНИЙ:")
        logger.info("-" * 40)
        
        if overall_roc_auc > 0.55:
            logger.info(f"✅ ROC-AUC улучшен до {overall_roc_auc:.3f} (было ~0.50)")
        else:
            logger.warning(f"⚠️ ROC-AUC {overall_roc_auc:.3f} все еще близок к случайному")
        
        if high_conf_accuracy > 0.60:
            logger.info(f"✅ Точность на высокой уверенности: {high_conf_accuracy:.3f}")
        else:
            logger.warning(f"⚠️ Низкая точность высокоуверенных предсказаний: {high_conf_accuracy:.3f}")
        
        if coverage > 0.30:
            logger.info(f"✅ Покрытие высокоуверенных предсказаний: {coverage:.3f}")
        else:
            logger.warning(f"⚠️ Низкое покрытие: {coverage:.3f}")
        
        model_results['metrics_summary'] = metrics_summary
    
    def _step4_generate_report(self, model_results: Dict, mode: str, symbols: List[str]):
        """Шаг 4: Генерация финального отчета"""
        logger.info("\n" + "="*60)
        logger.info("📝 ШАГ 4: ГЕНЕРАЦИЯ ОТЧЕТА")
        logger.info("="*60)
        
        model_dir = model_results['model_dir']
        metrics_summary = model_results['metrics_summary']
        data_size = model_results['data_size']
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Создаем отчет
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    ML TRADING SYSTEM REPORT                  ║
╚══════════════════════════════════════════════════════════════╝

Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Режим: {mode.upper()}
Символы: {', '.join(symbols)}
Размер данных: {data_size:,} записей
Время выполнения: {elapsed:.1f} секунд

РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:
{'='*50}
ROC-AUC:                    {metrics_summary['roc_auc']['mean']:.3f} ± {metrics_summary['roc_auc']['std']:.3f}
Общая точность:             {metrics_summary['accuracy']['mean']:.3f} ± {metrics_summary['accuracy']['std']:.3f}
Точность (высокая увер.):   {metrics_summary['high_confidence_accuracy']['mean']:.3f} ± {metrics_summary['high_confidence_accuracy']['std']:.3f}
Покрытие:                   {metrics_summary['coverage']['mean']:.3f} ± {metrics_summary['coverage']['std']:.3f}

ОЦЕНКА УЛУЧШЕНИЙ:
{'='*50}
"""
        
        # Оценка результатов
        roc_auc = metrics_summary['roc_auc']['mean']
        if roc_auc > 0.60:
            report += "🚀 ОТЛИЧНЫЙ РЕЗУЛЬТАТ: ROC-AUC значительно выше случайного!\n"
        elif roc_auc > 0.55:
            report += "✅ ХОРОШИЙ РЕЗУЛЬТАТ: ROC-AUC улучшен по сравнению с базовой моделью.\n"
        elif roc_auc > 0.52:
            report += "📈 НЕБОЛЬШОЕ УЛУЧШЕНИЕ: ROC-AUC немного выше случайного.\n"
        else:
            report += "❌ ПРОБЛЕМА НЕ РЕШЕНА: ROC-AUC все еще близок к 0.5.\n"
        
        high_conf_acc = metrics_summary['high_confidence_accuracy']['mean']
        coverage = metrics_summary['coverage']['mean']
        
        if high_conf_acc > 0.65 and coverage > 0.25:
            report += "🎯 CONFIDENCE-ПОДХОД РАБОТАЕТ: Высокая точность при достаточном покрытии.\n"
        elif high_conf_acc > 0.60:
            report += "⚡ CONFIDENCE-ПОДХОД ЧАСТИЧНО РАБОТАЕТ: Хорошая точность, но низкое покрытие.\n"
        else:
            report += "⚠️ CONFIDENCE-ПОДХОД ТРЕБУЕТ ДОРАБОТКИ.\n"
        
        report += f"""
ФАЙЛЫ:
{'='*50}
Модель: {model_dir}/ensemble_model.pkl
Результаты CV: {model_dir}/cv_results.csv
Отчет: {model_dir}/final_report.txt

СЛЕДУЮЩИЕ ШАГИ:
{'='*50}
1. Проанализируйте результаты CV в деталях
2. Протестируйте модель на новых данных
3. Если результаты хорошие - интегрируйте в торговую стратегию
4. Если нужны улучшения - попробуйте:
   - Больше данных (все 51 символ)
   - Другие временные горизонты
   - Дополнительные признаки (order flow, social sentiment)
   - Другие архитектуры моделей (LSTM, Transformer)

КОНТАКТЫ ПОДДЕРЖКИ:
{'='*50}
Если нужна помощь с интерпретацией результатов или дальнейшими улучшениями,
обратитесь к документации или сообществу разработчиков.
"""
        
        # Сохраняем отчет
        report_path = model_dir / 'final_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Выводим ключевые результаты
        logger.info("\n🎯 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
        logger.info(f"ROC-AUC: {roc_auc:.3f}")
        logger.info(f"Точность (высокая уверенность): {high_conf_acc:.3f}")
        logger.info(f"Покрытие: {coverage:.3f}")
        logger.info(f"\n📁 Полный отчет: {report_path}")
    
    def _get_header(self) -> str:
        """Возвращает заголовок системы"""
        return """
╔════════════════════════════════════════════════════════════════╗
║                     🚀 ML TRADING SYSTEM v3.0                 ║
║                                                                ║
║  Решение проблемы ROC-AUC 0.5 через:                         ║
║  • Адаптивные пороги на основе волатильности                  ║
║  • Confidence-based предсказания                              ║
║  • Ансамбль стратегий (trend, reversion, breakout, momentum) ║
║  • Walk-forward анализ с временной валидацией                 ║
║  • Учет рыночного режима                                      ║
╚════════════════════════════════════════════════════════════════╝
        """


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="ML Trading System v3.0 - Решение проблемы ROC-AUC 0.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:

# Быстрый тест (1 символ, 50k записей, 3 CV folds)
python run_ml_trading.py --mode test

# Полное обучение (несколько символов, все данные)  
python run_ml_trading.py --mode full --symbols BTCUSDT ETHUSDT BNBUSDT

# Кастомные параметры
python run_ml_trading.py --mode full --symbols BTCUSDT --horizon 4hour --cv-splits 10

РЕЖИМЫ:
test  - Быстрый тест для проверки работоспособности
full  - Полное обучение для продакшена

ГОРИЗОНТЫ:
15min, 1hour, 4hour, 16hour
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['test', 'full'],
        default='test',
        help='Режим запуска (test - быстрый тест, full - полное обучение)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Список символов для обучения (по умолчанию зависит от режима)'
    )
    
    parser.add_argument(
        '--horizon',
        choices=['15min', '1hour', '4hour', '16hour'],
        default='1hour',
        help='Временной горизонт для предсказания'
    )
    
    parser.add_argument(
        '--cv-splits',
        type=int,
        default=5,
        help='Количество fold для кросс-валидации'
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Путь к файлу конфигурации'
    )
    
    args = parser.parse_args()
    
    # Создаем и запускаем пайплайн
    pipeline = MLTradingPipeline(args.config)
    
    pipeline.run_pipeline(
        mode=args.mode,
        symbols=args.symbols,
        target_horizon=args.horizon,
        cv_splits=args.cv_splits
    )


if __name__ == "__main__":
    main()