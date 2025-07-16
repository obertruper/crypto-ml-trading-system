#!/usr/bin/env python3
"""
Комплексная оценка модели для продакшена
Анализирует проблемы с предсказаниями и готовит модель к развертыванию
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Настройка визуализации
plt.style.use('dark_background')
sns.set_palette("husl")

class ModelProductionEvaluator:
    """Оценщик модели для продакшена"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Инициализация оценщика"""
        # Загружаем конфигурацию
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Используется устройство: {self.device}")
        
        # Пути
        self.models_dir = Path("models_saved")
        self.logs_dir = Path("logs")
        self.cache_dir = Path("cache/precomputed")
        self.plots_dir = Path("evaluation_plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Результаты анализа
        self.evaluation_results = {}
        
    def find_best_model(self) -> Optional[Path]:
        """Находит лучшую сохраненную модель"""
        print("\n🔍 Поиск лучшей модели...")
        
        if not self.models_dir.exists():
            print("❌ Директория models_saved не существует!")
            return None
        
        # Ищем файлы моделей
        model_files = list(self.models_dir.glob("*.pth"))
        
        if not model_files:
            print("❌ Не найдено сохраненных моделей!")
            return None
        
        # Сортируем по времени модификации (последняя - самая новая)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"✅ Найдено моделей: {len(model_files)}")
        for i, model_file in enumerate(model_files[:5]):  # Показываем топ-5
            size_mb = model_file.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
            print(f"   {i+1}. {model_file.name} ({size_mb:.1f} MB) - {mod_time}")
        
        best_model = model_files[0]
        print(f"\n📦 Выбрана модель: {best_model.name}")
        return best_model
    
    def load_model(self, model_path: Path) -> torch.nn.Module:
        """Загружает модель"""
        print(f"\n📥 Загрузка модели: {model_path.name}")
        
        # Загружаем checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Информация о checkpoint
        if isinstance(checkpoint, dict):
            print("📊 Информация о checkpoint:")
            for key in ['epoch', 'best_val_loss', 'model_config']:
                if key in checkpoint:
                    print(f"   {key}: {checkpoint[key]}")
        
        # Создаем модель с правильной конфигурацией
        from models.patchtst_unified import UnifiedPatchTSTForTrading
        
        # Проверяем конфигурацию в checkpoint
        if isinstance(checkpoint, dict):
            # Если есть сохраненная конфигурация, используем её
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                # Обновляем размер входа из сохраненной конфигурации
                if 'model' in saved_config and 'input_size' in saved_config['model']:
                    self.config['model']['input_size'] = saved_config['model']['input_size']
                    print(f"   📊 Использую input_size из checkpoint: {saved_config['model']['input_size']}")
            
            # Альтернативно, определяем размер из весов
            elif 'model_state_dict' in checkpoint:
                if 'revin.affine_weight' in checkpoint['model_state_dict']:
                    input_size = checkpoint['model_state_dict']['revin.affine_weight'].shape[0]
                    self.config['model']['input_size'] = input_size
                    print(f"   📊 Определен input_size из весов модели: {input_size}")
        
        model = UnifiedPatchTSTForTrading(self.config)
        
        # Загружаем веса
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print("✅ Модель загружена успешно")
        return model
    
    def analyze_model_weights(self, model: torch.nn.Module):
        """Анализирует веса модели для понимания bias"""
        print("\n🔬 Анализ весов модели...")
        
        # Анализируем direction head
        if hasattr(model, 'direction_head'):
            print("\n📊 Direction Head анализ:")
            
            # Получаем последний слой
            for name, param in model.direction_head.named_parameters():
                if 'bias' in name and param.shape[0] >= 12:  # 4 таймфрейма × 3 класса
                    biases = param.detach().cpu().numpy()
                    
                    # Разбиваем на таймфреймы
                    for tf_idx in range(4):
                        tf_biases = biases[tf_idx*3:(tf_idx+1)*3]
                        print(f"\n   Таймфрейм {tf_idx+1}:")
                        print(f"      LONG bias:  {tf_biases[0]:.4f}")
                        print(f"      SHORT bias: {tf_biases[1]:.4f}")
                        print(f"      FLAT bias:  {tf_biases[2]:.4f}")
                        
                        # Определяем предпочтительный класс
                        preferred_class = ['LONG', 'SHORT', 'FLAT'][np.argmax(tf_biases)]
                        print(f"      ⚠️ Предпочтительный класс: {preferred_class}")
        
        # Анализируем градиенты (если есть)
        total_grad_norm = 0
        num_params = 0
        
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                num_params += 1
        
        if num_params > 0:
            avg_grad_norm = total_grad_norm / num_params
            print(f"\n📈 Средняя норма градиентов: {avg_grad_norm:.6f}")
            if avg_grad_norm < 0.0001:
                print("   ⚠️ Очень маленькие градиенты - модель может не обучаться!")
    
    def load_validation_data(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Загружает валидационные данные"""
        print("\n📂 Загрузка валидационных данных...")
        
        # Ищем предвычисленные данные - сначала h5, потом pt
        val_files = list(self.cache_dir.glob("val_*.h5"))
        if not val_files:
            val_files = list(self.cache_dir.glob("*_val.pt"))
        
        if not val_files:
            print("❌ Валидационные данные не найдены!")
            return None
        
        # Загружаем первый файл для анализа
        val_file = val_files[0]
        print(f"📄 Загружаем: {val_file.name}")
        
        # Загружаем h5 файл
        import h5py
        
        with h5py.File(val_file, 'r') as f:
            # Проверяем структуру файла
            print(f"   Ключи в файле: {list(f.keys())}")
            
            # Загружаем данные - пробуем разные ключи
            if 'X' in f and 'y' in f:
                # Берем первые 1000 сэмплов для анализа
                num_samples = min(1000, f['X'].shape[0])
                
                inputs = torch.tensor(f['X'][:num_samples], dtype=torch.float32)
                targets = torch.tensor(f['y'][:num_samples], dtype=torch.float32)
                
                print(f"✅ Загружено сэмплов: {num_samples}")
                print(f"   Размер входов: {inputs.shape}")
                print(f"   Размер целей: {targets.shape}")
                
                return inputs, targets
            elif 'windows' in f and 'targets' in f:
                # Альтернативные ключи
                num_samples = min(1000, f['windows'].shape[0])
                
                inputs = torch.tensor(f['windows'][:num_samples], dtype=torch.float32)
                targets = torch.tensor(f['targets'][:num_samples], dtype=torch.float32)
                
                print(f"✅ Загружено сэмплов: {num_samples}")
                print(f"   Размер входов: {inputs.shape}")
                print(f"   Размер целей: {targets.shape}")
                
                return inputs, targets
            else:
                print("❌ Неверная структура h5 файла!")
                return None
    
    def analyze_class_distribution(self, targets: torch.Tensor):
        """Анализирует распределение классов в данных"""
        print("\n📊 Анализ распределения классов...")
        
        # Проверяем размерность targets и корректируем если нужно
        if len(targets.shape) == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # Убираем измерение размера 1
        
        # Direction классы находятся в позициях 4-7
        direction_targets = targets[:, 4:8].numpy()
        
        class_counts = {
            'LONG': 0,
            'SHORT': 0,
            'FLAT': 0
        }
        
        # Подсчитываем классы для каждого таймфрейма
        for tf_idx in range(4):
            tf_classes = direction_targets[:, tf_idx]
            
            counts = Counter(tf_classes)
            total = len(tf_classes)
            
            print(f"\n   Таймфрейм {tf_idx+1}:")
            for class_idx, class_name in enumerate(['LONG', 'SHORT', 'FLAT']):
                count = counts.get(float(class_idx), 0)
                percent = (count / total) * 100
                print(f"      {class_name}: {count} ({percent:.1f}%)")
                class_counts[class_name] += count
        
        # Общая статистика
        total_counts = sum(class_counts.values())
        print("\n   📈 Общее распределение:")
        for class_name, count in class_counts.items():
            percent = (count / total_counts) * 100
            print(f"      {class_name}: {count} ({percent:.1f}%)")
        
        # Визуализация
        self._plot_class_distribution(class_counts)
        
        # Анализ дисбаланса
        max_class = max(class_counts.values())
        min_class = min(class_counts.values())
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        print(f"\n   ⚠️ Коэффициент дисбаланса: {imbalance_ratio:.2f}")
        if imbalance_ratio > 2:
            print("   ❌ Сильный дисбаланс классов!")
        
        self.evaluation_results['class_distribution'] = class_counts
        self.evaluation_results['imbalance_ratio'] = imbalance_ratio
    
    def test_model_predictions(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor):
        """Тестирует предсказания модели"""
        print("\n🧪 Тестирование предсказаний модели...")
        
        # Проверяем размерность targets и корректируем если нужно
        if len(targets.shape) == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # Убираем измерение размера 1
        
        model.eval()
        with torch.no_grad():
            # Переносим на устройство
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Предсказания
            outputs = model(inputs)
            
            # Анализируем direction предсказания
            if hasattr(outputs, '_direction_logits'):
                direction_logits = outputs._direction_logits
                print(f"✅ Direction logits shape: {direction_logits.shape}")
                
                # Применяем softmax для получения вероятностей
                direction_probs = torch.softmax(direction_logits, dim=-1)
                
                # Анализируем предсказания для каждого таймфрейма
                pred_distribution = {}
                
                for tf_idx in range(4):
                    tf_probs = direction_probs[:, tf_idx, :]
                    tf_preds = torch.argmax(tf_probs, dim=-1)
                    
                    # Подсчет предсказаний
                    pred_counts = Counter(tf_preds.cpu().numpy())
                    
                    print(f"\n   Таймфрейм {tf_idx+1} предсказания:")
                    for class_idx, class_name in enumerate(['LONG', 'SHORT', 'FLAT']):
                        count = pred_counts.get(class_idx, 0)
                        percent = (count / len(tf_preds)) * 100
                        print(f"      {class_name}: {count} ({percent:.1f}%)")
                    
                    # Средняя уверенность
                    max_probs = torch.max(tf_probs, dim=-1)[0]
                    avg_confidence = max_probs.mean().item()
                    print(f"      Средняя уверенность: {avg_confidence:.3f}")
                    
                    # Конвертируем numpy типы в Python типы для JSON
                    pred_counts_dict = {}
                    for k, v in pred_counts.items():
                        pred_counts_dict[str(k)] = int(v)
                    pred_distribution[f'tf_{tf_idx}'] = pred_counts_dict
                
                self.evaluation_results['prediction_distribution'] = pred_distribution
                
                # Проверяем разнообразие предсказаний
                self._check_prediction_diversity(direction_probs)
    
    def _check_prediction_diversity(self, direction_probs: torch.Tensor):
        """Проверяет разнообразие предсказаний"""
        print("\n🎲 Анализ разнообразия предсказаний...")
        
        # Энтропия предсказаний
        eps = 1e-8
        entropy = -torch.sum(direction_probs * torch.log(direction_probs + eps), dim=-1)
        avg_entropy = entropy.mean().item()
        
        max_entropy = np.log(3)  # Максимальная энтропия для 3 классов
        normalized_entropy = avg_entropy / max_entropy
        
        print(f"   Средняя энтропия: {avg_entropy:.3f}")
        print(f"   Нормализованная энтропия: {normalized_entropy:.3f}")
        
        if normalized_entropy < 0.3:
            print("   ⚠️ Низкое разнообразие - модель слишком уверена в одном классе!")
        elif normalized_entropy > 0.9:
            print("   ⚠️ Слишком высокая неопределенность - модель не может выбрать!")
        else:
            print("   ✅ Хорошее разнообразие предсказаний")
        
        self.evaluation_results['prediction_entropy'] = normalized_entropy
    
    def _plot_class_distribution(self, class_counts: Dict[str, int]):
        """Визуализирует распределение классов"""
        plt.figure(figsize=(10, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = plt.bar(classes, counts)
        
        # Раскрашиваем в зависимости от класса
        colors = ['green', 'red', 'gray']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('Распределение классов в данных', fontsize=16)
        plt.xlabel('Класс направления', fontsize=12)
        plt.ylabel('Количество', fontsize=12)
        
        # Добавляем проценты
        total = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percent = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{percent:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'class_distribution.png', dpi=150)
        plt.close()
    
    def generate_production_recommendations(self):
        """Генерирует рекомендации для продакшена"""
        print("\n📋 Генерация рекомендаций для продакшена...")
        
        recommendations = []
        
        # Анализ дисбаланса классов
        if self.evaluation_results.get('imbalance_ratio', 0) > 2:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Сильный дисбаланс классов',
                'solution': 'Использовать weighted loss или SMOTE для балансировки',
                'file': 'models/patchtst_unified.py',
                'code_change': 'Добавить class_weights в CrossEntropyLoss'
            })
        
        # Анализ энтропии предсказаний
        entropy = self.evaluation_results.get('prediction_entropy', 0)
        if entropy < 0.3:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Модель предсказывает только один класс',
                'solution': 'Изменить инициализацию bias и увеличить learning rate',
                'file': 'config/config.yaml',
                'code_change': 'learning_rate: 0.0001 -> 0.001'
            })
        
        # Learning rate
        learning_rate = self.config.get('model', {}).get('learning_rate', 0.0001)
        if learning_rate < 0.0001:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': 'Слишком низкий learning rate',
                'solution': 'Увеличить learning rate и использовать warmup',
                'file': 'config/config.yaml',
                'code_change': 'Добавить warmup_steps: 1000'
            })
        
        # Direction loss weight
        direction_weight = self.config['loss']['task_weights'].get('directions', 1.0)
        if direction_weight < 5.0:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': 'Низкий вес для direction loss',
                'solution': 'Увеличить вес direction в loss функции',
                'file': 'config/config.yaml',
                'code_change': 'directions: 3.0 -> 10.0'
            })
        
        # Сохраняем рекомендации
        self._save_recommendations(recommendations)
        
        return recommendations
    
    def _save_recommendations(self, recommendations: List[Dict]):
        """Сохраняет рекомендации в файл"""
        report_path = self.plots_dir / 'production_recommendations.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🚀 Рекомендации для продакшена\n\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Группируем по приоритету
            high_priority = [r for r in recommendations if r['priority'] == 'HIGH']
            medium_priority = [r for r in recommendations if r['priority'] == 'MEDIUM']
            
            if high_priority:
                f.write("## 🔴 Высокий приоритет\n\n")
                for rec in high_priority:
                    f.write(f"### {rec['issue']}\n")
                    f.write(f"**Решение:** {rec['solution']}\n")
                    f.write(f"**Файл:** `{rec['file']}`\n")
                    f.write(f"**Изменение:** `{rec['code_change']}`\n\n")
            
            if medium_priority:
                f.write("## 🟡 Средний приоритет\n\n")
                for rec in medium_priority:
                    f.write(f"### {rec['issue']}\n")
                    f.write(f"**Решение:** {rec['solution']}\n")
                    f.write(f"**Файл:** `{rec['file']}`\n")
                    f.write(f"**Изменение:** `{rec['code_change']}`\n\n")
            
            # Добавляем результаты анализа
            f.write("## 📊 Результаты анализа\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.evaluation_results, indent=2))
            f.write("\n```\n")
        
        print(f"✅ Рекомендации сохранены: {report_path}")
    
    def run_full_evaluation(self):
        """Запускает полную оценку модели"""
        print("="*80)
        print("🚀 ПОЛНАЯ ОЦЕНКА МОДЕЛИ ДЛЯ ПРОДАКШЕНА")
        print("="*80)
        
        # 1. Находим модель
        model_path = self.find_best_model()
        if not model_path:
            return
        
        # 2. Загружаем модель
        model = self.load_model(model_path)
        
        # 3. Анализируем веса
        self.analyze_model_weights(model)
        
        # 4. Загружаем данные
        data = self.load_validation_data()
        if data is None:
            print("❌ Не удалось загрузить данные для оценки")
            return
        
        inputs, targets = data
        
        # 5. Анализируем распределение классов
        self.analyze_class_distribution(targets)
        
        # 6. Тестируем предсказания
        self.test_model_predictions(model, inputs, targets)
        
        # 7. Генерируем рекомендации
        recommendations = self.generate_production_recommendations()
        
        print("\n" + "="*80)
        print("📊 ИТОГОВАЯ ОЦЕНКА")
        print("="*80)
        
        print(f"\n🎯 Основные проблемы:")
        for rec in recommendations[:3]:  # Топ-3 проблемы
            print(f"   • {rec['issue']}")
        
        print(f"\n✅ Следующие шаги:")
        print("   1. Исправить дисбаланс классов через weighted loss")
        print("   2. Увеличить learning rate до 0.001")
        print("   3. Изменить инициализацию direction head")
        print("   4. Увеличить вес direction loss до 10.0")
        print("   5. Добавить мониторинг разнообразия предсказаний")
        
        print(f"\n📁 Результаты сохранены в: {self.plots_dir}")
        print("="*80)


def main():
    """Главная функция"""
    evaluator = ModelProductionEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()