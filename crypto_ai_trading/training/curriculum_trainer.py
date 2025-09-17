"""
Curriculum Learning Trainer для поэтапного обучения
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import logging
import yaml
from typing import Dict, Optional, List, Tuple

from utils.logger import get_logger
from training.optimized_trainer import OptimizedTrainer

class CurriculumTrainer(OptimizedTrainer):
    """
    Расширенный тренер с поддержкой Curriculum Learning

    Поэтапное обучение от простых примеров к сложным:
    1. Явные сигналы (большие движения > 2%)
    2. Средние сигналы (движения > 1%)
    3. Полный датасет (все сигналы)
    """

    def __init__(self, model, train_dataset, val_loader, config, device='cuda'):
        """
        Args:
            model: Модель для обучения
            train_dataset: Полный тренировочный датасет (не DataLoader!)
            val_loader: DataLoader для валидации
            config: Конфигурация
            device: Устройство
        """
        # Сохраняем полный датасет
        self.full_train_dataset = train_dataset

        # Загружаем конфигурацию curriculum
        self.curriculum_config = self._load_curriculum_config(config)

        # Текущий этап обучения
        self.current_stage = 0
        self.stages = self.curriculum_config['stages']

        # Создаем DataLoader для первого этапа
        stage_loader = self._create_stage_dataloader(0)

        # Инициализируем родительский класс
        super().__init__(model, stage_loader, val_loader, config, device)

        self.logger = get_logger("CurriculumTrainer")

        # История по этапам
        self.stage_history = []

    def _load_curriculum_config(self, config):
        """Загружает конфигурацию curriculum learning"""

        # Проверяем наличие файла curriculum.yaml
        curriculum_path = Path('config/curriculum.yaml')
        if curriculum_path.exists():
            with open(curriculum_path, 'r') as f:
                curriculum_data = yaml.safe_load(f)
                return curriculum_data.get('curriculum_learning', self._get_default_curriculum())

        # Или берем из основного конфига
        if 'curriculum_learning' in config:
            return config['curriculum_learning']

        # Дефолтная конфигурация
        return self._get_default_curriculum()

    def _get_default_curriculum(self):
        """Возвращает дефолтную конфигурацию curriculum"""
        return {
            'enabled': True,
            'stages': [
                {
                    'name': 'Явные сигналы',
                    'description': 'Обучение на сильных движениях > 2%',
                    'epochs': 5,
                    'data_filter': {
                        'min_return_threshold': 0.02,
                        'confidence_threshold': 0.7,
                        'exclude_flat_ratio': 0.5  # Исключаем 50% FLAT
                    },
                    'loss_weights': {
                        'directions': 1.5,
                        'returns': 0.5
                    },
                    'learning_rate_scale': 1.0
                },
                {
                    'name': 'Средние сигналы',
                    'description': 'Добавление движений > 1%',
                    'epochs': 10,
                    'data_filter': {
                        'min_return_threshold': 0.01,
                        'confidence_threshold': 0.5,
                        'exclude_flat_ratio': 0.3  # Исключаем 30% FLAT
                    },
                    'loss_weights': {
                        'directions': 1.2,
                        'returns': 0.8
                    },
                    'learning_rate_scale': 0.5
                },
                {
                    'name': 'Полный датасет',
                    'description': 'Обучение на всех данных',
                    'epochs': 15,
                    'data_filter': {
                        'min_return_threshold': 0.0,
                        'confidence_threshold': 0.3,
                        'exclude_flat_ratio': 0.0  # Используем все FLAT
                    },
                    'loss_weights': {
                        'directions': 1.0,
                        'returns': 1.0
                    },
                    'learning_rate_scale': 0.2
                }
            ]
        }

    def _filter_dataset_indices(self, stage_idx):
        """
        Фильтрует индексы датасета для текущего этапа

        Returns:
            list: Список индексов для обучения
        """
        stage = self.stages[stage_idx]
        data_filter = stage['data_filter']

        # Получаем параметры фильтрации
        min_return = data_filter.get('min_return_threshold', 0.0)
        confidence = data_filter.get('confidence_threshold', 0.0)
        exclude_flat_ratio = data_filter.get('exclude_flat_ratio', 0.0)

        valid_indices = []
        flat_indices = []

        # Проходим по всему датасету
        for idx in range(len(self.full_train_dataset)):
            # Получаем sample
            features, targets = self.full_train_dataset[idx]

            # targets shape: [1, 20]
            if isinstance(targets, torch.Tensor):
                targets = targets.squeeze(0) if targets.dim() > 1 else targets

                # Извлекаем целевые переменные
                future_return_15m = abs(targets[0].item()) if len(targets) > 0 else 0
                direction_15m = targets[4].item() if len(targets) > 4 else 2

                # Фильтр по возврату
                if future_return_15m >= min_return:
                    if direction_15m == 2:  # FLAT
                        flat_indices.append(idx)
                    else:
                        valid_indices.append(idx)

        # Применяем exclude_flat_ratio
        if exclude_flat_ratio > 0 and flat_indices:
            # Оставляем только часть FLAT примеров
            n_flat_to_keep = int(len(flat_indices) * (1 - exclude_flat_ratio))
            np.random.seed(42)
            flat_to_keep = np.random.choice(flat_indices, n_flat_to_keep, replace=False)
            valid_indices.extend(flat_to_keep)
        else:
            valid_indices.extend(flat_indices)

        # Перемешиваем
        np.random.seed(42 + stage_idx)
        np.random.shuffle(valid_indices)

        self.logger.info(f"📊 Этап {stage_idx + 1}: отфильтровано {len(valid_indices):,} примеров из {len(self.full_train_dataset):,}")

        return valid_indices

    def _create_stage_dataloader(self, stage_idx):
        """Создает DataLoader для текущего этапа"""

        # Получаем индексы для этапа
        indices = self._filter_dataset_indices(stage_idx)

        # Создаем Subset
        stage_dataset = Subset(self.full_train_dataset, indices)

        # Создаем DataLoader
        stage_loader = DataLoader(
            stage_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=False,  # Отключаем из-за custom collate
            drop_last=True,
            persistent_workers=True
        )

        return stage_loader

    def _update_stage_parameters(self, stage_idx):
        """Обновляет параметры для текущего этапа"""

        stage = self.stages[stage_idx]

        # Обновляем learning rate
        lr_scale = stage.get('learning_rate_scale', 1.0)
        base_lr = self.config['model'].get('learning_rate', 1e-4)
        new_lr = base_lr * lr_scale

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Обновляем веса loss функции
        loss_weights = stage.get('loss_weights', {})
        if hasattr(self.criterion, 'direction_weight'):
            self.criterion.direction_weight = loss_weights.get('directions', 1.0)
        if hasattr(self.criterion, 'return_weight'):
            self.criterion.return_weight = loss_weights.get('returns', 1.0)

        self.logger.info(f"📝 Параметры этапа {stage_idx + 1}:")
        self.logger.info(f"   Learning rate: {new_lr:.6f}")
        self.logger.info(f"   Direction weight: {loss_weights.get('directions', 1.0)}")
        self.logger.info(f"   Return weight: {loss_weights.get('returns', 1.0)}")

    def train(self, num_epochs=None):
        """
        Основной цикл обучения с curriculum learning
        """

        self.logger.info("="*60)
        self.logger.info("🎓 Запуск Curriculum Learning")
        self.logger.info(f"📚 Всего этапов: {len(self.stages)}")
        self.logger.info("="*60)

        total_epochs = 0

        # Проходим по всем этапам
        for stage_idx in range(len(self.stages)):
            stage = self.stages[stage_idx]
            stage_epochs = stage['epochs']

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎯 ЭТАП {stage_idx + 1}/{len(self.stages)}: {stage['name']}")
            self.logger.info(f"📝 {stage['description']}")
            self.logger.info(f"⏱️ Эпох: {stage_epochs}")
            self.logger.info(f"{'='*60}")

            # Обновляем DataLoader для этапа
            self.train_loader = self._create_stage_dataloader(stage_idx)

            # Обновляем параметры
            self._update_stage_parameters(stage_idx)

            # Сбрасываем счетчики patience для нового этапа
            self.patience_counter = 0
            self.best_val_loss = float('inf')

            # Обучение на этапе
            stage_metrics = {
                'stage_name': stage['name'],
                'train_losses': [],
                'val_losses': [],
                'best_val_loss': float('inf')
            }

            for epoch in range(stage_epochs):
                total_epochs += 1

                self.logger.info(f"\n📅 Эпоха {epoch + 1}/{stage_epochs} (Общая: {total_epochs})")

                # Обучение
                train_metrics = self.train_epoch(epoch)
                stage_metrics['train_losses'].append(train_metrics['loss'])

                # Валидация
                val_metrics = self.validate()
                stage_metrics['val_losses'].append(val_metrics['loss'])

                # Обновляем лучший результат
                if val_metrics['loss'] < stage_metrics['best_val_loss']:
                    stage_metrics['best_val_loss'] = val_metrics['loss']

                    # Сохраняем чекпоинт
                    checkpoint_name = f"stage_{stage_idx + 1}_best_model.pth"
                    self.save_checkpoint(checkpoint_name, {
                        'stage': stage_idx + 1,
                        'stage_name': stage['name'],
                        'epoch': epoch + 1,
                        'val_loss': val_metrics['loss']
                    })

                # Early stopping внутри этапа
                if self.patience_counter >= self.patience // 2:
                    self.logger.info(f"⚠️ Early stopping на этапе {stage_idx + 1}")
                    break

            # Сохраняем историю этапа
            self.stage_history.append(stage_metrics)

            # Отчет по этапу
            self._report_stage_results(stage_idx, stage_metrics)

            # Проверка на глобальный early stopping
            if self.patience_counter >= self.patience:
                self.logger.info("🛑 Глобальный early stopping")
                break

        # Финальный отчет
        self._report_final_results()

        return self.stage_history

    def _report_stage_results(self, stage_idx, metrics):
        """Выводит результаты этапа"""

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Результаты этапа {stage_idx + 1}: {metrics['stage_name']}")
        self.logger.info(f"   Лучший val_loss: {metrics['best_val_loss']:.4f}")
        self.logger.info(f"   Финальный train_loss: {metrics['train_losses'][-1]:.4f}")
        self.logger.info(f"   Улучшение: {(metrics['train_losses'][0] - metrics['train_losses'][-1]):.4f}")
        self.logger.info(f"{'='*60}")

    def _report_final_results(self):
        """Выводит финальные результаты обучения"""

        self.logger.info(f"\n{'='*60}")
        self.logger.info("🏁 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ CURRICULUM LEARNING")
        self.logger.info(f"{'='*60}")

        for idx, stage_metrics in enumerate(self.stage_history):
            self.logger.info(f"\n📚 Этап {idx + 1}: {stage_metrics['stage_name']}")
            self.logger.info(f"   Лучший val_loss: {stage_metrics['best_val_loss']:.4f}")
            self.logger.info(f"   Эпох пройдено: {len(stage_metrics['train_losses'])}")

        # Общее улучшение
        if self.stage_history:
            initial_loss = self.stage_history[0]['train_losses'][0]
            final_loss = self.stage_history[-1]['train_losses'][-1]
            self.logger.info(f"\n🎯 Общее улучшение:")
            self.logger.info(f"   Начальный loss: {initial_loss:.4f}")
            self.logger.info(f"   Финальный loss: {final_loss:.4f}")
            self.logger.info(f"   Снижение: {((initial_loss - final_loss) / initial_loss * 100):.1f}%")

        self.logger.info(f"{'='*60}")