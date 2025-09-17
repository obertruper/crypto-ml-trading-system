#!/usr/bin/env python3
"""
Менеджер диагностики обучения - находит точную причину проблем
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import traceback
from typing import Dict, Any, List, Optional, Tuple
import logging
import gc

class TrainingDiagnosticManager:
    """Полная диагностика каждого шага обучения"""

    def __init__(self, log_dir: str = "diagnostic_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Создаем файл логов с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"diagnostic_{timestamp}.log"
        self.json_log = self.log_dir / f"diagnostic_{timestamp}.json"

        # Настройка логгера
        self.setup_logger()

        # Хранилище диагностики
        self.diagnostics = {
            "start_time": timestamp,
            "steps": [],
            "errors": [],
            "warnings": [],
            "data_checks": [],
            "model_checks": [],
            "gradient_checks": [],
            "loss_history": [],
            "prediction_distribution": []
        }

    def setup_logger(self):
        """Настройка детального логгирования"""
        self.logger = logging.getLogger("DiagnosticManager")
        self.logger.setLevel(logging.DEBUG)

        # Файловый хэндлер
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)

        # Консольный хэндлер
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Формат
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def check_data_loader(self, dataloader) -> Dict[str, Any]:
        """Проверка DataLoader"""
        self.logger.info("=" * 80)
        self.logger.info("🔍 ПРОВЕРКА DATALOADER")
        self.logger.info("=" * 80)

        result = {
            "timestamp": datetime.now().isoformat(),
            "check_type": "dataloader",
            "status": "ok",
            "details": {}
        }

        try:
            # Получаем первый батч
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            else:
                inputs = batch
                targets = None

            # Анализ входных данных
            result["details"]["batch_size"] = inputs.shape[0]
            result["details"]["sequence_length"] = inputs.shape[1] if len(inputs.shape) > 2 else None
            result["details"]["feature_dim"] = inputs.shape[-1]
            result["details"]["input_shape"] = str(inputs.shape)
            result["details"]["input_dtype"] = str(inputs.dtype)
            result["details"]["input_device"] = str(inputs.device)

            # Статистика входных данных
            result["details"]["input_stats"] = {
                "mean": float(inputs.mean().item()),
                "std": float(inputs.std().item()),
                "min": float(inputs.min().item()),
                "max": float(inputs.max().item()),
                "nan_count": int(torch.isnan(inputs).sum().item()),
                "inf_count": int(torch.isinf(inputs).sum().item()),
                "zero_ratio": float((inputs == 0).float().mean().item())
            }

            # Проверка targets
            if targets is not None:
                result["details"]["target_shape"] = str(targets.shape)
                result["details"]["target_dtype"] = str(targets.dtype)

                # Для классификации
                if targets.dtype in [torch.long, torch.int32, torch.int64]:
                    unique_targets = torch.unique(targets)
                    result["details"]["unique_targets"] = unique_targets.tolist()
                    result["details"]["target_distribution"] = {
                        str(int(t)): int((targets == t).sum().item())
                        for t in unique_targets
                    }

                    # Проверка баланса классов
                    counts = [int((targets == t).sum().item()) for t in unique_targets]
                    max_count = max(counts)
                    min_count = min(counts)
                    imbalance_ratio = max_count / max(min_count, 1)
                    result["details"]["class_imbalance_ratio"] = float(imbalance_ratio)

                    if imbalance_ratio > 10:
                        self.logger.warning(f"⚠️ СИЛЬНЫЙ ДИСБАЛАНС КЛАССОВ: {imbalance_ratio:.1f}x")
                        result["warnings"] = result.get("warnings", [])
                        result["warnings"].append(f"Class imbalance: {imbalance_ratio:.1f}x")

                # Для регрессии
                else:
                    result["details"]["target_stats"] = {
                        "mean": float(targets.mean().item()),
                        "std": float(targets.std().item()),
                        "min": float(targets.min().item()),
                        "max": float(targets.max().item())
                    }

            # Логирование результатов
            self.logger.info(f"✅ Batch size: {result['details']['batch_size']}")
            self.logger.info(f"✅ Input shape: {result['details']['input_shape']}")
            self.logger.info(f"✅ Feature dim: {result['details']['feature_dim']}")
            self.logger.info(f"✅ Input stats: mean={result['details']['input_stats']['mean']:.4f}, "
                           f"std={result['details']['input_stats']['std']:.4f}")

            if "target_distribution" in result["details"]:
                self.logger.info(f"✅ Target distribution: {result['details']['target_distribution']}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            self.logger.error(f"❌ ОШИБКА В DATALOADER: {str(e)}")
            self.logger.debug(traceback.format_exc())

        self.diagnostics["data_checks"].append(result)
        return result

    def check_model_initialization(self, model: nn.Module) -> Dict[str, Any]:
        """Проверка инициализации модели"""
        self.logger.info("=" * 80)
        self.logger.info("🔍 ПРОВЕРКА МОДЕЛИ")
        self.logger.info("=" * 80)

        result = {
            "timestamp": datetime.now().isoformat(),
            "check_type": "model_init",
            "status": "ok",
            "details": {}
        }

        try:
            # Базовая информация о модели
            result["details"]["model_class"] = model.__class__.__name__
            result["details"]["device"] = str(next(model.parameters()).device)

            # Подсчет параметров
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            result["details"]["total_params"] = total_params
            result["details"]["trainable_params"] = trainable_params
            result["details"]["frozen_params"] = total_params - trainable_params

            # Проверка параметров
            param_stats = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    stats = {
                        "name": name,
                        "shape": str(param.shape),
                        "mean": float(param.mean().item()),
                        "std": float(param.std().item()),
                        "min": float(param.min().item()),
                        "max": float(param.max().item()),
                        "nan_count": int(torch.isnan(param).sum().item()),
                        "inf_count": int(torch.isinf(param).sum().item()),
                        "zero_ratio": float((param == 0).float().mean().item())
                    }
                    param_stats.append(stats)

                    # Проверка на проблемы
                    if stats["nan_count"] > 0:
                        self.logger.error(f"❌ NaN в параметре {name}")
                        result["status"] = "error"
                    if stats["inf_count"] > 0:
                        self.logger.error(f"❌ Inf в параметре {name}")
                        result["status"] = "error"
                    if stats["std"] < 1e-6:
                        self.logger.warning(f"⚠️ Очень малая дисперсия в {name}: {stats['std']:.2e}")
                    if stats["zero_ratio"] > 0.9:
                        self.logger.warning(f"⚠️ Много нулей в {name}: {stats['zero_ratio']:.1%}")

            result["details"]["param_stats"] = param_stats[:10]  # Первые 10 для краткости

            # Проверка выходного слоя
            if hasattr(model, 'output_projection') or hasattr(model, 'fc') or hasattr(model, 'classifier'):
                output_layer = getattr(model, 'output_projection',
                                     getattr(model, 'fc',
                                           getattr(model, 'classifier', None)))
                if output_layer:
                    if hasattr(output_layer, 'out_features'):
                        result["details"]["output_size"] = output_layer.out_features
                        self.logger.info(f"✅ Output size: {output_layer.out_features}")

            self.logger.info(f"✅ Model: {result['details']['model_class']}")
            self.logger.info(f"✅ Total params: {total_params:,}")
            self.logger.info(f"✅ Trainable params: {trainable_params:,}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            self.logger.error(f"❌ ОШИБКА ПРОВЕРКИ МОДЕЛИ: {str(e)}")

        self.diagnostics["model_checks"].append(result)
        return result

    def check_forward_pass(self, model: nn.Module, inputs: torch.Tensor) -> Dict[str, Any]:
        """Проверка forward pass"""
        self.logger.info("=" * 80)
        self.logger.info("🔍 ПРОВЕРКА FORWARD PASS")
        self.logger.info("=" * 80)

        result = {
            "timestamp": datetime.now().isoformat(),
            "check_type": "forward_pass",
            "status": "ok",
            "details": {}
        }

        try:
            model.eval()
            with torch.no_grad():
                # Forward pass
                outputs = model(inputs)

                # Анализ выходов
                if isinstance(outputs, torch.Tensor):
                    result["details"]["output_shape"] = str(outputs.shape)
                    result["details"]["output_dtype"] = str(outputs.dtype)
                    result["details"]["output_stats"] = {
                        "mean": float(outputs.mean().item()),
                        "std": float(outputs.std().item()),
                        "min": float(outputs.min().item()),
                        "max": float(outputs.max().item()),
                        "nan_count": int(torch.isnan(outputs).sum().item()),
                        "inf_count": int(torch.isinf(outputs).sum().item())
                    }

                    # Проверка распределения предсказаний для классификации
                    if outputs.dim() == 2 and outputs.shape[-1] <= 10:  # Вероятно классификация
                        probs = torch.softmax(outputs, dim=-1)
                        preds = torch.argmax(probs, dim=-1)
                        unique_preds = torch.unique(preds)

                        result["details"]["unique_predictions"] = unique_preds.tolist()
                        result["details"]["prediction_distribution"] = {
                            str(int(p)): int((preds == p).sum().item())
                            for p in unique_preds
                        }

                        # Проверка на схлопывание
                        max_class_ratio = max(result["details"]["prediction_distribution"].values()) / len(preds)
                        if max_class_ratio > 0.9:
                            self.logger.error(f"❌ СХЛОПЫВАНИЕ! {max_class_ratio:.1%} предсказаний в одном классе")
                            result["status"] = "collapsed"
                            result["error"] = f"Model collapsed: {max_class_ratio:.1%} in one class"

                        # Проверка уверенности
                        max_probs = probs.max(dim=-1)[0]
                        result["details"]["confidence_stats"] = {
                            "mean": float(max_probs.mean().item()),
                            "std": float(max_probs.std().item()),
                            "min": float(max_probs.min().item()),
                            "max": float(max_probs.max().item())
                        }

                    self.logger.info(f"✅ Output shape: {result['details']['output_shape']}")
                    self.logger.info(f"✅ Output stats: mean={result['details']['output_stats']['mean']:.4f}, "
                                   f"std={result['details']['output_stats']['std']:.4f}")

                    if "prediction_distribution" in result["details"]:
                        self.logger.info(f"📊 Predictions: {result['details']['prediction_distribution']}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            self.logger.error(f"❌ ОШИБКА FORWARD PASS: {str(e)}")
            self.logger.debug(traceback.format_exc())

        self.diagnostics["model_checks"].append(result)
        return result

    def check_loss_computation(self, outputs: torch.Tensor, targets: torch.Tensor,
                              criterion: nn.Module) -> Dict[str, Any]:
        """Проверка вычисления loss"""
        self.logger.info("=" * 80)
        self.logger.info("🔍 ПРОВЕРКА LOSS")
        self.logger.info("=" * 80)

        result = {
            "timestamp": datetime.now().isoformat(),
            "check_type": "loss_computation",
            "status": "ok",
            "details": {}
        }

        try:
            # Вычисление loss
            loss = criterion(outputs, targets)

            result["details"]["loss_value"] = float(loss.item())
            result["details"]["loss_finite"] = bool(torch.isfinite(loss).item())
            result["details"]["criterion_type"] = criterion.__class__.__name__

            # Проверка на проблемы
            if not result["details"]["loss_finite"]:
                self.logger.error("❌ Loss is NaN or Inf!")
                result["status"] = "error"
                result["error"] = "Non-finite loss"

            elif result["details"]["loss_value"] < 1e-8:
                self.logger.warning(f"⚠️ Очень малый loss: {result['details']['loss_value']:.2e}")
                result["warnings"] = ["Very small loss"]

            elif result["details"]["loss_value"] > 100:
                self.logger.warning(f"⚠️ Очень большой loss: {result['details']['loss_value']:.2f}")
                result["warnings"] = ["Very large loss"]

            self.logger.info(f"✅ Loss: {result['details']['loss_value']:.6f}")
            self.logger.info(f"✅ Criterion: {result['details']['criterion_type']}")

            # Дополнительная диагностика для CrossEntropy
            if 'CrossEntropy' in result["details"]["criterion_type"]:
                with torch.no_grad():
                    probs = torch.softmax(outputs, dim=-1)
                    preds = torch.argmax(probs, dim=-1)
                    accuracy = (preds == targets).float().mean().item()
                    result["details"]["accuracy"] = float(accuracy)
                    self.logger.info(f"✅ Accuracy: {accuracy:.2%}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            self.logger.error(f"❌ ОШИБКА LOSS: {str(e)}")

        self.diagnostics["loss_history"].append(result)
        return result

    def check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Проверка градиентов после backward"""
        self.logger.info("=" * 80)
        self.logger.info("🔍 ПРОВЕРКА ГРАДИЕНТОВ")
        self.logger.info("=" * 80)

        result = {
            "timestamp": datetime.now().isoformat(),
            "check_type": "gradients",
            "status": "ok",
            "details": {}
        }

        try:
            grad_stats = []
            zero_grad_params = []
            large_grad_params = []

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad
                    stats = {
                        "name": name,
                        "shape": str(grad.shape),
                        "mean": float(grad.mean().item()),
                        "std": float(grad.std().item()),
                        "min": float(grad.min().item()),
                        "max": float(grad.max().item()),
                        "norm": float(grad.norm().item()),
                        "nan_count": int(torch.isnan(grad).sum().item()),
                        "inf_count": int(torch.isinf(grad).sum().item())
                    }
                    grad_stats.append(stats)

                    # Проверка на проблемы
                    if stats["nan_count"] > 0:
                        self.logger.error(f"❌ NaN градиенты в {name}")
                        result["status"] = "error"
                    if stats["inf_count"] > 0:
                        self.logger.error(f"❌ Inf градиенты в {name}")
                        result["status"] = "error"
                    if abs(stats["mean"]) < 1e-8 and stats["std"] < 1e-8:
                        zero_grad_params.append(name)
                    if stats["norm"] > 100:
                        large_grad_params.append(name)

            result["details"]["grad_stats"] = grad_stats[:10]  # Первые 10
            result["details"]["zero_grad_params_count"] = len(zero_grad_params)
            result["details"]["large_grad_params_count"] = len(large_grad_params)

            if zero_grad_params:
                self.logger.warning(f"⚠️ {len(zero_grad_params)} параметров с нулевыми градиентами")
                result["details"]["zero_grad_params"] = zero_grad_params[:5]

            if large_grad_params:
                self.logger.warning(f"⚠️ {len(large_grad_params)} параметров с большими градиентами")
                result["details"]["large_grad_params"] = large_grad_params[:5]

            # Общая статистика
            if grad_stats:
                avg_norm = np.mean([s["norm"] for s in grad_stats])
                max_norm = max(s["norm"] for s in grad_stats)
                result["details"]["avg_grad_norm"] = float(avg_norm)
                result["details"]["max_grad_norm"] = float(max_norm)
                self.logger.info(f"✅ Avg grad norm: {avg_norm:.4f}")
                self.logger.info(f"✅ Max grad norm: {max_norm:.4f}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            self.logger.error(f"❌ ОШИБКА ГРАДИЕНТОВ: {str(e)}")

        self.diagnostics["gradient_checks"].append(result)
        return result

    def save_diagnostics(self):
        """Сохранение диагностики в файл"""
        with open(self.json_log, 'w') as f:
            json.dump(self.diagnostics, f, indent=2)
        self.logger.info(f"📝 Диагностика сохранена в {self.json_log}")

    def summary(self):
        """Вывод итогового отчета"""
        self.logger.info("=" * 80)
        self.logger.info("📊 ИТОГОВЫЙ ОТЧЕТ ДИАГНОСТИКИ")
        self.logger.info("=" * 80)

        # Подсчет проблем
        errors = []
        warnings = []

        for checks in [self.diagnostics["data_checks"],
                      self.diagnostics["model_checks"],
                      self.diagnostics["gradient_checks"],
                      self.diagnostics["loss_history"]]:
            for check in checks:
                if check.get("status") == "error":
                    errors.append(f"{check['check_type']}: {check.get('error', 'unknown')}")
                elif check.get("status") == "collapsed":
                    errors.append(f"{check['check_type']}: MODEL COLLAPSED")
                if "warnings" in check:
                    warnings.extend(check["warnings"])

        self.logger.info(f"❌ Ошибок найдено: {len(errors)}")
        for error in errors[:5]:
            self.logger.error(f"  • {error}")

        self.logger.info(f"⚠️ Предупреждений: {len(warnings)}")
        for warning in warnings[:5]:
            self.logger.warning(f"  • {warning}")

        # Рекомендации
        self.logger.info("\n💡 РЕКОМЕНДАЦИИ:")

        if any("collapsed" in str(e).lower() for e in errors):
            self.logger.info("  1. Модель схлопывается - проверьте:")
            self.logger.info("     • Инициализацию весов")
            self.logger.info("     • Learning rate (попробуйте уменьшить)")
            self.logger.info("     • Баланс классов в данных")
            self.logger.info("     • Loss function и веса классов")

        if any("nan" in str(e).lower() or "inf" in str(e).lower() for e in errors):
            self.logger.info("  2. Численная нестабильность:")
            self.logger.info("     • Добавьте gradient clipping")
            self.logger.info("     • Уменьшите learning rate")
            self.logger.info("     • Проверьте нормализацию данных")

        if len(warnings) > 0:
            self.logger.info("  3. Есть предупреждения - проверьте логи выше")

        self.save_diagnostics()


def run_diagnostic_training():
    """Запуск обучения с полной диагностикой"""
    import argparse
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))

    # Инициализация диагностики
    diagnostics = TrainingDiagnosticManager()

    try:
        diagnostics.logger.info("🚀 ЗАПУСК ДИАГНОСТИКИ ОБУЧЕНИЯ")

        # Импорты
        from data.data_loader import create_dataloaders
        from models.patchtst_unified import UnifiedPatchTST
        from training.optimized_trainer import OptimizedTrainer
        from config.config import load_config

        # Загрузка конфигурации
        config = load_config("config/config.yaml")
        diagnostics.logger.info("✅ Конфигурация загружена")

        # Создание DataLoader
        diagnostics.logger.info("📦 Создание DataLoader...")
        train_loader, val_loader, test_loader, feature_columns, num_features = create_dataloaders(
            config=config,
            batch_size=config.training.batch_size,
            num_workers=0,  # Для диагностики без многопоточности
            pin_memory=False
        )

        # Проверка DataLoader
        diagnostics.check_data_loader(train_loader)

        # Создание модели
        diagnostics.logger.info("🏗️ Создание модели...")
        model = UnifiedPatchTST(
            num_features=num_features,
            pred_len=config.model.pred_len,
            patch_len=config.model.patch_len,
            stride=config.model.stride,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            n_layers=config.model.n_layers,
            dropout=config.model.dropout,
            activation=config.model.activation,
            norm_type=config.model.norm_type,
            num_targets=config.data.num_targets
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        diagnostics.logger.info(f"✅ Модель на устройстве: {device}")

        # Проверка модели
        diagnostics.check_model_initialization(model)

        # Тестовый forward pass
        batch = next(iter(train_loader))
        inputs = batch[0].to(device)
        targets = batch[1].to(device) if len(batch) > 1 else None

        diagnostics.check_forward_pass(model, inputs)

        # Проверка loss
        if targets is not None:
            outputs = model(inputs)
            criterion = nn.CrossEntropyLoss()
            diagnostics.check_loss_computation(outputs, targets, criterion)

            # Backward pass
            model.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()

            # Проверка градиентов
            diagnostics.check_gradients(model)

        # Итоговый отчет
        diagnostics.summary()

    except Exception as e:
        diagnostics.logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        diagnostics.logger.error(traceback.format_exc())
        diagnostics.diagnostics["errors"].append({
            "type": "critical",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        diagnostics.save_diagnostics()
        sys.exit(1)


if __name__ == "__main__":
    run_diagnostic_training()