#!/usr/bin/env python3
"""
Монитор обучения с полным отслеживанием всех вызовов и потоков
"""
import sys
import os
import time
import threading
import traceback
import functools
import inspect
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict
import json
import psutil
import torch
import torch.nn as nn
from contextlib import contextmanager

# Цвета для консоли
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TrainingMonitor:
    """Полный мониторинг процесса обучения"""

    def __init__(self):
        self.start_time = time.time()
        self.call_stack = []
        self.file_access_log = []
        self.error_log = []
        self.warning_log = []
        self.gpu_usage_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=1000)
        self.batch_times = deque(maxlen=100)
        self.current_phase = "initialization"
        self.files_accessed = defaultdict(int)
        self.function_calls = defaultdict(int)
        self.thread_info = {}

        # Создание директории для логов
        self.log_dir = Path("training_monitor_logs")
        self.log_dir.mkdir(exist_ok=True)

        # Файлы логов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log = self.log_dir / f"monitor_{timestamp}.log"
        self.trace_log = self.log_dir / f"trace_{timestamp}.log"
        self.error_log_file = self.log_dir / f"errors_{timestamp}.log"
        self.perf_log = self.log_dir / f"performance_{timestamp}.json"

        # Настройка логгеров
        self.setup_loggers()

        # Отслеживание проблемы схлопывания
        self.collapse_detector = {
            "predictions": [],
            "loss_values": [],
            "gradient_norms": [],
            "last_collapse_batch": None,
            "collapse_count": 0
        }

        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}🔍 TRAINING MONITOR АКТИВИРОВАН{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"📁 Логи сохраняются в: {self.log_dir}")
        print(f"📝 Основной лог: {self.main_log.name}")
        print(f"🔎 Трассировка: {self.trace_log.name}")
        print("")

    def setup_loggers(self):
        """Настройка всех логгеров"""
        # Основной логгер
        self.logger = logging.getLogger("TrainingMonitor")
        self.logger.setLevel(logging.DEBUG)

        # Файловый хэндлер для основных логов
        fh_main = logging.FileHandler(self.main_log)
        fh_main.setLevel(logging.INFO)

        # Файловый хэндлер для трассировки
        fh_trace = logging.FileHandler(self.trace_log)
        fh_trace.setLevel(logging.DEBUG)

        # Файловый хэндлер для ошибок
        fh_error = logging.FileHandler(self.error_log_file)
        fh_error.setLevel(logging.WARNING)

        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S.%f'
        )

        fh_main.setFormatter(formatter)
        fh_trace.setFormatter(formatter)
        fh_error.setFormatter(formatter)

        self.logger.addHandler(fh_main)
        self.logger.addHandler(fh_trace)
        self.logger.addHandler(fh_error)

        # Отдельный логгер для трассировки
        self.trace_logger = logging.getLogger("Trace")
        self.trace_logger.setLevel(logging.DEBUG)
        self.trace_logger.addHandler(fh_trace)

    def log_call(self, func_name: str, file_path: str, line_no: int, args: tuple = None, kwargs: dict = None):
        """Логирование вызова функции"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name

        call_info = {
            "time": time.time() - self.start_time,
            "thread": f"{thread_name}({thread_id})",
            "function": func_name,
            "file": file_path,
            "line": line_no
        }

        self.call_stack.append(call_info)
        self.function_calls[func_name] += 1
        self.files_accessed[file_path] += 1

        # Логирование в trace
        self.trace_logger.debug(
            f"[{thread_name}] {file_path}:{line_no} -> {func_name}()"
        )

        # Вывод важных вызовов в консоль
        if any(key in func_name.lower() for key in ['forward', 'backward', 'loss', 'optimizer', 'train', 'eval']):
            elapsed = time.time() - self.start_time
            print(f"{Colors.OKCYAN}[{elapsed:7.2f}s] {file_path.split('/')[-1]}:{line_no} -> {func_name}(){Colors.ENDC}")

    def check_collapse(self, outputs: Optional[torch.Tensor] = None, loss: Optional[float] = None,
                      batch_idx: Optional[int] = None):
        """Проверка на схлопывание модели"""
        if outputs is not None and isinstance(outputs, torch.Tensor):
            # Анализ предсказаний
            if outputs.dim() >= 2:  # Проверяем что есть классы
                with torch.no_grad():
                    # Получаем предсказания
                    if outputs.shape[-1] == 3:  # 3 класса (LONG, SHORT, FLAT)
                        probs = torch.softmax(outputs, dim=-1)
                        preds = torch.argmax(probs, dim=-1)

                        # Подсчет распределения
                        total = preds.numel()
                        flat_count = (preds == 2).sum().item()  # FLAT обычно индекс 2
                        flat_ratio = flat_count / total if total > 0 else 0

                        # Детектирование схлопывания
                        if flat_ratio > 0.9:  # Более 90% FLAT
                            self.collapse_detector["collapse_count"] += 1
                            self.collapse_detector["last_collapse_batch"] = batch_idx

                            # КРИТИЧЕСКОЕ предупреждение
                            msg = (f"🚨 СХЛОПЫВАНИЕ ОБНАРУЖЕНО! Батч {batch_idx}: "
                                  f"{flat_ratio:.1%} предсказаний FLAT")

                            print(f"\n{Colors.FAIL}{Colors.BOLD}{msg}{Colors.ENDC}\n")
                            self.logger.critical(msg)

                            # Детальная диагностика
                            self.diagnose_collapse(outputs, preds, probs, batch_idx)

                            return True

                        # Предупреждение о риске схлопывания
                        elif flat_ratio > 0.7:
                            msg = f"⚠️ Риск схлопывания! Батч {batch_idx}: {flat_ratio:.1%} FLAT"
                            print(f"{Colors.WARNING}{msg}{Colors.ENDC}")
                            self.logger.warning(msg)

        # Проверка loss
        if loss is not None:
            self.collapse_detector["loss_values"].append(loss)

            # Проверка на нулевой или очень малый loss
            if loss < 1e-7:
                msg = f"⚠️ Очень малый loss: {loss:.2e} на батче {batch_idx}"
                print(f"{Colors.WARNING}{msg}{Colors.ENDC}")
                self.logger.warning(msg)

        return False

    def diagnose_collapse(self, outputs, preds, probs, batch_idx):
        """Детальная диагностика схлопывания"""
        print(f"\n{Colors.FAIL}{'='*60}")
        print(f"📊 ДЕТАЛЬНАЯ ДИАГНОСТИКА СХЛОПЫВАНИЯ")
        print(f"{'='*60}{Colors.ENDC}")

        # Статистика по классам
        unique, counts = torch.unique(preds, return_counts=True)
        for i, count in zip(unique.tolist(), counts.tolist()):
            class_name = ['LONG', 'SHORT', 'FLAT'][i] if i < 3 else f'Class_{i}'
            ratio = count / len(preds)
            bar = '█' * int(ratio * 40)
            print(f"{class_name:6s}: {bar:40s} {count:5d} ({ratio:6.2%})")

        # Статистика уверенности
        max_probs = probs.max(dim=-1)[0]
        print(f"\n📈 Уверенность предсказаний:")
        print(f"  Средняя: {max_probs.mean():.3f}")
        print(f"  Мин:     {max_probs.min():.3f}")
        print(f"  Макс:    {max_probs.max():.3f}")
        print(f"  Std:     {max_probs.std():.3f}")

        # Проверка логитов
        print(f"\n🔢 Статистика логитов (outputs):")
        print(f"  Mean: {outputs.mean():.3f}")
        print(f"  Std:  {outputs.std():.3f}")
        print(f"  Min:  {outputs.min():.3f}")
        print(f"  Max:  {outputs.max():.3f}")

        # Сохранение в файл
        collapse_data = {
            "batch_idx": batch_idx,
            "timestamp": datetime.now().isoformat(),
            "class_distribution": {
                i: count.item() for i, count in zip(unique.tolist(), counts.tolist())
            },
            "confidence": {
                "mean": float(max_probs.mean()),
                "std": float(max_probs.std()),
                "min": float(max_probs.min()),
                "max": float(max_probs.max())
            },
            "logits": {
                "mean": float(outputs.mean()),
                "std": float(outputs.std()),
                "min": float(outputs.min()),
                "max": float(outputs.max())
            }
        }

        collapse_file = self.log_dir / f"collapse_batch_{batch_idx}.json"
        with open(collapse_file, 'w') as f:
            json.dump(collapse_data, f, indent=2)

        print(f"\n💾 Диагностика сохранена в: {collapse_file.name}")
        print(f"{Colors.FAIL}{'='*60}{Colors.ENDC}\n")

    def monitor_resources(self):
        """Мониторинг системных ресурсов"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory
            mem = psutil.virtual_memory()

            # GPU (если доступно)
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_util = (gpu_mem / gpu_mem_total) * 100
                gpu_info = f" | GPU: {gpu_mem:.1f}/{gpu_mem_total:.1f}GB ({gpu_util:.0f}%)"

                self.gpu_usage_history.append({
                    "time": time.time() - self.start_time,
                    "memory_gb": gpu_mem,
                    "utilization": gpu_util
                })

            # Вывод только при изменениях
            if len(self.gpu_usage_history) % 10 == 0:  # Каждые 10 проверок
                print(f"{Colors.OKBLUE}📊 CPU: {cpu_percent:.0f}% | RAM: {mem.percent:.0f}%{gpu_info}{Colors.ENDC}")

        except Exception as e:
            self.trace_logger.error(f"Resource monitoring error: {e}")

    def wrap_function(self, func):
        """Обертка для отслеживания вызовов функций"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Получаем информацию о вызове
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            filename = caller_frame.f_code.co_filename
            line_no = caller_frame.f_lineno
            func_name = func.__name__

            # Логируем вызов
            self.log_call(func_name, filename, line_no, args, kwargs)

            # Мониторинг ресурсов
            self.monitor_resources()

            try:
                # Вызов оригинальной функции
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                # Логирование времени выполнения для важных функций
                if elapsed > 0.1:  # Функции дольше 100ms
                    self.logger.info(f"⏱️ {func_name} took {elapsed:.3f}s")

                # Проверка на схлопывание для forward/loss функций
                if 'forward' in func_name.lower() and result is not None:
                    if isinstance(result, torch.Tensor):
                        self.check_collapse(outputs=result)

                return result

            except Exception as e:
                # Логирование ошибки
                error_msg = f"❌ ERROR in {func_name}: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
                print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
                self.error_log.append({
                    "time": time.time() - self.start_time,
                    "function": func_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                raise

        return wrapper

    def patch_training_functions(self):
        """Патчинг критических функций для мониторинга"""
        import training.optimized_trainer as trainer_module
        import models.patchtst_unified as model_module

        # Список функций для патчинга
        modules_to_patch = [
            (trainer_module, ['train_epoch', 'evaluate', '_train_batch']),
            (model_module, ['forward']),
        ]

        for module, func_names in modules_to_patch:
            for func_name in func_names:
                if hasattr(module, func_name):
                    original_func = getattr(module, func_name)
                    wrapped_func = self.wrap_function(original_func)
                    setattr(module, func_name, wrapped_func)
                    self.logger.info(f"✅ Patched {module.__name__}.{func_name}")

    def save_performance_report(self):
        """Сохранение отчета о производительности"""
        report = {
            "total_time": time.time() - self.start_time,
            "function_calls": dict(self.function_calls),
            "files_accessed": dict(self.files_accessed),
            "errors": self.error_log,
            "warnings": self.warning_log,
            "collapse_events": {
                "count": self.collapse_detector["collapse_count"],
                "last_batch": self.collapse_detector["last_collapse_batch"]
            },
            "gpu_usage": list(self.gpu_usage_history)[-50:],  # Последние 50 записей
            "batch_times": list(self.batch_times)[-50:]
        }

        with open(self.perf_log, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Отчет сохранен в: {self.perf_log}")

    def __del__(self):
        """Финализация при завершении"""
        try:
            self.save_performance_report()
            print(f"\n{Colors.OKGREEN}✅ Мониторинг завершен. Логи в {self.log_dir}{Colors.ENDC}")
        except:
            pass


# Глобальный экземпляр монитора
monitor = None


def run_training_with_monitor():
    """Запуск обучения с полным мониторингом"""
    global monitor

    print(f"{Colors.BOLD}🚀 ЗАПУСК ОБУЧЕНИЯ С ПОЛНЫМ МОНИТОРИНГОМ{Colors.ENDC}\n")

    # Создаем монитор
    monitor = TrainingMonitor()

    # Патчим функции для отслеживания
    monitor.patch_training_functions()

    # Добавляем текущую директорию в path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        # Устанавливаем аргументы
        sys.argv = ['main.py', '--mode', 'staged']

        # Импортируем main
        import main

        print(f"{Colors.OKGREEN}✅ Main модуль загружен{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

        # Запускаем обучение
        main.main()

    except Exception as e:
        print(f"\n{Colors.FAIL}❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}{Colors.ENDC}")
        print(traceback.format_exc())
        monitor.logger.critical(f"CRITICAL ERROR: {str(e)}")
        monitor.logger.critical(traceback.format_exc())

    finally:
        # Сохраняем отчет
        if monitor:
            monitor.save_performance_report()

            # Выводим итоговую статистику
            print(f"\n{Colors.HEADER}{'='*80}")
            print(f"📊 ИТОГОВАЯ СТАТИСТИКА")
            print(f"{'='*80}{Colors.ENDC}")

            print(f"⏱️ Общее время: {time.time() - monitor.start_time:.1f}с")
            print(f"📁 Файлов обращений: {len(monitor.files_accessed)}")
            print(f"🔧 Вызовов функций: {sum(monitor.function_calls.values())}")
            print(f"❌ Ошибок: {len(monitor.error_log)}")
            print(f"⚠️ Предупреждений: {len(monitor.warning_log)}")
            print(f"💥 Схлопываний: {monitor.collapse_detector['collapse_count']}")

            # Топ вызываемых функций
            print(f"\n📈 Топ-5 вызываемых функций:")
            for func, count in sorted(monitor.function_calls.items(),
                                     key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {func}: {count} вызовов")

            print(f"\n{Colors.OKGREEN}✅ Мониторинг завершен{Colors.ENDC}")
            print(f"📁 Все логи в: {monitor.log_dir}/")


if __name__ == "__main__":
    run_training_with_monitor()