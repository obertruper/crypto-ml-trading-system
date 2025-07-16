"""
Продвинутая система логирования для AI торговой системы
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import colorlog
from pythonjsonlogger import jsonlogger

class TradingLogger:
    """Централизованная система логирования для торговой системы"""
    
    def __init__(self, name: str, config_path: str = "config/config.yaml", is_subprocess: bool = False):
        self.name = name
        self.config = self._load_config(config_path)
        self.is_subprocess = is_subprocess  # Флаг для отключения консольного вывода в подпроцессах
        self.logger = self._setup_logger()
        self.stage_timers = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        import yaml
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Fallback конфигурация
            return {
                'logging': {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'log_dir': 'experiments/logs'
                }
            }
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера с несколькими обработчиками"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.config['logging']['level'])
        logger.handlers = []
        
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # В подпроцессах отключаем консольный вывод для предотвращения дублирования
        if 'console' in self.config['logging']['handlers'] and not self.is_subprocess:
            console_handler = self._create_console_handler()
            logger.addHandler(console_handler)
        
        if 'file' in self.config['logging']['handlers']:
            file_handler = self._create_file_handler(log_dir)
            logger.addHandler(file_handler)
        
        # JSON логирование только в основном процессе
        if not self.is_subprocess:
            json_handler = self._create_json_handler(log_dir)
            logger.addHandler(json_handler)
        
        return logger
    
    def _create_console_handler(self) -> logging.Handler:
        """Создание цветного консольного обработчика"""
        console_handler = colorlog.StreamHandler(sys.stdout)
        
        console_format = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        console_handler.setFormatter(console_format)
        return console_handler
    
    def _create_file_handler(self, log_dir: Path) -> logging.Handler:
        """Создание файлового обработчика с ротацией"""
        log_file = log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(file_format)
        return file_handler
    
    def _create_json_handler(self, log_dir: Path) -> logging.Handler:
        """Создание JSON обработчика для структурированных логов"""
        json_file = log_dir / f"{self.name}_structured_{datetime.now().strftime('%Y%m%d')}.json"
        
        json_handler = logging.handlers.RotatingFileHandler(
            json_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5,
            encoding='utf-8'
        )
        
        json_formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            rename_fields={'timestamp': '@timestamp', 'level': 'level'}
        )
        
        json_handler.setFormatter(json_formatter)
        return json_handler
    
    def start_stage(self, stage_name: str, **kwargs):
        """Начало этапа обработки"""
        self.stage_timers[stage_name] = datetime.now()
        
        self.logger.info(
            f"🚀 Начало этапа: {stage_name}",
            extra={
                'stage': stage_name,
                'stage_type': 'start',
                'parameters': kwargs
            }
        )
    
    def end_stage(self, stage_name: str, **results):
        """Завершение этапа обработки"""
        if stage_name in self.stage_timers:
            duration = (datetime.now() - self.stage_timers[stage_name]).total_seconds()
            
            self.logger.info(
                f"✅ Завершение этапа: {stage_name} (время: {duration:.2f}с)",
                extra={
                    'stage': stage_name,
                    'stage_type': 'end',
                    'duration': duration,
                    'results': results
                }
            )
            
            del self.stage_timers[stage_name]
    
    def log_model_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Логирование метрик модели"""
        # Специальное форматирование для разных метрик
        formatted_metrics = []
        for k, v in metrics.items():
            if k == 'learning_rate':
                # Используем научную нотацию для малых значений LR
                formatted_metrics.append(f"{k}: {v:.2e}")
            elif k == 'epoch_time':
                # Время в секундах с 2 знаками
                formatted_metrics.append(f"{k}: {v:.2f}s")
            else:
                # Остальные метрики с 4 знаками
                formatted_metrics.append(f"{k}: {v:.4f}")
        
        metrics_str = " | ".join(formatted_metrics)
        
        self.logger.info(
            f"📊 Эпоха {epoch} | {metrics_str}",
            extra={
                'epoch': epoch,
                'metrics': metrics,
                'log_type': 'model_metrics'
            }
        )
    
    def log_trade(self, symbol: str, action: str, price: float, 
                  stop_loss: float, take_profit: float, confidence: float):
        """Логирование торговой операции"""
        risk_reward = (take_profit - price) / (price - stop_loss)
        
        self.logger.info(
            f"💰 {action.upper()} {symbol} @ {price:.4f} | "
            f"SL: {stop_loss:.4f} | TP: {take_profit:.4f} | "
            f"RR: {risk_reward:.2f} | Conf: {confidence:.2%}",
            extra={
                'trade_type': 'signal',
                'symbol': symbol,
                'action': action,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward,
                'confidence': confidence
            }
        )
    
    def log_backtest_results(self, results: Dict[str, Any]):
        """Логирование результатов бэктеста"""
        self.logger.info(
            f"📈 Результаты бэктеста:\n"
            f"   - Общая доходность: {results.get('total_return', 0):.2%}\n"
            f"   - Коэффициент Шарпа: {results.get('sharpe_ratio', 0):.2f}\n"
            f"   - Максимальная просадка: {results.get('max_drawdown', 0):.2%}\n"
            f"   - Win Rate: {results.get('win_rate', 0):.2%}\n"
            f"   - Profit Factor: {results.get('profit_factor', 0):.2f}",
            extra={
                'log_type': 'backtest_results',
                'results': results
            }
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Логирование ошибок с контекстом"""
        self.logger.error(
            f"❌ Ошибка в {context}: {type(error).__name__}: {str(error)}",
            exc_info=True,
            extra={
                'error_type': type(error).__name__,
                'error_context': context
            }
        )
    
    def log_data_info(self, symbol: str, records: int, date_range: tuple):
        """Логирование информации о данных"""
        self.logger.info(
            f"📊 Данные {symbol}: {records:,} записей | "
            f"Период: {date_range[0]} - {date_range[1]}",
            extra={
                'log_type': 'data_info',
                'symbol': symbol,
                'records': records,
                'start_date': str(date_range[0]),
                'end_date': str(date_range[1])
            }
        )
    
    def log_feature_importance(self, features: Dict[str, float], top_n: int = 10):
        """Логирование важности признаков"""
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features_str = "\n".join([f"     {i+1}. {name}: {score:.4f}" 
                                  for i, (name, score) in enumerate(sorted_features)])
        
        self.logger.info(
            f"🔍 Топ-{top_n} важных признаков:\n{features_str}",
            extra={
                'log_type': 'feature_importance',
                'features': dict(sorted_features)
            }
        )
    
    def debug(self, message: str, **kwargs):
        """Debug уровень"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Info уровень"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning уровень"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exc_info=None, **kwargs):
        """Error уровень"""
        if exc_info is not None:
            self.logger.error(message, exc_info=exc_info, extra=kwargs)
        else:
            self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical уровень"""
        self.logger.critical(message, extra=kwargs)


def get_logger(name: str, is_subprocess: bool = False) -> TradingLogger:
    """Получить экземпляр логгера"""
    return TradingLogger(name, is_subprocess=is_subprocess)


def setup_logging(config: Dict[str, Any] = None) -> None:
    """Глобальная настройка логирования"""
    if config is None:
        config = {
            'logging': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'log_dir': 'experiments/logs'
            }
        }
    
    # Создание директории для логов
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Базовая настройка логирования
    logging.basicConfig(
        level=config['logging']['level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def log_execution_time(logger: TradingLogger):
    """Декоратор для автоматического логирования времени выполнения"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.debug(f"Начало выполнения {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Завершение {func.__name__} за {duration:.2f}с")
                return result
            except Exception as e:
                logger.log_error(e, context=func.__name__)
                raise
        
        return wrapper
    return decorator