#!/usr/bin/env python3
"""
Скрипт для запуска LSP сервера как сервиса
Автоматически перезапускает при сбоях
"""

import subprocess
import time
import os
import sys
from pathlib import Path
import logging
import signal
import psutil

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lsp_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LSPService:
    def __init__(self):
        self.lsp_dir = Path(__file__).parent
        self.venv_python = self.lsp_dir / "venv" / "bin" / "python"
        self.lsp_script = self.lsp_dir / "enhanced_lsp_server.py"
        self.pid_file = self.lsp_dir / "lsp.pid"
        self.process = None
        
    def is_running(self):
        """Проверяет, запущен ли сервер"""
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text())
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    if "python" in process.name().lower():
                        return True
            except:
                pass
        return False
        
    def stop(self):
        """Останавливает сервер"""
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text())
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Остановлен LSP сервер с PID {pid}")
                self.pid_file.unlink()
            except Exception as e:
                logger.error(f"Ошибка при остановке: {e}")
                
    def start(self):
        """Запускает сервер"""
        if self.is_running():
            logger.info("LSP сервер уже запущен")
            return
            
        logger.info("Запуск LSP сервера...")
        
        # Настройка окружения
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.lsp_dir.parent)  # crypto_ai_trading
        
        # Запуск процесса
        self.process = subprocess.Popen(
            [str(self.venv_python), str(self.lsp_script), "--stdio"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        
        # Сохраняем PID
        self.pid_file.write_text(str(self.process.pid))
        logger.info(f"LSP сервер запущен с PID {self.process.pid}")
        
    def run_forever(self):
        """Запускает сервер и перезапускает при сбоях"""
        logger.info("Запуск LSP сервиса...")
        
        def signal_handler(signum, frame):
            logger.info("Получен сигнал остановки")
            self.stop()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while True:
            try:
                if not self.is_running():
                    self.start()
                    # Ждем немного после запуска
                    time.sleep(5)
                    
                # Проверяем каждые 10 секунд
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("Остановка по Ctrl+C")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Ошибка в главном цикле: {e}")
                time.sleep(5)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LSP Service Manager')
    parser.add_argument('command', choices=['start', 'stop', 'restart', 'status'],
                      help='Команда для выполнения')
    
    args = parser.parse_args()
    
    service = LSPService()
    
    if args.command == 'start':
        if service.is_running():
            print("LSP сервер уже запущен")
        else:
            print("Запуск LSP сервера...")
            service.run_forever()
            
    elif args.command == 'stop':
        if service.is_running():
            service.stop()
            print("LSP сервер остановлен")
        else:
            print("LSP сервер не запущен")
            
    elif args.command == 'restart':
        if service.is_running():
            service.stop()
            time.sleep(2)
        service.run_forever()
        
    elif args.command == 'status':
        if service.is_running():
            print("LSP сервер запущен")
        else:
            print("LSP сервер не запущен")

if __name__ == "__main__":
    main()