#!/usr/bin/env python3
"""
CLI для Universal LSP Server
"""

import click
import logging
import sys
import os
from pathlib import Path
import asyncio
import signal
import json
import yaml
from typing import Optional

from .config import Config
from .server import UniversalLSPServer
from .utils import setup_logging, find_project_root

logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version="1.0.0", prog_name="Universal LSP Server")
def cli():
    """Universal LSP Server - улучшенный Language Server для работы с кодом"""
    pass

@cli.command()
@click.option('--host', default=None, help='Хост для привязки сервера')
@click.option('--port', type=int, default=None, help='Порт для привязки сервера')
@click.option('--stdio', is_flag=True, help='Использовать stdio вместо TCP')
@click.option('--config', type=click.Path(exists=True), help='Путь к файлу конфигурации')
@click.option('--project-root', type=click.Path(exists=True), help='Корневая директория проекта')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), help='Уровень логирования')
@click.option('--log-file', type=click.Path(), help='Файл для логов')
def start(host: Optional[str], port: Optional[int], stdio: bool, 
          config: Optional[str], project_root: Optional[str],
          log_level: Optional[str], log_file: Optional[str]):
    """Запуск LSP сервера"""
    
    # Загружаем конфигурацию
    config_path = Path(config) if config else None
    cfg = Config.load(config_path)
    
    # Переопределяем параметры из командной строки
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
    if stdio:
        cfg.server.enable_stdio = True
    if project_root:
        cfg.project_root = project_root
    if log_level:
        cfg.server.log_level = log_level
    if log_file:
        cfg.server.log_file = log_file
    
    # Настраиваем логирование
    setup_logging(cfg.server.log_level, cfg.server.log_file)
    
    # Автоопределение project_root если не задан
    if not cfg.project_root:
        cfg.project_root = find_project_root() or os.getcwd()
    
    logger.info(f"Запуск Universal LSP Server v{cfg.server.version}")
    logger.info(f"Корневая директория проекта: {cfg.project_root}")
    
    # Создаем и запускаем сервер
    server = UniversalLSPServer(cfg)
    
    # Обработка сигналов для корректного завершения
    def signal_handler(sig, frame):
        logger.info("Получен сигнал завершения, останавливаем сервер...")
        asyncio.create_task(server.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if cfg.server.enable_stdio:
            logger.info("Запуск в режиме stdio")
            server.start_io()
        else:
            logger.info(f"Запуск TCP сервера на {cfg.server.host}:{cfg.server.port}")
            server.start_tcp(cfg.server.host, cfg.server.port)
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}", exc_info=True)
        sys.exit(1)

@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Файл для сохранения конфигурации')
@click.option('--format', type=click.Choice(['yaml', 'json']), default='yaml', help='Формат файла')
def init(output: Optional[str], format: str):
    """Создать файл конфигурации по умолчанию"""
    
    cfg = Config()
    
    if output:
        output_path = Path(output)
    else:
        output_path = Path.cwd() / f"lsp-server.{format}"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'yaml':
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
    
    click.echo(f"Конфигурация создана: {output_path}")

@cli.command()
@click.argument('directory', type=click.Path(exists=True), required=False)
@click.option('--config', type=click.Path(exists=True), help='Путь к файлу конфигурации')
@click.option('--output', '-o', type=click.Path(), help='Файл для сохранения индекса')
@click.option('--format', type=click.Choice(['json', 'yaml', 'summary']), default='summary', help='Формат вывода')
def index(directory: Optional[str], config: Optional[str], output: Optional[str], format: str):
    """Проиндексировать проект"""
    
    # Загружаем конфигурацию
    config_path = Path(config) if config else None
    cfg = Config.load(config_path)
    
    # Определяем директорию
    if directory:
        project_dir = Path(directory).resolve()
    else:
        project_dir = Path(cfg.project_root or os.getcwd()).resolve()
    
    click.echo(f"Индексация проекта: {project_dir}")
    
    # Импортируем здесь чтобы избежать циклических импортов
    from .indexer import ProjectIndexer
    
    indexer = ProjectIndexer(cfg.indexing)
    index_data = asyncio.run(indexer.index_directory(project_dir))
    
    if format == 'summary':
        click.echo(f"\nРезультаты индексации:")
        click.echo(f"Файлов проиндексировано: {index_data['total_files']}")
        click.echo(f"Общий размер: {index_data['total_size'] / 1024 / 1024:.2f} MB")
        click.echo(f"Символов найдено: {index_data['total_symbols']}")
        click.echo(f"Время индексации: {index_data['indexing_time']:.2f} сек")
        
        if index_data['errors']:
            click.echo(f"\nОшибки ({len(index_data['errors'])}):")
            for error in index_data['errors'][:5]:
                click.echo(f"  - {error}")
    else:
        # Сохраняем в файл
        if output:
            output_path = Path(output)
        else:
            output_path = Path.cwd() / f"index.{format}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False, default=str)
        else:  # yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(index_data, f, default_flow_style=False, allow_unicode=True)
        
        click.echo(f"Индекс сохранен: {output_path}")

@cli.command()
@click.option('--host', default='127.0.0.1', help='Хост сервера')
@click.option('--port', type=int, default=2087, help='Порт сервера')
def check(host: str, port: int):
    """Проверить состояние LSP сервера"""
    
    import socket
    import json
    
    click.echo(f"Проверка сервера на {host}:{port}...")
    
    try:
        # Пытаемся подключиться
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            click.echo(f"✅ Сервер доступен на {host}:{port}")
        else:
            click.echo(f"❌ Сервер недоступен на {host}:{port}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Ошибка подключения: {e}")
        sys.exit(1)

@cli.command()
def install():
    """Установить LSP сервер как системный сервис"""
    
    system = sys.platform
    
    if system == 'darwin':
        # macOS - используем launchd
        from .install import install_macos
        try:
            install_macos()
            click.echo("✅ LSP сервер установлен как служба macOS")
            click.echo("Используйте 'launchctl start com.universal.lsp' для запуска")
        except Exception as e:
            click.echo(f"❌ Ошибка установки: {e}")
            sys.exit(1)
            
    elif system == 'linux':
        # Linux - используем systemd
        from .install import install_linux
        try:
            install_linux()
            click.echo("✅ LSP сервер установлен как systemd служба")
            click.echo("Используйте 'systemctl --user start universal-lsp' для запуска")
        except Exception as e:
            click.echo(f"❌ Ошибка установки: {e}")
            sys.exit(1)
            
    elif system == 'win32':
        # Windows - используем Windows Service
        click.echo("❌ Установка как Windows службы пока не поддерживается")
        click.echo("Используйте 'lsp-server start' для ручного запуска")
        sys.exit(1)
    else:
        click.echo(f"❌ Неподдерживаемая система: {system}")
        sys.exit(1)

def main():
    """Главная точка входа"""
    cli()

if __name__ == '__main__':
    main()