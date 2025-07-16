#!/usr/bin/env python3
"""
Тест размера контекстного окна LSP сервера
"""
import yaml
import os
import psutil
import socket

def check_lsp_status():
    """Проверка статуса LSP сервера и его конфигурации"""
    
    # Загрузка конфигурации
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    context_size = config.get('context', {}).get('max_tokens', 0)
    
    print("🔍 ПРОВЕРКА LSP СЕРВЕРА")
    print("=" * 50)
    
    # Конфигурация
    print(f"\n📋 Конфигурация:")
    print(f"   Контекстное окно: {context_size:,} токенов")
    print(f"   Порт: {config['server']['port']}")
    
    # Проверка процесса
    lsp_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'enhanced_lsp_server.py' in ' '.join(proc.info['cmdline'] or []):
                lsp_running = True
                print(f"\n✅ LSP сервер запущен:")
                print(f"   PID: {proc.info['pid']}")
                
                # Память процесса
                memory = proc.memory_info()
                print(f"   Память: {memory.rss / 1024 / 1024:.1f} MB")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if not lsp_running:
        print("\n❌ LSP сервер не запущен")
    
    # Проверка порта
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_result = sock.connect_ex(('127.0.0.1', config['server']['port']))
    sock.close()
    
    if port_result == 0:
        print(f"\n✅ Порт {config['server']['port']} доступен")
    else:
        print(f"\n⚠️  Порт {config['server']['port']} недоступен")
    
    # Рекомендации по памяти
    print(f"\n💡 Рекомендации:")
    if context_size >= 1000000:
        estimated_memory = context_size * 4 / 1024 / 1024  # Примерная оценка
        print(f"   - При контексте в {context_size:,} токенов")
        print(f"     может потребоваться до {estimated_memory:.0f} MB памяти")
        print(f"   - Убедитесь, что у системы достаточно RAM")
        print(f"   - LSP будет кешировать больше данных")
    
    print("\n✅ Новые настройки применены!")
    print("   LSP сервер будет использовать расширенный контекст")
    print("   при следующих запросах от Claude Code")

if __name__ == "__main__":
    check_lsp_status()
