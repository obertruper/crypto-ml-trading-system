"""
Установщики для Universal LSP Server как системного сервиса
"""

import os
import sys
import subprocess
from pathlib import Path
import pwd
import grp

def get_executable_path() -> str:
    """Получить путь к исполняемому файлу"""
    # Пытаемся найти установленный lsp-server
    try:
        result = subprocess.run(['which', 'lsp-server'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    # Используем текущий Python
    return f"{sys.executable} -m lsp_server"

def install_macos():
    """Установка для macOS через launchd"""
    
    # Определяем пути
    home = Path.home()
    plist_name = "com.universal.lsp.plist"
    plist_path = home / "Library" / "LaunchAgents" / plist_name
    log_dir = home / "Library" / "Logs" / "UniversalLSP"
    
    # Создаем директорию для логов
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем plist файл
    executable = get_executable_path()
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.universal.lsp</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{executable}</string>
        <string>start</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>2087</string>
        <string>--log-file</string>
        <string>{log_dir}/lsp-server.log</string>
    </array>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}</string>
        <key>HOME</key>
        <string>{home}</string>
    </dict>
    
    <key>WorkingDirectory</key>
    <string>{home}</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>{log_dir}/stdout.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/stderr.log</string>
    
    <key>ThrottleInterval</key>
    <integer>30</integer>
</dict>
</plist>
"""
    
    # Сохраняем plist
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)
    
    # Загружаем службу
    subprocess.run(['launchctl', 'unload', str(plist_path)], capture_output=True)
    subprocess.run(['launchctl', 'load', str(plist_path)], check=True)
    
    print(f"✅ Служба установлена: {plist_path}")
    print("\nУправление службой:")
    print(f"  Запуск:    launchctl start com.universal.lsp")
    print(f"  Остановка: launchctl stop com.universal.lsp")
    print(f"  Статус:    launchctl list | grep com.universal.lsp")
    print(f"  Логи:      tail -f {log_dir}/lsp-server.log")

def install_linux():
    """Установка для Linux через systemd"""
    
    # Определяем пути
    home = Path.home()
    service_name = "universal-lsp.service"
    service_path = home / ".config" / "systemd" / "user" / service_name
    
    # Получаем информацию о пользователе
    user_info = pwd.getpwuid(os.getuid())
    username = user_info.pw_name
    
    # Создаем systemd unit файл
    executable = get_executable_path()
    
    service_content = f"""[Unit]
Description=Universal LSP Server
Documentation=https://github.com/yourusername/universal-lsp-server
After=network.target

[Service]
Type=simple
ExecStart={executable} start --host 127.0.0.1 --port 2087
Restart=on-failure
RestartSec=10

# Окружение
Environment="PATH={os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"
Environment="HOME={home}"

# Логирование
StandardOutput=journal
StandardError=journal
SyslogIdentifier=universal-lsp

# Ограничения
MemoryLimit=1G
CPUQuota=50%

# Безопасность
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={home}/.lsp_cache

[Install]
WantedBy=default.target
"""
    
    # Создаем директорию для systemd
    service_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем service файл
    service_path.write_text(service_content)
    
    # Перезагружаем systemd и включаем службу
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', service_name], check=True)
    
    print(f"✅ Служба установлена: {service_path}")
    print("\nУправление службой:")
    print(f"  Запуск:    systemctl --user start {service_name}")
    print(f"  Остановка: systemctl --user stop {service_name}")
    print(f"  Статус:    systemctl --user status {service_name}")
    print(f"  Логи:      journalctl --user -u {service_name} -f")

def create_startup_script():
    """Создать скрипт для ручного запуска"""
    
    script_content = """#!/bin/bash
# Скрипт запуска Universal LSP Server

# Определяем директорию скрипта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Настройки по умолчанию
HOST="${LSP_HOST:-127.0.0.1}"
PORT="${LSP_PORT:-2087}"
LOG_FILE="${LSP_LOG_FILE:-$SCRIPT_DIR/lsp-server.log}"

# Проверяем, запущен ли сервер
check_server() {
    nc -z "$HOST" "$PORT" 2>/dev/null
    return $?
}

# Запуск сервера
start_server() {
    if check_server; then
        echo "LSP сервер уже запущен на $HOST:$PORT"
        return 1
    fi
    
    echo "Запускаем LSP сервер на $HOST:$PORT..."
    nohup lsp-server start --host "$HOST" --port "$PORT" --log-file "$LOG_FILE" > /dev/null 2>&1 &
    
    # Сохраняем PID
    echo $! > "$SCRIPT_DIR/lsp.pid"
    
    # Ждем запуска
    sleep 2
    
    if check_server; then
        echo "✅ LSP сервер успешно запущен"
        echo "PID: $(cat "$SCRIPT_DIR/lsp.pid")"
        echo "Логи: $LOG_FILE"
    else
        echo "❌ Не удалось запустить сервер"
        return 1
    fi
}

# Остановка сервера
stop_server() {
    if [ -f "$SCRIPT_DIR/lsp.pid" ]; then
        PID=$(cat "$SCRIPT_DIR/lsp.pid")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Останавливаем LSP сервер (PID: $PID)..."
            kill "$PID"
            rm -f "$SCRIPT_DIR/lsp.pid"
            echo "✅ Сервер остановлен"
        else
            echo "Процесс не найден"
            rm -f "$SCRIPT_DIR/lsp.pid"
        fi
    else
        echo "PID файл не найден"
    fi
}

# Статус сервера
status_server() {
    if check_server; then
        echo "✅ LSP сервер работает на $HOST:$PORT"
        if [ -f "$SCRIPT_DIR/lsp.pid" ]; then
            echo "PID: $(cat "$SCRIPT_DIR/lsp.pid")"
        fi
    else
        echo "❌ LSP сервер не запущен"
    fi
}

# Обработка команд
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 1
        start_server
        ;;
    status)
        status_server
        ;;
    *)
        echo "Использование: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
"""
    
    script_path = Path.cwd() / "lsp-server.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    print(f"✅ Скрипт запуска создан: {script_path}")
    print("\nИспользование:")
    print(f"  ./lsp-server.sh start    # Запустить сервер")
    print(f"  ./lsp-server.sh stop     # Остановить сервер")
    print(f"  ./lsp-server.sh restart  # Перезапустить сервер")
    print(f"  ./lsp-server.sh status   # Проверить статус")