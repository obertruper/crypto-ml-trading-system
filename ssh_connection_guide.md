# Руководство по SSH подключению с Mac к Linux

## Текущий статус системы:
- ✅ SSH сервер активен и работает на порту 22
- ✅ SSH ключ от Mac уже добавлен в authorized_keys
- ✅ IP адрес системы: 192.168.10.101
- ✅ Firewall отключен (нет блокировок)

## Для подключения с вашего Mac:

### 1. Проверьте SSH ключ на Mac:
```bash
# На вашем Mac выполните:
ls -la ~/.ssh/
# Должен быть файл id_ed25519 или похожий
```

### 2. Подключение к Linux системе:
```bash
# Базовая команда для подключения:
ssh obertruper@192.168.10.101

# Если ключ в нестандартном месте:
ssh -i ~/.ssh/ваш_ключ obertruper@192.168.10.101
```

### 3. Если подключение не работает, проверьте:

#### На Mac:
```bash
# Проверить SSH ключ
cat ~/.ssh/id_ed25519.pub

# Проверить соединение
ssh -v obertruper@192.168.10.101
```

#### Возможные проблемы:
1. **Permission denied** - проверьте, что ключ на Mac соответствует ключу в authorized_keys
2. **Connection refused** - проверьте IP адрес и сетевое подключение
3. **Host key verification failed** - очистите known_hosts: `ssh-keygen -R 192.168.10.101`

### 4. Дополнительные настройки на Mac:
Создайте/отредактируйте файл `~/.ssh/config` на Mac:
```
Host linux-ml
    HostName 192.168.10.101
    User obertruper
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

После этого можно подключаться просто: `ssh linux-ml`

### 5. Передача файлов:
```bash
# Копирование файла на Linux:
scp file.txt obertruper@192.168.10.101:/home/obertruper/

# Копирование папки:
scp -r folder/ obertruper@192.168.10.101:/home/obertruper/

# Или используя alias:
scp file.txt linux-ml:~/
```

### 6. VS Code Remote SSH:
1. Установите расширение "Remote - SSH" в VS Code
2. Нажмите F1 и выберите "Remote-SSH: Connect to Host"
3. Введите: `obertruper@192.168.10.101`
4. Или используйте сохраненную конфигурацию `linux-ml`

## Устранение проблем:

### Если всё равно не подключается:
1. Проверьте, что оба устройства в одной сети
2. Попробуйте ping: `ping 192.168.10.101`
3. Проверьте SSH порт: `nc -zv 192.168.10.101 22`
4. Посмотрите логи SSH на Linux: `sudo journalctl -u ssh -f`