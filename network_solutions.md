# Решения для подключения между разными подсетями

## Текущая ситуация:
- 🖥️ Linux: 192.168.10.101 (подсеть 192.168.10.x)
- 💻 Mac: 192.168.88.94 (подсеть 192.168.88.x)
- ❌ Прямое подключение невозможно

## Решение 1: Tailscale VPN (Рекомендуется)
**Самый простой и безопасный способ**

1. На Linux выполните:
```bash
# Авторизация в Tailscale
sudo tailscale up

# После авторизации проверьте IP
tailscale ip -4
```

2. На Mac:
- Скачайте Tailscale: https://tailscale.com/download/mac
- Войдите в тот же аккаунт
- Получите IP вашего Linux в Tailscale

3. Подключайтесь через Tailscale IP:
```bash
ssh obertruper@100.x.x.x  # Tailscale IP
```

## Решение 2: Перенести Mac в ту же подсеть
1. Подключите Mac к тому же роутеру, что и Linux
2. Или измените настройки сети Mac на 192.168.10.x

## Решение 3: Port Forwarding на роутере
Если у вас есть доступ к роутеру 192.168.10.254:
1. Настройте проброс портов (port forwarding)
2. Пробросьте внешний порт на 192.168.10.101:22
3. Подключайтесь через внешний IP роутера

## Решение 4: SSH Jump Host
Если есть сервер, доступный из обеих сетей:
```bash
# На Linux
ssh -R 2222:localhost:22 user@jump-server

# На Mac
ssh -p 2222 obertruper@jump-server
```

## Решение 5: ZeroTier (альтернатива Tailscale)
1. Установите ZeroTier на обоих устройствах
2. Создайте сеть на https://my.zerotier.com
3. Подключите оба устройства к сети

## Быстрая проверка после настройки:
```bash
# Проверка доступности
ping [новый-ip-адрес]

# SSH подключение
ssh obertruper@[новый-ip-адрес]
```

## Для разработки рекомендую Tailscale:
- ✅ Работает через любые NAT
- ✅ Автоматическая настройка
- ✅ Безопасное шифрование
- ✅ Не нужно менять настройки сети
- ✅ Бесплатно для личного использования