# 🚀 Metabase успешно установлен и запущен!

## 📊 Доступ к Metabase
- **URL**: http://localhost:3333
- **Статус**: ✅ Работает

## 🔧 Настройка подключения к БД в Metabase

При первом входе в Metabase:

1. **Создайте аккаунт администратора**
   - Введите ваш email и пароль

2. **Настройте подключение к базе данных:**
   - **Database type**: PostgreSQL
   - **Name**: Crypto Trading DB
   - **Host**: 172.18.0.1
   - **Port**: 5555
   - **Database name**: crypto_trading
   - **Username**: ruslan
   - **Password**: ruslan
   
   **Дополнительные параметры подключения (Advanced options):**
   - Нажмите "Show advanced options"
   - В поле "Additional JDBC connection string options" добавьте:
     ```
     ?TimeZone=UTC
     ```

3. **Проверьте подключение**
   - Нажмите "Test Connection"
   - Должно появиться "Success!"

## 📈 Рекомендуемые дашборды для создания

### 1. Обзор рынка
- Количество данных по символам
- Временной охват данных
- Активность по дням недели

### 2. Анализ цен
- График цен по символам
- Волатильность
- Объемы торгов

### 3. Производительность модели
- Метрики обучения
- Точность предсказаний
- Сравнение моделей

## 🛠️ Полезные SQL запросы для начала

### Топ символов по количеству данных:
```sql
SELECT symbol, 
       COUNT(*) as data_points, 
       MIN(timestamp) as first_date,
       MAX(timestamp) as last_date
FROM raw_market_data
GROUP BY symbol
ORDER BY data_points DESC
LIMIT 20;
```

### Статистика по дням:
```sql
SELECT DATE(to_timestamp(timestamp/1000)) as date,
       COUNT(DISTINCT symbol) as symbols,
       COUNT(*) as total_records
FROM raw_market_data
GROUP BY DATE(to_timestamp(timestamp/1000))
ORDER BY date DESC
LIMIT 30;
```

### Средние цены по символам:
```sql
SELECT symbol,
       AVG(close) as avg_price,
       MIN(close) as min_price,
       MAX(close) as max_price,
       AVG(volume) as avg_volume
FROM raw_market_data
WHERE timestamp > (EXTRACT(EPOCH FROM NOW() - INTERVAL '30 days') * 1000)
GROUP BY symbol
ORDER BY avg_volume DESC;
```

## 🔄 Управление Metabase

### Остановка:
```bash
cd /mnt/SSD/PYCHARMPRODJECT/LLM\ TRANSFORM/crypto_ai_trading
sudo docker-compose -f docker-compose-metabase.yml down
```

### Перезапуск:
```bash
cd /mnt/SSD/PYCHARMPRODJECT/LLM\ TRANSFORM/crypto_ai_trading
sudo docker-compose -f docker-compose-metabase.yml restart
```

### Просмотр логов:
```bash
sudo docker logs -f metabase_crypto_ai
```

## 📝 Примечания
- Данные Metabase сохраняются в папке `metabase-data/`
- Бэкап базы данных Metabase делается автоматически
- Рекомендуется регулярно экспортировать важные дашборды

---
*Дата установки: $(date)*