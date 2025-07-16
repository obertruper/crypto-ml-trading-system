# Настройки подключения к PostgreSQL

## Параметры подключения:
- **Host:** localhost (127.0.0.1)
- **Port:** 5555
- **Database:** crypto_trading
- **Username:** ruslan
- **Password:** ruslan

## Рекомендуемые Desktop клиенты:

### 1. TablePlus (macOS/Windows/Linux)
**Скачать:** https://tableplus.com/

**Настройка:**
1. Create new connection → PostgreSQL
2. Введите параметры выше
3. Test connection → Save

### 2. DBeaver (Бесплатный, все платформы)
**Скачать:** https://dbeaver.io/

**Настройка:**
1. New Database Connection → PostgreSQL
2. Server: localhost, Port: 5555
3. Database: crypto_trading
4. Username/Password: ruslan/ruslan

### 3. pgAdmin 4 (Официальный, бесплатный)
**Скачать:** https://www.pgadmin.org/

**Настройка:**
1. Add New Server
2. Name: Crypto Trading
3. Connection tab: заполните параметры

### 4. Postico 2 (macOS)
**Скачать:** https://eggerapps.at/postico2/

### 5. DataGrip (JetBrains, платный)
**Если у вас есть лицензия JetBrains**

## SQL запросы для быстрой проверки:

```sql
-- Статистика загрузки
SELECT 
    symbol,
    COUNT(*) as records,
    to_char(MIN(to_timestamp(timestamp/1000)), 'YYYY-MM-DD') as start_date,
    to_char(MAX(to_timestamp(timestamp/1000)), 'YYYY-MM-DD') as end_date,
    ROUND(AVG(close)::numeric, 4) as avg_price
FROM raw_market_data
GROUP BY symbol
ORDER BY symbol;

-- Проверка обработанных данных
SELECT 
    symbol,
    COUNT(*) as total,
    SUM(buy_profit_target) as buy_profits,
    SUM(sell_profit_target) as sell_profits,
    ROUND(100.0 * SUM(buy_profit_target) / NULLIF(SUM(buy_profit_target + buy_loss_target), 0), 2) as buy_win_rate
FROM processed_market_data
GROUP BY symbol;

-- Последние цены
SELECT 
    symbol,
    to_timestamp(timestamp/1000) as time,
    close as price,
    volume
FROM raw_market_data
WHERE timestamp = (SELECT MAX(timestamp) FROM raw_market_data r2 WHERE r2.symbol = raw_market_data.symbol)
ORDER BY symbol;
```

## Полезные представления (Views):

```sql
-- Создание представления для удобного анализа
CREATE OR REPLACE VIEW v_market_summary AS
SELECT 
    r.symbol,
    COUNT(DISTINCT DATE(to_timestamp(r.timestamp/1000))) as trading_days,
    MIN(r.close) as min_price,
    MAX(r.close) as max_price,
    AVG(r.close) as avg_price,
    SUM(r.volume) as total_volume,
    COUNT(p.id) as processed_records
FROM raw_market_data r
LEFT JOIN processed_market_data p ON r.symbol = p.symbol
GROUP BY r.symbol;
```