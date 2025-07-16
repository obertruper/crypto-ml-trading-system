-- 📊 ПОЛЕЗНЫЕ SQL ЗАПРОСЫ ДЛЯ METABASE

-- 1. Общая статистика по символам
SELECT 
    symbol,
    COUNT(*) as total_records,
    MIN(datetime) as first_date,
    MAX(datetime) as last_date,
    ROUND(AVG(close)::numeric, 2) as avg_price,
    ROUND(MIN(close)::numeric, 2) as min_price,
    ROUND(MAX(close)::numeric, 2) as max_price,
    ROUND(AVG(volume)::numeric, 2) as avg_volume
FROM raw_market_data
WHERE market_type = 'futures'
GROUP BY symbol
ORDER BY total_records DESC;

-- 2. Динамика цен по дням
SELECT 
    symbol,
    DATE(datetime) as date,
    ROUND(AVG(close)::numeric, 2) as avg_daily_price,
    ROUND(MAX(high)::numeric, 2) as daily_high,
    ROUND(MIN(low)::numeric, 2) as daily_low,
    SUM(volume) as daily_volume
FROM raw_market_data
WHERE market_type = 'futures'
GROUP BY symbol, DATE(datetime)
ORDER BY symbol, date DESC;

-- 3. Топ-10 символов по волатильности
SELECT 
    symbol,
    ROUND(((MAX(close) - MIN(close)) / AVG(close) * 100)::numeric, 2) as volatility_percent,
    ROUND(STDDEV(close)::numeric, 4) as price_std_dev,
    COUNT(*) as data_points
FROM raw_market_data
WHERE market_type = 'futures'
GROUP BY symbol
HAVING COUNT(*) > 1000
ORDER BY volatility_percent DESC
LIMIT 10;

-- 4. Анализ доходности (для обработанных данных)
SELECT 
    symbol,
    COUNT(*) as total_signals,
    SUM(CASE WHEN buy_return > 0 THEN 1 ELSE 0 END) as profitable_buy_signals,
    SUM(CASE WHEN sell_return > 0 THEN 1 ELSE 0 END) as profitable_sell_signals,
    ROUND(AVG(buy_return)::numeric, 4) as avg_buy_return,
    ROUND(AVG(sell_return)::numeric, 4) as avg_sell_return,
    ROUND(MAX(buy_return)::numeric, 4) as max_buy_return,
    ROUND(MAX(sell_return)::numeric, 4) as max_sell_return
FROM processed_market_data
GROUP BY symbol
ORDER BY avg_buy_return DESC;

-- 5. Распределение доходности по часам
SELECT 
    EXTRACT(HOUR FROM datetime) as hour,
    ROUND(AVG(buy_return)::numeric, 4) as avg_buy_return,
    ROUND(AVG(sell_return)::numeric, 4) as avg_sell_return,
    COUNT(*) as signal_count
FROM processed_market_data
GROUP BY EXTRACT(HOUR FROM datetime)
ORDER BY hour;

-- 6. Корреляция объема и доходности
SELECT 
    symbol,
    ROUND(CORR(volume, buy_return)::numeric, 4) as volume_buy_correlation,
    ROUND(CORR(volume, sell_return)::numeric, 4) as volume_sell_correlation,
    COUNT(*) as data_points
FROM processed_market_data
WHERE buy_return IS NOT NULL AND sell_return IS NOT NULL
GROUP BY symbol
HAVING COUNT(*) > 1000
ORDER BY volume_buy_correlation DESC;

-- 7. Последние данные по каждому символу
WITH latest_data AS (
    SELECT 
        symbol,
        datetime,
        close,
        volume,
        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY datetime DESC) as rn
    FROM raw_market_data
    WHERE market_type = 'futures'
)
SELECT 
    symbol,
    datetime as latest_datetime,
    close as latest_price,
    volume as latest_volume,
    EXTRACT(EPOCH FROM (NOW() - datetime))/3600 as hours_since_update
FROM latest_data
WHERE rn = 1
ORDER BY hours_since_update;

-- 8. Технические индикаторы (последние значения)
SELECT 
    symbol,
    datetime,
    ROUND(rsi_14::numeric, 2) as RSI,
    ROUND(macd::numeric, 4) as MACD,
    ROUND(bb_upper::numeric, 2) as BB_Upper,
    ROUND(close::numeric, 2) as Price,
    ROUND(bb_lower::numeric, 2) as BB_Lower,
    CASE 
        WHEN rsi_14 < 30 THEN 'Oversold'
        WHEN rsi_14 > 70 THEN 'Overbought'
        ELSE 'Neutral'
    END as RSI_Signal
FROM processed_market_data
WHERE datetime = (SELECT MAX(datetime) FROM processed_market_data WHERE symbol = processed_market_data.symbol)
ORDER BY symbol;

-- 9. Производительность модели (если есть предсказания)
SELECT 
    m.model_name,
    m.model_type,
    m.metrics->>'mae' as MAE,
    m.metrics->>'rmse' as RMSE,
    m.metrics->>'r2' as R2,
    m.metrics->>'direction_accuracy' as Direction_Accuracy,
    m.created_at
FROM model_metadata m
ORDER BY m.created_at DESC;

-- 10. Пропуски в данных
WITH time_gaps AS (
    SELECT 
        symbol,
        datetime,
        LAG(datetime) OVER (PARTITION BY symbol ORDER BY datetime) as prev_datetime,
        EXTRACT(EPOCH FROM (datetime - LAG(datetime) OVER (PARTITION BY symbol ORDER BY datetime)))/60 as gap_minutes
    FROM raw_market_data
    WHERE market_type = 'futures'
)
SELECT 
    symbol,
    COUNT(*) as gap_count,
    MAX(gap_minutes) as max_gap_minutes,
    AVG(gap_minutes) as avg_gap_minutes
FROM time_gaps
WHERE gap_minutes > 15  -- Больше интервала в 15 минут
GROUP BY symbol
ORDER BY gap_count DESC;