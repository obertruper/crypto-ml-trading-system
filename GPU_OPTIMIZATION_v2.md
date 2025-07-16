# 🚀 Оптимизация GPU памяти для Enhanced TFT v2.1

## ✅ Внесенные изменения

### 1. **Уменьшены параметры модели** (строки 761-770):
- `batch_size`: 16 → 8
- `gradient_accumulation_steps`: 8 → 4 (эффективный batch = 32 вместо 128)
- `d_model`: 256 → 128
- `num_heads`: 8 (без изменений)
- `num_transformer_blocks`: 6 → 4
- `ff_dim`: 512 → 256
- `dropout_rate`: 0.3 (без изменений)

### 2. **Включен Mixed Precision Training** (строки 82-86):
- Использование float16 вместо float32 для экономии памяти
- Автоматическое масштабирование loss для стабильности
- Финальные слои остаются в float32 для точности

### 3. **Добавлен MemoryCleanupCallback** (строки 91-108):
- Очистка TensorFlow сессии после каждой эпохи
- Сборка мусора Python (gc.collect())
- Логирование использования GPU памяти

### 4. **Оптимизирован оптимизатор** (строки 896-898):
- Добавлен LossScaleOptimizer для mixed precision
- Сохранены оригинальные параметры Adam

## 📊 Ожидаемые результаты

### Использование памяти:
- **До оптимизации**: ~22 GB (OOM на 24GB GPU)
- **После оптимизации**: ~8-10 GB (комфортно для RTX 4090)

### Производительность:
- Скорость обучения может увеличиться на 20-30% благодаря float16
- Качество модели должно остаться на том же уровне

## 🔧 Дополнительные рекомендации

### 1. **Если все еще возникает OOM**:
```python
# Уменьшить sequence_length в config.yaml
sequence_length: 30  # вместо 60
```

### 2. **Для увеличения скорости**:
```python
# Использовать tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.cache()
```

### 3. **Мониторинг GPU**:
```bash
# В отдельном терминале на сервере
watch -n 1 nvidia-smi
```

### 4. **Автоматический перезапуск при OOM**:
```python
# Обернуть обучение в try-except
try:
    history = model.fit(...)
except tf.errors.ResourceExhaustedError:
    # Очистить память
    tf.keras.backend.clear_session()
    gc.collect()
    # Уменьшить batch_size и попробовать снова
```

## 📈 Тестирование

1. **Сначала запустите в тестовом режиме**:
```bash
python train_universal_transformer_v2.py --test_mode --task classification_binary
```

2. **Если тест прошел успешно, запускайте полное обучение**:
```bash
python train_universal_transformer_v2.py --task classification_binary --ensemble_size 1
```

3. **Мониторьте логи**:
```bash
tail -f logs/training_*/training.log
```

## 🎯 Целевые метрики

- GPU Memory Usage: < 20 GB
- Training Speed: > 100 samples/sec
- No OOM errors
- Model quality: сохранить текущий уровень accuracy/MAE