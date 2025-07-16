# 🎯 Методы обучения и параметры

## 📊 Обзор методов обучения

### 1. Temporal Fusion Transformer (TFT) - Основной метод

**Описание**: Современная архитектура для прогнозирования временных рядов, комбинирующая LSTM, Attention и статический контекст.

**Преимущества**:
- Автоматический выбор важных признаков
- Обработка долгосрочных зависимостей
- Интерпретируемость через attention веса

**Архитектура**:
```python
def create_temporal_fusion_transformer(input_shape):
    # 1. Variable Selection Network
    vsn_weights = Dense(n_features, activation='softmax')
    selected_features = Multiply()([inputs, vsn_weights])
    
    # 2. Temporal Processing
    lstm_out = LSTM(128, return_sequences=True, dropout=0.2)
    lstm_out = LSTM(64, return_sequences=True, dropout=0.2)
    
    # 3. Self-Attention
    attention = MultiHeadAttention(num_heads=4, key_dim=16)
    
    # 4. Feed Forward
    ff_out = Dense(128, activation='relu')
    ff_out = Dense(64)
    
    # 5. Output
    output = Dense(1, activation='sigmoid')
```

### 2. Упрощенная LSTM модель (запасной вариант)

**Когда использовать**: При недостатке данных или проблемах с обучением TFT

**Архитектура**:
```python
def create_simple_model(input_shape):
    x = LSTM(64, return_sequences=True, dropout=0.2)
    x = LSTM(32, dropout=0.2)
    x = Dense(32, activation='relu')
    x = Dropout(0.3)
    x = Dense(16, activation='relu')
    x = Dropout(0.2)
    output = Dense(1, activation='sigmoid')
```

## ⚙️ Параметры обучения

### Базовые параметры (config.yaml)

```yaml
model:
  # Параметры данных
  sequence_length: 60        # Длина входной последовательности
                            # 60 = 15 часов для 15m таймфрейма
                            # Рекомендации: 40-100
  
  prediction_horizon: 100    # Горизонт прогнозирования
                            # 100 = 25 часов для поиска SL/TP
                            # Рекомендации: 50-200
  
  # Параметры обучения
  batch_size: 32            # Размер батча
                           # Меньше = медленнее, но стабильнее
                           # Больше = быстрее, но нужно больше памяти
                           # Рекомендации: 16-64
  
  epochs: 100              # Количество эпох
                          # С EarlyStopping обычно останавливается на 30-50
                          # Рекомендации: 50-200
  
  learning_rate: 0.001     # Скорость обучения
                          # Начальная скорость для Adam optimizer
                          # Рекомендации: 0.0001-0.01
```

### Дополнительные параметры в коде

```python
# Оптимизатор
optimizer = Adam(
    learning_rate=0.001,
    clipnorm=1.0,          # Gradient clipping для стабильности
    beta_1=0.9,            # Момент для градиентов
    beta_2=0.999,          # Момент для квадратов градиентов
    epsilon=1e-7           # Для численной стабильности
)

# Функция потерь
loss = 'binary_crossentropy'  # Для бинарной классификации
# Альтернативы:
# - 'focal_loss' - лучше для дисбаланса классов
# - 'weighted_binary_crossentropy' - с весами классов

# Метрики
metrics = [
    'accuracy',            # Общая точность
    'precision',           # Точность положительных предсказаний
    'recall',              # Полнота (процент найденных)
    tf.keras.metrics.AUC() # Площадь под ROC кривой
]
```

### Callbacks (автоматические действия при обучении)

```python
# 1. Early Stopping - остановка при отсутствии улучшений
EarlyStopping(
    monitor='val_loss',       # Метрика для отслеживания
    patience=15,              # Эпох без улучшения до остановки
    restore_best_weights=True # Восстановить лучшие веса
)

# 2. Reduce Learning Rate - уменьшение скорости обучения
ReduceLROnPlateau(
    monitor='val_loss',    # Метрика для отслеживания
    factor=0.5,           # Во сколько раз уменьшать LR
    patience=8,           # Эпох без улучшения
    min_lr=1e-6          # Минимальная скорость обучения
)

# 3. Model Checkpoint - сохранение лучшей модели
ModelCheckpoint(
    filepath='models/{model_name}_best.h5',
    monitor='val_loss',
    save_best_only=True
)
```

### Балансировка классов

```python
# Автоматический расчет весов классов
pos_samples = sum(y_train)
neg_samples = len(y_train) - pos_samples
class_weight = {
    0: 1.0,
    1: neg_samples / pos_samples  # Больший вес для редкого класса
}

# Пример для наших данных:
# BUY signals: 8.1% positive -> weight = 11.3
# SELL signals: 15.6% positive -> weight = 5.4
```

## 🔧 Настройка параметров

### Для улучшения качества модели:

1. **При переобучении** (val_loss растет, train_loss падает):
   - Увеличить dropout: 0.2 → 0.3-0.5
   - Уменьшить размер модели (меньше нейронов)
   - Увеличить patience в EarlyStopping
   - Добавить L2 регуляризацию

2. **При недообучении** (обе метрики плохие):
   - Увеличить размер модели
   - Увеличить epochs
   - Увеличить learning_rate
   - Уменьшить dropout

3. **При дисбалансе классов**:
   - Использовать focal loss
   - Увеличить веса редких классов
   - Использовать SMOTE для генерации примеров
   - Изменить threshold (вместо 0.5)

### Для ускорения обучения:

1. **Увеличить batch_size** (если хватает памяти)
2. **Использовать mixed precision training**:
   ```python
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

3. **Уменьшить sequence_length** (но не меньше 40)

### Для разных условий рынка:

1. **Высокая волатильность**:
   - Уменьшить prediction_horizon (50-75)
   - Увеличить веса последних данных
   - Использовать более короткие последовательности

2. **Низкая волатильность**:
   - Увеличить prediction_horizon (100-150)
   - Использовать более длинные последовательности
   - Уменьшить learning_rate

## 📈 Мониторинг обучения

### Ключевые метрики для отслеживания:

1. **Loss** - основная метрика
   - Должна уменьшаться
   - val_loss не должна расти

2. **Accuracy** - общая точность
   - Целевое значение: >65%
   - Но может быть обманчива при дисбалансе

3. **Precision/Recall** - для оценки качества
   - Precision: из предсказанных positive сколько правильных
   - Recall: из всех positive сколько нашли

4. **AUC** - интегральная оценка
   - >0.7 - хорошо
   - >0.8 - отлично

### Проблемы и решения:

1. **NaN в loss**:
   - Уменьшить learning_rate
   - Проверить данные на inf/nan
   - Использовать gradient clipping

2. **Метрики не улучшаются**:
   - Проверить качество данных
   - Изменить архитектуру
   - Попробовать другие признаки

3. **Очень долгое обучение**:
   - Уменьшить размер модели
   - Использовать GPU
   - Уменьшить batch_size

## 🎯 Рекомендации по выбору параметров

### Для начала (тестирование):
```yaml
sequence_length: 40
batch_size: 64
epochs: 30
learning_rate: 0.001
```

### Для продакшена:
```yaml
sequence_length: 60-80
batch_size: 32
epochs: 100
learning_rate: 0.0005
```

### Для исследований:
```yaml
sequence_length: 100-120
batch_size: 16
epochs: 200
learning_rate: 0.0001
```