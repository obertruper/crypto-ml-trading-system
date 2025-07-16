"""
Адаптер для преобразования выходов UnifiedPatchTST модели в формат для торговых сигналов
ВАЖНО: Этот файл НЕ влияет на процесс обучения, только на тестирование!
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd

class ModelOutputAdapter:
    """
    Преобразует 20 выходов UnifiedPatchTST в формат, ожидаемый SignalGenerator
    
    20 выходов модели:
    - [0-3]: future_return_15m, 1h, 4h, 12h
    - [4-7]: direction_15m, 1h, 4h, 12h  
    - [8-11]: volatility_15m, 1h, 4h, 12h
    - [12-15]: volume_change_15m, 1h, 4h, 12h
    - [16-19]: price_range_15m, 1h, 4h, 12h
    """
    
    def __init__(self):
        self.output_mapping = {
            'future_return': slice(0, 4),    # индексы 0-3
            'direction': slice(4, 8),         # индексы 4-7
            'volatility': slice(8, 12),       # индексы 8-11
            'volume_change': slice(12, 16),   # индексы 12-15
            'price_range': slice(16, 20)      # индексы 16-19
        }
        
        self.timeframes = ['15m', '1h', '4h', '12h']
        
    def adapt_model_outputs(self, raw_outputs: torch.Tensor, symbols: List[str]) -> Dict:
        """
        Преобразует сырые выходы модели в структурированный формат
        
        Args:
            raw_outputs: Тензор формы [batch_size, 20] с выходами модели
            symbols: Список символов для каждого элемента в батче
            
        Returns:
            Dict с предсказаниями по символам в формате для SignalGenerator
        """
        # Переводим в numpy для удобства
        if isinstance(raw_outputs, torch.Tensor):
            outputs = raw_outputs.detach().cpu().numpy()
        else:
            outputs = raw_outputs
            
        # Группируем по символам
        predictions_by_symbol = {}
        
        unique_symbols = list(set(symbols))
        
        for symbol in unique_symbols:
            # Находим индексы для данного символа
            symbol_indices = [i for i, s in enumerate(symbols) if s == symbol]
            
            if not symbol_indices:
                continue
                
            # Берем последнее предсказание для символа (самое актуальное)
            symbol_outputs = outputs[symbol_indices[-1]]
            
            # Преобразуем в нужный формат
            predictions_by_symbol[symbol] = self._create_signal_format(symbol_outputs)
            
        return predictions_by_symbol
    
    def _create_signal_format(self, outputs: np.ndarray) -> Dict:
        """
        Создает структуру предсказаний, ожидаемую SignalGenerator
        """
        # Извлекаем компоненты
        future_returns = outputs[self.output_mapping['future_return']]
        directions = outputs[self.output_mapping['direction']]
        volatilities = outputs[self.output_mapping['volatility']]
        volume_changes = outputs[self.output_mapping['volume_change']]
        price_ranges = outputs[self.output_mapping['price_range']]
        
        # Анализируем направления для определения вероятностей TP/SL
        # direction > 0.5 означает рост, < 0.5 - падение
        bullish_prob = np.mean(directions > 0.5)
        bearish_prob = 1.0 - bullish_prob
        
        # Рассчитываем вероятности достижения целей на основе future_returns
        # Нормализуем returns в проценты (предполагаем, что модель выдает доли)
        returns_pct = future_returns * 100
        
        # Вероятности достижения TP на основе предсказанных доходностей
        tp_probs = []
        tp_targets = [1.5, 2.5, 4.0]  # Целевые уровни TP в процентах
        
        for tp_target in tp_targets:
            # Вероятность = доля временных горизонтов, где return > tp_target
            prob = np.mean(returns_pct > tp_target)
            tp_probs.append(prob)
            
        # Вероятность SL на основе отрицательных returns
        sl_target = -2.0  # Stop loss на -2%
        sl_prob = np.mean(returns_pct < sl_target)
        
        # Средняя волатильность
        avg_volatility = np.mean(volatilities)
        
        # Предсказание цены (используем средний return)
        avg_return = np.mean(future_returns)
        
        # Уверенность модели на основе согласованности предсказаний
        # Если все направления согласованы - высокая уверенность
        direction_consistency = np.std(directions)
        confidence = 1.0 - min(direction_consistency * 2, 1.0)  # Инвертируем std
        
        return {
            'tp_probs': tp_probs,              # [TP1, TP2, TP3] вероятности
            'sl_prob': sl_prob,                # Вероятность stop loss
            'volatility': avg_volatility,      # Ожидаемая волатильность
            'price_pred': avg_return,          # Предсказание изменения цены
            'confidence': confidence,          # Уверенность модели
            'direction': 'long' if bullish_prob > 0.5 else 'short',
            'signal_strength': abs(bullish_prob - 0.5) * 2,  # 0 to 1
            
            # Дополнительная информация для анализа
            'detailed_predictions': {
                'future_returns': {tf: float(ret) for tf, ret in zip(self.timeframes, future_returns)},
                'directions': {tf: float(dir) for tf, dir in zip(self.timeframes, directions)},
                'volatilities': {tf: float(vol) for tf, vol in zip(self.timeframes, volatilities)},
                'volume_changes': {tf: float(vc) for tf, vc in zip(self.timeframes, volume_changes)},
                'price_ranges': {tf: float(pr) for tf, pr in zip(self.timeframes, price_ranges)}
            }
        }
    
    def get_batch_predictions(self, model, data_loader, device='cuda') -> Tuple[Dict, pd.DataFrame]:
        """
        Получает предсказания модели для всего датасета
        
        Returns:
            predictions: Dict с предсказаниями по символам
            raw_data: DataFrame с сырыми предсказаниями для анализа
        """
        model.eval()
        
        all_predictions = []
        all_symbols = []
        all_timestamps = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Извлекаем данные из батча
                if isinstance(batch, dict):
                    features = batch['features'].to(device)
                    symbols = batch.get('symbol', ['BTCUSDT'] * len(features))
                    timestamps = batch.get('timestamp', [None] * len(features))
                else:
                    # Старый формат (X, y, meta)
                    features = batch[0].to(device)
                    symbols = batch[2].get('symbol', ['BTCUSDT'] * len(features)) if len(batch) > 2 else ['BTCUSDT'] * len(features)
                    timestamps = batch[2].get('timestamp', [None] * len(features)) if len(batch) > 2 else [None] * len(features)
                
                # Предсказание модели
                outputs = model(features)
                
                all_predictions.append(outputs.cpu())
                all_symbols.extend(symbols)
                all_timestamps.extend(timestamps)
        
        # Объединяем все предсказания
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # Адаптируем выходы
        adapted_predictions = self.adapt_model_outputs(all_predictions, all_symbols)
        
        # Создаем DataFrame для анализа
        raw_df = self._create_raw_dataframe(all_predictions, all_symbols, all_timestamps)
        
        return adapted_predictions, raw_df
    
    def _create_raw_dataframe(self, predictions: torch.Tensor, symbols: List[str], timestamps: List) -> pd.DataFrame:
        """Создает DataFrame с сырыми предсказаниями для анализа"""
        
        predictions_np = predictions.numpy()
        
        # Создаем колонки для каждого выхода
        columns = []
        for category, indices in self.output_mapping.items():
            for i, tf in enumerate(self.timeframes):
                columns.append(f'{category}_{tf}')
        
        df = pd.DataFrame(predictions_np, columns=columns)
        df['symbol'] = symbols
        df['timestamp'] = timestamps
        
        return df