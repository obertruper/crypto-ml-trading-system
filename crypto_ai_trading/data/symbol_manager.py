"""
SymbolManager: централизованное управление маппингом символов.

Особенности:
- Работает без БД (fallback): строит маппинг по данным и кэширует в cache/features/symbol_mapping.pkl
- При наличии подключения к БД может читать/инициализировать таблицу symbol_mapping
- Предоставляет encode_symbols(df): добавляет столбцы symbol_index, symbol_id, а также базовые статистики (если доступны)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
from utils.logger import get_logger


SYMBOL_CACHE_DIR = Path("cache/features")
SYMBOL_CACHE_FILE = SYMBOL_CACHE_DIR / "symbol_mapping.pkl"


@dataclass
class SymbolInfo:
    symbol: str
    symbol_index: int
    sector: Optional[str] = None
    market_cap_rank: Optional[int] = None
    avg_volume: Optional[float] = None
    avg_volatility: Optional[float] = None
    correlation_cluster: Optional[int] = None
    is_active: bool = True
    metadata: Optional[Dict] = None


class SymbolManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("SymbolManager")
        SYMBOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._mapping_df: Optional[pd.DataFrame] = None

    # --------------------- Public API ---------------------
    def initialize_mappings(self, df: Optional[pd.DataFrame] = None, engine=None) -> pd.DataFrame:
        """Инициализирует/загружает маппинг.

        Порядок приоритета:
        1) symbol_mapping.pkl
        2) таблица symbol_mapping в БД (если есть engine)
        3) построение из df (fallback) + сохранение в pkl
        """
        # 1) Кэш
        cached = self._load_cache()
        if cached is not None:
            self._mapping_df = cached
            return self._mapping_df

        # 2) БД
        if engine is not None:
            try:
                tbl = pd.read_sql("SELECT * FROM symbol_mapping ORDER BY symbol_index", engine)
                if len(tbl) > 0:
                    self._mapping_df = tbl
                    self._save_cache(tbl)
                    self.logger.info("✅ Загружен mapping символов из БД")
                    return self._mapping_df
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось прочитать symbol_mapping из БД: {e}")

        # 3) Fallback из df
        if df is None or 'symbol' not in df.columns:
            raise ValueError("Для инициализации маппинга без БД требуется DataFrame с колонкой 'symbol'")

        symbols = sorted(df['symbol'].dropna().unique().tolist())
        mapping = pd.DataFrame({
            'symbol': symbols,
            'symbol_index': list(range(len(symbols))),
            'sector': [None] * len(symbols),
            'market_cap_rank': [None] * len(symbols),
            'avg_volume': [None] * len(symbols),
            'avg_volatility': [None] * len(symbols),
            'correlation_cluster': [None] * len(symbols),
            'is_active': [True] * len(symbols),
            'metadata': [None] * len(symbols),
        })
        self._mapping_df = mapping
        self._save_cache(mapping)
        self.logger.info("✅ Построен новый mapping символов (fallback)")
        return self._mapping_df

    def encode_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет в df колонки символов: symbol_index, symbol_id (тождественно),
        а также опциональные статистики если они есть в mapping.
        """
        if self._mapping_df is None:
            # Попробуем инициализировать по df
            self.initialize_mappings(df)

        out = df.copy()
        # Джоин по symbol
        out = out.merge(
            self._mapping_df[['symbol', 'symbol_index', 'sector', 'market_cap_rank']],
            on='symbol', how='left'
        )

        # Переименуем для единообразия
        out.rename(columns={'symbol_index': 'symbol_id'}, inplace=True)
        # Сохраним дублирующую колонку с индексом если нужно в дальнейшем
        out['symbol_index'] = out['symbol_id']

        # Простые численные фичи из метаданных
        if 'market_cap_rank' in out.columns:
            ranks = out['market_cap_rank'].copy()
            # Если есть хоть одно ненулевое значение — нормализуем, заполняя пропуски max'ом
            if ranks.notna().any():
                max_rank = ranks.max()
                # Защита от NaN/некорректного max
                if pd.isna(max_rank) or max_rank <= 0:
                    unique_cnt = out['symbol'].nunique()
                    max_rank = max(int(unique_cnt), 1)
                ranks = ranks.fillna(max_rank)
                out['symbol_rank_norm'] = ranks.astype('float32') / float(max_rank)
            else:
                # Полностью отсутствуют ранги — детерминированный fallback по алфавиту
                sym_order = {s: i + 1 for i, s in enumerate(sorted(out['symbol'].unique()))}
                max_rank = max(sym_order.values())
                out['symbol_rank_norm'] = out['symbol'].map(sym_order).astype('float32') / float(max_rank)

        # Если сектора нет — заполним 'unknown'
        if 'sector' in out.columns:
            out['sector'] = out['sector'].fillna('unknown')

        return out

    def get_symbol_id(self, symbol: str) -> Optional[int]:
        if self._mapping_df is None:
            cached = self._load_cache()
            if cached is None:
                return None
            self._mapping_df = cached
        row = self._mapping_df[self._mapping_df['symbol'] == symbol]
        if row.empty:
            return None
        return int(row.iloc[0]['symbol_index'])

    # --------------------- Internal utils ---------------------
    def _load_cache(self) -> Optional[pd.DataFrame]:
        if SYMBOL_CACHE_FILE.exists():
            try:
                return pd.read_pickle(SYMBOL_CACHE_FILE)
            except Exception as e:
                self.logger.warning(f"Ошибка чтения кэша symbol_mapping: {e}")
        return None

    def _save_cache(self, df: pd.DataFrame):
        try:
            df.to_pickle(SYMBOL_CACHE_FILE)
            self.logger.info(f"💾 Сохранён symbol_mapping кэш: {SYMBOL_CACHE_FILE}")
        except Exception as e:
            self.logger.warning(f"Ошибка сохранения кэша symbol_mapping: {e}")
