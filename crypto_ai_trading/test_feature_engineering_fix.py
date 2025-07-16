import pandas as pd
import numpy as np

from data.feature_engineering import FeatureEngineer


def make_df():
    n = 14
    data = {
        'open': np.full(n, 100.0),
        'high': np.full(n, 101.0),
        'low': np.full(n, 99.0),
        'close': np.full(n, 100.0),
        'volume': np.arange(1, n+1),
        'turnover': np.arange(1, n+1) * 100.0,
        'returns': np.full(n, 0.01),
        'atr': np.full(n, 1.0),
        'bb_width': [0.05]*5 + [0.01]*4 + [0.03]*2 + [0.01]*3,
        'rsi': np.linspace(30, 70, n),
        'macd': np.linspace(-1, 1, n),
    }
    return pd.DataFrame(data)


def test_price_impact_and_toxicity():
    df = make_df()
    engineer = FeatureEngineer({'features': {}})
    result = engineer._create_microstructure_features(df.copy())
    assert result['price_impact'].max() < 0.001
    assert result['toxicity'].mean() > 0.99


def test_volatility_squeeze_duration():
    df = make_df()
    engineer = FeatureEngineer({'features': {}})
    # bb_width already set to produce squeeze pattern
    df['bb_width'] = [0.05]*5 + [0.01]*4 + [0.03]*2 + [0.01]*3
    result = engineer._create_rally_detection_features(df.copy())
    expected = [0,0,0,0,0,1,2,3,4,0,0,1,2,3]
    assert result['volatility_squeeze_duration'].tolist() == expected
