# Crypto AI Trading System

Advanced machine learning system for cryptocurrency trading using PatchTST architecture with multi-task learning for 20 target variables prediction.

## 🚀 Project Overview

This repository contains a sophisticated cryptocurrency trading system that leverages:
- **PatchTST Architecture**: State-of-the-art time series transformer
- **Multi-Task Learning**: Simultaneous prediction of 20 target variables
- **171 Technical Indicators**: Comprehensive feature engineering
- **GPU Optimized**: Built for RTX 5090 with advanced optimizations

## 📁 Project Structure

```
LLM TRANSFORM/
└── crypto_ai_trading/       # Main trading system
    ├── config/              # Configuration files
    │   └── config.yaml     # Main configuration
    ├── data/               # Data processing modules
    │   ├── data_loader.py
    │   ├── dataset.py
    │   ├── feature_engineering.py
    │   └── precomputed_dataset.py
    ├── models/             # Model architecture
    │   ├── patchtst_unified.py  # Main PatchTST model
    │   ├── losses.py
    │   └── components.py
    ├── training/           # Training modules
    │   ├── optimized_trainer.py
    │   ├── optimizer.py
    │   └── validator.py
    ├── trading/            # Trading logic
    │   ├── unified_backtester.py
    │   ├── risk_manager.py
    │   └── signals.py
    ├── utils/              # Utilities
    │   ├── logger.py
    │   ├── metrics.py
    │   └── visualization.py
    ├── validation/         # Validation suite
    │   └── comprehensive_validator.py
    ├── main.py            # Main entry point
    ├── README.md          # Project documentation
    ├── CLAUDE.md          # Claude AI instructions
    └── requirements.txt   # Dependencies
```

## 🎯 Key Features

### Model Architecture
- **UnifiedPatchTST**: Transformer-based model with patch embedding
- **20 Target Variables**: Returns, directions, TP/SL levels, risk metrics
- **Context Window**: 96 timesteps (24 hours on 15-min candles)
- **Multi-Head Architecture**: Specialized heads for different prediction tasks

### Technical Indicators (171 total)
- **Trend**: EMA, SMA, ADX, MACD, Ichimoku, SAR, Aroon
- **Oscillators**: RSI, Stochastic, CCI, Williams %R, MFI
- **Volatility**: ATR, Bollinger Bands, Donchian, Keltner
- **Volume**: OBV, CMF, VWAP, Volume Profile
- **Patterns**: Candlestick patterns, support/resistance levels
- **Microstructure**: Bid/ask spread, volume imbalance, flow toxicity

### Performance Optimizations
- **RTX 5090 Optimized**: Mixed precision (FP16), TF32 enabled
- **Batch Size**: 8192 with gradient accumulation
- **Speed**: 60,000+ samples/s on batches
- **GPU Utilization**: 90-98% target utilization

## 🚀 Quick Start

### 1. Installation

```bash
cd crypto_ai_trading
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/config.yaml` to set:
- Database connection (PostgreSQL port 5555)
- GPU settings
- Training parameters
- Target symbols list

### 3. Running the System

```bash
# Full pipeline: data + training
python main.py --mode full

# Training only (if data is ready)
python main.py --mode train

# Data preparation only
python main.py --mode data

# Demo mode (10 symbols)
python main.py --mode demo

# Backtesting
python main.py --mode backtest

# Interactive menu
python main.py --mode interactive
```

### 4. Monitoring

```bash
# TensorBoard monitoring
tensorboard --logdir logs/

# Real-time monitoring
python monitor_training.py

# GPU monitoring
nvidia-smi -l 1
```

## 📊 Target Variables (20)

1. **Returns** (4): future_return_15m, 1h, 4h, 12h
2. **Directions** (4): direction_15m, 1h, 4h, 12h (LONG/SHORT/FLAT)
3. **LONG Levels** (4): will_reach_1%, 2%, 3%, 5% in 4h/12h
4. **SHORT Levels** (4): equivalent for short positions
5. **Risk Metrics** (2): max_drawdown, max_rally in 1h/4h
6. **Trading Signals** (2): best_action, signal_strength

## 🔧 System Requirements

- **Python**: 3.10+
- **PostgreSQL**: 14+ (port 5555)
- **GPU**: NVIDIA RTX 3090+ (24GB+ VRAM recommended)
- **RAM**: 32GB+
- **Storage**: 100GB+ for data cache
- **CUDA**: 12.1+
- **PyTorch**: 2.2.0+

## 📈 Performance Metrics

### Target Performance
- **Win Rate**: 50-52% (optimal for crypto)
- **F1 Score**: > 0.42
- **Precision**: > 0.30
- **Train Loss**: 1.41-1.42 (stable)
- **Val Loss**: 1.34-1.35 (lower than train = good generalization)

### Training Speed
- **Samples/s**: 28,000-30,000 per epoch
- **Batch Processing**: 60,000+ samples/s
- **Epoch Time**: ~5-10 minutes (full dataset)

## 🛠️ Development

### Database Schema

```sql
crypto_trading/
├── raw_market_data         # Raw OHLCV data
├── processed_market_data   # Data with indicators
├── model_metadata          # Model metadata
├── model_predictions       # Predictions
└── training_sequences      # Training sequences
```

### Adding New Features

1. Update feature engineering in `data/feature_engineering.py`
2. Modify config in `config/config.yaml`
3. Retrain the model with new features

### Custom Trading Strategies

Implement custom strategies in `trading/signals.py`:
```python
class CustomStrategy(BaseStrategy):
    def generate_signals(self, predictions):
        # Your strategy logic here
        pass
```

## 🔍 Troubleshooting

### CUDA Pin Memory Error
Fixed via custom_collate_fn in PrecomputedDataset

### RTX 5090 torch.compile Issue
Disabled as sm_120 is not supported

### Large Batch OOM
Use gradient accumulation in config

### Low Entropy Predictions
Increase dropout, add label smoothing

### Class Imbalance
Use weighted loss function in config

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation in CLAUDE.md
- Review config examples in config/config.yaml

## 🚨 Disclaimer

This system is for educational purposes. Cryptocurrency trading involves significant risk. Always do your own research and never invest more than you can afford to lose.

---

**Repository**: [github.com/obertruper/crypto-ml-trading-system](https://github.com/obertruper/crypto-ml-trading-system)

**Last Updated**: September 2025