# DeepTrade

A deep learning framework for stock trading with OHLCV data, featuring model training and backtesting capabilities.

## Project Structure

```
DeepTrade/
├── main.py              # Main training script
├── downloader.py        # Stock data downloading script
├── example_backtest.py  # Example backtesting script
├── data/                # Data loading and preprocessing
│   ├── ohlcv_dataset.py      # OHLCV dataset implementation
│   ├── data_interface.py     # PyTorch Lightning DataModule
│   ├── download.py           # Stock data download utilities
│   └── registry.py           # Dataset registry pattern
├── model/              # Deep learning models
│   ├── base_model.py         # Base Lightning module
│   ├── model_interface.py    # Model factory
│   ├── metrics.py            # Custom metrics
│   └── registry.py           # Model registry pattern
├── backtesting/         # Backtesting engine
│   ├── strategy.py           # Trading strategies
│   └── engine.py             # Backtrader integration
├── tests/               # Unit tests
└── README.md            # This file
```

## Features

- **OHLCV Data Support**: Works with standard OHLCV (Open, High, Low, Close, Volume) data
- **Data Download**: Download stock data from external sources (Yahoo Finance, etc.)
- **Model Training**: Train deep learning models for stock prediction
- **Backtesting**: Backtest trading strategies using backtrader
- **Modular Design**: Easy to extend with new models and datasets

## Quick Start

### UV Setup

```bash
cd DeepTrade
uv sync
source .venv/bin/activate # check if env is loaded by 'which python'
```

### Basic Usage
```bash
python downloader.py         # download MSFT by default
python main.py               # train a transformer model (rename the best training version as best_msft)
python example_backtest.py   # run backtest on trained model
```