# DeepTrade

A deep learning framework for stock trading with OHLCV data, featuring model training and backtesting capabilities.

## Project Structure

```
DeepTrade/
├── data/              # Data loading and preprocessing
│   ├── ohlcv_dataset.py    # OHLCV dataset implementation
│   ├── data_interface.py   # PyTorch Lightning DataModule
│   ├── download.py         # Stock data download utilities
│   └── registry.py         # Dataset registry pattern
├── model/             # Deep learning models
│   ├── base_model.py       # Base Lightning module
│   ├── model_interface.py  # Model factory
│   ├── metrics.py          # Custom metrics
│   └── registry.py         # Model registry pattern
├── backtesting/       # Backtesting engine
│   ├── strategy.py         # Trading strategies
│   └── engine.py           # Backtrader integration
├── tests/             # Unit tests
└── README.md          # This file
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
git clone
cd DeepTrade
uv sync
source .venv/bin/activate
```


### Download Stock Data

```python
from data.download import download_stock_data

# Download data from Yahoo Finance
df = download_stock_data('AAPL', start='2020-01-01', end='2023-12-31')
df.to_csv('data/AAPL.csv', index=False)
```

### Train a Model

```python
from data import DInterface
from model import MInterface
import lightning as pl

# Setup data
dataset_config = {
    "datasetname": "OHLCV",
    "data_path": "data/AAPL.csv",
    "sliding_window": 60,
    "features": ["open", "high", "low", "close", "volume"],
    "normalization": "Zscore",
    "train_ratio": 0.8,
}

dm = DInterface(batch_size=32, dataset_config=dataset_config)

# Setup model
model_config = {
    "modelname": "lstm",
    "d_input": 5,
    "hidden_size": 64,
    "num_layers": 2,
    "num_classes": 3,  # UP, FLAT, DOWN
    "lr": 1e-3,
}

model = MInterface(model_config)

# Train
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, datamodule=dm)
```

### Backtest Strategy

```python
from backtesting.engine import BacktestEngine
from backtesting.strategy import MLStrategy

engine = BacktestEngine(
    data_path="data/AAPL.csv",
    initial_cash=10000,
    commission=0.001
)

strategy = MLStrategy(model=model, lookback=60)
engine.add_strategy(strategy)
results = engine.run()

print(f"Final Portfolio Value: {results['final_value']}")
print(f"Total Return: {results['total_return']:.2%}")
```

## Requirements

- PyTorch
- PyTorch Lightning
- backtrader
- pandas
- numpy
- yfinance (for data download)


