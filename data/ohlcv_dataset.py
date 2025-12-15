"""
OHLCV Dataset for stock price prediction
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional
from .registry import register_dataset, register_stage_helper


def parse_ohlcv_config(config: dict) -> tuple:
    """
    Parse OHLCV dataset configuration.
    
    Returns:
        (data_path, features, sliding_window, normalization, k)
    """
    assert config['datasetname'] == 'OHLCV'
    assert 'data_path' in config
    assert 'sliding_window' in config
    
    data_path = config['data_path']
    features = config.get('features', ['open', 'high', 'low', 'close', 'volume'])
    sliding_window = config['sliding_window']
    normalization = config.get('normalization', 'Zscore')
    k = config.get('k', 1)  # Prediction horizon
    
    return data_path, features, sliding_window, normalization, k


@register_dataset("OHLCV")
class OHLCVDataset(Dataset):
    """
    Dataset for OHLCV (Open, High, Low, Close, Volume) stock data.
    
    Args:
        ohlcv_config: Dictionary containing:
            - data_path: Path to CSV file with OHLCV data
            - features: List of feature columns to use (default: ['open', 'high', 'low', 'close', 'volume'])
            - sliding_window: Sequence length
            - normalization: 'Zscore' or 'MinMax' (default: 'Zscore')
            - k: Prediction horizon (default: 1)
            - datatype: 'train' or 'test' (for stage helper)
    """
    
    def __init__(self, ohlcv_config: dict):
        self.config = ohlcv_config
        self.data_path, self.features, self.sliding_window, self.normalization, self.k = \
            parse_ohlcv_config(ohlcv_config)
        
        # Load data
        print(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Ensure date column exists
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Select features
        assert all(f in self.df.columns for f in self.features), \
            f"Missing features. Available: {self.df.columns.tolist()}"
        
        self.data = self.df[self.features].values.astype(np.float32)
        
        # Normalize
        if self.normalization == 'Zscore':
            self.mean = self.data.mean(axis=0)
            self.std = self.data.std(axis=0) + 1e-8
            self.data = (self.data - self.mean) / self.std
        elif self.normalization == 'MinMax':
            self.min = self.data.min(axis=0)
            self.max = self.data.max(axis=0) + 1e-8
            self.data = (self.data - self.min) / (self.max - self.min)
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        
        # Generate labels: 0=DOWN, 1=FLAT, 2=UP
        # Based on close price change over k periods
        close_idx = self.features.index('close')
        close_prices = self.data[:, close_idx]
        
        # Future close price
        future_close = np.roll(close_prices, -self.k)
        future_close[-self.k:] = close_prices[-self.k:]  # Handle boundary
        
        # Calculate returns
        returns = (future_close - close_prices) / (close_prices + 1e-8)
        
        # Create labels: threshold-based classification
        threshold = 0.001  # 0.1% threshold
        self.labels = np.zeros(len(returns), dtype=np.int64)
        self.labels[returns > threshold] = 2  # UP
        self.labels[returns < -threshold] = 0  # DOWN
        self.labels[(returns >= -threshold) & (returns <= threshold)] = 1  # FLAT
        
        self.num_datapoints = len(self.data) - self.sliding_window - self.k + 1
    
    def __len__(self):
        return max(0, self.num_datapoints)
    
    def __getitem__(self, idx):
        """
        Returns:
            sequence: (sliding_window, num_features) tensor
            label: scalar (0=DOWN, 1=FLAT, 2=UP)
        """
        if idx >= self.num_datapoints:
            raise IndexError(f"Index {idx} out of range")
        
        start = idx
        end = idx + self.sliding_window
        
        sequence = torch.from_numpy(self.data[start:end]).float()
        label = torch.tensor(self.labels[end - 1], dtype=torch.long)
        
        return sequence, label


@register_stage_helper("OHLCV")
def OHLCV_dataset_stage_helper(Dataset, config: dict, stage: Optional[str] = None) -> OHLCVDataset:
    """
    Stage helper for OHLCV dataset.
    Splits data into train/val/test sets.
    """
    train_ratio = config.pop('train_ratio', 0.8)
    val_ratio = config.pop('val_ratio', 0.1)
    
    if stage == 'fit' or stage is None:
        # Load full dataset
        full_dataset = Dataset({**config, 'datatype': 'train'})
        
        # Split indices
        total_size = len(full_dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # Create indices
        indices = np.arange(total_size)
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_ds = Subset(full_dataset, train_indices)
        val_ds = Subset(full_dataset, val_indices)
        test_ds = Subset(full_dataset, test_indices)
        
        return train_ds, val_ds, test_ds
    
    elif stage == 'test':
        test_ds = Dataset({**config, 'datatype': 'test'})
        return test_ds
    
    elif stage == 'predict':
        raise NotImplementedError("Predict stage not implemented yet.")


if __name__ == "__main__":
    # Test dataset
    config = {
        "datasetname": "OHLCV",
        "data_path": "data/AAPL.csv",
        "features": ["open", "high", "low", "close", "volume"],
        "sliding_window": 60,
        "normalization": "Zscore",
        "k": 1,
        "train_ratio": 0.8,
    }
    
    dataset = OHLCVDataset(config)
    print(f"Dataset length: {len(dataset)}")
    print(f"Sequence shape: {dataset[0][0].shape}")
    print(f"Label: {dataset[0][1]}")


