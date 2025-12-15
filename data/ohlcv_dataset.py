"""
OHLCV Dataset for multi-stock price prediction (sliding window)
"""
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from registry import register_dataset, register_stage_helper


def parse_ohlcv_config(config: dict):
    assert config["datasetname"] == "OHLCV"
    assert "data_path" in config
    assert "sliding_window" in config

    data_path = config["data_path"]
    features = config.get(
        "features", ["open", "high", "low", "close", "volume"]
    )
    sliding_window = config["sliding_window"]
    normalization = config.get("normalization", "Zscore")
    k = config.get("k", 1)

    return data_path, features, sliding_window, normalization, k


def load_csv_list(data_path) -> List[str]:
    """
    data_path can be:
      - directory: data/*.csv
      - list of csv paths
      - single csv file
    """
    if isinstance(data_path, list):
        return data_path

    if os.path.isdir(data_path):
        return [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".csv")
        ]

    if data_path.endswith(".csv"):
        return [data_path]

    raise ValueError(f"Invalid data_path: {data_path}")


@register_dataset("OHLCV")
class OHLCVDataset(Dataset):
    """
    Multi-stock OHLCV Dataset with per-stock sliding windows
    """

    def __init__(self, ohlcv_config: dict):
        self.config = ohlcv_config
        (
            self.data_path,
            self.features,
            self.sliding_window,
            self.normalization,
            self.k,
        ) = parse_ohlcv_config(ohlcv_config)

        self.stock_data = []      # List[np.ndarray] (T_i, d)
        self.stock_labels = []    # List[np.ndarray] (T_i,)
        self.index_map = []       # List[(stock_id, local_idx)]

        csv_files = load_csv_list(self.data_path)
        assert len(csv_files) > 0, "No CSV files found"

        for stock_id, csv_path in enumerate(csv_files):
            self._load_single_stock(csv_path, stock_id)

        print(
            f"[OHLCVDataset] Loaded {len(csv_files)} stocks, "
            f"{len(self.index_map)} total samples"
        )

    def _load_single_stock(self, csv_path: str, stock_id: int):
        df = pd.read_csv(csv_path)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values("date").reset_index(drop=True)

        assert all(f in df.columns for f in self.features), \
            f"{csv_path} missing features"

        data = df[self.features].values.astype(np.float32)

        # ----- label generation -----
        close_idx = self.features.index("close")
        close = data[:, close_idx]

        future_close = np.roll(close, -self.k)
        future_close[-self.k:] = close[-self.k:]

        returns = (future_close - close) / (close + 1e-8)

        labels = np.zeros(len(returns), dtype=np.int64)
        threshold = 0.001
        labels[returns > threshold] = 2  # UP
        labels[returns < -threshold] = 0  # DOWN
        labels[
            (returns >= -threshold) & (returns <= threshold)
        ] = 1  # FLAT

        self.stock_data.append(data)
        self.stock_labels.append(labels)

        # ----- build index map -----
        max_t = len(data) - self.sliding_window - self.k + 1
        for t in range(max(0, max_t)):
            self.index_map.append((stock_id, t))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        stock_id, t = self.index_map[idx]

        data = self.stock_data[stock_id]
        labels = self.stock_labels[stock_id]

        x = data[t : t + self.sliding_window]
        y = labels[t + self.sliding_window - 1]

        return (
            torch.from_numpy(x).float(),      # (seq_len, d)
            torch.tensor(y, dtype=torch.long)
        )


@register_stage_helper("OHLCV")
def OHLCV_dataset_stage_generator(Dataset, config: dict, stage: Optional[str] = None) -> OHLCVDataset:
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
        "data_path": "./data/sp500_3stocks",
        "features": ["open", "high", "low", "close", "volume"],
        "sliding_window": 60,
        "k": 1,
        "train_ratio": 0.8,
    }
    
    dataset = OHLCVDataset(config)
    print(f"Dataset length: {len(dataset)}")
    print(f"Sequence shape: {dataset[0][0].shape}")
    print(f"Label: {dataset[0][1]}")


