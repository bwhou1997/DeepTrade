"""
OHLCV Dataset for multi-stock price prediction (sliding window)
"""
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
from .registry import register_dataset, register_stage_helper
from .indicator import IndicatorBundle


# def normalizer(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Z-score normalization for all features.
#     Assumes df is time-sorted.
#     """

#     eps = 1e-8
#     norm_df = pd.DataFrame(index=df.index)

#     for col in df.columns:
#         x = df[col].values.astype(np.float32)
#         mean = x.mean()
#         std = x.std()
#         norm_df[col] = (x - mean) / (std + eps)

#     return norm_df

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
        self.data_path = ohlcv_config["data_path"]
        self.features = ohlcv_config["features"]
        self.sliding_window = ohlcv_config["sliding_window"]
        self.k = ohlcv_config["k"]

        # ---------- Indicator bundle ----------
        indicator_cfg = ohlcv_config.get("indicator_bundle", None)
        self.indicator_bundle = (
            IndicatorBundle(indicator_cfg)
            if indicator_cfg is not None
            else None
        )

        self.stock_data = []      # List[np.ndarray] (T_i, d)
        self.stock_labels = []    # List[np.ndarray] (T_i,)
        self.index_map = []       # List[(stock_id, local_idx)]

        csv_files = load_csv_list(self.data_path)
        assert len(csv_files) > 0, "No CSV files found"

        for stock_id, csv_path in enumerate(csv_files):
            self._load_single_stock(csv_path, stock_id)

        # print statistic of labels (up, down, flat)
        total_counts = np.zeros(3, dtype=np.int64)
        for labels in self.stock_labels:
            for c in range(3):
                total_counts[c] += np.sum(labels == c)
        total = total_counts.sum()
        print(
            f"[OHLCVDataset] Label distribution: "
            f"DOWN={total_counts[0]} ({total_counts[0]/total:.2%}), "
            f"FLAT={total_counts[1]} ({total_counts[1]/total:.2%}), "
            f"UP={total_counts[2]} ({total_counts[2]/total:.2%})"
        )

        print(
            f"[OHLCVDataset] Loaded {len(csv_files)} stocks, "
            f"{len(self.index_map)} total samples"
        )

    def _load_single_stock(self, csv_path: str, stock_id: int):
        df = pd.read_csv(csv_path)

        # ---------- sort by time ----------
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values("date").reset_index(drop=True)

        # ---------- apply indicators ----------
        if self.indicator_bundle is not None:
            df = self.indicator_bundle.transform(df)

        # ---------- drop NaN (VERY IMPORTANT) ----------
        # caused by rolling indicators
        df = df.dropna().reset_index(drop=True)

        # ---------- feature selection ----------
        assert all(f in df.columns for f in self.features), \
            f"{csv_path} missing features"

        # get rid of data and transform to numpy
        df = df.drop(columns=["date"])
        # df = normalizer(df)
        data = df.values.astype(np.float32)

        # ---------- label generation ----------
        close_idx = self.features.index("close")
        close = data[:, close_idx]

        future_close = np.roll(close, -self.k)
        future_close[-self.k:] = close[-self.k:]

        returns = (future_close - close) / (close + 1e-8)

        labels = np.zeros(len(returns), dtype=np.int64)
        threshold = 0.005
        labels[returns > threshold] = 2  # UP
        labels[returns < -threshold] = 0  # DOWN
        labels[
            (returns >= -threshold) & (returns <= threshold)
        ] = 1  # FLAT

        self.stock_data.append(data)
        self.stock_labels.append(labels)

        # ---------- build index map ----------
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

        # normalize x by z-score per sample (normalize within sliding window)
        x_mean = x.mean(axis=0, keepdims=True)
        x_std = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - x_mean) / x_std

        return (
            torch.from_numpy(x).float(),      # (seq_len, d)
            torch.tensor(y, dtype=torch.long)
        )

@register_stage_helper("OHLCV")
def OHLCV_dataset_stage_generator(
    Dataset,
    config: dict,
    stage: Optional[str] = None,
):
    train_ratio = config.get("train_ratio", 0.8)
    val_ratio = config.get("val_ratio", 0.1)

    full_dataset = Dataset(config)

    # -------- collect indices by stock --------

    stock_to_indices = defaultdict(list)

    for global_idx, (stock_id, t) in enumerate(full_dataset.index_map):
        stock_to_indices[stock_id].append((t, global_idx))

    train_indices = []
    val_indices = []
    test_indices = []

    # -------- per-stock time split --------
    for stock_id, items in stock_to_indices.items():
        # items: [(t, global_idx), ...]
        items.sort(key=lambda x: x[0])  # sort by time

        n = len(items)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_indices.extend([idx for _, idx in items[:train_end]])
        val_indices.extend([idx for _, idx in items[train_end:val_end]])
        test_indices.extend([idx for _, idx in items[val_end:]])

    from torch.utils.data import Subset
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Test dataset
    import matplotlib.pyplot as plt
    config = {
        "data_path": "./data/sp500_1h",
        "features": ["open", "high", "low", "close", "volume"],
        "sliding_window": 60,
        "k": 5,
        "train_ratio": 0.8,
        "valid_ratio": 0.1,
        "indicator_bundle": {
            "use": True,
            "indicators": {
                "sma": [5, 10],
                "rsi": [14],
                "ema": [10],
                "macd": [(12, 26, 9)],
                # "bbands": [20],
                "atr": [14],
            }
        }
    }
    
    dataset = OHLCVDataset(config)
    print(f"Dataset length: {len(dataset)}")
    print(f"Sequence shape: {dataset[0][0].shape}")
    print(f"Label: {dataset[0][1]}")


