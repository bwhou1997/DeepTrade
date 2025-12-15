"""
Data Interface for PyTorch Lightning
"""
import torch 
import lightning as pl
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from .registry import DATASET_REGISTRY


class DInterface(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for OHLCV datasets.
    
    Args:
        batch_size: Batch size for dataloaders
        dataset_config: Configuration dictionary for dataset
    """
    
    def __init__(
        self,
        batch_size: int,
        dataset_config: dict,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.dataset_config = dataset_config
        self.num_workers = num_workers

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        assert stage in ['fit', 'test', 'predict', None]
        
        datasetname = self.dataset_config['datasetname']
        assert datasetname in DATASET_REGISTRY, \
            f"Dataset {datasetname} not registered. Available: {list(DATASET_REGISTRY.keys())}"
        
        entry = DATASET_REGISTRY[datasetname]
        Dataset = entry['dataset_cls']
        get_dataset_by_stage = entry['stage_helper']

        if stage == 'fit' or stage is None:
            self.train_ds, self.val_ds, self.test_ds = get_dataset_by_stage(
                Dataset, self.dataset_config, stage='fit'
            )
        elif stage == 'test':
            self.test_ds = get_dataset_by_stage(
                Dataset, self.dataset_config, stage='test'
            )
        elif stage == 'predict':
            self.predict_x_ds = get_dataset_by_stage(
                Dataset, self.dataset_config, stage='predict'
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    # Example usage
    dataset_config = {
        "datasetname": "OHLCV",
        "data_path": "data/AAPL.csv",
        "features": ["open", "high", "low", "close", "volume"],
        "sliding_window": 60,
        "normalization": "Zscore",
        "k": 1,
        "train_ratio": 0.8,
    }
    
    dm = DInterface(batch_size=32, dataset_config=dataset_config)
    dm.setup()
    train_loader = dm.train_dataloader()
    print(f"Number of batches: {len(train_loader)}")


