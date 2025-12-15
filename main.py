"""
Main training script for DeepTrade
"""
from data import DInterface
from model import MInterface
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


# Dataset configuration
dataset_config = {
    "datasetname": "OHLCV",
    "data_path": "./data/AAPL.csv",
    "features": ["open", "high", "low", "close", "volume"],
    "sliding_window": 60,
    "normalization": "Zscore",
    "k": 1,
    "train_ratio": 0.8,
}

# Model configuration
model_config = {
    "modelname": "lstm",
    "d_input": 5,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "bidirectional": False,
    "num_classes": 3,
    "lr": 1e-3,
    "pooling": "last",
}

# Setup data and model
dm = DInterface(batch_size=32, dataset_config=dataset_config)
model = MInterface(model_config)

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val/precision_up",
    mode="max",
    save_top_k=1,
    save_last=True,
    filename="best-{epoch}-{val_precision_up:.4f}",
)

# Trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu" if pl.utilities.device_parser.num_cuda_devices() > 0 else "cpu",
    devices="auto",
    callbacks=[checkpoint_callback],
    log_every_n_steps=10,
)

# Train
trainer.fit(model, datamodule=dm)

# Test
trainer.test(model, datamodule=dm)


