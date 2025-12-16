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
    "data_path": "data/data/sp500_1h",
    "features": ["open", "high", "low", "close", "volume"],
    "sliding_window": 60,
    "k": 1,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "indicator_bundle": {
        "use": True,
        "indicators": {
            "sma": [5, 10],
            "rsi": [14],
            "ema": [10],
            "macd": [(12, 26, 9)],
            "bbands": [20],
            "atr": [14],
        }
    }
}

# Model configuration
# model_config = {
#     "modelname": "lstm",
#     "d_input": 15,
#     "hidden_size": 64,
#     "num_layers": 2,
#     "dropout": 0.1,
#     "bidirectional": False,
#     "num_classes": 3,
#     "lr": 1e-3,
#     "pooling": "last",
# }
model_config = {
    "modelname": "transformer_encoder",
    "d_input": 15,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "num_classes": 3,
    "lr": 5e-4,
}

# Setup data and model
dm = DInterface(batch_size=1280, dataset_config=dataset_config)
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
    accelerator="auto",
    callbacks=[checkpoint_callback],
    log_every_n_steps=1,
)

# Train
trainer.fit(model, datamodule=dm)

# Test
# trainer.test(model, datamodule=dm)


