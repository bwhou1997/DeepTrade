"""
Main training script for DeepTrade
"""
from data import DInterface
from model import MInterface
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint


dataset_config = {
    "datasetname": "OHLCV",
    "data_path": "./data/data/msft",
    "features": ["open", "high", "low", "close", "volume"],
    "sliding_window": 60,
    "k": 5,
    "train_ratio": 0.8,
    "valid_ratio": 0.1,
    "indicator_bundle": {
        "use": False,
        "indicators": {
            "sma": [5, 10, 20],
            "rsi": [14],
            "ema": [10],
            "macd": [(12, 26, 9)],
            "atr": [14],
        }
    }
}

# Model configuration
# model_config = {
#     "modelname": "lstm",
#     "d_input": 13,
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
    "d_input": 5,
    "d_model": 32,
    "nhead": 2,
    "num_layers": 1,
    "dim_feedforward": 64,
    "dropout": 0.3,
    "num_classes": 3,
    "lr": 1e-3,
}

# Setup data and model
dm = DInterface(batch_size=256, dataset_config=dataset_config)
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
    max_epochs=100,
    accelerator="auto",
    callbacks=[checkpoint_callback],
    log_every_n_steps=1,
)

# Train
trainer.fit(model, datamodule=dm)

# Test
# trainer.test(model, datamodule=dm)


