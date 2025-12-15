"""
Model Interface for creating models from configuration
"""
import torch
import lightning as pl
from typing import Optional
from .registry import MODEL_REGISTRY


def MInterface(model_config: dict) -> pl.LightningModule:
    """
    Create a model from configuration dictionary.
    
    Args:
        model_config: Dictionary containing 'modelname' and model-specific parameters
    
    Returns:
        PyTorch Lightning module
    """
    modelname = model_config.get("modelname")
    
    if modelname not in MODEL_REGISTRY:
        raise KeyError(
            f"Model {modelname} not registered. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_class = MODEL_REGISTRY[modelname]
    return model_class(**model_config)


# Import all model classes to ensure they are registered
from .lstm import LSTMClassifier
# Add more imports here as you create new models
# from .transformer import TransformerEncoder
# from .cnn import SimpleCNN

