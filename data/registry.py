from typing import Callable, Dict, Type
from torch.utils.data import Dataset

DATASET_REGISTRY: Dict[str, Dict] = {}

def register_dataset(name: str):
    """
    Register a dataset and its stage helper
    """
    def decorator(
        dataset_cls: Type[Dataset]
    ):
        if name in DATASET_REGISTRY:
            raise KeyError(f"Dataset {name} already registered")

        DATASET_REGISTRY[name] = {
            "dataset_cls": dataset_cls,
            "stage_helper": None,
        }
        return dataset_cls
    return decorator

def register_stage_helper(name: str):
    def decorator(func: Callable):
        if name not in DATASET_REGISTRY:
            raise KeyError(f"Dataset {name} must be registered first")
        DATASET_REGISTRY[name]["stage_helper"] = func
        return func
    return decorator


