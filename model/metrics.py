"""
Metrics bundle for model evaluation
"""
import torch.nn as nn
from torchmetrics.classification import Accuracy, F1Score, CohenKappa, Precision, Recall


class MetricsBundle(nn.Module):
    """
    Metrics bundle for classification tasks.
    Tracks per-class and global metrics.
    """
    
    def __init__(self, prefix: str, num_classes: int):
        super().__init__()
        self.prefix = prefix.rstrip("/") + "/"

        # Global metrics
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.kappa = CohenKappa(task="multiclass", num_classes=num_classes)

        # Per-class metrics (return metrics for each class)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average=None)

        # Macro metrics
        self.f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def update_and_log(self, pl_module, logits, targets):
        """
        Update metrics and log to PyTorch Lightning.
        
        Args:
            pl_module: PyTorch Lightning module
            logits: Model predictions (B, num_classes)
            targets: Ground truth labels (B,)
        """
        acc = self.acc(logits, targets)
        precision = self.precision(logits, targets)   # shape = (num_classes,)
        recall = self.recall(logits, targets)         # shape = (num_classes,)
        f1 = self.f1(logits, targets)                 # shape = (num_classes,)
        f1_macro = self.f1_macro(logits, targets)
        kappa = self.kappa(logits, targets)

        # Global Metrics
        pl_module.log(self.prefix + "acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(self.prefix + "kappa", kappa, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(self.prefix + "f1_macro", f1_macro, on_step=False, on_epoch=True, sync_dist=True)

        # Per-class Precision
        pl_module.log(self.prefix + "precision_down", precision[0], on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(self.prefix + "precision_flat", precision[1], on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(self.prefix + "precision_up", precision[2], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Per-class Recall
        pl_module.log(self.prefix + "recall_down", recall[0], on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(self.prefix + "recall_flat", recall[1], on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(self.prefix + "recall_up", recall[2], on_step=False, on_epoch=True, sync_dist=True)

        # Per-class F1
        pl_module.log(self.prefix + "f1_down", f1[0], on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(self.prefix + "f1_flat", f1[1], on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log(self.prefix + "f1_up", f1[2], on_step=False, on_epoch=True, sync_dist=True)

        return acc, f1_macro, kappa


