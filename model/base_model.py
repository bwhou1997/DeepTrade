"""
Base model class for PyTorch Lightning
"""
import lightning as pl
import torch
import torch.nn.functional as F

from .metrics import MetricsBundle


class BaseModel(pl.LightningModule):
    """
    A reusable Lightning base class.
    Provides:
        - automatic metrics
        - automatic loss selection
        - shared train/val/test step
    """

    def init_task(self, task_type: str, num_classes: int = None):
        """
        Initialize task type and metrics.
        
        Args:
            task_type: "classification" or "regression"
            num_classes: Number of classes (required for classification)
        """
        self.task_type = task_type
        self.num_classes = num_classes

        if task_type == "classification":
            assert num_classes is not None
            self.train_metrics = MetricsBundle("train", num_classes)
            self.val_metrics = MetricsBundle("val", num_classes)
            self.test_metrics = MetricsBundle("test", num_classes)

    def compute_loss(self, logits, targets):
        """Automatic loss selection."""
        if self.task_type == "classification":
            return F.cross_entropy(logits, targets)

        elif self.task_type == "regression":
            return F.mse_loss(logits, targets)

        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    ## -------------------------------
    ## Lightning Steps
    ## -------------------------------
    def shared_step(self, batch, stage: str):
        """
        Shared step for train/val/test.
        
        Args:
            batch: (x, y) where x.shape = (B, seq_len, d_model), y.shape = (B,)
            stage: 'train', 'val', or 'test'
        """
        x, y = batch

        # normalization
        seq_len_mean = x.mean(dim=1, keepdim=True)
        seq_len_std = x.std(dim=1, keepdim=True) + 1e-6
        x = (x - seq_len_mean) / seq_len_std


        logits = self(x)
        loss = self.compute_loss(logits, y)

        # Classification metrics
        if self.task_type == "classification":
            metrics = {
                "train": self.train_metrics,
                "val": self.val_metrics,
                "test": self.test_metrics,
            }[stage]

            metrics.update_and_log(self, logits, y)
            self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Regression metrics (extendable)
        elif self.task_type == "regression":
            self.log(f"{stage}/mse", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")


