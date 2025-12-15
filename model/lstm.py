"""
LSTM model for OHLCV sequence prediction
"""
import torch
import torch.nn as nn
from .registry import register_model
from .base_model import BaseModel


@register_model("lstm")
class LSTMClassifier(BaseModel):
    """
    LSTM classifier for OHLCV sequences.
    
    Args:
        d_input: Input feature dimension (e.g., 5 for OHLCV)
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
        num_classes: Number of output classes (default: 3 for UP/FLAT/DOWN)
        lr: Learning rate
        pooling: 'last' or 'mean' pooling strategy
    """
    
    def __init__(
        self,
        d_input=5,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
        num_classes=3,
        lr=1e-3,
        pooling="last",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.init_task("classification", num_classes=num_classes)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, seq_len, d_input)
        
        Returns:
            logits: (B, num_classes)
        """
        # LSTM output: (B, seq_len, hidden * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.pooling == "last":
            if self.bidirectional:
                # Last layer, forward + backward
                h_forward = h_n[-2]
                h_backward = h_n[-1]
                feat = torch.cat([h_forward, h_backward], dim=1)
            else:
                feat = h_n[-1]  # (B, hidden)
        elif self.pooling == "mean":
            feat = lstm_out.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.classifier(feat)
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


