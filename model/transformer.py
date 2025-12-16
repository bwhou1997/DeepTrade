import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model
import lightning as pl
from .base_model import BaseModel

# @register_model("transformer_encoder")
# class TransformerEncoder(BaseModel):
#     def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, num_classes=3, lr=1e-3):
#         super().__init__()

#         self.save_hyperparameters()

#         # Initialize base model for classification task
#         self.init_task("classification", num_classes=num_classes)

#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
#                                                    nhead=nhead, 
#                                                    dim_feedforward=dim_feedforward, 
#                                                    dropout=dropout,
#                                                    batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.classifier = nn.Linear(d_model, num_classes)

#     def forward(self, x):
#         """
#         x: (B, seq_len, d_model)
#         """
#         encoded = self.transformer_encoder(x)  # (B, seq_len, d_model)
#         encoded = encoded.mean(dim=1)  # mean over seq_len
#         logits = self.classifier(encoded)
#         return logits

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         return optimizer


class FlashMHALayer(nn.Module):
    """MHA using PyTorch 2.x SDPA (Flash Attention Kernel if available)"""
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = dropout

    def forward(self, x):
        """
        x: (B, L, d_model)
        """
        B, L, _ = x.shape

        # (B, L, 3*d_model)
        qkv = self.qkv(x)

        # split → q, k, v (B, L, d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape into heads (B, nhead, L, head_dim)
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # Flash/SDPA kernel
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        # reshape back
        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.d_model)

        return self.out_proj(attn_output)


class FlashTransformerLayer(nn.Module):
    """Drop-in replacement of TransformerEncoderLayer using SDPA"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()

        self.self_attn = FlashMHALayer(d_model, nhead, dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FlashAttention
        attn_out = self.self_attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feedforward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


@register_model("transformer_encoder")
class TransformerEncoder(BaseModel):
    def __init__(self, d_input=40, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1,
                 num_classes=3, lr=1e-3,
                 max_seq_len=250,
                 **kwargs):   
        super().__init__()
        self.save_hyperparameters()
        self.init_task("classification", num_classes=num_classes)

        # proj 40 → 128
        self.input_proj = nn.Linear(d_input, d_model)

        #learnable positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Build Transformer layers
        layers = []
        for _ in range(num_layers):
            layers.append(
                FlashTransformerLayer(
                    d_model, nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                )
            )

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (B, seq_len, d_input=40)
        """
        B, L, _ = x.shape

        x = self.input_proj(x)

        x = x + self.pos_emb[:, :L, :]

        encoded = self.encoder(x)
        encoded = encoded.mean(dim=1)
        logits = self.classifier(encoded)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
