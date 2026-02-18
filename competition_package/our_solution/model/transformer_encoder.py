"""
Transformer Encoder с learnable positional encoding.
Вход/выход: (B, T, d_model).
"""

import math
import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """Один слой: Self-Attention → Residual → LN → FFN → Residual → LN."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(residual + self.dropout(x))
        residual = x
        x = self.ff(x)
        x = self.norm2(residual + self.dropout(x))
        return x


class TransformerEncoderWithPE(nn.Module):
    """Transformer encoder с learnable positional encoding. Вход: (B, T, d_model)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        n_layers: int,
        max_len: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        x = x + self.pos_embed[:, :T, :]
        for layer in self.layers:
            x = layer(x, mask)
        return x
