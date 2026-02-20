"""
Transformer Encoder с causal mask для архитектуры Conv + Transformer.
Обрабатывает последовательность [B, T, d_model], выход [B, T, d_model].
Используется выбор последнего токена для предсказания.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Маска вида (size, size): True там, где внимание запрещено (будущее)."""
    return torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device),
        diagonal=1
    )


class SinusoidalPositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование (добавляется к эмбеддингам)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Один блок: causal self-attention + FFN (GELU), с residual и LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, T, d_model)
        # Causal: use attn_mask so position i cannot attend to j > i
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder (causal): 2–6 блоков self-attention + FFN с GELU.
    Вход: [B, T, d_model], выход: [B, T, d_model].
    Позиционное кодирование добавляется к входу внутри модуля.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, T, d_model)
        key_padding_mask: (B, T), True = маскировать позицию (не использовать)
        """
        x = self.pos_encoding(x)
        T = x.size(1)
        # Causal mask: (T, T), True = не смотреть
        attn_mask = causal_mask(T, x.device)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.norm(x)
