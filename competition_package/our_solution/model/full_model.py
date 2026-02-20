"""
Полная модель Conv + Transformer по description.md.
Цепочка: FeatureProjection → TemporalConvStack → TransformerEncoder → last token → Head.
"""

import torch
import torch.nn as nn
from typing import Optional

from .feature_projection import FeatureProjection
from .conv_blocks import TemporalConvStack
from .transformer_encoder import TransformerEncoder
from .model_state import ModelState


class PredictionHead(nn.Module):
    """MLP: d_model → 64 → 2 для предсказания двух таргетов (t0, t1)."""

    def __init__(self, d_model: int, hidden_size: int = 64, num_targets: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FullModel(nn.Module):
    """
    Полная модель по description.md:
    Вход [B, T, D] → проекция → Conv1D по времени → позиционное кодирование + Transformer (causal) →
    последний токен → Head → [B, 2].
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        conv_window: int = 3,
        trans_window: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: Optional[int] = None,
        head_hidden: int = 64,
        num_targets: int = 2,
        dropout: float = 0.1,
        conv_kernel_sizes: Optional[list] = None,
        conv_dilations: Optional[list] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.conv_window = conv_window
        self.trans_window = trans_window

        self.projection = FeatureProjection(input_dim, d_model)
        self.conv = TemporalConvStack(
            d_model,
            kernel_sizes=conv_kernel_sizes,
            dilations=conv_dilations,
        )
        self.transformer = TransformerEncoder(
            d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.head = PredictionHead(d_model, hidden_size=head_hidden, num_targets=num_targets)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, T, D) — окно состояний (без таргет-фич).
        key_padding_mask: (B, T), True = маскировать позицию (опционально).
        Возвращает: (B, num_targets) — предсказания для t0, t1.
        """
        x = self.projection(x)           # (B, T, d_model)
        x = self.conv(x)                 # (B, T, d_model)
        x = self.transformer(x, key_padding_mask=key_padding_mask)  # (B, T, d_model)
        x = x[:, -1]                     # (B, d_model) — последний токен
        return self.head(x)              # (B, num_targets)

    def create_state(self, device: str = "cpu") -> ModelState:
        """
        Создаёт объект состояния для пошагового инференса.
        conv_buffer хранит векторы размера d_model (выход Conv).
        """
        return ModelState(
            input_dim=self.input_dim,
            conv_window=self.conv_window,
            trans_window=self.trans_window,
            conv_feat_dim=self.d_model,
            device=device,
        )
