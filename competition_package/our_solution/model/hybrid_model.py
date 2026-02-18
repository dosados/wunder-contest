"""
Полный поток: X (B, T, D) → Projection → Conv → Transformer → Pooling → Head → (B, 2).
"""

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from .feature_projection import FeatureProjection
from .conv_blocks import TemporalConvBackbone
from .transformer_encoder import TransformerEncoderWithPE
from .pooling import get_pooling
from .head import PredictionHead

if TYPE_CHECKING:
    from ..config import ModelConfig


class HybridConvTransformer(nn.Module):
    """Hybrid Conv + Transformer для предсказания (t0, t1)."""

    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.config = config
        self.projection = FeatureProjection(config.input_dim, config.d_model)
        self.conv = TemporalConvBackbone(
            config.d_model,
            config.conv_kernel,
            config.conv_dilations[: config.n_conv_blocks],
        )
        self.transformer = TransformerEncoderWithPE(
            d_model=config.d_model,
            n_heads=config.n_heads,
            ff_dim=config.ff_dim,
            n_layers=config.n_transformer_layers,
            max_len=config.seq_length,
            dropout=config.dropout,
        )
        self.pooling = get_pooling(config.pooling, config.d_model)
        self.head = PredictionHead(
            config.d_model,
            hidden=config.head_hidden,
            num_targets=2,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.projection(x)
        x = self.conv(x)
        x = self.transformer(x)
        x = self.pooling(x)
        return self.head(x)
