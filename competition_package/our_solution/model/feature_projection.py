"""Feature projection: Linear(D → d_model)."""

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """Преобразование (B, T, D) → (B, T, d_model)."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
