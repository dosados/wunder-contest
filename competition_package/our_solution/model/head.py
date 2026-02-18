"""Prediction head: Linear(d_model → 128) → GELU → Dropout → Linear(128 → 2)."""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128, num_targets: int = 2, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
