"""Pooling: (B, T, d_model) → (B, d_model)."""

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class AttentionPooling(nn.Module):
    """α_t = softmax(W · H_t), out = Σ α_t H_t."""

    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        scores = self.w(x).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, d_model)


def get_pooling(kind: str, d_model: int) -> nn.Module:
    if kind == "mean":
        return MeanPooling()
    if kind == "attention":
        return AttentionPooling(d_model)
    raise ValueError(f"Unknown pooling: {kind}")
