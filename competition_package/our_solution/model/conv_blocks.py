"""
Temporal Convolution Block: residual Conv1D с dilation.
Conv1D(kernel, dilation=k) → GELU → LayerNorm → Conv1D(kernel, dilation=2k) → Residual.
"""

import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    """Один residual Conv-блок с двумя свёртками и разными dilation."""

    def __init__(self, d_model: int, kernel_size: int, dilation1: int, dilation2: int):
        super().__init__()
        pad1 = (kernel_size - 1) * dilation1 // 2
        pad2 = (kernel_size - 1) * dilation2 // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=pad1, dilation=dilation1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=pad2, dilation=dilation2)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        residual = x
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.conv1(x)
        x = x.transpose(1, 2)  # (B, T, d_model)
        x = self.act(x)
        x = self.norm(x)
        residual2 = x
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)
        return x + residual2


class TemporalConvBackbone(nn.Module):
    """Стек из нескольких TemporalConvBlock с заданными dilation."""

    def __init__(self, d_model: int, kernel_size: int, dilations: tuple):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TemporalConvBlock(d_model, kernel_size, d, d * 2) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
