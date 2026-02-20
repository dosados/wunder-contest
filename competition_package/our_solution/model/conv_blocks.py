"""
Блок свёрток по времени (1D Conv over time) для архитектуры Conv + Transformer.
Извлекает локальные временные паттерны из последовательности [B, T, d_model].
Свёртка выполняется по оси времени (T).
"""

import torch
import torch.nn as nn
from typing import List, Optional


class TemporalConvBlock(nn.Module):
    """
    Один блок: Conv1D по времени + GELU + residual (если размерность совпадает).
    Вход/выход по каналам: d_model.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        dilation: int = 1,
        residual: bool = True,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # для сохранения длины при causal-подобной свёртке
        # padding слева, чтобы не заглядывать в будущее (causal)
        self.pad_left = padding
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation)
        self.activation = nn.GELU()
        self.residual = residual and (kernel_size > 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> (B, d_model, T)
        x_t = x.transpose(1, 2)
        if self.pad_left > 0:
            x_t = nn.functional.pad(x_t, (self.pad_left, 0), mode="constant", value=0)
        out = self.conv(x_t)
        out = self.activation(out)
        # при pad_left = (kernel_size-1)*dilation выходная длина = T
        # (B, d_model, T) -> (B, T, d_model)
        out = out.transpose(1, 2)
        if self.residual:
            out = out + x
        return out


class TemporalConvStack(nn.Module):
    """
    Несколько слоёв 1D свёрток по времени (kernel 3–5, с dilation, residual, GELU).
    Вход: [B, T, d_model], выход: [B, T, d_model].
    Подходит для использования после линейной проекции и перед Transformer.
    """

    def __init__(
        self,
        d_model: int,
        kernel_sizes: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        residual: bool = True,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes or [5, 3, 3]
        dilations = dilations or [1, 2, 1]
        if len(dilations) != len(kernel_sizes):
            dilations = [1] * len(kernel_sizes)
        self.blocks = nn.ModuleList([
            TemporalConvBlock(d_model, kernel_size=k, dilation=d, residual=residual)
            for k, d in zip(kernel_sizes, dilations)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model) — эмбеддинги после линейной проекции.
        Возвращает: (B, T, d_model) — признаки после свёрток по времени.
        """
        for block in self.blocks:
            x = block(x)
        return x
