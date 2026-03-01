"""
Состояние для streaming-режима SSM: буферы causal conv и скрытые состояния SSM по слоям.
По аналогии с model.model_state.ModelState — кольцевой буфер для conv + состояние h для каждого блока.
"""
from __future__ import annotations

from typing import Any

import torch


class _RingBuffer:
    """Кольцевой буфер для causal conv: хранит последние window векторов размерности feat_dim."""

    def __init__(self, window: int, feat_dim: int, device: torch.device | str = "cpu"):
        self.window = window
        self.feat_dim = feat_dim
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.buffer = torch.zeros(window, feat_dim, device=self.device)
        self.pos = 0
        self.full = False

    def add(self, x: torch.Tensor) -> None:
        self.buffer[self.pos] = x.detach().to(self.device, dtype=self.buffer.dtype)
        self.pos = (self.pos + 1) % self.window
        if self.pos == 0:
            self.full = True

    def get_sequence(self) -> torch.Tensor:
        if not self.full:
            return self.buffer[: self.pos]
        return torch.cat([self.buffer[self.pos :], self.buffer[: self.pos]], dim=0)

    def reset(self) -> None:
        self.buffer.zero_()
        self.pos = 0
        self.full = False


class MambaBlockState:
    """Состояние одного MambaBlock: буфер causal conv и скрытое состояние SSM h."""

    def __init__(self, d_conv: int, d_inner: int, d_state: int, device: torch.device | str = "cpu"):
        self.conv_buffer = _RingBuffer(d_conv, d_inner, device)
        self.h: torch.Tensor | None = None
        self.d_inner = d_inner
        self.d_state = d_state
        self._device = device if isinstance(device, torch.device) else torch.device(device)

    def get_or_create_h(self, dtype: torch.dtype) -> torch.Tensor:
        if self.h is None:
            self.h = torch.zeros(
                self.d_inner, self.d_state, device=self._device, dtype=dtype
            )
        return self.h

    def reset(self) -> None:
        self.conv_buffer.reset()
        self.h = None


class SSMModelState:
    """
    Состояние всей SSM-модели при streaming: по одному MambaBlockState на каждый блок.
    При смене seq_ix вызывается reset() перед началом новой последовательности.
    """

    def __init__(self, block_states: list[MambaBlockState]):
        self.block_states = block_states

    def reset(self) -> None:
        for st in self.block_states:
            st.reset()
