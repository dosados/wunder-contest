"""
Полная модель на базе Mamba-like SSM для предсказания t0, t1.
Интерфейс совместим с model.FullModel:
  - forward_sequence(x) -> (B, T, 2) для обучения;
  - forward(x, seq_ix=None) — streaming: один вектор (D,) или (B, D), хранит состояние между вызовами,
    при смене seq_ix состояние сбрасывается.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .mamba_block import MambaBlock
from .model_state import SSMModelState, MambaBlockState

try:
    from constants import INPUT_DIM, TARGET_DIM
except ImportError:
    INPUT_DIM = 32
    TARGET_DIM = 2

try:
    from constants import ssm_constants
except (ImportError, AttributeError):
    ssm_constants = {}


def _get_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    defaults = {
        "input_dim": INPUT_DIM,
        "d_model": 64,
        "n_layers": 2,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dt_rank": None,
        "output_dim": TARGET_DIM,
    }
    defaults.update(ssm_constants)
    if config:
        defaults.update(config)
    return defaults


class SSMModel(nn.Module):
    """
    Mamba-style SSM: input_proj -> стек MambaBlock -> два головных слоя (t0, t1) + residual.
    Вход: (B, T, input_dim), выход: (B, T, output_dim).
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = _get_config(config)
        self.input_dim = cfg["input_dim"]
        self.d_model = cfg["d_model"]
        self.n_layers = cfg["n_layers"]
        self.output_dim = cfg["output_dim"]

        self.input_proj = nn.Linear(cfg["input_dim"], self.d_model)

        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=self.d_model,
                d_state=cfg["d_state"],
                d_conv=cfg["d_conv"],
                expand=cfg["expand"],
                dt_rank=cfg.get("dt_rank"),
                layer_idx=i,
            )
            for i in range(self.n_layers)
        ])

        self.head0 = nn.Linear(self.d_model, 1)
        self.head1 = nn.Linear(self.d_model, 1)
        self.residual_proj = nn.Linear(cfg["input_dim"], self.output_dim)

        self._d_conv = cfg["d_conv"]
        self._d_inner = int(cfg["expand"] * self.d_model)
        self._d_state = cfg["d_state"]
        self._streaming_state: Optional[SSMModelState] = None
        self._last_seq_ix: Optional[int] = None

    def _get_state(self, device: torch.device) -> SSMModelState:
        if self._streaming_state is None:
            self._streaming_state = SSMModelState([
                MambaBlockState(
                    self._d_conv, self._d_inner, self._d_state, device
                )
                for _ in range(self.n_layers)
            ])
        return self._streaming_state

    def reset_state(self) -> None:
        """Сброс состояния при смене последовательности (новый seq_ix)."""
        if self._streaming_state is not None:
            self._streaming_state.reset()
        self._last_seq_ix = None

    def forward(
        self,
        x: torch.Tensor,
        seq_ix: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Streaming: один шаг или несколько подряд в одной последовательности.
        x: (input_dim,) один шаг или (B, input_dim) — B шагов подряд.
        seq_ix: при передаче состояние сбрасывается при смене значения (новая последовательность).
        Возвращает (output_dim,) или (B, output_dim).
        """
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix

        device = x.device
        state = self._get_state(device)
        residual = self.residual_proj(x)
        if x.dim() == 1:
            x_proj = self.input_proj(x.unsqueeze(0)).squeeze(0)
            for block, block_state in zip(self.blocks, state.block_states):
                x_proj = block.forward_step(x_proj.unsqueeze(0), block_state).squeeze(0)
            out = torch.stack([
                self.head0(x_proj.unsqueeze(0)).squeeze(0).squeeze(-1),
                self.head1(x_proj.unsqueeze(0)).squeeze(0).squeeze(-1),
            ], dim=-1)
            return out + residual

        outs = []
        for i in range(x.shape[0]):
            xi = x[i]
            ri = residual[i]
            xi_proj = self.input_proj(xi.unsqueeze(0)).squeeze(0)
            for block, block_state in zip(self.blocks, state.block_states):
                xi_proj = block.forward_step(xi_proj.unsqueeze(0), block_state).squeeze(0)
            out_i = torch.stack([
                self.head0(xi_proj.unsqueeze(0)).squeeze(0).squeeze(-1),
                self.head1(xi_proj.unsqueeze(0)).squeeze(0).squeeze(-1),
            ], dim=-1)
            outs.append(out_i + ri)
        return torch.stack(outs, dim=0)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Один проход по всей последовательности (обучение / батч-инференс).
        x: (B, T, input_dim) или (T, input_dim)
        Возвращает (B, T, output_dim) или (T, output_dim).
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        residual = self.residual_proj(x)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        out0 = self.head0(x)
        out1 = self.head1(x)
        out = torch.cat([out0, out1], dim=-1) + residual

        if squeeze:
            out = out.squeeze(0)
        return out
