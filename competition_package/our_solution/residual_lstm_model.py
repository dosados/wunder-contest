"""
Вторая LSTM для стека: обучается на остатках первой LSTM (аналог SSM в gym_ssm).
Интерфейс совместим с ssm_model.SSMModel и model.FullModel:
  - forward_sequence(x) -> (B, T, 2) для обучения;
  - forward(x, seq_ix=None) — streaming: один шаг или батч шагов, состояние сбрасывается при смене seq_ix.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

try:
    from constants import INPUT_DIM, TARGET_DIM
except ImportError:
    INPUT_DIM = 32
    TARGET_DIM = 2

try:
    from constants import lstm2_constants
except (ImportError, AttributeError):
    lstm2_constants = {}


def _get_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    defaults = {
        "input_dim": INPUT_DIM,
        "linear_dim": 64,
        "lstm_hidden": 128,
        "lstm_num_layers": 2,
        "output_dim": TARGET_DIM,
    }
    defaults.update(lstm2_constants)
    if config:
        defaults.update(config)
    return defaults


class ResidualLSTM(nn.Module):
    """
    LSTM без Conv: input_proj -> LSTM -> два головных слоя (t0, t1) + residual_proj.
    Вход: (B, T, input_dim), выход: (B, T, output_dim).
    Используется как вторая модель стека на остатках первой LSTM.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = _get_config(config)
        self.input_dim = cfg["input_dim"]
        self.linear_dim = cfg["linear_dim"]
        self.lstm_hidden = cfg["lstm_hidden"]
        self.num_layers = cfg["lstm_num_layers"]
        self.output_dim = cfg["output_dim"]

        self.input_proj = nn.Linear(self.input_dim, self.linear_dim)
        self.lstm = nn.LSTM(
            input_size=self.linear_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
        )
        self.head0 = nn.Linear(self.lstm_hidden, 1)
        self.head1 = nn.Linear(self.lstm_hidden, 1)
        self.residual_proj = nn.Linear(self.input_dim, self.output_dim)

        self._h: Optional[torch.Tensor] = None
        self._c: Optional[torch.Tensor] = None
        self._last_seq_ix: Optional[int] = None

    def reset_state(self) -> None:
        """Сброс состояния при смене последовательности (новый seq_ix)."""
        self._h = None
        self._c = None
        self._last_seq_ix = None

    def forward(
        self,
        x: torch.Tensor,
        seq_ix: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Streaming: один шаг или несколько подряд в одной последовательности.
        x: (input_dim,) один шаг или (B, input_dim) — B шагов подряд.
        seq_ix: при передаче состояние сбрасывается при смене значения.
        Возвращает (output_dim,) или (B, output_dim).
        """
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix

        residual = self.residual_proj(x)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
            squeeze = True
        else:
            x = x.unsqueeze(1)  # (B, 1, input_dim)
            squeeze = False

        x_proj = self.input_proj(x)
        device = x_proj.device
        dtype = x_proj.dtype
        batch = x_proj.shape[0]
        if self._h is None or self._c is None:
            self._h = torch.zeros(
                self.num_layers, batch, self.lstm_hidden,
                device=device, dtype=dtype,
            )
            self._c = torch.zeros(
                self.num_layers, batch, self.lstm_hidden,
                device=device, dtype=dtype,
            )
        out, (self._h, self._c) = self.lstm(x_proj, (self._h, self._c))
        out = out.squeeze(1)  # (B, lstm_hidden)
        head = torch.cat([self.head0(out), self.head1(out)], dim=-1)  # (B, 2)
        result = head + residual
        if squeeze:
            result = result.squeeze(0)
        return result

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
        out, _ = self.lstm(x)
        head = torch.cat([self.head0(out), self.head1(out)], dim=-1)
        result = head + residual

        if squeeze:
            result = result.squeeze(0)
        return result
