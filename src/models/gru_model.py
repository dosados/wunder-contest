from __future__ import annotations
import torch
import torch.nn as nn
from constants import gru_constants


class GRUModel(nn.Module):

    def __init__(self):
        super().__init__()
        cfg = gru_constants
        self.input_dim = cfg["input_dim"]
        self.linear_dim = cfg.get("linear_dim", self.input_dim)
        self.gru_hidden = cfg["gru_hidden"]
        self.gru_num_layers = cfg.get("gru_num_layers", 1)
        self.output_dim = cfg["output_dim"]
        self.input_proj = (
            nn.Linear(self.input_dim, self.linear_dim)
            if self.linear_dim != self.input_dim
            else nn.Identity()
        )
        self.gru = nn.GRU(
            self.linear_dim,
            self.gru_hidden,
            self.gru_num_layers,
            batch_first=True,
            bias=True,
        )
        self.head0 = nn.Linear(self.gru_hidden, 1)
        self.head1 = nn.Linear(self.gru_hidden, 1)
        self.residual_proj = nn.Linear(self.input_dim, self.output_dim)
        self._h = None
        self._last_seq_ix = None

    def reset_state(self):
        self._h = None
        self._last_seq_ix = None

    def forward(self, x, seq_ix=None):
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix
        residual = self.residual_proj(x)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze = True
        else:
            x = x.unsqueeze(1)
            squeeze = False
        x = self.input_proj(x)
        if self._h is None:
            self._h = torch.zeros(
                self.gru_num_layers,
                x.shape[0],
                self.gru_hidden,
                device=x.device,
                dtype=x.dtype,
            )
        out, self._h = self.gru(x, self._h)
        out = (
            torch.cat([self.head0(out.squeeze(1)), self.head1(out.squeeze(1))], dim=-1)
            + residual
        )
        return out.squeeze(0) if squeeze else out

    def forward_sequence(self, x):
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        residual = self.residual_proj(x)
        out, _ = self.gru(self.input_proj(x))
        out = torch.cat([self.head0(out), self.head1(out)], dim=-1) + residual
        return out.squeeze(0) if squeeze else out
