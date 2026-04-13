from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from constants import gru_constants
from models.configurable_blocks import build_input_projection, merge_block_cfg
from models.configurable_blocks import ConfigurableLinearBlock


def _build_post_trunk(
    in_dim: int,
    post_stack: dict[str, Any] | None,
    global_block: dict[str, Any] | None,
) -> tuple[nn.Module, int]:
    if not post_stack:
        return nn.Identity(), in_dim
    layer_dims = list(post_stack.get("hidden_dims") or [])
    if not layer_dims:
        return nn.Identity(), in_dim
    cfg = merge_block_cfg(post_stack, global_block)
    dims = [in_dim] + layer_dims
    blocks = [
        ConfigurableLinearBlock(dims[i], dims[i + 1], cfg)
        for i in range(len(dims) - 1)
    ]
    trunk = nn.Sequential(*blocks) if len(blocks) > 1 else blocks[0]
    return trunk, layer_dims[-1]


class GRUModel(nn.Module):

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        base = dict(gru_constants)
        if config:
            base.update(config)
        self._cfg = base
        gb = base.get("block")
        self.input_dim = base["input_dim"]
        self.linear_dim = base.get("linear_dim", self.input_dim)
        self.gru_hidden = base["gru_hidden"]
        self.gru_num_layers = base.get("gru_num_layers", 1)
        self.output_dim = base["output_dim"]
        self.input_proj = build_input_projection(
            self.input_dim, self.linear_dim, base.get("input_stack"), gb
        )
        self.gru = nn.GRU(
            self.linear_dim,
            self.gru_hidden,
            self.gru_num_layers,
            batch_first=True,
            bias=True,
        )
        self.post_trunk, head_in = _build_post_trunk(
            self.gru_hidden, base.get("post_stack"), gb
        )
        self.head0 = nn.Linear(head_in, 1)
        self.head1 = nn.Linear(head_in, 1)
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
        out = self.post_trunk(out.squeeze(1))
        out = (
            torch.cat([self.head0(out), self.head1(out)], dim=-1) + residual
        )
        return out.squeeze(0) if squeeze else out

    def forward_sequence(self, x):
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        residual = self.residual_proj(x)
        out, _ = self.gru(self.input_proj(x))
        out = self.post_trunk(out)
        out = torch.cat([self.head0(out), self.head1(out)], dim=-1) + residual
        return out.squeeze(0) if squeeze else out
