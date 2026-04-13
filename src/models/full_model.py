from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from constants import convo_constants
from models.conv_blocks import StreamingTemporalConvModel
from models.configurable_blocks import build_input_projection, merge_block_cfg
from models.configurable_blocks import ConfigurableLinearBlock
from models.lstm_blocks import StreamingLSTM


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


def _build_pre_lstm_trunk(
    conv_dim: int,
    pre_stack: dict[str, Any] | None,
    global_block: dict[str, Any] | None,
) -> tuple[nn.Module, int]:
    return _build_post_trunk(conv_dim, pre_stack, global_block)


class FullModel(nn.Module):

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg = dict(convo_constants)
        if config:
            cfg.update(config)
        self._cfg = cfg
        gb = cfg.get("block")
        self.input_dim = cfg["input_dim"]
        self.linear1_dim = cfg["linear1_dim"]
        self.lstm_hidden = cfg["lstm_hidden"]
        self.output_dim = cfg["output_dim"]
        self.input_proj = build_input_projection(
            self.input_dim, self.linear1_dim, cfg.get("input_stack"), gb
        )
        self.conv_model = StreamingTemporalConvModel(cfg)
        self.pre_lstm_trunk, lstm_in = _build_pre_lstm_trunk(
            cfg["conv_dim"], cfg.get("pre_lstm_stack"), gb
        )
        lstm_cfg = dict(cfg)
        lstm_cfg["lstm_input_size"] = lstm_in
        self.lstm0 = StreamingLSTM(lstm_cfg)
        self.lstm1 = StreamingLSTM(lstm_cfg)
        self.post_lstm_trunk, head_in = _build_post_trunk(
            self.lstm_hidden, cfg.get("post_lstm_stack"), gb
        )
        self.linear0 = nn.Linear(head_in, 1)
        self.linear1 = nn.Linear(head_in, 1)
        self.residual_proj = nn.Linear(self.input_dim, self.output_dim)
        self._h0 = self._c0 = self._h1 = self._c1 = None
        self._last_seq_ix = None

    def reset_state(self):
        self.conv_model.reset_state()
        self._h0 = self._c0 = self._h1 = self._c1 = None

    def forward(self, x: torch.Tensor, seq_ix: int | None = None):
        if seq_ix is not None and seq_ix != self._last_seq_ix:
            self.reset_state()
            self._last_seq_ix = seq_ix
        residual = self.residual_proj(x)
        if x.dim() == 1:
            x_conv = self.conv_model(self.input_proj(x))
            x_conv = self.pre_lstm_trunk(x_conv)
            o0, self._h0, self._c0 = self.lstm0(x_conv, self._h0, self._c0)
            o1, self._h1, self._c1 = self.lstm1(x_conv, self._h1, self._c1)
            o0 = self.post_lstm_trunk(o0)
            o1 = self.post_lstm_trunk(o1)
            return torch.cat([self.linear0(o0), self.linear1(o1)], dim=-1) + residual
        outs = []
        for i in range(x.shape[0]):
            xi = x[i]
            ri = residual[i]
            x_conv = self.conv_model(self.input_proj(xi))
            x_conv = self.pre_lstm_trunk(x_conv)
            o0, self._h0, self._c0 = self.lstm0(x_conv, self._h0, self._c0)
            o1, self._h1, self._c1 = self.lstm1(x_conv, self._h1, self._c1)
            o0 = self.post_lstm_trunk(o0)
            o1 = self.post_lstm_trunk(o1)
            outs.append(torch.cat([self.linear0(o0), self.linear1(o1)], dim=-1) + ri)
        return torch.stack(outs, dim=0)

    def forward_sequence(self, x: torch.Tensor):
        residual = self.residual_proj(x)
        x = self.conv_model.forward_sequence(self.input_proj(x))
        x = self.pre_lstm_trunk(x)
        out0 = self.lstm0.forward_sequence(x)
        out1 = self.lstm1.forward_sequence(x)
        out0 = self.post_lstm_trunk(out0)
        out1 = self.post_lstm_trunk(out1)
        return torch.cat([self.linear0(out0), self.linear1(out1)], dim=-1) + residual
